# -*- coding: utf-8 -*-
"""Minimal bridge between nao_motion (Python 3) and NAOqi.

This file is deliberately compatible with Python 2.7 AND Python 3 (no
f-strings, no annotations), because the NAOqi Python SDK for NAO is often
limited to Python 2.7 depending on the version. It only depends on the
standard library and the NAOqi SDK.

Two possible locations:
  - ON THE ROBOT (recommended): the `naoqi` module is already present in the
    NAO's Python 2.7, nothing to install. `python naoqi_bridge.py --on-robot`
    connects to NAOqi locally (127.0.0.1) and listens on the network (port 9600).
    This is also ia-agent's choice (embedded nao_ws_bridge.py).
  - on the PC: requires the NAOqi Python SDK (pynaoqi, Python 2.7), which has
    become hard to find and install.

Protocol: TCP, one JSON request per line -> one JSON response per line.
    {"cmd": "read_angles", "joints": [...]} -> {"ok": true, "result": [...]}

Safety:
  - NO movement at startup nor when a client connects.
  - Watchdog: if a movement command was sent and no message (heartbeat or
    command) arrives for --watchdog seconds, the bridge stops the movement
    tasks and holds the MEASURED posture. It does not send the robot back to
    rest and does not relax anything.
  - Legs are refused.

Usage:
    python naoqi_bridge.py --on-robot          # on the NAO (recommended)
    python naoqi_bridge.py                     # on the PC: the robot address is given by the interface
    python naoqi_bridge.py --robot-ip nao.local  # or given at launch (IP or DNS name)
    python naoqi_bridge.py --fake              # without a robot (protocol tests)
"""
from __future__ import print_function

import argparse
import json
import re
import socket
import sys
import threading
import time

LEG_PREFIXES = ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")
PROTOCOL_VERSION = 1
HOST_RE = re.compile(r"^(?=.{1,253}$)([A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?$|^[0-9A-Fa-f:]{2,39}$")


def native_str(x):
    """Python 2: json yields unicode strings, which NAOqi 2.1 (ALProxy) refuses for
    String parameters; they are converted to str (UTF-8 bytes). Python 3: unchanged."""
    if sys.version_info[0] < 3 and isinstance(x, type(u"")):
        return x.encode("utf-8")
    return x


def valid_host(host):
    return isinstance(host, (str, type(u""))) and bool(HOST_RE.match(host))


class FakeBackend(object):
    """Fake robot for testing the protocol (no physics)."""
    real = False

    def __init__(self):
        self.q = {}
        self.stiff = {}
        self.killed = 0

    def info(self):
        return {"backend": "fake", "naoqi_version": None, "robot": None, "robot_name": "fake"}

    def get_limits(self, names):
        return [[-1.0, 1.0, 5.0, 1.0] for _ in names]

    def get_angles(self, names):
        return [self.q.get(n, 0.0) for n in names]

    def set_stiffness(self, names, value):
        for n in names:
            self.stiff[n] = value

    def get_stiffness(self, names):
        return [self.stiff.get(n, 0.0) for n in names]

    def set_angles(self, names, angles, speed):
        for n, a in zip(names, angles):
            self.q[n] = a

    def angle_interpolation(self, names, angles, times):
        for n, a in zip(names, angles):
            self.q[n] = a[-1]

    def stop(self, names):
        self.killed += 1


class NaoqiBackend(object):
    real = True

    def __init__(self, ip, port, api="auto"):
        self.motion = None
        self.system = None
        # Order: `naoqi` (ALProxy) first, present on the robot for NAOqi 2.1 and
        # used by ia-agent on the same robot; `qi` (libqi) as a fallback.
        order = {"auto": ("naoqi", "qi"), "naoqi": ("naoqi",), "qi": ("qi",)}[api]
        last = None
        for name in order:
            try:
                if name == "naoqi":
                    from naoqi import ALProxy  # SDK historique
                    self.motion = ALProxy("ALMotion", ip, port)
                    try:
                        self.system = ALProxy("ALSystem", ip, port)
                    except Exception:
                        self.system = None
                else:
                    import qi  # libqi
                    session = qi.Session()
                    session.connect("tcp://%s:%d" % (ip, port))
                    self.motion = session.service("ALMotion")
                    try:
                        self.system = session.service("ALSystem")
                    except Exception:
                        self.system = None
                self.api = name
                return
            except ImportError as e:
                last = e
        raise RuntimeError("aucun module NAOqi (naoqi ou qi) dans ce Python : %s. "
                           "Lancer le pont sur le robot (--on-robot)." % last)

    def info(self):
        version = None
        robot = None
        try:
            version = self.system.systemVersion() if self.system else None
        except Exception:
            pass
        try:
            robot = self.motion.getRobotConfig()
        except Exception:
            pass
        name = None
        try:
            name = self.system.robotName() if self.system else None
        except Exception:
            pass
        return {"backend": self.api, "naoqi_version": version, "robot": robot, "robot_name": name}

    def get_limits(self, names):
        # ALMotion.getLimits(nom) -> [[min, max, vitesse max, couple max]] (lecture seule)
        out = []
        for n in names:
            lim = self.motion.getLimits(n)
            out.append([float(x) for x in lim[0]])
        return out

    def get_angles(self, names):
        return list(self.motion.getAngles(names, True))  # True = capteurs

    def set_stiffness(self, names, value):
        self.motion.setStiffnesses(names, value)

    def get_stiffness(self, names):
        return list(self.motion.getStiffnesses(names))

    def set_angles(self, names, angles, speed):
        self.motion.setAngles(names, angles, speed)

    def angle_interpolation(self, names, angles, times):
        # non-blocking; interrupted by stop()
        if self.api == "qi":
            # libqi fallback: `_async` only exists on recent versions of qi
            self.motion.angleInterpolation(names, angles, times, True, _async=True)
        else:
            self.motion.post.angleInterpolation(names, angles, times, True)

    def stop(self, names):
        try:
            self.motion.killTasksUsingResources(names)
        except Exception:
            pass
        current = self.get_angles(names)
        # hold the measured posture, without changing stiffness
        self.motion.setAngles(names, current, 0.1)


class Bridge(object):
    """backend can be None: the robot address (IP or DNS name) is then provided
    later by the interface through the connect_robot command."""

    def __init__(self, backend, watchdog_s, factory=None):
        self.backend = backend
        self.factory = factory or (lambda host, port: NaoqiBackend(host, port))
        self.robot_address = None
        self.watchdog_s = watchdog_s
        self.lock = threading.Lock()
        self.last_msg = time.time()
        self.motion_active = False
        self.commanded = set()
        self.watchdog_trips = 0

    def check_names(self, names):
        for n in names:
            if n.startswith(LEG_PREFIXES):
                raise ValueError("jambes hors perimetre : %s" % n)

    def handle(self, req):
        cmd = req.get("cmd")
        with self.lock:
            self.last_msg = time.time()
            if cmd == "hello":
                info = self.backend.info() if self.backend is not None else {"backend": None}
                info["protocol"] = PROTOCOL_VERSION
                info["real"] = self.backend.real if self.backend is not None else True
                info["robot_connected"] = self.backend is not None
                info["robot_address"] = self.robot_address
                info["watchdog_s"] = self.watchdog_s
                return info
            if cmd == "heartbeat":
                return {"watchdog_trips": self.watchdog_trips}
            if cmd == "connect_robot":
                # NAOqi connection only: no movement, no stiffness change
                if self.motion_active:
                    raise ValueError("mouvement en cours : arreter avant de changer de robot")
                host, port = req.get("host"), int(req.get("port") or 9559)
                if not valid_host(host):
                    raise ValueError("adresse invalide : %r (IP ou nom DNS attendu)" % (host,))
                if not 0 < port < 65536:
                    raise ValueError("port invalide")
                self.backend = self.factory(native_str(host), port)
                self.robot_address = "%s:%d" % (host, port)
                self.commanded = set()
                info = self.backend.info()
                info["robot_address"] = self.robot_address
                return info
            if self.backend is None:
                raise ValueError("robot non connecte : indiquer son adresse (IP ou DNS) dans l'interface")
            names = [native_str(n) for n in req.get("joints", [])]
            self.check_names(names)
            if cmd == "get_limits":
                return self.backend.get_limits(names)
            if cmd == "read_angles":
                return self.backend.get_angles(names)
            if cmd == "get_stiffness":
                return self.backend.get_stiffness(names)
            if cmd == "set_stiffness":
                self.backend.set_stiffness(names, float(req["value"]))
                return True
            if cmd == "set_angles":
                self.backend.set_angles(names, [float(a) for a in req["angles"]], float(req["speed"]))
                self.motion_active = True
                self.commanded.update(names)
                return True
            if cmd == "angle_interpolation":
                self.backend.angle_interpolation(names, req["angles"], req["times"])
                self.motion_active = True
                self.commanded.update(names)
                return True
            if cmd == "stop":
                self._stop()
                return True
            raise ValueError("commande inconnue : %s" % cmd)

    def _stop(self):
        if self.commanded:
            self.backend.stop(sorted(self.commanded))
        self.motion_active = False

    def watchdog_loop(self):
        while True:
            time.sleep(self.watchdog_s / 4.0)
            with self.lock:
                if self.motion_active and time.time() - self.last_msg > self.watchdog_s:
                    self.watchdog_trips += 1
                    sys.stderr.write("[bridge] watchdog: silent client, stop and hold\n")
                    try:
                        self._stop()
                    except Exception as e:
                        sys.stderr.write("[bridge] stop failed: %s\n" % e)


def _set_keepalive(conn):
    """Short TCP keepalive when the system allows it (the NAO's Linux): a link
    dropped without a clean close is detected even without traffic."""
    try:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt, val in (("TCP_KEEPIDLE", 2), ("TCP_KEEPINTVL", 1), ("TCP_KEEPCNT", 3)):
            if hasattr(socket, opt):
                conn.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
    except Exception:
        pass


def _serve_client(bridge, conn, state, idle_s):
    """Serves a client until it disconnects, goes silent (> idle_s) or a new client
    arrives (and replaces it)."""
    conn.settimeout(idle_s)
    _set_keepalive(conn)
    f = conn.makefile("rwb") if sys.version_info[0] >= 3 else conn.makefile("rw")
    try:
        for line in f:
            if state.get("current") is not conn:
                break  # replaced by a new client
            if not line.strip():
                continue
            try:
                req = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
                resp = {"ok": True, "result": bridge.handle(req)}
            except Exception as e:
                resp = {"ok": False, "error": str(e)}
            out = json.dumps(resp) + "\n"
            f.write(out.encode("utf-8") if sys.version_info[0] >= 3 else out)
            f.flush()
    except socket.timeout:
        sys.stderr.write("[bridge] client silent for %.0f s: connection closed\n" % idle_s)
    except Exception:
        pass
    finally:
        for c in (f, conn):
            try:
                c.close()
            except Exception:
                pass
    # client disconnected: if a movement was in progress, stop and hold. A replaced
    # client touches nothing: the stop was done when the new client arrived.
    with bridge.lock:
        if state.get("current") is not conn:
            return
        state["current"] = None
        if bridge.motion_active:
            sys.stderr.write("[bridge] client disconnected during a movement: stop and hold\n")
            try:
                bridge._stop()
            except Exception as e:
                sys.stderr.write("[bridge] stop failed: %s\n" % e)


def serve(bridge, host, port, ready_event=None, idle_s=10.0):
    """Only one client is served at a time; a NEW client replaces the old one. After a
    network drop, the old connection may stay "open" on the robot side (no
    close received): without this replacement, the bridge would stay busy and the PC
    could no longer reconnect. A client silent for more than idle_s seconds (the client
    sends a heartbeat every 0.2 s) is also disconnected."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    t = threading.Thread(target=bridge.watchdog_loop)
    t.daemon = True
    t.start()
    if ready_event is not None:
        ready_event.port = srv.getsockname()[1]
        ready_event.set()
    state = {"current": None}
    while True:
        conn, addr = srv.accept()
        with bridge.lock:
            old = state.get("current")
            state["current"] = conn
            if old is not None:
                sys.stderr.write("[bridge] new client %s: the old connection is closed\n" % (addr,))
                if bridge.motion_active:  # the old client was driving: stop and hold
                    try:
                        bridge._stop()
                    except Exception as e:
                        sys.stderr.write("[bridge] stop failed: %s\n" % e)
        if old is not None:
            try:
                old.shutdown(socket.SHUT_RDWR)  # unblocks its pending read
            except Exception:
                pass
        th = threading.Thread(target=_serve_client, args=(bridge, conn, state, idle_s))
        th.daemon = True
        th.start()


def read_env(path):
    """Minimal reader for a .env file (NAO_IP, NAO_PORT, NAO_BRIDGE_PORT, ...)."""
    out = {}
    try:
        f = open(path)
    except IOError:
        return out
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.split(" #", 1)[0].strip().strip('"').strip("'")
    f.close()
    return out


def main():
    import os
    env = read_env(os.environ.get("NAO_MOTION_ENV",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")))
    env.update(dict((k, v) for k, v in os.environ.items() if k.startswith("NAO_")))
    p = argparse.ArgumentParser()
    p.add_argument("--robot-ip", default=env.get("NAO_IP") or None)
    p.add_argument("--robot-port", type=int, default=int(env.get("NAO_PORT") or 9559))
    p.add_argument("--host", default=env.get("NAO_BRIDGE_HOST") or "127.0.0.1")
    p.add_argument("--port", type=int, default=int(env.get("NAO_BRIDGE_PORT") or 9600))
    p.add_argument("--watchdog", type=float, default=float(env.get("NAO_BRIDGE_WATCHDOG_S") or 0.6))
    p.add_argument("--idle", type=float, default=float(env.get("NAO_BRIDGE_IDLE_S") or 10.0),
                   help="client muet plus longtemps (s) : connexion fermee, le pont se libere")
    p.add_argument("--api", choices=("auto", "naoqi", "qi"), default=env.get("NAO_BRIDGE_API") or "auto")
    p.add_argument("--on-robot", action="store_true",
                   help="pont lance sur le NAO : NAOqi en local (127.0.0.1), ecoute sur le reseau (0.0.0.0)")
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()
    if a.on_robot:
        a.robot_ip = a.robot_ip or "127.0.0.1"
        if a.host == "127.0.0.1":
            a.host = "0.0.0.0"
    factory = (lambda host, port: FakeBackend()) if a.fake else (lambda host, port: NaoqiBackend(host, port, a.api))
    bridge = Bridge(None, a.watchdog, factory)
    if a.robot_ip:  # optional: otherwise the address is given by the interface
        bridge.handle({"cmd": "connect_robot", "host": a.robot_ip, "port": a.robot_port})
    print("[bridge] ready on %s:%d, robot: %s. No movement is sent at startup."
          % (a.host, a.port, bridge.robot_address or "address to be given by the interface"))
    serve(bridge, a.host, a.port, idle_s=a.idle)


if __name__ == "__main__":
    main()
