# -*- coding: utf-8 -*-
"""Pont minimal entre nao_motion (Python 3) et NAOqi.

Ce fichier est volontairement compatible Python 2.7 ET Python 3 (pas de
f-strings, pas d'annotations), car le SDK Python de NAOqi pour NAO est
souvent limité à Python 2.7 selon la version. Il ne dépend que de la
bibliothèque standard et du SDK NAOqi.

Il peut tourner sur le PC (connexion au robot par IP) ou sur le robot.

Protocole : TCP, une requête JSON par ligne -> une réponse JSON par ligne.
    {"cmd": "read_angles", "joints": [...]} -> {"ok": true, "result": [...]}

Sécurité :
  - AUCUN mouvement au démarrage ni à la connexion d'un client.
  - Chien de garde : si une commande de mouvement a été envoyée et qu'aucun
    message (battement ou commande) n'arrive pendant --watchdog secondes, le
    pont arrête les tâches de mouvement et maintient la posture MESURÉE. Il
    ne renvoie pas le robot au repos et ne relâche rien.
  - Les jambes sont refusées.

Usage :
    python naoqi_bridge.py                     # l'adresse du robot est donnée par l'interface
    python naoqi_bridge.py --robot-ip nao.local  # ou fournie au lancement (IP ou nom DNS)
    python naoqi_bridge.py --fake              # sans robot (tests du protocole)
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


def valid_host(host):
    return isinstance(host, (str, type(u""))) and bool(HOST_RE.match(host))


class FakeBackend(object):
    """Robot factice pour tester le protocole (aucune physique)."""
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

    def __init__(self, ip, port):
        self.motion = None
        self.system = None
        try:
            import qi  # NAOqi 2.x (libqi)
            session = qi.Session()
            session.connect("tcp://%s:%d" % (ip, port))
            self.motion = session.service("ALMotion")
            try:
                self.system = session.service("ALSystem")
            except Exception:
                self.system = None
            self.api = "qi"
        except ImportError:
            from naoqi import ALProxy  # SDK historique
            self.motion = ALProxy("ALMotion", ip, port)
            try:
                self.system = ALProxy("ALSystem", ip, port)
            except Exception:
                self.system = None
            self.api = "naoqi"

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
        # non bloquant ; interrompu par stop()
        if self.api == "qi":
            self.motion.angleInterpolation(names, angles, times, True, _async=True)
        else:
            self.motion.post.angleInterpolation(names, angles, times, True)

    def stop(self, names):
        try:
            self.motion.killTasksUsingResources(names)
        except Exception:
            pass
        current = self.get_angles(names)
        # maintien de la posture mesurée, sans changer la raideur
        self.motion.setAngles(names, current, 0.1)


class Bridge(object):
    """backend peut être None : l'adresse du robot (IP ou nom DNS) est alors fournie
    plus tard par l'interface via la commande connect_robot."""

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
                # connexion NAOqi seulement : aucun mouvement, aucune raideur modifiée
                if self.motion_active:
                    raise ValueError("mouvement en cours : arreter avant de changer de robot")
                host, port = req.get("host"), int(req.get("port") or 9559)
                if not valid_host(host):
                    raise ValueError("adresse invalide : %r (IP ou nom DNS attendu)" % (host,))
                if not 0 < port < 65536:
                    raise ValueError("port invalide")
                self.backend = self.factory(host, port)
                self.robot_address = "%s:%d" % (host, port)
                self.commanded = set()
                info = self.backend.info()
                info["robot_address"] = self.robot_address
                return info
            if self.backend is None:
                raise ValueError("robot non connecte : indiquer son adresse (IP ou DNS) dans l'interface")
            names = req.get("joints", [])
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
                    sys.stderr.write("[bridge] chien de garde : client muet, arret et maintien\n")
                    try:
                        self._stop()
                    except Exception as e:
                        sys.stderr.write("[bridge] echec arret : %s\n" % e)


def serve(bridge, host, port, ready_event=None):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    t = threading.Thread(target=bridge.watchdog_loop)
    t.daemon = True
    t.start()
    if ready_event is not None:
        ready_event.port = srv.getsockname()[1]
        ready_event.set()
    while True:
        conn, _ = srv.accept()
        f = conn.makefile("rwb") if sys.version_info[0] >= 3 else conn.makefile("rw")
        try:
            for line in f:
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
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        # client déconnecté : si un mouvement était en cours, arrêt et maintien
        with bridge.lock:
            if bridge.motion_active:
                sys.stderr.write("[bridge] client deconnecte pendant un mouvement : arret et maintien\n")
                bridge._stop()


def read_env(path):
    """Lecture minimale d'un fichier .env (NAO_IP, NAO_PORT, NAO_BRIDGE_PORT, ...)."""
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
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()
    factory = (lambda host, port: FakeBackend()) if a.fake else None
    bridge = Bridge(None, a.watchdog, factory)
    if a.robot_ip:  # optionnel : sinon l'adresse est donnée par l'interface
        bridge.handle({"cmd": "connect_robot", "host": a.robot_ip, "port": a.robot_port})
    print("[bridge] pret sur %s:%d, robot : %s. Aucun mouvement n'est envoye au demarrage."
          % (a.host, a.port, bridge.robot_address or "adresse a fournir par l'interface"))
    serve(bridge, a.host, a.port)


if __name__ == "__main__":
    main()
