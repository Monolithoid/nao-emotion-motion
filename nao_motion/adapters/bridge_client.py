"""Adapter to the NAOqi bridge (bridge/naoqi_bridge.py) over TCP/JSON.

All NAOqi/SDK version constraints stay in the bridge. This client sends a
periodic heartbeat; if the bridge stops hearing it during a movement, it stops
and holds the measured posture.

Drops: no call blocks for more than `timeout_s` (socket AND the lock shared by the
recording loop, the heartbeat and the interface requests). Any I/O error closes the
connection and raises `ConnectionLost`; the cause is kept in `lost_reason` (shown by
the interface) until the reconnection, which happens without restarting the server.
"""
from __future__ import annotations

import json
import socket
import threading
from typing import Optional
import time

import numpy as np

from .base import AdapterError, ConnectionLost, RobotAdapter

PROTOCOL_VERSION = 1


class BridgeAdapter(RobotAdapter):
    name = "bridge"

    def __init__(self, cfg, host=None, port=None, timeout_s: Optional[float] = None):
        super().__init__(cfg)
        b = cfg.robot.get("bridge", {})
        self.host = host or b.get("host", "127.0.0.1")
        self.port = int(port or b.get("port", 9600))
        self.heartbeat_s = float(b.get("heartbeat_s", 0.2))
        # bridge response timeout: the NAO's Wi-Fi has latency spikes; a late response
        # would desynchronise the stream, so the connection is then closed cleanly
        self.timeout_s = float(timeout_s if timeout_s is not None else b.get("timeout_s", 3.0))
        self._sock = None
        self._file = None
        self._lock = threading.Lock()
        self._hb_thread = None
        self._alive = False
        self._gen = 0              # connection number: an old heartbeat stops by itself
        self.info = {}
        self.lost_reason: Optional[str] = None  # cause of the last drop (None if connected)
        self.lost_at: Optional[float] = None

    @property
    def is_real(self):
        return bool(self.info.get("real", True))

    def connect(self):
        self._close()  # releases an old connection: the bridge serves only one client at a time
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        except OSError as e:
            raise ConnectionLost(f"Pont injoignable sur {self.host}:{self.port} : {e}") from e
        self._file = self._sock.makefile("rwb")
        self._alive = True
        self._gen += 1
        try:
            self.info = self._call("hello")
        except ConnectionLost as e:
            # the bridge accepts the connection but does not answer (e.g. still busy with a
            # dead connection, older bridge version)
            raise ConnectionLost(f"Le pont {self.host}:{self.port} ne répond pas : {e}") from e
        if self.info.get("protocol") != PROTOCOL_VERSION:
            self.disconnect()
            raise AdapterError(f"Version de protocole du pont incompatible : {self.info.get('protocol')}")
        self.lost_reason = self.lost_at = None
        self._hb_thread = threading.Thread(target=self._hb_loop, args=(self._gen,), daemon=True)
        self._hb_thread.start()

    def _hb_loop(self, gen):
        while self._alive and gen == self._gen:
            time.sleep(self.heartbeat_s)
            if not self._alive or gen != self._gen:
                return
            try:
                self._call("heartbeat")
            except Exception:
                # ConnectionLost: the connection is already closed and the cause recorded;
                # other error: also stop the heartbeat (the bridge will stop the movement)
                return

    def disconnect(self):
        self._close()
        self.lost_reason = self.lost_at = None  # deliberate disconnection: not a drop

    def _lost(self, reason: str) -> ConnectionLost:
        """Closes the dead connection (the bridge, which serves one client, is freed) and records the cause."""
        self._close()
        if self.lost_reason is None:
            self.lost_reason = reason
            self.lost_at = time.time()
        return ConnectionLost(reason)

    def _close(self):
        self._alive = False
        try:
            if self._file:
                self._file.close()  # otherwise the socket stays open as long as the file exists
            if self._sock:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
        except OSError:
            pass
        finally:
            self._sock = None
            self._file = None

    def is_connected(self) -> bool:
        return self._alive and self._sock is not None

    def _call(self, cmd, **kw):
        if not self.is_connected():
            raise ConnectionLost("Pont non connecté")
        msg = (json.dumps(dict(cmd=cmd, **kw)) + "\n").encode()
        # bounded lock: without a bridge response, the holder gives up after
        # timeout_s; waiting longer would be pointless
        if not self._lock.acquire(timeout=self.timeout_s + 1.0):
            raise self._lost(f"Le pont ne répond plus ({self.host}:{self.port}, plus de {self.timeout_s:.0f} s)")
        try:
            f = self._file
            if f is None or not self._alive:  # closed while waiting for the lock
                raise ConnectionLost(self.lost_reason or "Pont non connecté")
            try:
                f.write(msg)
                f.flush()
                line = f.readline()
            except socket.timeout as e:
                # late response: the stream would be desynchronised, close it
                raise self._lost(f"Le pont ne répond plus ({self.host}:{self.port}, délai de "
                                 f"{self.timeout_s:.0f} s dépassé)") from e
            except (OSError, ValueError) as e:  # ValueError: file closed by another thread
                raise self._lost(f"Connexion au pont perdue : {e}") from e
            if not line:
                raise self._lost("Connexion au pont fermée par le pont")
        finally:
            self._lock.release()
        try:
            resp = json.loads(line)
        except ValueError as e:
            raise self._lost(f"Réponse illisible du pont : {e}") from e
        if not resp.get("ok"):
            raise AdapterError(f"Pont : {resp.get('error')}")
        return resp["result"]

    def heartbeat(self):
        self._call("heartbeat")

    def capabilities(self) -> dict:
        return {"adapter": "bridge", **self.info}

    def connect_robot(self, host: str, port: int = 9559) -> dict:
        """Asks the bridge to connect to NAOqi (IP or DNS name). No movement."""
        from ..config.hosts import validate_host
        info = self._call("connect_robot", host=validate_host(host), port=int(port))
        self.info = self._call("hello")
        return info

    @property
    def robot_connected(self) -> bool:
        return bool(self.info.get("robot_connected"))

    def factory_limits(self, joints):
        """Limites d'usine lues via ALMotion.getLimits (aucune commande)."""
        lim = self._call("get_limits", joints=list(joints))
        return {j: [float(x) for x in l] for j, l in zip(joints, lim)}

    def _read(self, joints):
        return np.array(self._call("read_angles", joints=joints), dtype=np.float64)

    def _set_stiffness(self, joints, value):
        self._call("set_stiffness", joints=joints, value=float(value))

    def _send(self, joints, angles, fraction_speed):
        self._call("set_angles", joints=joints, angles=[float(a) for a in angles], speed=float(fraction_speed))

    def _stop(self):
        if self.is_connected():
            self._call("stop")
