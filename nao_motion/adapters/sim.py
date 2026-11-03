"""Adaptateur simulé : robot virtuel sans aucune dépendance matérielle.

- Les articulations pilotées suivent leur consigne avec une vitesse limitée.
- Les articulations guidées suivent un « démonstrateur » virtuel (fonction du
  temps) qui remplace la main de l'opérateur ; sinon elles restent immobiles.
- `virtual_time=True` fait avancer une horloge virtuelle au lieu d'attendre :
  tests rapides et déterministes.
- `fail_after_s` simule une perte de connexion.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from .base import AdapterError, ConnectionLost, RobotAdapter


class SimAdapter(RobotAdapter):
    name = "sim"
    is_real = False

    def __init__(self, cfg, virtual_time: bool = False, max_speed_rad_s: float = 3.0,
                 sensor_noise_rad: float = 0.0, seed: int = 0, initial_pose: str = "rest"):
        super().__init__(cfg)
        self.virtual_time = virtual_time
        self.max_speed = max_speed_rad_s
        self.noise = sensor_noise_rad
        self.rng = np.random.default_rng(seed)
        self._t_virtual = 0.0
        self._last_update = None
        self._connected = False
        self._lock = threading.RLock()
        joints = cfg.all_joints
        self.q: Dict[str, float] = dict(zip(joints, cfg.pose(initial_pose, joints)))
        self.target: Dict[str, float] = dict(self.q)
        self.stiffness: Dict[str, float] = {j: 0.0 for j in joints}
        self.demonstrator: Optional[Callable[[float], Dict[str, float]]] = None
        self._demo_t0 = 0.0
        self.fail_after_s: Optional[float] = None
        self._connect_time = 0.0
        self.command_log: List[tuple] = []  # (t, joints, angles) pour les tests

    # -- horloge ----------------------------------------------------------
    def now(self) -> float:
        return self._t_virtual if self.virtual_time else time.monotonic()

    def sleep(self, dt: float):
        if dt <= 0:
            return
        if self.virtual_time:
            with self._lock:
                self._t_virtual += dt
        else:
            time.sleep(dt)

    # -- connexion ----------------------------------------------------------
    def connect(self):
        self._connected = True
        self._connect_time = self.now()
        self._last_update = self.now()

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        if self._connected and self.fail_after_s is not None and self.now() - self._connect_time > self.fail_after_s:
            self._connected = False
        return self._connected

    def _require(self):
        if not self.is_connected():
            raise ConnectionLost("Simulation : connexion perdue")

    def heartbeat(self):
        self._require()

    def capabilities(self) -> dict:
        return {"adapter": "sim", "real_robot": False, "stream": True, "interpolation": False}

    # -- démonstrateur virtuel ---------------------------------------------
    def set_demonstrator(self, fn: Optional[Callable[[float], Dict[str, float]]]):
        with self._lock:
            self.demonstrator = fn
            self._demo_t0 = self.now()

    # -- physique simplifiée -------------------------------------------------
    def _update(self):
        with self._lock:
            t = self.now()
            dt = max(0.0, t - (self._last_update or t))
            self._last_update = t
            step = self.max_speed * dt
            for j in self.q:
                role = self.roles.get(j)
                if role == "driven" and self.stiffness[j] > 0:
                    d = self.target[j] - self.q[j]
                    self.q[j] += float(np.clip(d, -step, step))
            if self.demonstrator is not None:
                pose = self.demonstrator(t - self._demo_t0)
                for j, v in pose.items():
                    if self.roles.get(j) == "guided":
                        self.q[j] = float(v)

    def _read(self, joints):
        self._require()
        self._update()
        q = np.array([self.q[j] for j in joints])
        if self.noise:
            q = q + self.rng.normal(0, self.noise, size=q.shape)
        return q

    def _set_stiffness(self, joints, value):
        self._require()
        self._update()
        for j in joints:
            self.stiffness[j] = float(value)

    def _send(self, joints, angles, fraction_speed):
        self._require()
        self._update()
        for j, a in zip(joints, angles):
            if self.stiffness[j] <= 0:
                raise AdapterError(f"Simulation : {j} est relâchée, elle ne peut pas suivre une consigne")
            self.target[j] = float(a)
        self.command_log.append((self.now(), list(joints), np.array(angles)))

    def _stop(self):
        if not self._connected:
            return
        self._update()
        with self._lock:
            for j in self.q:
                self.target[j] = self.q[j]
