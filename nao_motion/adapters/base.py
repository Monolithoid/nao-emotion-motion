"""Interface commune des adaptateurs robot.

Règle des rôles : chaque articulation est soit
  - « guided » : relâchée (raideur nulle), uniquement LUE, jamais commandée ;
  - « driven » : raide, commandée ;
  - « idle »   : ni commandée ni relâchée par ce logiciel (état laissé tel quel).
Une articulation relâchée ne peut pas suivre une commande motorisée : toute
commande vers une articulation guidée est refusée ici, avant d'atteindre le robot.
Les jambes ne sont jamais adressées.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Sequence

import numpy as np

LEG_PREFIXES = ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")


class AdapterError(RuntimeError):
    pass


class ConnectionLost(AdapterError):
    pass


class RoleError(AdapterError):
    pass


class RobotAdapter:
    name = "abstract"
    is_real = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.roles: Dict[str, str] = {j: "idle" for j in cfg.all_joints}

    # -- à implémenter --------------------------------------------------------
    def connect(self): raise NotImplementedError
    def disconnect(self): raise NotImplementedError
    def is_connected(self) -> bool: raise NotImplementedError
    def _read(self, joints: List[str]) -> np.ndarray: raise NotImplementedError
    def _set_stiffness(self, joints: List[str], value: float): raise NotImplementedError
    def _send(self, joints: List[str], angles: np.ndarray, fraction_speed: float): raise NotImplementedError
    def _stop(self): raise NotImplementedError
    def heartbeat(self): pass
    def capabilities(self) -> dict: return {}

    # -- horloge (surchargée par la simulation en temps virtuel) -------------
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, dt: float):
        if dt > 0:
            time.sleep(dt)

    # -- API publique contrôlée ----------------------------------------------
    def _check_joints(self, joints: Iterable[str]):
        for j in joints:
            if j.startswith(LEG_PREFIXES):
                raise RoleError(f"Les jambes sont hors périmètre : {j}")
            if j not in self.roles:
                raise RoleError(f"Articulation non configurée : {j}")

    def read_angles(self, joints: Sequence[str]) -> np.ndarray:
        """Angles MESURÉS par les capteurs (rad)."""
        joints = list(joints)
        self._check_joints(joints)
        q = np.asarray(self._read(joints), dtype=np.float64)
        if q.shape != (len(joints),) or not np.all(np.isfinite(q)):
            raise AdapterError("Lecture capteurs invalide")
        return q

    def set_roles(self, guided: Sequence[str] = (), driven: Sequence[str] = (), hold: bool = True):
        """Applique les rôles et les raideurs correspondantes (action physique sur un robot réel)."""
        guided, driven = list(guided), list(driven)
        if set(guided) & set(driven):
            raise RoleError("Une articulation ne peut être à la fois guidée et pilotée")
        self._check_joints(guided + driven)
        st = self.cfg.robot.get("stiffness", {})
        if driven:
            # on fige d'abord la consigne sur la position mesurée pour éviter un saut
            q = self.read_angles(driven)
            self._set_stiffness(driven, float(st.get("driven", 0.8)))
            self._send(driven, q, 0.05)
            for j in driven:
                self.roles[j] = "driven"
        if guided:
            for j in guided:
                self.roles[j] = "guided"
            self._set_stiffness(guided, float(st.get("guided", 0.0)))

    def send_angles(self, joints: Sequence[str], angles, fraction_speed: float):
        joints = list(joints)
        self._check_joints(joints)
        bad = [j for j in joints if self.roles.get(j) != "driven"]
        if bad:
            raise RoleError(f"Commande refusée : articulations non pilotées {bad}")
        a = np.asarray(angles, dtype=np.float64)
        if a.shape != (len(joints),) or not np.all(np.isfinite(a)):
            raise AdapterError("Consigne invalide")
        if not 0 < fraction_speed <= 1:
            raise AdapterError("fraction_speed hors de ]0, 1]")
        self._send(joints, a, fraction_speed)

    def stop_motion(self):
        """Arrête toute exécution en cours et maintient la posture actuelle (pas de retour au repos)."""
        self._stop()

    def relax(self, joints: Sequence[str]):
        """Raideur nulle (les bras tombent sous l'effet de la gravité) : action explicite uniquement."""
        joints = list(joints)
        self._check_joints(joints)
        self._set_stiffness(joints, 0.0)
        for j in joints:
            self.roles[j] = "guided"
