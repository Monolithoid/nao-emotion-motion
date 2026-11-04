"""Calibration du guardian pilotée depuis l'interface web (onglet Calibration).

Même procédure et même calcul que `nao-motion calibrate` (CalibrationSession,
compute_calibration) : seule la conduite change. L'opérateur avance étape par
étape, groupe par groupe et dans l'ordre qu'il choisit ; la page interroge
`state()` pour afficher les angles en direct, les min/max qui s'élargissent
pendant l'exploration et le nuage de points de chaque couple.

Sécurité (inchangée) :
  - aucune commande de mouvement : on relâche un groupe (raideur nulle) ou on
    le raidit SUR PLACE (la consigne est d'abord la position mesurée) ;
  - sur le robot réel, relâcher un groupe exige une confirmation explicite ;
  - fermer ou annuler laisse chaque articulation dans son état : pas de retour
    au repos ;
  - une calibration n'est jamais écrasée ; l'activation est une action à part.

En simulation, un démonstrateur synthétique remplace la main de l'opérateur
(exploration) et les poses de remplacement sont « placées » avant capture :
les valeurs obtenues n'ont aucune portée réelle.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .calibration import (Calibration, CalibrationError, CalibrationSession, activate, convex_hull,
                          load_procedure, sim_sweeper)

POSE_NAMES = ("rest", "start")
POSE_LABEL = {"rest": "repos", "start": "départ"}
GROUP_LABEL = {"head": "tête", "left_arm": "bras gauche", "right_arm": "bras droit"}


def _g(group: str) -> str:
    return GROUP_LABEL.get(group, group)
MAX_POINTS = 400  # points envoyés à la page par couple


def _downsample(q: np.ndarray, n: int = MAX_POINTS) -> np.ndarray:
    if len(q) <= n:
        return q
    idx = np.linspace(0, len(q) - 1, n).round().astype(int)
    return q[idx]


class CalibrationWizard:
    def __init__(self, wb):
        self.wb = wb
        self.cfg = wb.cfg
        self.adapter = wb.adapter
        self.proc = load_procedure(self.cfg.root)
        self.session = CalibrationSession(self.cfg, self.adapter, self.proc)
        self.robot = self.session.robot_info()
        self.sim = self.adapter.name == "sim"
        self.thread: Optional[threading.Thread] = None
        self.sweep_error: Optional[str] = None
        self.result: Optional[Calibration] = None
        self.saved_path: Optional[Path] = None
        self.activated = False
        self.message = "Calibration ouverte : choisir un groupe."
        self._lock = threading.RLock()

    # -- état -----------------------------------------------------------------------
    @property
    def sweeping(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _joints(self, group: str) -> List[str]:
        if group not in self.cfg.groups:
            raise CalibrationError(f"Groupe inconnu : {group}")
        return list(self.cfg.groups[group])

    def _not_sweeping(self):
        if self.sweeping:
            raise CalibrationError("Exploration en cours : l'arrêter d'abord")

    def _invalidate(self):
        self.result, self.saved_path, self.activated = None, None, False

    # -- étapes (aucun mouvement commandé) -------------------------------------------
    def release(self, group: str, confirm: bool = False):
        """Relâche un groupe pour le placer à la main (raideur nulle)."""
        self._not_sweeping()
        js = self._joints(group)
        if getattr(self.adapter, "is_real", False) and not confirm:
            raise CalibrationError(f"Confirmer que le groupe {_g(group)} est tenu avant de le relâcher")
        self.adapter.set_roles(guided=js)
        self.message = f"Groupe {_g(group)} relâché : le placer à la main."

    def hold(self, group: str):
        """Raidit un groupe SUR PLACE (consigne = position mesurée, sans saut)."""
        self._not_sweeping()
        self.adapter.set_roles(driven=self._joints(group))
        self.message = f"Groupe {_g(group)} raidi sur place."

    def sweep_start(self, group: str, confirm: bool = False, max_duration_s: float = 600.0,
                    background: bool = True):
        """Relâche le groupe (les autres sont raidis sur place) et enregistre les angles."""
        self._not_sweeping()
        self._joints(group)
        if getattr(self.adapter, "is_real", False) and not confirm:
            raise CalibrationError(f"Confirmer que le groupe {_g(group)} est tenu avant de le relâcher")
        self.sweep_error = None
        self._invalidate()
        if self.sim:
            self.adapter.set_demonstrator(sim_sweeper(self.cfg, group))

        def work():
            try:
                self.session.sweep(group, max_duration_s=max_duration_s, confirm=True)
            except Exception as e:  # remonté à la page
                self.sweep_error = str(e)
            finally:
                if self.sim:
                    self.adapter.set_demonstrator(None)

        self.message = (f"Exploration du groupe {_g(group)} : amener lentement chaque articulation jusqu'à ses butées utiles, "
                        "et parcourir les combinaisons (par ex. tête tournée ET baissée).")
        if background:
            self.thread = threading.Thread(target=work, daemon=True)
            self.thread.start()
        else:
            work()
            self._after_sweep(group)

    def sweep_stop(self):
        group = self.session.current
        self.session.request_stop()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if group is not None:
            self._after_sweep(group)

    def _after_sweep(self, group: str):
        # raidi sur place, comme dans la procédure en ligne de commande
        try:
            self.adapter.set_roles(driven=self._joints(group))
        except Exception as e:
            self.sweep_error = self.sweep_error or f"raidir le groupe {_g(group)} : {e}"
        n = len(self.session.sweeps.get(group, {}).get("q", []))
        self.message = f"Exploration du groupe {_g(group)} terminée : {n} mesures. Groupe raidi sur place."

    def capture(self, name: str, group: str) -> Dict[str, float]:
        """Lit la pose actuelle d'un groupe (moyenne de quelques lectures)."""
        self._not_sweeping()
        if name not in POSE_NAMES:
            raise CalibrationError(f"Pose inconnue : {name} (rest | start)")
        js = self._joints(group)
        if self.sim:  # en simulation, on « place » le groupe sur la pose de remplacement
            for j in js:
                v = self.cfg.poses[name][group][j]
                self.adapter.q[j] = v
                self.adapter.target[j] = v
        pose = self.session.capture_pose(name, group)
        self._invalidate()
        self.message = f"Pose de {POSE_LABEL[name]} du groupe {_g(group)} capturée."
        return pose

    def compute(self, version: Optional[str] = None) -> dict:
        self._not_sweeping()
        self.result = self.session.result(version=version or None)
        self.saved_path, self.activated = None, False
        self.message = f"Calibration {self.result.version} calculée : relire puis enregistrer."
        return self.result.to_dict()

    def save(self) -> str:
        if self.result is None:
            raise CalibrationError("Calculer la calibration d'abord")
        if self.saved_path is None:
            self.saved_path = self.result.save(self.cfg.root / "calibration" / f"{self.result.version}.yaml")
        self.message = f"Enregistrée : {self.saved_path.name} (non active)."
        return str(self.saved_path)

    def activate(self) -> str:
        if self.saved_path is None:
            raise CalibrationError("Enregistrer la calibration d'abord")
        if self.sim:
            raise CalibrationError("Calibration simulée : elle ne peut pas être activée")
        p = activate(self.cfg.root, self.saved_path)
        self.activated = True
        self.message = f"{self.saved_path.name} activée pour le guardian."
        return str(p)

    def stop(self):
        self.session.request_stop()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.sim:
            self.adapter.set_demonstrator(None)

    # -- état pour la page -------------------------------------------------------------
    def _pairs(self, group: Optional[str], live: Optional[np.ndarray], live_joints: List[str]) -> List[dict]:
        out = []
        for a, b in self.proc.get("pairs", []):
            src, js = None, None
            if live is not None and a in live_joints and b in live_joints:
                src, js = live, live_joints
            else:
                for g, s in self.session.sweeps.items():
                    if a in s["joints"] and b in s["joints"] and len(s["q"]):
                        src, js = np.asarray(s["q"]), s["joints"]
            if src is None or not len(src):
                continue
            pts = src[:, [js.index(a), js.index(b)]]
            hull = convex_hull(pts) if len(pts) >= 3 else pts
            out.append({"joints": [a, b], "points": np.round(_downsample(pts), 4).tolist(),
                        "hull": np.round(hull, 4).tolist(), "n": int(len(pts)),
                        "limits": [list(self.cfg.limits[a]), list(self.cfg.limits[b])]})
        return out

    def state(self) -> dict:
        s = self.session
        joints = self.cfg.all_joints
        angles = None
        if self.adapter.is_connected():
            try:
                angles = dict(zip(joints, np.round(self.adapter.read_angles(joints), 4).tolist()))
            except Exception:
                angles = None
        live = np.array(list(s.live)) if self.sweeping and s.live else None
        live_joints = list(s.live_joints) if live is not None else []
        ranges: Dict[str, dict] = {}
        for g, sw in s.sweeps.items():
            q = np.asarray(sw["q"])
            if len(q):
                for k, j in enumerate(sw["joints"]):
                    ranges[j] = {"min": round(float(q[:, k].min()), 4), "max": round(float(q[:, k].max()), 4),
                                 "n": int(len(q))}
        if live is not None and len(live):
            for k, j in enumerate(live_joints):
                ranges[j] = {"min": round(float(live[:, k].min()), 4), "max": round(float(live[:, k].max()), 4),
                             "n": int(len(live)), "live": True}
        res = None
        if self.result is not None:
            r = self.result
            res = {"version": r.version, "limits": {j: list(v) for j, v in r.limits.items()},
                   "measured": r.measured, "coupled": len(r.coupled), "margin_rad": r.margin_rad}
        missing = [f"{n}/{g}" for n in POSE_NAMES for g in self.cfg.groups if g not in s.poses.get(n, {})]
        return {"active": True, "sim": self.sim, "robot": self.robot, "groups": self.cfg.groups,
                "sweeping": s.current if self.sweeping else None, "sweep_n": s.n if self.sweeping else 0,
                "swept": sorted(s.sweeps), "angles": angles, "roles": dict(self.adapter.roles),
                "ranges": ranges, "poses": s.poses, "missing_poses": missing,
                "pairs": self._pairs(s.current, live, live_joints),
                "procedure": {k: self.proc.get(k) for k in ("margin_rad", "min_samples", "min_range_rad", "rate_hz")},
                "result": res, "saved": self.saved_path.name if self.saved_path else None,
                "activated": self.activated, "message": self.message, "error": self.sweep_error}
