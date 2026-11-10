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

Progression gardée : les explorations, poses capturées et groupes repris sont
sauvegardés à chaque étape dans configs/calibration/drafts/<robot>.npz et
repris à la prochaine ouverture (fermeture de l'onglet, redémarrage du serveur).
Un groupe déjà calibré peut aussi être repris d'un fichier enregistré
(`import_group`) : on garde par exemple les bras et on refait la tête.

En simulation, un démonstrateur synthétique remplace la main de l'opérateur
(exploration) et les poses de remplacement sont « placées » avant capture :
les valeurs obtenues n'ont aucune portée réelle.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .calibration import (Calibration, CalibrationError, CalibrationSession, convex_hull,
                          hull_constraints, in_zone, load_procedure, sim_sweeper, simplify_hull, zone_polygon)

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
    PERSIST_SIM = False  # en simulation, la progression n'est pas écrite sur disque (tests : True)

    def __init__(self, wb):
        self.wb = wb
        self.cfg = wb.cfg
        self.adapter = wb.adapter
        self.proc = load_procedure(self.cfg.root)
        self.session = CalibrationSession(self.cfg, self.adapter, self.proc)
        self.robot = self.session.robot_info()
        # limites d'usine (ALMotion.getLimits), lues une fois à l'ouverture : affichage seulement
        try:
            self.factory = self.session.factory_limits()
        except Exception:
            self.factory = None
        self.sim = self.adapter.name == "sim"
        self.thread: Optional[threading.Thread] = None
        self.sweep_error: Optional[str] = None
        self.result: Optional[Calibration] = None
        self.saved_path: Optional[Path] = None
        self.activated = False
        self.message = "Calibration ouverte : choisir un groupe."
        self._lock = threading.RLock()
        self.restored = self._restore_draft()

    # -- progression gardée sur disque ---------------------------------------------------
    @property
    def draft_path(self) -> Path:
        name = re.sub(r"[^\w\-]+", "_", str(self.robot.get("name") or "nao")) or "nao"
        return self.cfg.root / "calibration" / "drafts" / f"{name}.npz"

    def _persist(self) -> bool:
        return (not self.sim) or self.PERSIST_SIM

    def _save_draft(self):
        if not self._persist():
            return
        s = self.session
        if not s.sweeps and not s.poses and not s.imported:
            self.draft_path.unlink(missing_ok=True)
            return
        meta = {"robot": self.robot, "poses": s.poses, "imported": s.imported,
                "sweeps": {g: list(sw["joints"]) for g, sw in s.sweeps.items()}}
        arrays = {f"q_{g}": np.asarray(sw["q"], dtype=float) for g, sw in s.sweeps.items()}
        p = self.draft_path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.replace(p)

    def _restore_draft(self) -> List[str]:
        p = self.draft_path
        if not self._persist() or not p.exists():
            return []
        try:
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                sweeps = {g: {"joints": list(js), "q": np.asarray(z[f"q_{g}"], dtype=float)}
                          for g, js in meta.get("sweeps", {}).items()}
        except Exception as e:
            self.message = f"Progression enregistrée illisible ({p.name}) : {e}. Calibration repartie de zéro."
            return []
        s = self.session
        groups = self.cfg.groups
        s.sweeps = {g: sw for g, sw in sweeps.items() if g in groups and sw["joints"] == list(groups[g])}
        s.imported = {g: d for g, d in (meta.get("imported") or {}).items() if g in groups}
        s.poses = {n: {g: {j: float(v) for j, v in jq.items()} for g, jq in gp.items() if g in groups}
                   for n, gp in (meta.get("poses") or {}).items() if n in POSE_NAMES}
        done = sorted(set(s.sweeps) | set(s.imported))
        if done or any(s.poses.values()):
            self.message = ("Progression précédente reprise : " +
                            (", ".join(_g(g) for g in done) or "aucun groupe exploré") +
                            ". « Repartir de zéro » pour tout effacer.")
        return done

    def discard_draft(self):
        """Efface toute la progression (explorations, poses, groupes repris)."""
        self._not_sweeping()
        s = self.session
        s.sweeps, s.poses, s.imported = {}, {}, {}
        self._invalidate()
        if self._persist():
            self.draft_path.unlink(missing_ok=True)
        self.restored = []
        self.message = "Progression effacée : calibration repartie de zéro."

    def import_group(self, file: str, group: str) -> dict:
        """Reprend un groupe (limites, enveloppes, poses) d'une calibration enregistrée,
        sans l'explorer à nouveau. Aucun mouvement."""
        self._not_sweeping()
        self._joints(group)
        d = self.cfg.root / "calibration"
        f = d / Path(str(file)).name
        if not f.exists():
            raise CalibrationError(f"Calibration introuvable : {f.name}")
        cal = Calibration.load(f)
        imp = self.session.import_group(cal, group)
        self._invalidate()
        self._save_draft()
        miss = [POSE_LABEL[n] for n in POSE_NAMES if n not in imp["poses"]]
        self.message = (f"Groupe {_g(group)} repris de {cal.version}" +
                        (f" ; poses à capturer : {', '.join(miss)}." if miss else " (exploration et poses)."))
        return imp

    def drop_import(self, group: str):
        self._not_sweeping()
        self._joints(group)
        self.session.drop_import(group)
        self._invalidate()
        self._save_draft()
        self.message = f"Reprise du groupe {_g(group)} annulée : l'explorer ou le reprendre d'un autre fichier."

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

    def sweep_clear(self, group: str):
        """Efface les mesures d'exploration d'un groupe (aucune commande)."""
        self._not_sweeping()
        self._joints(group)
        self.session.clear_sweep(group)
        self._invalidate()
        self._save_draft()
        self.message = f"Exploration du groupe {_g(group)} effacée."

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
        try:
            self._save_draft()
        except Exception as e:
            self.sweep_error = self.sweep_error or f"sauvegarde de la progression : {e}"
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
        self._save_draft()
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
        self.message = f"Enregistrée : {self.saved_path.name} (non active). La progression reste gardée jusqu'à « Repartir de zéro »."
        return str(self.saved_path)

    def activate(self) -> str:
        if self.saved_path is None:
            raise CalibrationError("Enregistrer la calibration d'abord")
        if self.sim:
            raise CalibrationError("Calibration simulée : elle ne peut pas être activée")
        self.wb.activate_calibration(self.saved_path)  # valide, active, recharge (rollback si échec)
        p = self.cfg.root / "calibration" / "ACTIVE"
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
    def _frame(self, j: str) -> List[float]:
        f = (self.factory or {}).get(j)
        return [float(f[0]), float(f[1])] if f else list(self.cfg.limits[j])

    def _zone(self, a: str, b: str, pts: np.ndarray, hull: np.ndarray) -> dict:
        """Zone que retiendra le calcul pour ce couple (même règle que compute_calibration :
        enveloppe simplifiée rétrécie de la marge, ∩ plages explorées ∩ usine, moins la marge)
        et position des poses capturées par rapport à elle."""
        m = float(self.proc.get("margin_rad", 0.05))
        box = []
        for k, j in enumerate((a, b)):
            flo, fhi = self._frame(j) if self.factory else (-np.inf, np.inf)
            box += [max(float(pts[:, k].min()), flo) + m, min(float(pts[:, k].max()), fhi) - m]
        cons = []
        if len(hull) >= 3:
            cons = hull_constraints(simplify_hull(hull, int(self.proc.get("max_hull_edges", 12))), m)
        poly = zone_polygon(cons, tuple(box)) if box[0] < box[1] and box[2] < box[3] else np.zeros((0, 2))
        poses = {}
        for name in POSE_NAMES:
            flat = {j: v for gq in self.session.poses.get(name, {}).values() for j, v in gq.items()}
            if a in flat and b in flat:
                p = [flat[a], flat[b]]
                # le repos peut être hors zone (contact) : seul le départ doit y être
                poses[name] = {"q": [round(p[0], 4), round(p[1], 4)],
                               "inside": bool(len(poly) and in_zone(p, cons, box))}
        cur = None
        ang = getattr(self, "_angles", None) or {}
        if a in ang and b in ang:  # angles déjà lus pour cet état (pas de lecture en plus)
            cur = [ang[a], ang[b]]
        return {"zone": np.round(poly, 4).tolist(), "poses": poses, "current": cur,
                "current_inside": bool(cur is not None and len(poly) and in_zone(cur, cons, box))}

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
                        "limits": [self._frame(a), self._frame(b)], **self._zone(a, b, pts, hull)})
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
        self._angles = angles
        live = np.array(list(s.live)) if self.sweeping and s.live else None
        live_joints = list(s.live_joints) if live is not None else []
        if live is not None and s.current in s.sweeps and len(s.sweeps[s.current]["q"]):
            live = np.vstack([np.asarray(s.sweeps[s.current]["q"]), live])  # complète l'exploration précédente
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
                   "measured": r.measured, "coupled": len(r.coupled), "margin_rad": r.margin_rad,
                   "warnings": list(r.warnings)}
        missing = [f"{n}/{g}" for n in POSE_NAMES for g in self.cfg.groups if g not in s.poses.get(n, {})]
        return {"active": True, "sim": self.sim, "robot": self.robot, "groups": self.cfg.groups,
                "sweeping": s.current if self.sweeping else None, "sweep_n": s.n if self.sweeping else 0,
                "swept": sorted(s.sweeps), "imported": {g: {"source": d["source"], "file": d.get("file"),
                                                            "poses": sorted(d["poses"])}
                                                        for g, d in s.imported.items()},
                "restored": list(self.restored), "persist": self._persist(), "angles": angles, "roles": dict(self.adapter.roles),
                "ranges": ranges, "poses": s.poses,
                "factory": ({j: [round(float(v[0]), 4), round(float(v[1]), 4)] for j, v in self.factory.items()}
                            if self.factory else None), "missing_poses": missing,
                "pairs": self._pairs(s.current, live, live_joints),
                "procedure": {k: self.proc.get(k) for k in ("margin_rad", "min_samples", "min_range_rad", "rate_hz")},
                "result": res, "saved": self.saved_path.name if self.saved_path else None,
                "activated": self.activated, "message": self.message, "error": self.sweep_error}
