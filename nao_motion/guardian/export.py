"""Export du guardian pour un autre projet (ia-agent, pont embarqué sur NAO...).

Produit un dossier contenant :
  - guardian.json   : limites calibrées, marge, vitesses, accélération, durées,
                      contraintes couplées, poses calibrées, provenance ;
  - nao_guardian.py : vérificateur autonome (Python 2.7 et 3, sans dépendance)
                      qui applique les mêmes règles que `nao_motion.guardian`.

Par défaut, l'export est refusé sans calibration active : des limites de
remplacement ne doivent pas protéger un robot réel.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Optional

from ..adapters.base import LEG_PREFIXES
from ..config import Config

KIND = "nao_motion.guardian_export"
SCHEMA_VERSION = 1
CHECKER = Path(__file__).resolve().parents[1] / "standalone" / "nao_guardian.py"


class ExportError(ValueError):
    pass


def guardian_dict(cfg: Config, allow_uncalibrated: bool = False) -> dict:
    if not cfg.calibrated and not allow_uncalibrated:
        raise ExportError("Aucune calibration active : calibrer le robot (onglet Calibration) avant d'exporter. "
                          "--allow-uncalibrated exporte les limites de remplacement, marquées comme telles.")
    g = cfg.guardian
    calib_names = {c["name"] for c in (cfg.calibration.coupled if cfg.calibration else [])}
    poses = {}
    for name, groups in cfg.poses.items():
        poses[name] = {j: float(v) for gq in groups.values() for j, v in gq.items()}
    return {
        "kind": KIND, "schema_version": SCHEMA_VERSION, "unit": "rad", "time_unit": "s",
        "exported_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": {"project": "nao-emotion-motion", "calibration_version": cfg.calibration_version,
                   "calibrated": bool(cfg.calibrated), "poses_calibrated": bool(cfg.poses_calibrated),
                   "robot": (cfg.calibration.robot if cfg.calibration else None),
                   "config_fingerprint": cfg.fingerprint()},
        "groups": dict({k: list(v) for k, v in cfg.groups.items()}, **({"hands": list(cfg.hand_joints)}
                                                                        if cfg.hand_joints else {})),
        "forbidden_joint_prefixes": list(LEG_PREFIXES),
        "limits": {j: [float(lo), float(hi)] for j, (lo, hi) in cfg.limits.items()},
        "limit_margin_rad": g.limit_margin_rad,
        # mains (LHand, RHand) : ouverture 0..1 et non radians ; marge et accélération propres
        "per_joint_margin": dict(g.per_joint_margin),
        "velocity_rad_s": {"default": g.max_velocity_rad_s, "per_joint": dict(g.per_joint_velocity_rad_s)},
        "max_acceleration_rad_s2": g.max_acceleration_rad_s2,
        "per_joint_acceleration": dict(g.per_joint_acceleration),
        "check_rate_hz": g.check_rate_hz,
        "duration_s": [g.min_duration_s, g.max_duration_s],
        "start_tolerance_rad": g.start_tolerance_rad,
        "transition_velocity_rad_s": g.transition_velocity_rad_s,
        # le repos peut être hors zone (contact) : segments repos <-> départ bornés par
        # ces limites (usine) et cette vitesse, sans contraintes couplées
        "rest_limits": {j: [float(lo), float(hi)] for j, (lo, hi) in cfg.rest_limits.items()},
        "rest_velocity_rad_s": g.rest_velocity_rad_s,
        "coupled_constraints": [{"name": c.name, "terms": dict(c.terms), "bound": c.bound,
                                 "description": c.description,
                                 "origin": "calibration" if c.name in calib_names else "manual"}
                                for c in g.coupled_constraints],
        "poses": poses,
    }


def export_guardian(cfg: Config, out_dir, allow_uncalibrated: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = guardian_dict(cfg, allow_uncalibrated)
    (out / "guardian.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(CHECKER, out / "nao_guardian.py")
    return {"guardian": str(out / "guardian.json"), "checker": str(out / "nao_guardian.py"),
            "calibrated": d["source"]["calibrated"], "calibration_version": d["source"]["calibration_version"],
            "n_constraints": len(d["coupled_constraints"])}
