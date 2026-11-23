"""Guardian export for another project (ia-agent, bridge embedded on NAO...).

Produces a folder containing:
  - guardian.json   : calibrated limits, margin, velocities, acceleration, durations,
                      coupled constraints, calibrated poses, provenance;
  - nao_guardian.py : standalone checker (Python 2.7 and 3, no dependency)
                      that applies the same rules as `nao_motion.guardian`.

By default, the export is refused without an active calibration: placeholder
limits must not protect a real robot.
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
        # allowed range, margins already included (calibrated zone minus zone_margin_rad, factory
        # stops minus limit_margin_rad): the checker removes nothing more
        "limits": {j: [float(lo), float(hi)] for j, (lo, hi) in cfg.guardian_bounds.items()},
        "limit_margin_rad": 0.0,
        "per_joint_margin": {},
        "velocity_rad_s": {"default": g.max_velocity_rad_s, "per_joint": dict(g.per_joint_velocity_rad_s)},
        "max_acceleration_rad_s2": g.max_acceleration_rad_s2,
        "per_joint_acceleration": dict(g.per_joint_acceleration),
        "check_rate_hz": g.check_rate_hz,
        "duration_s": [g.min_duration_s, g.max_duration_s],
        "start_tolerance_rad": g.start_tolerance_rad,
        "transition_velocity_rad_s": g.transition_velocity_rad_s,
        # rest may be outside the zone (contact): rest <-> start segments bounded by
        # these (factory) limits and this velocity ("factory": the per-joint velocity_rad_s),
        # without coupled constraints
        "rest_limits": {j: [float(lo), float(hi)] for j, (lo, hi) in cfg.rest_limits.items()},
        "rest_velocity_rad_s": "factory" if g.rest_velocity_rad_s is None else g.rest_velocity_rad_s,
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
