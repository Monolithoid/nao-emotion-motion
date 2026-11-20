"""Export of the gesture generator for another project (ia-agent...): TWO files.

  - nao_gesture.py   : standalone generator + guardian (Python 2.7 and 3, standard
                       library only, no PyTorch, no numpy);
  - nao_gesture.json : the arms / head / hands models (network weights), their settings
                       (emotions, normalisation, B-spline, combination) and the guardian
                       (calibrated limits, constraints, velocities, poses).

The weights come from the numpy export written by every training run (`model_numpy.npz`
of the latest run of each group): the export does not need PyTorch. The bundled
guardian is the very text of `standalone/nao_guardian.py` (<guardian-core> block):
same rules, no second copy to maintain.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import __version__
from .config import Config
from .guardian.export import ExportError, guardian_dict

STANDALONE = Path(__file__).resolve().parent / "standalone"
TEMPLATE = STANDALONE / "nao_gesture_template.py"
GUARDIAN_SRC = STANDALONE / "nao_guardian.py"
BUNDLE_KIND = "nao_motion.gesture_bundle"
BUNDLE_SCHEMA = 1
COMPONENTS = ("arms", "head", "hands")
LIB_NAME, DATA_NAME = "nao_gesture.py", "nao_gesture.json"


def build_library_source() -> str:
    """nao_gesture.py = template + standalone guardian core inserted in place of #@@GUARDIAN@@."""
    g = GUARDIAN_SRC.read_text(encoding="utf-8")
    start, end = g.index("# <guardian-core>"), g.index("# </guardian-core>")
    core = g[start:end].rstrip() + "\n"
    banner = ("# ======================================================================================\n"
              "# Guardian (copy of the <guardian-core> block of nao_guardian.py, same rules)\n"
              "# ======================================================================================\n")
    src = TEMPLATE.read_text(encoding="utf-8")
    if src.count("#@@GUARDIAN@@") != 1:
        raise ExportError("Invalid nao_gesture template")
    return src.replace("#@@GUARDIAN@@", banner + core)


def latest_numpy_models(runs_dir) -> Dict[str, Optional[Path]]:
    """Numpy export of the latest run of each group (None if never trained)."""
    from .training import latest
    out = {}
    for comp in COMPONENTS:
        p = latest(runs_dir, comp)
        out[comp] = p.with_name("model_numpy.npz") if p else None
        if out[comp] is not None and not out[comp].exists():
            raise ExportError(f"{out[comp]} missing: retrain the {comp} group (no numpy export)")
    return out


def _model_entry(cfg: Config, comp: str, path: Path) -> dict:
    from .models.features import Norm
    from .models.nets import load_model
    m = load_model(path, cfg, comp)  # NumpyModel: checks emotions, joints, format
    joints = {"arms": cfg.arm_joints, "head": cfg.head_joints, "hands": cfg.hand_joints}[comp]
    norm = Norm.from_config(cfg, joints)
    clip = None
    if comp == "hands":
        g = cfg.guardian
        clip = [[cfg.limits[j][0] + g.margin(j) for j in joints], [cfg.limits[j][1] - g.margin(j) for j in joints]]
    return {
        "joints": list(joints), "center": norm.center.tolist(), "scale": norm.scale.tolist(),
        "latent_dim": int(m.arch.get("latent_dim", 0)), "activation": "tanh", "clip": clip,
        # the model's own number of control points (a training setting, can differ per group)
        "n_control_points": int((m.meta.get("representation") or {}).get("n_control_points")
                                or cfg.model["representation"]["n_control_points"]),
        "layers": [{"W": np.asarray(W, dtype=np.float64).tolist(), "b": np.asarray(b, dtype=np.float64).tolist()}
                   for W, b in zip(m.W, m.b)],
        "source": {"run": Path(path).parent.name, "digest": m.digest,
                   "calibration_version": m.meta.get("calibration_version"),
                   "n_takes": len(m.meta.get("train_take_ids", [])) + len(m.meta.get("val_take_ids", [])),
                   "synthetic_data": m.meta.get("synthetic_data")},
        "warnings": list(m.warnings),
    }


def bundle_dict(cfg: Config, models: Dict[str, Optional[Path]], allow_uncalibrated: bool = False) -> dict:
    ex = {"project_control_points": True, "adapt_generated": True, "slow_down": "local",
          **{k: v for k, v in (cfg.robot.get("execution") or {}).items()
             if k in ("project_control_points", "adapt_generated", "slow_down")}}
    entries = {c: _model_entry(cfg, c, Path(p)) for c, p in models.items()
               if p is not None and (c != "hands" or cfg.hand_joints)}
    if not entries:
        raise ExportError("No trained model to export (arms, head or hands)")
    return {
        "kind": BUNDLE_KIND, "schema_version": BUNDLE_SCHEMA,
        "exported_at": _dt.datetime.now().isoformat(timespec="seconds"), "package_version": __version__,
        "emotions": list(cfg.emotions),
        "input_layout": "[emotions (one-hot or normalised mix)..., intensity 0..1, event present, "
                        "sin(azimuth), cos(azimuth), elevation (rad), distance known, 1/(1+distance m)]",
        "n_control_points": int(cfg.model["representation"]["n_control_points"]),  # default; see models[*]
        "rate_hz": 50.0,
        "timing": str((cfg.model.get("combine") or {}).get("timing", "hold")),
        "groups": {"arms": list(cfg.arm_joints), "head": list(cfg.head_joints), "hands": list(cfg.hand_joints)},
        "models": entries,
        "guardian": guardian_dict(cfg, allow_uncalibrated),
        # playback adaptation applied by the library before the guardian check (robot.yaml execution)
        "playback": {k: ex[k] for k in ("project_control_points", "adapt_generated", "slow_down") if k in ex},
        # CVAE models: default temperature of the latent draw (model.yaml cvae.temperature)
        "latent": {"temperature": float((cfg.model.get("cvae") or {}).get("temperature", 1.0))},
    }


def _same_bundle(path: Path, data: dict) -> bool:
    """True if `path` already holds this bundle, export timestamps aside."""
    if not path.exists():
        return False
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False

    def strip(d):
        d = dict(d, exported_at=None)
        if isinstance(d.get("guardian"), dict):
            d["guardian"] = dict(d["guardian"], exported_at=None)
        return d
    return strip(old) == strip(json.loads(json.dumps(data)))


def export_generator(cfg: Config, out_dir, runs_dir="runs", models: Optional[Dict[str, Optional[Path]]] = None,
                     allow_uncalibrated: bool = False) -> dict:
    """Writes nao_gesture.py and nao_gesture.json into `out_dir`. `models`: {group: .npz path}
    (default: latest run of each group in `runs_dir`)."""
    models = latest_numpy_models(runs_dir) if models is None else models
    data = bundle_dict(cfg, models, allow_uncalibrated)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not _same_bundle(out / DATA_NAME, data):  # unchanged models: keep the file (no diff in a Git repo)
        (out / DATA_NAME).write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    lib = build_library_source()
    if not (out / LIB_NAME).exists() or (out / LIB_NAME).read_text(encoding="utf-8") != lib:
        (out / LIB_NAME).write_text(lib, encoding="utf-8")
    return {"library": str(out / LIB_NAME), "data": str(out / DATA_NAME),
            "models": {c: e["source"]["run"] for c, e in data["models"].items()},
            "missing": [c for c in COMPONENTS if c not in data["models"]],
            "warnings": [w for e in data["models"].values() for w in e["warnings"]],
            "calibrated": data["guardian"]["source"]["calibrated"],
            "calibration_version": data["guardian"]["source"]["calibration_version"]}
