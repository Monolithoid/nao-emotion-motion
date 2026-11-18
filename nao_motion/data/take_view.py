"""What the Takes tab shows for one take: its curves per joint, the B-spline fit the model
is trained on (the smoothing due to the representation alone), and optionally what the
active model generates for the same emotion, intensity and event (the smoothing due to
learning). Read only: nothing is written, nothing moves."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..trajectory import Trajectory
from .representation import fit_control_points, greville, reconstruct, resample_phase
from .trash import active_runs

GROUPS = ("arms", "head", "hands")
COMPARE_N = (6, 8, 12, 16, 24)   # control point counts compared in the fit summary
N_RANGE = (4, 40)


def group_joints(cfg, kind):
    return {"arms": cfg.arm_joints, "head": cfg.head_joints, "hands": cfg.hand_joints}[kind]


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _round(x, nd=4):
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def _run_info(runs_dir, kind) -> dict:
    """Active model of the group: its name, its number of control points (when the run
    recorded it) and the role of each take in it."""
    run = active_runs(runs_dir).get(kind)
    if not run:
        return {}
    d = Path(runs_dir) / run
    out = {"run": run}
    try:
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8")).get("metrics", {})
        k = (m.get("settings") or {}).get("n_control_points") or (m.get("representation") or {}).get("n_control_points")
        if k:
            out["n_ctrl"] = int(k)
    except (OSError, ValueError):
        pass
    try:
        out["split"] = json.loads((d / "split.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return out


def _role(split: Optional[dict], take_id: str) -> Optional[str]:
    if not split:
        return None
    if take_id in (split.get("val") or []):
        return "validation"
    if any(take_id in part for part in (split.get("cv_val") or [])):
        return "train (cross-validated)"
    if take_id in (split.get("train") or []):
        mirrored = take_id in (split.get("train_mirrored") or [])
        return "train + mirror" if mirrored else "train"
    return "not used"


def target_trajectory(store, cfg, take, kind) -> tuple:
    """Group joints as the training sees them: measures, or the command sent (hands, target
    `command`). Full take, absolute time. Returns (trajectory, target name)."""
    joints = group_joints(cfg, kind)
    target = (cfg.model.get("hands_model") or {}).get("target", "measured") if kind == "hands" else "measured"
    if target == "command":
        ref = store.load_reference(take.meta.take_id)
        if ref is not None and set(joints) <= set(ref.joints):
            return ref.select(joints), "command"
    return take.traj.select(joints), "measured"


def take_view(wb, take_id: str, n_ctrl: Optional[int] = None, with_model: bool = False) -> dict:
    cfg, store = wb.cfg, wb.store
    take = store.load(take_id)
    m = take.meta
    kind = m.kind if m.kind in GROUPS else "arms"
    rep = cfg.model["representation"]
    n_samples = int(rep.get("n_eval_samples", 100))
    run = _run_info(wb.runs_dir, kind)
    default_n = int(run.get("n_ctrl") or rep["n_control_points"])
    n = int(n_ctrl) if n_ctrl else default_n
    if not N_RANGE[0] <= n <= N_RANGE[1]:
        raise ValueError(f"control points: {N_RANGE[0]} to {N_RANGE[1]}")

    traj, target = target_trajectory(store, cfg, take, kind)
    seg = tuple(m.segment)
    # full curves, at most ~600 points (enough for a 900 px wide plot)
    idx = np.unique(np.linspace(0, len(traj.t) - 1, min(len(traj.t), 600)).round().astype(int))
    t, raw, fitted = reconstruct(traj, n, n_samples, seg)
    qu, dur = resample_phase(traj.t, traj.q, n_samples, seg)
    ctrl = fit_control_points(qu, n)
    err = np.abs(fitted - raw)
    compare = []
    for k in sorted(set(COMPARE_N) | {n, default_n}):
        _, r, f = reconstruct(traj, k, n_samples, seg)
        compare.append({"n_ctrl": k, "rmse_rad": round(_rmse(f, r), 4), "max_err_rad": round(float(np.max(np.abs(f - r))), 4)})
    out = {
        "take_id": take_id, "group": kind, "joints": list(traj.joints), "target": target,
        "segment": list(seg), "duration_s": round(float(dur), 3),
        "t": _round(traj.t[idx]), "q": _round(traj.q[idx]),
        "fit": {"n_ctrl": n, "default_n_ctrl": default_n,
                "default_source": f"active model {run['run']}" if run.get("n_ctrl") else "configs/model.yaml",
                "n_samples": n_samples, "sample_rate_hz": round((n_samples - 1) / float(dur), 1),
                "t": _round(t), "q": _round(fitted), "resampled": _round(raw),
                "ctrl_t": _round(seg[0] + greville(n) * dur), "ctrl": _round(ctrl),
                "rmse_rad": round(_rmse(fitted, raw), 4), "max_err_rad": round(float(err.max()), 4),
                "rmse_per_joint": {j: round(_rmse(fitted[:, i], raw[:, i]), 4) for i, j in enumerate(traj.joints)},
                "compare": compare},
        "active_run": run.get("run"), "role": _role(run.get("split"), take_id),
    }
    if with_model:
        out["model"] = model_curves(wb, take, kind, traj.joints, seg[0], t, raw)
    return out


def model_curves(wb, take, kind, joints, t0, t_fit, raw) -> dict:
    """What the active model generates for the take's emotion, intensity and event,
    drawn from the start of the take's segment (its own learnt duration)."""
    m = take.meta
    try:
        eng = wb.get_engine()
        kw = {}
        model = eng.model_of(kind) if hasattr(eng, "model_of") else None
        if model is not None and (model.arch or {}).get("latent_dim"):
            # CVAE: the take's own z when the model has seen it (its reconstruction), else the
            # central gesture z = 0, never a random draw
            lat = (model.meta or {}).get("latent") or {}
            ids = lat.get("take_ids") or []
            z = lat["mu"][ids.index(m.take_id)] if m.take_id in ids else None
            kw = {"temperature": 0.0, "latent": {kind: z} if z is not None else None}
        gen = eng.generate(m.emotion, m.intensity, m.event_ctx, **kw)
    except Exception as e:  # no model, torch missing, remote engine down...
        return {"error": str(e)}
    traj = {"arms": gen.arms, "head": gen.head, "hands": gen.hands}.get(kind)
    plans = getattr(gen, "durations", {}) or {}
    if traj is None or kind not in plans:
        return {"error": f"no active {kind} model"}
    tr: Trajectory = traj.select(joints)
    d = float(plans[kind])
    keep = tr.t <= d + 1e-9
    # error against the take, on the same phase grid (model and take durations differ)
    u = (t_fit - t_fit[0]) / max(t_fit[-1] - t_fit[0], 1e-9)
    qm = tr.sample(np.clip(u * d, 0, tr.t[-1]))
    return {"run": active_runs(wb.runs_dir).get(kind), "duration_s": round(d, 3),
            "t": _round(t0 + tr.t[keep]), "q": _round(tr.q[keep]),
            "rmse_rad": round(_rmse(qm, raw), 4),
            "rmse_per_joint": {j: round(_rmse(qm[:, i], raw[:, i]), 4) for i, j in enumerate(joints)}}
