"""How faithfully K control points reproduce the recorded takes (no model involved).

Every gesture a model generates is a cubic B-spline with K control points per joint
(configs/model.yaml `representation.n_control_points`, or the `n_control_points`
setting of one training). Before any learning, each take is projected onto that
curve: whatever detail the projection loses (quick beats, waving, sharp peaks) can
never come out of the model, however well it learns.

This module measures that loss on the takes, for several K:
- `rmse_rad`: gap between the take and its K-point curve (all takes, all joints);
- `worst_take_rmse_rad`: the take that loses the most;
- `amplitude_kept`: median share of each moving joint's range (max - min) still there
  after projection (1.0 = nothing flattened); only joints that move more than 0.1 rad
  in the take count.

If the error is already small at the current K, a flat generated gesture comes from
the model (it predicts an average of the takes), not from the number of points.

Read only: nothing is written, nothing moves.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .representation import bspline_basis, fit_control_points, resample_phase

DEFAULT_KS = (8, 12, 16, 24, 32)
MOVING_RAD = 0.1
MIN_K, MAX_K = 4, 64


def _fit(seg, k, n):
    """(raw, k-point curve) on n uniform phase samples of a take segment."""
    raw, _ = resample_phase(seg.t, seg.q, n)
    return raw, bspline_basis(k, np.linspace(0, 1, n)) @ fit_control_points(raw, k)


def check_k(k) -> int:
    f = float(k)
    if f != int(f) or not MIN_K <= int(f) <= MAX_K:
        raise ValueError(f"n_control_points: whole number from {MIN_K} to {MAX_K} expected, got {k!r}")
    return int(f)


def _segments(store, cfg, component, statuses, take_ids=None):
    from ..models.datasets import _target_segment, group_joints
    joints = group_joints(cfg, component)
    target = "measured"
    if component == "hands":
        target = (cfg.model.get("hands_model") or {}).get("target", "measured")
    metas = [m for m in store.list(kind=component) if m.status in statuses]
    if take_ids is not None:
        keep = set(take_ids)
        metas = [m for m in metas if m.take_id in keep]
    real = [m for m in metas if not m.synthetic]
    used = real or metas  # simulated takes only when there is no real one
    out = [(m.take_id, _target_segment(store, store.load(m.take_id), joints, target)) for m in used]
    return joints, out, bool(real) or not metas


def fidelity(store, cfg, component: str, ks: Sequence[int] = DEFAULT_KS, statuses=("accepted",),
             take_ids: Optional[Sequence[str]] = None) -> dict:
    """Reconstruction error of the takes of one group for each K (see module doc)."""
    ks = sorted({check_k(k) for k in ks})
    n = int(cfg.model["representation"].get("n_eval_samples", 100))
    n = max(n, 2 * max(ks))  # never fewer samples than the curve can bend
    joints, segs, real = _segments(store, cfg, component, statuses, take_ids)
    cur = int(cfg.model["representation"]["n_control_points"])
    res = {"component": component, "n_takes": len(segs), "real_takes": real, "joints": list(joints),
           "config_n_control_points": cur, "n_samples": n, "rows": []}
    if not segs:
        res["warning"] = "No accepted take for this group"
        return res
    for k in ks:
        sq, per_take, kept, per_joint = 0.0, [], [], np.zeros(len(joints))
        count = 0
        for tid, seg in segs:
            raw, fit = _fit(seg, k, n)
            e = (fit - raw) ** 2
            sq += float(e.sum()); count += e.size
            per_joint += e.mean(axis=0)
            per_take.append((float(np.sqrt(e.mean())), tid))
            rng_raw = raw.max(axis=0) - raw.min(axis=0)
            rng_fit = fit.max(axis=0) - fit.min(axis=0)
            mv = rng_raw > MOVING_RAD
            kept += (rng_fit[mv] / rng_raw[mv]).tolist()
        worst = max(per_take)
        res["rows"].append({
            "n_control_points": k, "rmse_rad": float(np.sqrt(sq / count)),
            "worst_take_rmse_rad": worst[0], "worst_take_id": worst[1],
            "amplitude_kept": float(np.median(kept)) if kept else None,
            "amplitude_kept_min": float(np.min(kept)) if kept else None,
            "per_joint_rmse_rad": dict(zip(joints, np.sqrt(per_joint / len(segs)).round(4).tolist()))})
    return res


def reconstruction_rmse(store, cfg, component, k, take_ids) -> Optional[float]:
    """RMSE (rad) between the given takes and their k-point curves (training report)."""
    r = fidelity(store, cfg, component, [k], statuses=("accepted", "pending", "rejected"), take_ids=take_ids)
    return r["rows"][0]["rmse_rad"] if r["rows"] else None
