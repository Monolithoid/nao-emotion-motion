"""Project B-spline control points of a generated gesture into the guardian zone.

Each control point (one joint vector) that lies outside the calibrated zone is moved to the
NEAREST point inside it (shortest Euclidean translation); points already inside are not touched.
The zone is the box [limit + margin, limit - margin] of each joint intersected with the coupled
constraints (half-spaces sum(coef * q) <= bound) whose joints all belong to the group: a convex
set, so the nearest point is unique (Dykstra's alternating projections).

Because a B-spline stays inside the convex hull of its control points, the decoded curve then
stays inside the zone for every position constraint handled here. Velocities and accelerations
are NOT covered: the playback adaptation (playable.py) still slows the gesture down when needed,
and the guardian still checks the played result. Coupled constraints that mix groups (arm joints
with HeadYaw) cannot be enforced on control points, since the groups have their own time bases:
the sample-level adaptation keeps handling them.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..config import Config


def zone(cfg: Config, joints: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, float]]]:
    """(lo, hi, half-spaces [(a, b)]) of the zone for these joints, margins included."""
    g = cfg.guardian
    js = list(joints)
    lo = np.array([cfg.limits[j][0] + g.margin(j) for j in js])
    hi = np.array([cfg.limits[j][1] - g.margin(j) for j in js])
    half = []
    for c in g.coupled_constraints:
        if all(j in js for j in c.terms):
            a = np.zeros(len(js))
            for j, coef in c.terms.items():
                a[js.index(j)] = float(coef)
            half.append((a, float(c.bound) - 1e-3))
    return lo, hi, half


def _inside(x, lo, hi, half, tol=1e-9) -> bool:
    return bool(np.all(x >= lo - tol) and np.all(x <= hi + tol) and all(a @ x <= b + tol for a, b in half))


def project_point(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, half, iters: int = 200) -> np.ndarray:
    """Nearest point of {lo <= y <= hi, a.y <= b} to x (Dykstra)."""
    x = np.asarray(x, float)
    if _inside(x, lo, hi, half):
        return x.copy()
    sets = [("box", None)] + [("half", h) for h in half]
    y = x.copy()
    inc = [np.zeros_like(x) for _ in sets]
    for _ in range(iters):
        prev = y.copy()
        for k, (kind, h) in enumerate(sets):
            z = y + inc[k]
            if kind == "box":
                p = np.clip(z, lo, hi)
            else:
                a, b = h
                s = a @ z - b
                p = z - (s / float(a @ a)) * a if s > 0 else z
            inc[k] = z - p
            y = p
        if np.abs(y - prev).max() < 1e-10:
            break
    return np.clip(y, lo, hi)  # the box is never left, even if the half-spaces could not all be met


def project_control_points(cfg: Config, ctrl: np.ndarray, joints: Sequence[str]):
    """Returns (projected control points, max translation in rad, number of points moved)."""
    lo, hi, half = zone(cfg, joints)
    out = np.array([project_point(p, lo, hi, half) for p in np.asarray(ctrl, float)])
    d = np.linalg.norm(out - ctrl, axis=1)
    return out, float(d.max()) if d.size else 0.0, int((d > 1e-9).sum())
