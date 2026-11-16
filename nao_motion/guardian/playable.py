"""Make a generated gesture playable before the guardian checks it.

The guardian protects the robot when a gesture is PLAYED; it never filters the
training data. A model that imitates the recorded takes faithfully can still
propose gestures that are faster than the configured limits or that brush the
edge of the calibrated zone. Instead of refusing those gestures outright, the
playback path adapts them first:

1. shape: each joint is brought back inside its limits (margin included) with a
   smooth saturation (no kink, so no artificial acceleration spike), and the
   coupled constraints are enforced the same way along their normal direction;
2. timing: the gesture is slowed down uniformly, just enough for the velocity
   and acceleration limits (never sped up), within the duration bounds.

The guardian then validates the result as usual and stays the only judge: a
gesture that still fails is refused and nothing is sent. This module never
changes the guardian's rules, and it does not touch recorded takes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..trajectory import Trajectory

# width (rad) of the band, inside the allowed range, where saturation starts
SOFT_BAND_RAD = 0.06
# extra slow-down on top of the computed factor, so that the check passes with some room
TIME_SAFETY = 1.05


@dataclass
class Adaptation:
    traj: Trajectory
    time_scale: float = 1.0
    max_shape_change_rad: float = 0.0
    joints_changed: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.time_scale > 1.0 + 1e-9 or self.max_shape_change_rad > 1e-6

    def to_dict(self):
        return {"changed": self.changed, "time_scale": round(self.time_scale, 3),
                "max_shape_change_rad": round(self.max_shape_change_rad, 4),
                "joints_changed": self.joints_changed, "notes": self.notes}


def _soft_upper(x: np.ndarray, hi: float, band: float) -> np.ndarray:
    """Identity below hi - band; above, compressed smoothly so that it never exceeds hi.
    C1 continuous (slope 1 at the junction), so no velocity jump is introduced."""
    knee = hi - band
    out = x.copy()
    m = x > knee
    out[m] = knee + band * np.tanh((x[m] - knee) / band)
    return out


def _soft_box(x: np.ndarray, lo: float, hi: float, band: float) -> np.ndarray:
    band = max(1e-4, min(band, (hi - lo) / 4))
    return -_soft_upper(-_soft_upper(x, hi, band), -lo, band)


def _shape(cfg: Config, traj: Trajectory, context_pose: Dict[str, float], rounds: int = 6) -> np.ndarray:
    g = cfg.guardian
    q = traj.q.copy()
    bounds = []
    for j in traj.joints:
        lo, hi = cfg.limits[j]
        m = g.margin(j)
        bounds.append((lo + m, hi - m))
    for _ in range(rounds):
        for c in g.coupled_constraints:
            idx = [(traj.joints.index(j), coef) for j, coef in c.terms.items() if j in traj.joints]
            if not idx or any(j not in traj.joints and j not in context_pose for j in c.terms):
                continue  # the guardian skips it too (unknown joint)
            fixed = sum(coef * float(context_pose[j]) for j, coef in c.terms.items() if j not in traj.joints)
            cvec = np.array([coef for _, coef in idx])
            cols = [k for k, _ in idx]
            s = q[:, cols] @ cvec + fixed
            s_new = _soft_upper(s, c.bound - 1e-3, SOFT_BAND_RAD)
            delta = s_new - s
            if np.any(delta):
                q[:, cols] += np.outer(delta, cvec / float(cvec @ cvec))
        for k, (lo, hi) in enumerate(bounds):
            q[:, k] = _soft_box(q[:, k], lo, hi, SOFT_BAND_RAD)
    return q


def _time_scale(cfg: Config, t: np.ndarray, q: np.ndarray, joints) -> float:
    g = cfg.guardian
    dt = np.diff(t)
    if dt.size == 0:
        return 1.0
    vel = np.diff(q, axis=0) / dt[:, None]
    s = 1.0
    for k, j in enumerate(joints):
        s = max(s, float(np.abs(vel[:, k]).max()) / g.velocity_limit(j))
    if vel.shape[0] >= 2:
        acc = np.diff(vel, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]
        for k, j in enumerate(joints):
            s = max(s, float(np.sqrt(np.abs(acc[:, k]).max() / g.acceleration_limit(j))))
    return s


def _cubic_resample(t: np.ndarray, q: np.ndarray, tt: np.ndarray) -> np.ndarray:
    """Cubic Hermite interpolation (tangents from np.gradient). Linear interpolation
    between samples would add kinks, i.e. artificial acceleration spikes."""
    m = np.gradient(q, t, axis=0)
    i = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, len(t) - 2)
    h = (t[i + 1] - t[i])[:, None]
    u = ((tt - t[i]) / h[:, 0])[:, None]
    h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
    h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * q[i] + h10 * h * m[i] + h01 * q[i + 1] + h11 * h * m[i + 1]


def _grid(duration: float, rate_hz: float) -> np.ndarray:
    # same grid as Trajectory.resampled, so that the controller's own resampling is a no-op
    return np.linspace(0.0, duration, max(2, int(np.ceil(duration * rate_hz)) + 1))


def make_playable(cfg: Config, traj: Trajectory, context_pose: Optional[Dict[str, float]] = None,
                  resample_hz: float = 50.0) -> Adaptation:
    """Adapted copy of `traj` (see module docstring). The guardian must still validate it."""
    g = cfg.guardian
    ctx = dict(context_pose or {})
    t0 = traj.t - traj.t[0]
    tg = _grid(float(t0[-1]), resample_hz)
    src = Trajectory(traj.joints, tg, _cubic_resample(t0, traj.q, tg))
    q = _shape(cfg, src, ctx)
    diff = np.abs(q - src.q).max(axis=0)
    changed = [j for j, d in zip(src.joints, diff) if d > 1e-3]
    notes = []
    dur = src.duration

    def stretch(s):
        tn = _grid(dur * s, resample_hz)
        return Trajectory(src.joints, tn, _cubic_resample(src.t, q, np.minimum(tn / s, src.t[-1])))

    s = max(1.0, g.min_duration_s / dur)
    slowed = stretch(s)
    # uniform slow-down (same path, longer time). Measured on the played samples themselves,
    # as the guardian does; a few rounds because the finer sampling can reveal sharper peaks.
    for _ in range(6):
        need = _time_scale(cfg, slowed.t, slowed.q, slowed.joints)
        if need <= 1.0:
            break
        s_new = s * need * TIME_SAFETY
        if dur * s_new > g.max_duration_s:
            s_new = max(s, g.max_duration_s / dur)
            if s_new == s:
                break
            notes.append(f"slow-down capped by max duration ({g.max_duration_s} s): the guardian may refuse")
            s, slowed = s_new, stretch(s_new)
            break
        s, slowed = s_new, stretch(s_new)
    if changed:
        notes.append(f"brought back inside limits / coupled zone: {changed} (max {diff.max():.3f} rad)")
    if s > 1.0 + 1e-9:
        notes.append(f"slowed down x{s:.2f} for velocity / acceleration limits")
    return Adaptation(slowed, s, float(diff.max()) if diff.size else 0.0, changed, notes)
