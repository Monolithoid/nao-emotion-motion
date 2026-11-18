"""Make a generated gesture playable before the guardian checks it.

The guardian protects the robot when a gesture is PLAYED; it never filters the
training data. A model that imitates the recorded takes faithfully can still
propose gestures that are faster than the configured limits or that brush the
edge of the calibrated zone. Instead of refusing those gestures outright, the
playback path adapts them first:

1. shape, only where the robot needs it: a joint that goes past its motor limit
   (margin included) is brought back with a smooth saturation (a joint that
   stays inside is not touched), and where two links would collide the path is
   deviated as little as possible around that moment (repair_collisions); the
   rest of the gesture is left exactly as the model made it;
2. timing: the gesture is slowed down just enough for the velocity and
   acceleration limits (never sped up), within the duration bounds. By default
   only the moments that exceed a limit are slowed down (`local`: a smooth time
   warp, the rest keeps the tempo of the model); `uniform` slows the whole
   gesture by one factor. In both cases the path is the same, only its timing
   changes. robot.yaml: execution.slow_down.

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
from . import collision

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


def _bump_spread(D: np.ndarray, mask: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Spread corrections D (N, J), non-zero where mask, over the neighbouring samples with a
    Gaussian bump: the correction reaches its full value on the corrected samples and fades out
    smoothly around them (no kink, so no velocity or acceleration jump)."""
    r = max(1, int(np.ceil(3 * sigma_samples)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    dens = np.convolve(mask.astype(float), k, mode="same")
    out = np.stack([np.convolve(D[:, j], k, mode="same") for j in range(D.shape[1])], axis=1)
    return out / np.maximum(1.0, dens)[:, None]


# how far (s) a collision correction spreads around the colliding moments
REPAIR_SIGMA_S = 0.12
# extra distance (m) above the clearance aimed at by the correction
REPAIR_MARGIN_M = 0.004


def repair_collisions(cfg: Config, joints, t: np.ndarray, q: np.ndarray, context_pose: Dict[str, float],
                      rounds: int = 40) -> np.ndarray:
    """Smallest smooth change of the path that keeps every link at the clearance distance.

    Only the moments where two links come too close are changed: at each of them the joints
    of the links involved move along the gradient of their distance (least-norm step), and
    the change is spread over ~REPAIR_SIGMA_S around it. Elsewhere the gesture is untouched,
    and its timing never changes."""
    body = collision.for_config(cfg)
    if not body.enabled or len(t) < 2:
        return q
    q = q.copy()
    sigma = REPAIR_SIGMA_S / float(np.median(np.diff(t)))
    target = body.clearance + REPAIR_MARGIN_M
    eps = 1e-3
    for _ in range(rounds):
        dist = body.distances(joints, q, context_pose)
        bad_any = np.zeros(len(t), bool)
        for d in dist.values():
            bad_any |= d < body.clearance
        if not bad_any.any():
            break
        D = np.zeros_like(q)
        mask = np.zeros(len(t), bool)
        for pair, d in dist.items():
            idx = np.nonzero(d < target)[0]
            if idx.size == 0:
                continue
            grad = np.zeros((idx.size, len(joints)))
            for k in range(len(joints)):
                qp = q[idx].copy()
                qp[:, k] += eps
                grad[:, k] = (body.distances(joints, qp, context_pose)[pair] - d[idx]) / eps
            n2 = (grad ** 2).sum(axis=1)
            ok = n2 > 1e-9
            step = np.zeros_like(grad)
            step[ok] = ((target - d[idx[ok]]) / n2[ok])[:, None] * grad[ok]
            D[idx] += step
            mask[idx] = True
        q = q + _bump_spread(D, mask, sigma)
    return q


def _shape(cfg: Config, traj: Trajectory, context_pose: Dict[str, float], rounds: int = 6) -> np.ndarray:
    """Path changes that the robot needs, and only those: a joint that goes past its limit
    (margin included) is brought back smoothly (only that joint, nothing changes when it stays
    inside), the coupled constraints chosen by the operator likewise, then collisions are
    removed with the least change (repair_collisions)."""
    g = cfg.guardian
    q = traj.q.copy()
    bounds = []
    for j in traj.joints:
        lo, hi = cfg.guardian_limits[j]
        m = g.margin(j)
        bounds.append((lo + m, hi - m))
    def limits(q):
        for _ in range(rounds):
            for c in g.coupled_constraints:
                idx = [(traj.joints.index(j), coef) for j, coef in c.terms.items() if j in traj.joints]
                if not idx or any(j not in traj.joints and j not in context_pose for j in c.terms):
                    continue  # the guardian skips it too (unknown joint)
                fixed = sum(coef * float(context_pose[j]) for j, coef in c.terms.items() if j not in traj.joints)
                cvec = np.array([coef for _, coef in idx])
                cols = [k for k, _ in idx]
                s = q[:, cols] @ cvec + fixed
                if not np.any(s > c.bound - 1e-3):
                    continue
                s_new = _soft_upper(s, c.bound - 1e-3, SOFT_BAND_RAD)
                delta = s_new - s
                if np.any(delta):
                    q[:, cols] += np.outer(delta, cvec / float(cvec @ cvec))
            for k, (lo, hi) in enumerate(bounds):
                if np.any(q[:, k] < lo) or np.any(q[:, k] > hi):
                    q[:, k] = _soft_box(q[:, k], lo, hi, SOFT_BAND_RAD)
        return q

    # the collision repair can push a joint past its limit, and the limit saturation can bring
    # two links together again: a few alternations, ending with the limits (never exceeded)
    for _ in range(3):
        q = repair_collisions(cfg, traj.joints, traj.t, limits(q), context_pose)
    return limits(q)


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


def _local_need(cfg: Config, t: np.ndarray, q: np.ndarray, joints) -> np.ndarray:
    """Slow-down needed on each interval [t_i, t_i+1]: velocity ratio, and the square root of
    the acceleration ratio at both ends (time x s divides velocity by s, acceleration by s²)."""
    g = cfg.guardian
    dt = np.diff(t)
    vel = np.diff(q, axis=0) / dt[:, None]
    vlim = np.array([g.velocity_limit(j) for j in joints])
    need = np.abs(vel / vlim).max(axis=1)
    if vel.shape[0] >= 2:
        alim = np.array([g.acceleration_limit(j) for j in joints])
        acc = np.diff(vel, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]
        ra = np.sqrt(np.abs(acc / alim).max(axis=1))
        need[:-1] = np.maximum(need[:-1], ra)
        need[1:] = np.maximum(need[1:], ra)
    return need


def _warp_rate(need: np.ndarray, window: int) -> np.ndarray:
    """Smooth local slow-down factor >= need (and >= 1): widened by a running max, then a
    running mean of the same width, so that the factor changes gradually (an abrupt change of
    tempo would itself create an acceleration peak)."""
    r = np.maximum(need * TIME_SAFETY, 1.0)
    r[need <= 1.0] = 1.0
    n = len(r)
    w = max(1, min(window, n))
    pad = np.pad(r, (w, w), mode="edge")
    mx = np.array([pad[i:i + 2 * w + 1].max() for i in range(n)])
    pad = np.pad(mx, (w, w), mode="edge")
    k = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(pad, k, mode="valid")


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


# width (s) of the smoothing of the local slow-down factor
WARP_WINDOW_S = 0.2


def slow_down_mode(cfg: Config) -> str:
    mode = str((cfg.robot.get("execution") or {}).get("slow_down", "local"))
    return mode if mode in ("local", "uniform") else "local"


def make_playable(cfg: Config, traj: Trajectory, context_pose: Optional[Dict[str, float]] = None,
                  resample_hz: float = 50.0, mode: Optional[str] = None) -> Adaptation:
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
    mode = mode or slow_down_mode(cfg)
    base = Trajectory(src.joints, src.t, q)
    s0 = max(1.0, g.min_duration_s / dur)
    if s0 > 1.0:
        base = _stretch(base, s0, resample_hz)
    local = 1.0  # duration ratio of the local time warp (1 = none)
    if mode == "local" and _time_scale(cfg, base.t, base.q, base.joints) > 1.0:
        warped = _local_warp(cfg, base, resample_hz)
        if warped is not None:
            local = warped.duration / base.duration
            base = warped
    # uniform slow-down (same path, longer time), alone or to finish the local warp. Measured
    # on the played samples themselves, as the guardian does; a few rounds because the finer
    # sampling can reveal sharper peaks.
    s, slowed = 1.0, base
    for _ in range(6):
        need = _time_scale(cfg, slowed.t, slowed.q, slowed.joints)
        if need <= 1.0:
            break
        s_new = s * need * TIME_SAFETY
        if base.duration * s_new > g.max_duration_s:
            s_new = max(s, g.max_duration_s / base.duration)
            if s_new == s:
                break
            notes.append(f"slow-down capped by max duration ({g.max_duration_s} s): the guardian may refuse")
            s, slowed = s_new, _stretch(base, s_new, resample_hz)
            break
        s, slowed = s_new, _stretch(base, s_new, resample_hz)
    if changed:
        notes.append(f"path changed for the motor limits / self-collision: {changed} (max {diff.max():.3f} rad)")
    if local > 1.0 + 1e-9:
        notes.append(f"fast moments slowed down locally: duration x{local:.2f}")
    if s > 1.0 + 1e-9:
        notes.append(f"slowed down uniformly x{s:.2f} for velocity / acceleration limits")
    # the cubic resampling of the time warp can overshoot a saturated joint by a few 1e-5 rad
    lo = np.array([cfg.guardian_limits[j][0] + g.margin(j) for j in slowed.joints])
    hi = np.array([cfg.guardian_limits[j][1] - g.margin(j) for j in slowed.joints])
    slowed = Trajectory(slowed.joints, slowed.t, np.clip(slowed.q, lo, hi))
    total = slowed.duration / dur
    return Adaptation(slowed, total if total > 1.0 + 1e-9 else 1.0,
                      float(diff.max()) if diff.size else 0.0, changed, notes)


def _stretch(traj: Trajectory, s: float, resample_hz: float) -> Trajectory:
    """Same path, played s times slower (uniform)."""
    tn = _grid(traj.duration * s, resample_hz)
    return Trajectory(traj.joints, tn, _cubic_resample(traj.t, traj.q, np.minimum(tn / s, traj.t[-1])))


def _local_warp(cfg: Config, traj: Trajectory, resample_hz: float, rounds: int = 8) -> Optional[Trajectory]:
    """Slow down only where a limit is exceeded (smooth time warp, same path). None if the
    result would exceed the maximum duration."""
    g = cfg.guardian
    w = int(round(WARP_WINDOW_S * resample_hz))
    out = traj
    for _ in range(rounds):
        need = _local_need(cfg, out.t, out.q, out.joints)
        if need.max() <= 1.0:
            break
        tw = np.concatenate([[0.0], np.cumsum(np.diff(out.t) * _warp_rate(need, w))])
        if tw[-1] > g.max_duration_s:
            return None
        tn = _grid(float(tw[-1]), resample_hz)
        out = Trajectory(out.joints, tn, _cubic_resample(tw, out.q, np.minimum(tn, tw[-1])))
    return out
