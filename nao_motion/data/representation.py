"""Compact temporal representation of trajectories.

Choice: time normalised into a phase u ∈ [0, 1] + a separate duration, and K
cubic B-spline control points (uniform, clamped) per joint.

Why: with few demonstrations, few output parameters are needed (K×J instead of
N×J), a curve that is smooth by construction (continuous velocities and
accelerations, good for the guardian) and a LINEAR relation between parameters
and trajectory: interpolating two sets of control points gives a consistent
intermediate trajectory, and fitting is a simple least-squares problem.

Different durations: each take is mapped to the phase of its useful segment (by
default the whole take; for a head take, the arms replay window), linearly
resampled on a uniform phase grid, then projected onto the basis. The real
duration is learned separately. The raw data is never modified. Known limitation:
this linear alignment does not realign the sub-phases of a gesture (no DTW); that
is a possible improvement.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from ..trajectory import Trajectory, TrajectoryError

DEGREE = 3


@lru_cache(maxsize=64)
def _knots(n_ctrl: int) -> Tuple[float, ...]:
    if n_ctrl < DEGREE + 1:
        raise ValueError("n_ctrl >= 4 requis pour une B-spline cubique")
    inner = np.linspace(0.0, 1.0, n_ctrl - DEGREE + 1)
    return tuple(np.concatenate([[0.0] * DEGREE, inner, [1.0] * DEGREE]).tolist())


def bspline_basis(n_ctrl: int, u: np.ndarray) -> np.ndarray:
    """Basis matrix (len(u), n_ctrl), Cox–de Boor algorithm."""
    k = np.array(_knots(n_ctrl))
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    # degree 0
    B = np.zeros((u.size, len(k) - 1))
    for i in range(len(k) - 1):
        B[:, i] = (u >= k[i]) & (u < k[i + 1])
    last_nonzero = np.max(np.nonzero(np.diff(k))[0])
    B[u >= 1.0, :] = 0.0  # end point included in the last non-empty interval
    B[u >= 1.0, last_nonzero] = 1.0
    for d in range(1, DEGREE + 1):
        Bn = np.zeros((u.size, len(k) - 1 - d))
        for i in range(len(k) - 1 - d):
            den1 = k[i + d] - k[i]
            den2 = k[i + d + 1] - k[i + 1]
            a = (u - k[i]) / den1 * B[:, i] if den1 > 0 else 0.0
            b = (k[i + d + 1] - u) / den2 * B[:, i + 1] if den2 > 0 else 0.0
            Bn[:, i] = a + b
        B = Bn
    return B


def resample_phase(t: np.ndarray, q: np.ndarray, n: int,
                   segment: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, float]:
    """Resamples q(t) on n uniform phase points within `segment`.

    Returns (uniform q (n, J), segment duration in s).
    """
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    t0, t1 = (float(t[0]), float(t[-1])) if segment is None else (float(segment[0]), float(segment[1]))
    if not (t[0] - 1e-9 <= t0 < t1 <= t[-1] + 1e-9):
        raise TrajectoryError(f"Segment [{t0}, {t1}] hors de l'enregistrement [{t[0]}, {t[-1]}]")
    tt = np.linspace(t0, t1, n)
    out = np.stack([np.interp(tt, t, q[:, j]) for j in range(q.shape[1])], axis=-1)
    return out, t1 - t0


def fit_control_points(q_uniform: np.ndarray, n_ctrl: int, ridge: float = 1e-6) -> np.ndarray:
    """Least squares: q ≈ B · C; C of shape (n_ctrl, J). End points fixed exactly."""
    n = q_uniform.shape[0]
    B = bspline_basis(n_ctrl, np.linspace(0, 1, n))
    A = B.T @ B + ridge * np.eye(n_ctrl)
    C = np.linalg.solve(A, B.T @ q_uniform)
    # clamped B-spline: first and last control points = curve end points
    C[0], C[-1] = q_uniform[0], q_uniform[-1]
    return C


def decode(ctrl: np.ndarray, duration: float, joints, rate_hz: float = 50.0) -> Trajectory:
    """Control points (K, J) + duration -> timestamped trajectory starting at t=0."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    if not np.all(np.isfinite(ctrl)) or not np.isfinite(duration) or duration <= 0:
        raise TrajectoryError("Paramètres de trajectoire invalides")
    n = max(4, int(np.ceil(duration * rate_hz)) + 1)
    u = np.linspace(0, 1, n)
    q = bspline_basis(ctrl.shape[0], u) @ ctrl
    return Trajectory(list(joints), u * duration, q)


def encode_trajectory(traj: Trajectory, n_ctrl: int, n_samples: int,
                      segment: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, float]:
    qu, dur = resample_phase(traj.t, traj.q, n_samples, segment)
    return fit_control_points(qu, n_ctrl), dur


def suggest_segment(traj: Trajectory, threshold_rad: float = 0.03, pad_s: float = 0.15) -> Tuple[float, float]:
    """Suggests the useful segment of a take: from the first to the last moment where
    a joint departs by more than `threshold_rad` from its initial
    (resp. final) posture, with a small margin. The raw data is not
    modified; only the segment is stored in the metadata."""
    d0 = np.max(np.abs(traj.q - traj.q[0]), axis=1)
    d1 = np.max(np.abs(traj.q - traj.q[-1]), axis=1)
    moving0 = np.nonzero(d0 > threshold_rad)[0]
    moving1 = np.nonzero(d1 > threshold_rad)[0]
    if moving0.size == 0 or moving1.size == 0:
        return float(traj.t[0]), float(traj.t[-1])
    t0 = max(float(traj.t[0]), float(traj.t[moving0[0]]) - pad_s)
    t1 = min(float(traj.t[-1]), float(traj.t[moving1[-1]]) + pad_s)
    if t1 - t0 < 0.2:
        return float(traj.t[0]), float(traj.t[-1])
    return t0, t1


def greville(n_ctrl: int) -> np.ndarray:
    """Phase (in [0, 1]) where each control point acts most: the Greville abscissae,
    used to draw the control points along the curve they shape."""
    k = np.array(_knots(n_ctrl))
    return np.array([k[i + 1:i + DEGREE + 1].mean() for i in range(n_ctrl)])


def reconstruct(traj: Trajectory, n_ctrl: int, n_samples: int = 100,
                segment: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the model is asked to learn from a take, against the take itself.

    The useful segment is resampled on `n_samples` uniform phase points (as for training),
    fitted with `n_ctrl` control points per joint, and the fitted B-spline evaluated on the
    same points. Returns (t (n,) in the take's own absolute time, raw (n, J), fitted (n, J)).
    The gap between raw and fitted is the smoothing due to the representation alone."""
    t0, t1 = (float(traj.t[0]), float(traj.t[-1])) if segment is None else (float(segment[0]), float(segment[1]))
    qu, dur = resample_phase(traj.t, traj.q, n_samples, (t0, t1))
    ctrl = fit_control_points(qu, n_ctrl)
    fitted = bspline_basis(n_ctrl, np.linspace(0, 1, n_samples)) @ ctrl
    return np.linspace(t0, t1, n_samples), qu, fitted
