"""Transitions gérées par le contrôleur (repos <-> départ <-> geste), non apprises."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ..trajectory import Trajectory


def min_jerk(q0: Sequence[float], q1: Sequence[float], joints, v_max: float, a_max: float,
             rate_hz: float = 50.0, min_duration: float = 0.4) -> Trajectory:
    """Profil à jerk minimal (polynôme de degré 5), vitesse et accélération nulles aux bouts.

    Vitesse crête = 1.875·Δ/T ; accélération crête ≈ 5.774·Δ/T². T est choisi
    pour respecter v_max et a_max sur l'articulation la plus contraignante.
    """
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = np.abs(q1 - q0)  # v_max / a_max: one value or one per joint
    T = max([min_duration] + ([float(np.max(1.875 * d / v_max)), float(np.max(np.sqrt(5.774 * d / a_max)))]
                              if q0.size else []))
    n = max(3, int(np.ceil(T * rate_hz)) + 1)
    s = np.linspace(0, 1, n)
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    return Trajectory(list(joints), s * T, q0[None, :] + h[:, None] * (q1 - q0)[None, :])


def _quintic_coeffs(q0, v0, a0, q1, v1, a1, T):
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    c0, c1, c2 = q0, v0, a0 / 2
    c3 = (20 * (q1 - q0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T2) / (2 * T3)
    c4 = (30 * (q0 - q1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T2) / (2 * T4)
    c5 = (12 * (q1 - q0) - (6 * v1 + 6 * v0) * T - (a0 - a1) * T2) / (2 * T5)
    return np.stack([c0, c1, c2, c3, c4, c5])


def quintic_matched(q0, v0, a0, q1, v1, a1, joints, v_max, a_max,
                    rate_hz: float = 50.0, min_duration: float = 0.4, max_duration: float = 6.0,
                    accept: Optional[Callable[[Trajectory], bool]] = None) -> Trajectory:
    """Transition matched in position, velocity and acceleration at both ends (see _candidates).

    v_max / a_max: caps, one value or one per joint (raised to the matched boundary values).
    Among the durations that respect the caps, the SHORTEST one that `accept` approves is kept
    (accept: e.g. "stays inside the guardian zone"); without accept, or when none is approved,
    the shortest one. A transition that must reach a moving gesture overshoots more the longer
    it lasts (it has to wind up to the gesture's entry speed), so a long transition leaves the
    zone: the old search started at a duration set by the global default speed (1.2 rad/s) and
    lengthened it, which pushed most transitions out of the calibrated zone.

    When no duration respects the caps while matching the boundary ACCELERATION too, the
    acceleration match is dropped (position and velocity stay continuous)."""
    z = np.zeros(np.shape(a0))
    leasts = []
    for a_s, a_e in ((a0, a1), (z, z)):
        cands, least = _candidates(q0, v0, a_s, q1, v1, a_e, joints, v_max, a_max, rate_hz,
                                   min_duration, max_duration)
        if cands:
            if accept is not None:
                for c in cands:
                    if accept(c):
                        return c
            return cands[0]
        leasts.append(least)
    return min(leasts, key=lambda x: x[1])[0]  # nothing within the caps: the least bad


def _candidates(q0, v0, a0, q1, v1, a1, joints, v_max, a_max, rate_hz, min_duration, max_duration):
    """Quintic transitions for durations min_duration x 1.15^k up to max_duration. Returns
    (those within the caps, shortest first; (least bad transition, its score))."""
    q0, v0, a0, q1, v1, a1 = (np.asarray(x, float) for x in (q0, v0, a0, q1, v1, a1))
    vm = np.broadcast_to(np.asarray(v_max, float), q0.shape)
    am = np.broadcast_to(np.asarray(a_max, float), q0.shape)
    # the gesture may enter or leave faster than the caps: never ask for less than its own speed
    vcap = np.maximum(vm, np.maximum(np.abs(v0), np.abs(v1)))
    acap = np.maximum(am, np.maximum(np.abs(a0), np.abs(a1)))
    T = float(min_duration)
    ok, least, least_score = [], None, np.inf
    while T <= max_duration + 1e-9:
        n = max(3, int(np.ceil(T * rate_hz)) + 1)
        t = np.linspace(0, T, n)
        C = _quintic_coeffs(q0, v0, a0, q1, v1, a1, T)
        P = np.stack([t**k for k in range(6)], axis=1) @ C
        V = np.stack([k * t**(k - 1) if k else 0 * t for k in range(6)], axis=1) @ C
        A = np.stack([k * (k - 1) * t**(k - 2) if k > 1 else 0 * t for k in range(6)], axis=1) @ C
        score = max(float((np.abs(V) / vcap).max()), float((np.abs(A) / acap).max())) if q0.size else 0.0
        tr = Trajectory(list(joints), t, P)
        if score <= 1.0:
            ok.append(tr)
        if score < least_score - 1e-9:
            least, least_score = tr, score
        T *= 1.15
    return ok, (least, least_score)


def ease(traj: Trajectory, ease_in_s: float = 0.0, ease_out_s: float = 0.0) -> Trajectory:
    """Same path, retimed so that it starts (ease_in_s > 0) and/or ends (ease_out_s > 0) at rest.

    The first ease_in_s seconds of the gesture are played over 2 x ease_in_s with a time warp
    t = Tw·(u³ - u⁴/2), u = τ/Tw, Tw = 2·ease_in_s: zero velocity and acceleration at τ = 0,
    velocity and acceleration of the gesture at τ = Tw. Symmetric at the end. The joint-space
    path, hence its place in the guardian zone, is unchanged; only its timing is. The eased ends
    can then be joined to the start pose by a straight min-jerk segment, which stays inside the
    zone (the zone is convex) where a velocity-matched quintic may overshoot out of it.
    Durations are rounded up to the trajectory's time step so that the middle samples are the
    original ones."""
    h = float(traj.t[1] - traj.t[0])
    D = traj.duration
    ti = float(np.ceil(ease_in_s / h - 1e-9) * h) if ease_in_s > 0 else 0.0
    to = float(np.ceil(ease_out_s / h - 1e-9) * h) if ease_out_s > 0 else 0.0
    if ti + to > D:
        raise ValueError("ease longer than the trajectory")
    N = D + ti + to
    tau = np.linspace(0.0, N, int(round(N / h)) + 1)
    t = tau - ti
    if ti > 0:
        m = tau < 2 * ti
        u = tau[m] / (2 * ti)
        t[m] = 2 * ti * (u**3 - u**4 / 2)
    if to > 0:
        m = tau > N - 2 * to
        u = (N - tau[m]) / (2 * to)
        t[m] = D - 2 * to * (u**3 - u**4 / 2)
    return Trajectory(traj.joints, tau, traj.sample(traj.t[0] + np.clip(t, 0.0, D)))


def ease_duration(v, a_max, min_s: float = 0.15) -> float:
    """Gesture time consumed by an ease from/to rest at boundary velocity v. The warp adds up to
    1.5·|v|/Tw = 0.75·|v|/ease of acceleration; this keeps it to half of a_max."""
    v = np.abs(np.asarray(v, float))
    return float(max(min_s, np.max(1.5 * v / np.asarray(a_max, float)))) if v.size else min_s


def boundary_derivatives(traj: Trajectory, at_end: bool = False):
    """Vitesse et accélération (différences finies) au début ou à la fin d'une trajectoire."""
    q, t = traj.q, traj.t
    if at_end:
        q, t = q[::-1], t[::-1]  # pas de temps négatifs : dérivées dans le bon sens
    if len(t) < 3:
        return np.zeros(q.shape[1]), np.zeros(q.shape[1])
    dt1, dt2 = t[1] - t[0], t[2] - t[1]
    v1 = (q[1] - q[0]) / dt1
    v2 = (q[2] - q[1]) / dt2
    a = (v2 - v1) / ((dt1 + dt2) / 2)
    v0 = v1 - a * dt1 / 2
    return v0, a
