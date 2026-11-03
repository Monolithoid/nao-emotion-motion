"""Transitions gérées par le contrôleur (repos <-> départ <-> geste), non apprises."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..trajectory import Trajectory


def min_jerk(q0: Sequence[float], q1: Sequence[float], joints, v_max: float, a_max: float,
             rate_hz: float = 50.0, min_duration: float = 0.4) -> Trajectory:
    """Profil à jerk minimal (polynôme de degré 5), vitesse et accélération nulles aux bouts.

    Vitesse crête = 1.875·Δ/T ; accélération crête ≈ 5.774·Δ/T². T est choisi
    pour respecter v_max et a_max sur l'articulation la plus contraignante.
    """
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.max(np.abs(q1 - q0))) if q0.size else 0.0
    T = max(min_duration, 1.875 * d / v_max, float(np.sqrt(5.774 * d / a_max)) if d > 0 else 0.0)
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


def quintic_matched(q0, v0, a0, q1, v1, a1, joints, v_max: float, a_max: float,
                    rate_hz: float = 50.0, min_duration: float = 0.4, max_duration: float = 6.0) -> Trajectory:
    """Transition quintique raccordée en position, vitesse et accélération aux deux bouts.

    Sert à entrer dans un geste généré (qui peut commencer avec une vitesse non
    nulle) et à en sortir, sans discontinuité de vitesse. La durée est allongée
    jusqu'à respecter v_max/a_max ; le guardian revalide de toute façon le tout.
    """
    q0, v0, a0, q1, v1, a1 = (np.asarray(x, float) for x in (q0, v0, a0, q1, v1, a1))
    d = float(np.max(np.abs(q1 - q0))) if q0.size else 0.0
    T = max(min_duration, 1.875 * d / v_max)
    best = None
    while T <= max_duration:
        n = max(3, int(np.ceil(T * rate_hz)) + 1)
        t = np.linspace(0, T, n)
        C = _quintic_coeffs(q0, v0, a0, q1, v1, a1, T)
        P = np.stack([t**k for k in range(6)], axis=1) @ C
        V = np.stack([k * t**(k - 1) if k else 0 * t for k in range(6)], axis=1) @ C
        A = np.stack([k * (k - 1) * t**(k - 2) if k > 1 else 0 * t for k in range(6)], axis=1) @ C
        best = Trajectory(list(joints), t, P)
        if np.abs(V).max() <= v_max and np.abs(A).max() <= a_max:
            break
        T *= 1.15
    return best


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
