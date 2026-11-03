"""Représentation temporelle compacte des trajectoires.

Choix : temps normalisé en phase u ∈ [0, 1] + durée séparée, et K points de
contrôle B-spline cubique (uniforme, bornée) par articulation.

Pourquoi : avec peu de démonstrations, il faut peu de paramètres de sortie
(K×J au lieu de N×J), une courbe lisse par construction (vitesses et
accélérations continues, bon pour le guardian) et une relation LINÉAIRE entre
paramètres et trajectoire : interpoler deux jeux de points de contrôle donne
une trajectoire intermédiaire cohérente, et l'ajustement est un simple
moindres carrés.

Durées différentes : chaque prise est ramenée à la phase de son segment utile
(par défaut toute la prise ; pour une prise de tête, la fenêtre de relecture
des bras), rééchantillonnée linéairement sur une grille uniforme de phase, puis
projetée sur la base. La durée réelle est apprise à part. Les données brutes
ne sont jamais modifiées. Limite connue : cet alignement linéaire ne recale pas
les sous-phases d'un geste (pas de DTW) ; c'est une évolution possible.
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
    """Matrice de base (len(u), n_ctrl), algorithme de Cox–de Boor."""
    k = np.array(_knots(n_ctrl))
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    # degré 0
    B = np.zeros((u.size, len(k) - 1))
    for i in range(len(k) - 1):
        B[:, i] = (u >= k[i]) & (u < k[i + 1])
    last_nonzero = np.max(np.nonzero(np.diff(k))[0])
    B[u >= 1.0, :] = 0.0  # point final inclus dans le dernier intervalle non vide
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
    """Rééchantillonne q(t) sur n points uniformes de phase dans `segment`.

    Retourne (q_uniforme (n, J), durée du segment en s).
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
    """Moindres carrés : q ≈ B · C ; C de forme (n_ctrl, J). Extrémités fixées exactement."""
    n = q_uniform.shape[0]
    B = bspline_basis(n_ctrl, np.linspace(0, 1, n))
    A = B.T @ B + ridge * np.eye(n_ctrl)
    C = np.linalg.solve(A, B.T @ q_uniform)
    # B-spline bornée : premier et dernier points de contrôle = extrémités de la courbe
    C[0], C[-1] = q_uniform[0], q_uniform[-1]
    return C


def decode(ctrl: np.ndarray, duration: float, joints, rate_hz: float = 50.0) -> Trajectory:
    """Points de contrôle (K, J) + durée -> trajectoire horodatée commençant à t=0."""
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
    """Propose le segment utile d'une prise : du premier au dernier instant où
    une articulation s'écarte de plus de `threshold_rad` de sa posture initiale
    (resp. finale), avec une petite marge. Les données brutes ne sont pas
    modifiées ; seul le segment est mémorisé dans les métadonnées."""
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
