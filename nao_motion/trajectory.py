"""Trajectoire articulaire horodatée (temps en s, angles en rad)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


class TrajectoryError(ValueError):
    pass


@dataclass
class Trajectory:
    joints: List[str]
    t: np.ndarray  # (N,) secondes, strictement croissant
    q: np.ndarray  # (N, J) radians

    def __post_init__(self):
        self.joints = list(self.joints)
        self.t = np.asarray(self.t, dtype=np.float64)
        self.q = np.asarray(self.q, dtype=np.float64)
        if self.q.ndim == 1 and len(self.joints) == 1:
            self.q = self.q[:, None]
        check_samples(self.t, self.q, self.joints)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def select(self, joints: Sequence[str]) -> "Trajectory":
        idx = [self.joints.index(j) for j in joints]
        return Trajectory(list(joints), self.t.copy(), self.q[:, idx].copy())

    def sample(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float64)
        return np.stack([np.interp(times, self.t, self.q[:, j]) for j in range(len(self.joints))], axis=-1)

    def resampled(self, rate_hz: float) -> "Trajectory":
        n = max(2, int(np.ceil(self.duration * rate_hz)) + 1)
        tt = np.linspace(self.t[0], self.t[-1], n)
        return Trajectory(self.joints, tt, self.sample(tt))

    def shifted(self, dt: float) -> "Trajectory":
        return Trajectory(self.joints, self.t + dt, self.q.copy())

    def at(self, i: int) -> Dict[str, float]:
        return dict(zip(self.joints, self.q[i].tolist()))

    def to_dict(self) -> dict:
        return {"joints": self.joints, "unit": "rad", "time_unit": "s",
                "t": self.t.tolist(), "q": self.q.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        if d.get("unit", "rad") != "rad" or d.get("time_unit", "s") != "s":
            raise TrajectoryError("Unités attendues : rad et s")
        return cls(d["joints"], np.array(d["t"], dtype=float), np.array(d["q"], dtype=float))

    @staticmethod
    def merge(a: "Trajectory", b: "Trajectory") -> "Trajectory":
        """Fusionne deux trajectoires de mêmes horodatages sur des articulations disjointes."""
        if set(a.joints) & set(b.joints):
            raise TrajectoryError("Articulations en double")
        if a.t.shape != b.t.shape or not np.allclose(a.t, b.t):
            raise TrajectoryError("Horodatages incompatibles")
        return Trajectory(a.joints + b.joints, a.t.copy(), np.concatenate([a.q, b.q], axis=1))

    @staticmethod
    def concat(parts: Sequence["Trajectory"]) -> "Trajectory":
        """Enchaîne des segments ; chaque segment commence là où le précédent finit."""
        joints = parts[0].joints
        ts, qs, offset = [], [], 0.0
        for k, p in enumerate(parts):
            if p.joints != joints:
                raise TrajectoryError("Segments avec articulations différentes")
            t = p.t - p.t[0] + offset
            if k > 0:
                if not np.allclose(p.q[0], qs[-1][-1], atol=1e-6):
                    raise TrajectoryError(f"Discontinuité entre segments {k-1} et {k}")
                t, q = t[1:], p.q[1:]
            else:
                q = p.q
            ts.append(t)
            qs.append(q)
            offset = float(t[-1]) if len(t) else offset
        return Trajectory(joints, np.concatenate(ts), np.concatenate(qs))


def check_samples(t: np.ndarray, q: np.ndarray, joints: Sequence[str]):
    if t.ndim != 1 or q.ndim != 2:
        raise TrajectoryError("t doit être (N,) et q (N, J)")
    if q.shape != (t.shape[0], len(joints)):
        raise TrajectoryError(f"Forme incohérente : t{t.shape}, q{q.shape}, {len(joints)} articulations")
    if t.shape[0] < 2:
        raise TrajectoryError("Au moins deux échantillons requis")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(q))):
        raise TrajectoryError("Valeurs non finies")
    if np.any(np.diff(t) <= 0):
        raise TrajectoryError("Horodatages non strictement croissants")
    if len(set(joints)) != len(joints):
        raise TrajectoryError("Noms d'articulations dupliqués")
