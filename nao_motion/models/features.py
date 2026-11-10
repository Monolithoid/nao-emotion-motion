"""Construction des entrées/sorties des modèles (indépendante de PyTorch).

Normalisation des angles : x = (q - pose_de_départ) / demi_plage(limites).
Les sorties sont des points de contrôle normalisés ; la durée est apprise en log.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..config import Config
from ..events import EVENT_FEATURES, EventContext


@dataclass
class Norm:
    joints: List[str]
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_config(cls, cfg: Config, joints: Sequence[str]) -> "Norm":
        joints = list(joints)
        center = np.array(cfg.pose("start", joints))
        scale = np.array([(cfg.limits[j][1] - cfg.limits[j][0]) / 2 for j in joints])
        return cls(joints, center, scale)

    def fwd(self, q):
        return (np.asarray(q) - self.center) / self.scale

    def inv(self, x):
        return np.asarray(x) * self.scale + self.center

    def to_dict(self):
        return {"joints": self.joints, "center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(list(d["joints"]), np.array(d["center"]), np.array(d["scale"]))


def emotion_vector(emotions: List[str], emotion: Union[str, Dict[str, float]]) -> np.ndarray:
    """One-hot, ou mélange pondéré {émotion: poids} normalisé (interpolation entre émotions)."""
    v = np.zeros(len(emotions), dtype=np.float32)
    if isinstance(emotion, str):
        if emotion not in emotions:
            raise ValueError(f"Émotion inconnue : {emotion}")
        v[emotions.index(emotion)] = 1.0
        return v
    total = 0.0
    for e, w in emotion.items():
        if e not in emotions:
            raise ValueError(f"Émotion inconnue : {e}")
        if not np.isfinite(w) or w < 0:
            raise ValueError("Poids d'émotion invalides")
        v[emotions.index(e)] = w
        total += w
    if total <= 0:
        raise ValueError("Mélange d'émotions vide")
    return v / total


def check_intensity(x: float) -> float:
    x = float(x)
    if not np.isfinite(x) or not 0.0 <= x <= 1.0:
        raise ValueError("Intensité hors de [0, 1]")
    return x


def motion_input(emotions, emotion, intensity, event: Optional[EventContext] = None) -> np.ndarray:
    """Entrée COMMUNE aux trois modèles (bras, tête, mains) : [émotion, intensité,
    événement (6)]. La direction et la distance de l'événement permettent d'apprendre
    un geste orienté (pointer, regarder, se protéger d'un côté...). Aucun modèle ne
    reçoit la sortie d'un autre : chacun s'entraîne seul."""
    ev = event if event is not None else EventContext.none()
    return np.concatenate([emotion_vector(emotions, emotion), [check_intensity(intensity)],
                           ev.features()]).astype(np.float32)


def motion_input_dim(n_emotions) -> int:
    return n_emotions + 1 + EVENT_FEATURES


# noms historiques (modèle des bras)
arm_input = motion_input
arm_input_dim = motion_input_dim
