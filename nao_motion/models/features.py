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


def arm_input(emotions, emotion, intensity) -> np.ndarray:
    return np.concatenate([emotion_vector(emotions, emotion), [check_intensity(intensity)]]).astype(np.float32)


def head_input(emotions, emotion, intensity, arm_ctrl_norm: np.ndarray, arm_log_dur: float,
               head_current_norm: np.ndarray, event: EventContext) -> np.ndarray:
    """[émotion, intensité, points de contrôle bras (K×Ja, normalisés), log durée,
    orientation actuelle de la tête (normalisée), événement (4)]."""
    return np.concatenate([
        emotion_vector(emotions, emotion), [check_intensity(intensity)],
        np.asarray(arm_ctrl_norm, dtype=np.float32).ravel(), [arm_log_dur],
        np.asarray(head_current_norm, dtype=np.float32).ravel(), event.features(),
    ]).astype(np.float32)


def head_input_dim(n_emotions, n_ctrl, n_arm_joints, n_head_joints) -> int:
    return n_emotions + 1 + n_ctrl * n_arm_joints + 1 + n_head_joints + EVENT_FEATURES
