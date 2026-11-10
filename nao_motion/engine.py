"""Moteur de génération : trois modèles INDÉPENDANTS (bras, tête, mains) combinés.

Chaque modèle reçoit la même consigne [émotion, intensité, événement] et produit
la trajectoire de son seul groupe, avec sa propre durée. Aucun modèle ne lit la
sortie d'un autre : bras, tête et mains s'entraînent séparément. La cohérence
entre groupes vient de la consigne partagée (le bras pointe vers l'événement,
la tête le regarde), pas d'une synchronisation apprise.

Assemblage (configs/model.yaml, `combine.timing`) :
- `hold` (défaut) : tous les groupes démarrent ensemble, chacun à son tempo
  appris ; un groupe plus court tient sa pose finale jusqu'à la fin du plus long ;
- `stretch` : chaque groupe est étiré sur la durée la plus longue (plus lent,
  jamais plus rapide que ce qui a été appris).

Un groupe sans modèle : bras ou tête maintenus en position de départ, mains non
commandées (laissées telles quelles). Il faut au moins un modèle.

Le moteur PROPOSE des trajectoires ; il n'exécute rien. L'exécution passe par
le contrôleur, qui fait valider le cycle complet par le guardian.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from .config import Config
from .data.representation import bspline_basis
from .events import EventContext
from .models.features import Norm, check_intensity, emotion_vector, motion_input
from .models.nets import CheckpointError, load_model
from .trajectory import Trajectory

Emotion = Union[str, Dict[str, float]]
TIMINGS = ("hold", "stretch")


@dataclass
class Generated:
    arms: Trajectory
    head: Trajectory
    combined: Trajectory
    arm_ctrl: np.ndarray
    head_ctrl: np.ndarray
    duration: float
    inputs: dict
    notes: list = field(default_factory=list)
    hands: Optional[Trajectory] = None
    hand_ctrl: Optional[np.ndarray] = None
    durations: Dict[str, float] = field(default_factory=dict)   # durée propre de chaque groupe généré

    def to_dict(self):
        return {"inputs": self.inputs, "duration_s": self.duration, "notes": self.notes,
                "group_durations_s": self.durations, "trajectory": self.combined.to_dict()}


class MotionEngine:
    def __init__(self, cfg: Config, arm_model=None, head_model=None, rate_hz: float = 50.0, hands_model=None):
        self.cfg = cfg
        self.arm_model = arm_model
        self.head_model = head_model
        self.hands_model = hands_model
        self.rate = rate_hz
        self.K = int(cfg.model["representation"]["n_control_points"])
        self.arm_norm = Norm.from_config(cfg, cfg.arm_joints)
        self.head_norm = Norm.from_config(cfg, cfg.head_joints)
        self.hand_norm = Norm.from_config(cfg, cfg.hand_joints) if cfg.hand_joints else None
        self.timing = str((cfg.model.get("combine") or {}).get("timing", "hold"))
        if self.timing not in TIMINGS:
            raise ValueError(f"combine.timing inconnu : {self.timing} (attendu {TIMINGS})")

    @classmethod
    def from_paths(cls, cfg, arm_path=None, head_path=None, hands_path=None):
        arm = load_model(arm_path, cfg, "arms") if arm_path else None
        head = load_model(head_path, cfg, "head") if head_path else None
        hands = load_model(hands_path, cfg, "hands") if hands_path and cfg.hand_joints else None
        return cls(cfg, arm, head, hands_model=hands)

    @property
    def has_model(self) -> bool:
        return any(m is not None for m in (self.arm_model, self.head_model, self.hands_model))

    def _plan(self, model, emotion: Emotion, intensity: float, event: Optional[EventContext],
              variation: float = 0.0, seed: Optional[int] = None):
        """Points de contrôle normalisés (K×J) et durée (s) d'un groupe."""
        out = model.predict(motion_input(self.cfg.emotions, emotion, intensity, event))[0]
        ctrl_n = out[:-1].reshape(self.K, -1)
        if variation > 0:
            ctrl_n = ctrl_n + self._smooth_noise(ctrl_n.shape, variation, seed)
        g = self.cfg.guardian
        dur = float(np.clip(np.exp(out[-1]), g.min_duration_s, g.max_duration_s))
        return ctrl_n, dur

    def plan_arms(self, emotion: Emotion, intensity: float, variation: float = 0.0, seed: Optional[int] = None,
                  event: Optional[EventContext] = None):
        if self.arm_model is None:
            raise CheckpointError("Aucun modèle des bras chargé")
        return self._plan(self.arm_model, emotion, intensity, event, variation, seed)

    def plan_head(self, emotion: Emotion, intensity: float, event: Optional[EventContext] = None,
                  variation: float = 0.0, seed: Optional[int] = None):
        if self.head_model is None:
            raise CheckpointError("Aucun modèle de la tête chargé")
        return self._plan(self.head_model, emotion, intensity, event, variation, seed)

    def plan_hands(self, emotion: Emotion, intensity: float, event: Optional[EventContext] = None,
                   variation: float = 0.0, seed: Optional[int] = None):
        if self.hands_model is None or not self.cfg.hand_joints:
            raise CheckpointError("Aucun modèle des mains chargé")
        return self._plan(self.hands_model, emotion, intensity, event, variation, seed)

    def _smooth_noise(self, shape, sigma, seed):
        """Variation non apprise : bruit lisse sur les points de contrôle intérieurs.

        C'est une première façon simple de produire des variantes ; une variante
        apprise (latent) est prévue dans les modèles (latent_dim)."""
        rng = np.random.default_rng(seed)
        n = rng.normal(0, sigma, size=shape)
        n = (n + np.roll(n, 1, axis=0) + np.roll(n, -1, axis=0)) / 3
        n[0] = n[-1] = 0.0
        return n

    def generate(self, emotion: Emotion, intensity: float, event: Optional[EventContext] = None,
                 head_current: Optional[Dict[str, float]] = None, variation: float = 0.0,
                 seed: Optional[int] = None, hands_current: Optional[Dict[str, float]] = None) -> Generated:
        """Geste combiné bras + tête + mains. `head_current` et `hands_current` ne sont plus des
        entrées des modèles (gardés pour compatibilité d'appel) : les transitions depuis la
        posture mesurée sont planifiées par le contrôleur."""
        check_intensity(intensity)
        emotion_vector(self.cfg.emotions, emotion)
        if not self.has_model:
            raise CheckpointError("Aucun modèle entraîné (bras, tête ou mains)")
        event = event or EventContext.none()
        cfg, notes = self.cfg, []
        sub = (lambda k: None if seed is None else int(seed) + k)  # une graine par groupe
        plans = {}
        if self.arm_model is not None:
            plans["arms"] = self.plan_arms(emotion, intensity, variation, sub(0), event)
        if self.head_model is not None:
            plans["head"] = self.plan_head(emotion, intensity, event, variation, sub(1))
        if self.hands_model is not None and cfg.hand_joints:
            plans["hands"] = self.plan_hands(emotion, intensity, event, variation, sub(2))
        T = max(d for _, d in plans.values())
        n = max(4, int(np.ceil(T * self.rate)) + 1)
        tt = np.linspace(0.0, T, n)

        def build(group, joints, norm, ctrl_n, dur, lo=None, hi=None):
            ctrl = norm.inv(ctrl_n)
            if lo is not None:
                # points de contrôle ramenés dans la plage : par la propriété d'enveloppe convexe
                # des B-splines, la courbe y reste aussi, sans cassure de vitesse
                ctrl = np.clip(ctrl, lo, hi)
            d = T if self.timing == "stretch" else dur
            # au-delà de sa durée, le groupe tient sa pose finale (phase bornée à 1)
            u = np.clip(tt / d, 0.0, 1.0)
            return Trajectory(joints, tt, bspline_basis(self.K, u) @ ctrl), ctrl

        def held(joints):
            q0 = np.array(cfg.pose("start", joints))
            return Trajectory(joints, tt, np.tile(q0, (n, 1))), np.tile(q0, (self.K, 1))

        if "arms" in plans:
            arms, arm_ctrl = build("arms", cfg.arm_joints, self.arm_norm, *plans["arms"])
        else:
            arms, arm_ctrl = held(cfg.arm_joints)
            notes.append("Aucun modèle des bras : bras maintenus en position de départ")
        if "head" in plans:
            head, head_ctrl = build("head", cfg.head_joints, self.head_norm, *plans["head"])
        else:
            head, head_ctrl = held(cfg.head_joints)
            notes.append("Aucun modèle tête : tête maintenue en position de départ")
        combined = Trajectory.merge(arms, head)
        hands = hand_ctrl = None
        if "hands" in plans:
            hj, g = cfg.hand_joints, cfg.guardian
            lo = np.array([cfg.limits[j][0] + g.margin(j) for j in hj])
            hi = np.array([cfg.limits[j][1] - g.margin(j) for j in hj])
            hands, hand_ctrl = build("hands", hj, self.hand_norm, *plans["hands"], lo=lo, hi=hi)
            combined = Trajectory.merge(combined, hands)
        elif cfg.hand_joints:
            notes.append("Aucun modèle des mains : mains non commandées (laissées telles quelles)")
        for m in (self.arm_model, self.head_model, self.hands_model):
            notes += list(getattr(m, "warnings", []) or [])
        durations = {k: round(d, 3) for k, (_, d) in plans.items()}
        return Generated(arms, head, combined, arm_ctrl, head_ctrl, T,
                         {"emotion": emotion, "intensity": intensity, "event": event.to_dict(),
                          "variation": variation, "seed": seed, "timing": self.timing}, notes, hands, hand_ctrl,
                         durations)
