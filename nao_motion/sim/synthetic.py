"""Démonstrateurs SYNTHÉTIQUES pour la simulation.

Ils remplacent la main de l'opérateur afin de vérifier le fonctionnement
technique de la chaîne (collecte, données, entraînement, génération,
guardian). Ils ne disent RIEN de l'expressivité réelle : les formes ci-dessous
sont arbitraires. Les prises produites sont marquées `synthetic: true`.

Certaines émotions ont volontairement deux variantes de geste sous la même
étiquette, pour illustrer qu'un modèle de régression déterministe tend à en
produire la moyenne.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np

from ..config import Config
from ..events import EventContext

# amplitude (rad) par articulation "générique" côté gauche ; le côté droit est miroir
# clés : sp=ShoulderPitch, sr=ShoulderRoll, ey=ElbowYaw, er=ElbowRoll, wy=WristYaw
STYLES = {
    "joie": [dict(sp=-0.8, sr=0.30, er=0.25, freq=1.2, osc=0.05, dur=3.2),
             dict(sp=-0.6, sr=0.15, ey=0.4, er=-0.3, freq=1.2, osc=0.08, dur=3.4, one_arm=True)],
    "tristesse": [dict(sp=0.25, sr=-0.10, er=0.35, freq=0.5, osc=0.0, dur=3.8)],
    "colere": [dict(sp=-0.5, sr=0.05, er=0.5, freq=1.5, osc=0.04, dur=2.6)],
    "peur": [dict(sp=-0.4, sr=-0.10, er=-0.4, freq=2.0, osc=0.015, dur=2.8)],
    "surprise": [dict(sp=-0.7, sr=0.40, er=0.35, freq=1.0, osc=0.0, dur=2.6)],
    "calme": [dict(sp=-0.25, sr=0.20, ey=0.25, freq=0.8, osc=0.03, dur=3.2)],
}
_SIDE_JOINT = {"sp": "ShoulderPitch", "sr": "ShoulderRoll", "ey": "ElbowYaw", "er": "ElbowRoll", "wy": "WristYaw"}
# signes pour obtenir le miroir côté droit (convention NAOqi)
_MIRROR = {"sp": 1, "sr": -1, "ey": -1, "er": -1, "wy": -1}


def _env(u):
    return np.sin(np.pi * np.clip(u, 0, 1)) ** 2


def arm_gesture(cfg: Config, emotion: str, intensity: float, rng: np.random.Generator,
                style: Optional[int] = None):
    """Retourne (fonction t -> {joint: rad}, durée) pour un geste des bras."""
    styles = STYLES.get(emotion, STYLES["calme"])
    s = dict(styles[rng.integers(len(styles)) if style is None else style % len(styles)])
    amp = (0.35 + 0.65 * intensity) * rng.uniform(0.85, 1.15)
    dur = s["dur"] * (1.2 - 0.25 * intensity) * rng.uniform(0.9, 1.1)
    phase = rng.uniform(0, 2 * np.pi)
    start = dict(zip(cfg.arm_joints, cfg.pose("start", cfg.arm_joints)))
    m = cfg.guardian.limit_margin_rad + 0.02

    def fn(t: float) -> Dict[str, float]:
        u = t / dur
        e = float(_env(u))
        osc = s.get("osc", 0.0) * math.sin(2 * math.pi * s["freq"] * t + phase)
        out = {}
        for side, sign_side in (("L", 1), ("R", -1)):
            active = not (s.get("one_arm") and side == "R")
            for key, name in _SIDE_JOINT.items():
                j = side + name
                if j not in start:
                    continue
                a = s.get(key, 0.0)
                if key == "sp":
                    a = a + osc
                sign = 1 if side == "L" else _MIRROR[key]
                val = start[j] + (sign * amp * a * e if active else 0.0)
                if key == "er" and active:
                    val += sign * amp * osc * e * 0.5
                lo, hi = cfg.limits[j]
                out[j] = float(np.clip(val, lo + m, hi - m))
        return out

    return fn, float(dur)


def head_behaviour(cfg: Config, emotion: str, intensity: float, event: EventContext,
                   arm_fn: Callable[[float], Dict[str, float]], rng: np.random.Generator):
    """Politique de regard synthétique : regarder l'événement, suivre une main avec
    retard, ou détourner le regard, selon l'émotion. Arbitraire, pour les tests."""
    start = dict(zip(cfg.head_joints, cfg.pose("start", cfg.head_joints)))
    lag = rng.uniform(0.1, 0.35)
    mode = {"joie": "event", "surprise": "event", "colere": "event",
            "tristesse": "avert", "peur": "avert", "calme": "hands"}.get(emotion, "event")
    if mode == "event" and not event.present:
        mode = "hands" if emotion == "joie" else "front"
    m = cfg.guardian.limit_margin_rad + 0.02
    lim = cfg.limits

    def clipped(y, p):
        return {"HeadYaw": float(np.clip(y, lim["HeadYaw"][0] + m, lim["HeadYaw"][1] - m)),
                "HeadPitch": float(np.clip(p, lim["HeadPitch"][0] + m, lim["HeadPitch"][1] - m))}

    def ramp(t, T=1.2):
        x = np.clip(t / T, 0, 1)
        return 10 * x**3 - 15 * x**4 + 6 * x**5

    def fn(t: float) -> Dict[str, float]:
        r = float(ramp(t))
        y0, p0 = start["HeadYaw"], start["HeadPitch"]
        if mode == "event":
            ty, tp = np.clip(event.azimuth_rad, -1.2, 1.2), -event.elevation_rad
            return clipped(y0 + r * (ty - y0) * (0.6 + 0.4 * intensity), p0 + r * (tp - p0))
        if mode == "avert":
            away = -np.sign(event.azimuth_rad or 0.3) * 0.6 * intensity if event.present else 0.4 * intensity
            return clipped(y0 + r * away, p0 + r * 0.3 * intensity)
        if mode == "hands":
            a = arm_fn(max(0.0, t - lag))
            # regarde vers la main gauche, très approximativement
            y = 0.35 * (a.get("LShoulderRoll", 0.2) - 0.2) * 2
            p = 0.15 + 0.2 * (a.get("LShoulderPitch", 1.1) - 1.1)
            return clipped(y0 + r * (y - y0), p0 + r * (p - p0))
        return clipped(y0, p0)

    return fn
