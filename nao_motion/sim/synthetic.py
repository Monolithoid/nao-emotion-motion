"""Démonstrateurs SYNTHÉTIQUES pour la simulation.

Ils remplacent la main de l'opérateur afin de vérifier le fonctionnement
technique de la chaîne (collecte, données, entraînement, génération,
guardian). Ils ne disent RIEN de l'expressivité réelle : les formes ci-dessous
sont arbitraires. Les prises produites sont marquées `synthetic: true`.

« surprise » et « colère » pointent vers l'événement quand il y en a un (bras du
côté de l'événement, d'autant plus tendu que l'événement est proche), pour que
la chaîne puisse apprendre un geste orienté par l'angle et la distance.

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


POINTING = ("surprise", "colere")


def pointing_pose(cfg: Config, event: EventContext) -> Dict[str, float]:
    """Pose approximative du bras du côté de l'événement, tendu vers lui (convention NAOqi :
    ShoulderPitch = -élévation, ShoulderRoll = azimut, coude presque tendu)."""
    side = "L" if event.azimuth_rad >= 0 else "R"
    m = cfg.guardian.limit_margin_rad + 0.02
    want = {"ShoulderPitch": -event.elevation_rad, "ShoulderRoll": event.azimuth_rad,
            "ElbowYaw": 0.0, "ElbowRoll": -0.1 if side == "L" else 0.1}
    out = {}
    for name, v in want.items():
        j = side + name
        if j in cfg.limits:
            lo, hi = cfg.limits[j]
            out[j] = float(np.clip(v, lo + m, hi - m))
    return out


def arm_gesture(cfg: Config, emotion: str, intensity: float, rng: np.random.Generator,
                style: Optional[int] = None, event: Optional[EventContext] = None):
    """Retourne (fonction t -> {joint: rad}, durée) pour un geste des bras."""
    if event is not None and event.present and emotion in POINTING:
        return _pointing_gesture(cfg, emotion, intensity, rng, event)
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


def _pointing_gesture(cfg, emotion, intensity, rng, event: EventContext):
    """Pointe vers l'événement : aller, tenue, retour. Plus l'événement est proche, plus
    le bras va loin vers la pose de pointage (proximité inconnue : valeur moyenne)."""
    start = dict(zip(cfg.arm_joints, cfg.pose("start", cfg.arm_joints)))
    target = pointing_pose(cfg, event)
    close = 1.0 / (1.0 + event.distance_m) if event.distance_m else 0.4
    w = float(np.clip((0.45 + 0.55 * intensity) * (0.55 + 0.9 * close), 0.3, 1.0)) * rng.uniform(0.92, 1.05)
    dur = (2.8 if emotion == "surprise" else 2.4) * (1.15 - 0.2 * intensity) * rng.uniform(0.9, 1.1)
    # enveloppe sin² : vitesse de pointe = pi * amplitude / durée ; on la garde sous ~80 % de la limite
    vmax = 0.8 * cfg.guardian.max_velocity_rad_s
    amp = max(abs(v - start[j]) for j, v in target.items()) * min(w, 1.0)
    dur = max(dur, math.pi * amp / vmax)

    def fn(t: float) -> Dict[str, float]:
        e = float(_env(t / dur))
        out = dict(start)
        for j, v in target.items():
            out[j] = float(start[j] + min(w, 1.0) * e * (v - start[j]))
        return out

    return fn, float(dur)


def head_behaviour(cfg: Config, emotion: str, intensity: float, event: EventContext,
                   arm_fn: Callable[[float], Dict[str, float]], rng: np.random.Generator):
    """Politique de regard synthétique : regarder l'événement, suivre une main avec
    retard, ou détourner le regard, selon l'émotion. Un événement proche donne une
    réaction plus rapide et un évitement plus marqué. Arbitraire, pour les tests."""
    start = dict(zip(cfg.head_joints, cfg.pose("start", cfg.head_joints)))
    lag = rng.uniform(0.1, 0.35)
    mode = {"joie": "event", "surprise": "event", "colere": "event",
            "tristesse": "avert", "peur": "avert", "calme": "hands"}.get(emotion, "event")
    if mode == "event" and not event.present:
        mode = "hands" if emotion == "joie" else "front"
    m = cfg.guardian.limit_margin_rad + 0.02
    close = 1.0 / (1.0 + event.distance_m) if event.present and event.distance_m else 0.4  # proximité
    T = 1.6 - 1.2 * close  # 1,2 s à 1 m environ ; plus vite si proche
    lim = cfg.limits

    def clipped(y, p):
        return {"HeadYaw": float(np.clip(y, lim["HeadYaw"][0] + m, lim["HeadYaw"][1] - m)),
                "HeadPitch": float(np.clip(p, lim["HeadPitch"][0] + m, lim["HeadPitch"][1] - m))}

    def ramp(t, T=1.2):
        x = np.clip(t / T, 0, 1)
        return 10 * x**3 - 15 * x**4 + 6 * x**5

    def fn(t: float) -> Dict[str, float]:
        r = float(ramp(t, T))
        y0, p0 = start["HeadYaw"], start["HeadPitch"]
        if mode == "event":
            ty, tp = np.clip(event.azimuth_rad, -1.2, 1.2), -event.elevation_rad
            return clipped(y0 + r * (ty - y0) * (0.6 + 0.4 * intensity), p0 + r * (tp - p0))
        if mode == "avert":
            away = (-np.sign(event.azimuth_rad or 0.3) * (0.3 + 0.75 * close) * intensity if event.present
                    else 0.4 * intensity)
            return clipped(y0 + r * away, p0 + r * 0.3 * intensity)
        if mode == "hands":
            a = arm_fn(max(0.0, t - lag))
            # regarde vers la main gauche, très approximativement
            y = 0.35 * (a.get("LShoulderRoll", 0.2) - 0.2) * 2
            p = 0.15 + 0.2 * (a.get("LShoulderPitch", 1.1) - 1.1)
            return clipped(y0 + r * (y - y0), p0 + r * (p - p0))
        return clipped(y0, p0)

    return fn
