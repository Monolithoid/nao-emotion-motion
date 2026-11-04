"""Contexte événementiel (direction d'un événement extérieur) et repères.

Repère commun : FRAME_TORSO de NAOqi
  - origine au torse, x vers l'avant, y vers la gauche du robot, z vers le haut ;
  - azimut   = atan2(y, x), en radians, positif vers la GAUCHE du robot ;
  - élévation = atan2(z, sqrt(x²+y²)), en radians, positive vers le HAUT.

Un azimut de 0 signifie « devant le robot ». L'ABSENCE d'événement est
représentée explicitement par `present=False`, jamais par un angle nul.

Distance (optionnelle) : distance en mètres entre le torse et l'événement.
Inconnue -> `distance_m=None`, encodée explicitement (jamais par une valeur
arbitraire). Le modèle reçoit la proximité 1 / (1 + d), entre 0 (très loin)
et 1 (au contact), qui ne dépend d'aucune plage choisie à l'avance.

Conventions articulaires NAOqi utiles : HeadYaw > 0 tourne la tête vers la
gauche ; HeadPitch > 0 baisse la tête. Regarder une élévation e demande donc
approximativement HeadPitch = -e (la translation du cou est négligée).

Ce module ne fait aucune perception : il convertit ce que fournit un
composant caméra/microphones existant vers le repère commun.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

COMMON_FRAME = "torso"
SUPPORTED_FRAMES = ("torso", "head")
EVENT_FEATURES = 6  # [présent, sin(az), cos(az), élévation, distance connue, proximité 1/(1+d)]


class EventError(ValueError):
    pass


@dataclass
class EventContext:
    present: bool
    azimuth_rad: Optional[float] = None
    elevation_rad: Optional[float] = None
    frame: str = COMMON_FRAME
    source: str = "manual"           # manual | camera | microphones | test | ...
    timestamp: float = field(default_factory=time.time)
    confidence: Optional[float] = None
    distance_m: Optional[float] = None

    def __post_init__(self):
        if self.frame != COMMON_FRAME:
            raise EventError("EventContext est toujours exprimé dans le repère commun ; utiliser from_source()")
        if self.present:
            for v in (self.azimuth_rad, self.elevation_rad):
                if v is None or not math.isfinite(v):
                    raise EventError("Un événement présent exige azimut et élévation finis")
            if not -math.pi / 2 <= self.elevation_rad <= math.pi / 2:
                raise EventError("Élévation hors de [-pi/2, pi/2]")
            self.azimuth_rad = math.atan2(math.sin(self.azimuth_rad), math.cos(self.azimuth_rad))
            if self.distance_m is not None:
                self.distance_m = float(self.distance_m)
                if not math.isfinite(self.distance_m) or self.distance_m <= 0:
                    raise EventError("Distance invalide : nombre de mètres strictement positif attendu")
        else:
            self.azimuth_rad = None
            self.elevation_rad = None
            self.distance_m = None

    @classmethod
    def none(cls, source: str = "none") -> "EventContext":
        return cls(present=False, source=source)

    @classmethod
    def from_source(cls, *, present: bool, azimuth=None, elevation=None, frame: str = COMMON_FRAME,
                    unit: str = "rad", head_yaw_rad: Optional[float] = None,
                    head_pitch_rad: Optional[float] = None, source: str = "external",
                    timestamp: Optional[float] = None, confidence: Optional[float] = None,
                    distance=None) -> "EventContext":
        """Convertit une direction fournie par une source vers le repère commun.

        frame="head" : direction relative à la tête (caméra) ; il faut alors les
        angles HeadYaw/HeadPitch mesurés au moment de la détection.
        """
        ts = time.time() if timestamp is None else float(timestamp)
        if not present:
            return cls(present=False, source=source, timestamp=ts)
        if unit not in ("rad", "deg"):
            raise EventError(f"Unité inconnue {unit!r} (rad|deg)")
        if azimuth is None or elevation is None:
            raise EventError("azimut/élévation requis pour un événement présent")
        az, el = float(azimuth), float(elevation)
        if unit == "deg":
            az, el = math.radians(az), math.radians(el)
        if frame == "torso":
            pass
        elif frame == "head":
            if head_yaw_rad is None or head_pitch_rad is None:
                raise EventError("frame='head' exige head_yaw_rad et head_pitch_rad au moment de la mesure")
            az, el = head_to_torso(az, el, head_yaw_rad, head_pitch_rad)
        else:
            raise EventError(f"Repère inconnu {frame!r} ; supportés : {SUPPORTED_FRAMES}")
        dist = None if distance is None or distance == "" else float(distance)
        return cls(present=True, azimuth_rad=az, elevation_rad=el, source=source, timestamp=ts,
                   confidence=confidence, distance_m=dist)

    def features(self) -> np.ndarray:
        """Encodage pour le modèle :
        [présent, sin(az), cos(az), élévation, distance connue, proximité 1/(1+d)].

        Absent -> que des zéros ; devant, distance inconnue -> [1, 0, 1, 0, 0, 0] ;
        devant à 1 m -> [1, 0, 1, 0, 1, 0.5]. Tous distincts.
        """
        if not self.present:
            return np.zeros(EVENT_FEATURES, dtype=np.float32)
        known = self.distance_m is not None
        prox = 1.0 / (1.0 + self.distance_m) if known else 0.0
        return np.array([1.0, math.sin(self.azimuth_rad), math.cos(self.azimuth_rad), self.elevation_rad,
                         1.0 if known else 0.0, prox], dtype=np.float32)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unit"] = "rad"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EventContext":
        if d.get("frame", COMMON_FRAME) != COMMON_FRAME or d.get("unit", "rad") != "rad":
            return cls.from_source(present=d["present"], azimuth=d.get("azimuth_rad", d.get("azimuth")),
                                   elevation=d.get("elevation_rad", d.get("elevation")),
                                   frame=d.get("frame", COMMON_FRAME), unit=d.get("unit", "rad"),
                                   head_yaw_rad=d.get("head_yaw_rad"), head_pitch_rad=d.get("head_pitch_rad"),
                                   source=d.get("source", "external"), timestamp=d.get("timestamp"),
                                   distance=d.get("distance_m", d.get("distance")))
        return cls(present=bool(d["present"]), azimuth_rad=d.get("azimuth_rad"),
                   elevation_rad=d.get("elevation_rad"), source=d.get("source", "manual"),
                   timestamp=float(d.get("timestamp", time.time())), confidence=d.get("confidence"),
                   distance_m=d.get("distance_m"))


def _dir(az: float, el: float) -> np.ndarray:
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def head_to_torso(az_h: float, el_h: float, head_yaw: float, head_pitch: float):
    """Direction exprimée dans le repère tête -> repère torse (rotation pure).

    R = Rz(HeadYaw) · Ry(HeadPitch) ; HeadPitch > 0 incline l'axe x vers le bas,
    cohérent avec la convention NAOqi. La translation du cou est négligée
    (acceptable pour des événements à plus de ~50 cm).
    """
    cy, sy = math.cos(head_yaw), math.sin(head_yaw)
    cp, sp = math.cos(head_pitch), math.sin(head_pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    d = rz @ ry @ _dir(az_h, el_h)
    return math.atan2(d[1], d[0]), math.asin(max(-1.0, min(1.0, d[2])))


def gaze_angles(event: EventContext):
    """Angles (HeadYaw, HeadPitch) approximatifs pour regarder l'événement."""
    if not event.present:
        raise EventError("Pas d'événement")
    return event.azimuth_rad, -event.elevation_rad
