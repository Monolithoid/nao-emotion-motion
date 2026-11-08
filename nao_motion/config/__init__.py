"""Chargement et validation des configurations versionnées (dossier configs/)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

DEFAULT_CONFIG_DIR = Path(os.environ.get("NAO_MOTION_CONFIG", Path(__file__).resolve().parents[2] / "configs"))

ARM_GROUPS = ("left_arm", "right_arm")
HEAD_GROUP = "head"
HANDS_GROUP = "hands"   # LHand / RHand : ouverture 0..1, pilotée (jamais guidée), hors calibration


class ConfigError(ValueError):
    pass


@dataclass
class CoupledConstraint:
    name: str
    terms: Dict[str, float]
    bound: float
    description: str = ""


@dataclass
class GuardianConfig:
    limit_margin_rad: float
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float
    per_joint_velocity_rad_s: Dict[str, float]
    check_rate_hz: float
    min_duration_s: float
    max_duration_s: float
    start_tolerance_rad: float
    transition_velocity_rad_s: float
    coupled_constraints: List[CoupledConstraint]
    rest_velocity_rad_s: float = 0.3  # segments repos <-> départ (repos hors zone de mouvement)
    per_joint_margin: Dict[str, float] = field(default_factory=dict)
    per_joint_acceleration: Dict[str, float] = field(default_factory=dict)

    def velocity_limit(self, joint: str) -> float:
        return float(self.per_joint_velocity_rad_s.get(joint, self.max_velocity_rad_s))

    def margin(self, joint: str) -> float:
        return float(self.per_joint_margin.get(joint, self.limit_margin_rad))

    def acceleration_limit(self, joint: str) -> float:
        return float(self.per_joint_acceleration.get(joint, self.max_acceleration_rad_s2))


@dataclass
class Config:
    root: Path
    groups: Dict[str, List[str]]
    limits: Dict[str, Tuple[float, float]]
    calibration_version: str
    calibrated: bool
    poses: Dict[str, Dict[str, Dict[str, float]]]  # name -> group -> joint -> rad
    poses_calibrated: bool
    emotions: List[str]
    emotion_labels: Dict[str, str]
    intentions: List[str]
    guardian: GuardianConfig
    robot: dict
    model: dict
    raw: dict = field(default_factory=dict, repr=False)
    calibration: Optional[object] = None  # nao_motion.calibration.Calibration si une calibration est active
    # Le repos peut être hors de la zone de mouvement (ex. bras posés sur les genoux :
    # contact). Il est borné par rest_limits (limites d'usine si connues), et seuls les
    # segments repos <-> départ peuvent quitter la zone. Que le départ soit accessible
    # sans heurt depuis le repos est à la charge de l'opérateur qui calibre.
    rest_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Mains (LHand, RHand) : hors des groupes calibrés (on ne les guide pas à la main),
    # bornées par leur plage d'usine 0..1 et pilotées depuis l'interface.
    hand_joints: List[str] = field(default_factory=list)

    # -- accès pratiques -------------------------------------------------
    @property
    def arm_joints(self) -> List[str]:
        return [j for g in ARM_GROUPS for j in self.groups[g]]

    @property
    def head_joints(self) -> List[str]:
        return list(self.groups[HEAD_GROUP])

    @property
    def all_joints(self) -> List[str]:
        """Articulations calibrées (bras + tête). Les mains n'en font pas partie."""
        return self.arm_joints + self.head_joints

    @property
    def controlled_joints(self) -> List[str]:
        """Tout ce que le logiciel peut lire ou commander : bras, tête et mains."""
        return self.all_joints + self.hand_joints

    def is_hand(self, joint: str) -> bool:
        return joint in self.hand_joints

    def pose(self, name: str, joints: List[str]) -> List[float]:
        flat = {}
        for g in self.poses[name].values():
            flat.update(g)
        return [float(flat[j]) for j in joints]

    def emotion_index(self, emotion: str) -> int:
        if emotion not in self.emotions:
            raise ConfigError(f"Émotion inconnue : {emotion!r}. Connues : {self.emotions}")
        return self.emotions.index(emotion)

    def fingerprint(self) -> str:
        """Empreinte des éléments qui conditionnent la compatibilité d'un modèle."""
        payload = json.dumps(
            {"groups": self.groups, "emotions": self.emotions, "calibration": self.calibration_version},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if data.get("schema_version") != 1:
        raise ConfigError(f"{path.name}: schema_version 1 attendu")
    return data


def load_config(config_dir: Optional[os.PathLike] = None, calibration_file: Optional[os.PathLike] = None,
                use_calibration: bool = True) -> Config:
    """Charge configs/. Si une calibration est active (configs/calibration/ACTIVE)
    ou fournie, ses limites, poses et contraintes couplées mesurées REMPLACENT
    les valeurs de remplacement de joints.yaml / poses.yaml."""
    from ..calibration import Calibration, CalibrationError, active_calibration_path
    root = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    joints = _load(root / "joints.yaml")
    poses = _load(root / "poses.yaml")
    emos = _load(root / "emotions.yaml")
    guard = _load(root / "guardian.yaml")
    from .env import apply_env
    robot = apply_env(_load(root / "robot.yaml"), Path(os.environ.get("NAO_MOTION_ENV", root.parent / ".env")))
    model = _load(root / "model.yaml")

    groups = {k: list(v) for k, v in joints["groups"].items()}
    for g in ARM_GROUPS + (HEAD_GROUP,):
        if g not in groups or not groups[g]:
            raise ConfigError(f"joints.yaml : groupe {g} manquant")
    all_j = [j for g in groups.values() for j in g]
    if len(set(all_j)) != len(all_j):
        raise ConfigError("joints.yaml : articulation présente dans plusieurs groupes")
    for j in all_j:
        if any(j.startswith(p) for p in ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")):
            raise ConfigError(f"Les jambes sont hors périmètre : {j}")

    limits = {}
    for j in all_j:
        if j not in joints["limits"]:
            raise ConfigError(f"Limite manquante pour {j}")
        lo, hi = (float(x) for x in joints["limits"][j])
        if not lo < hi:
            raise ConfigError(f"Limite invalide pour {j}: {lo} >= {hi}")
        limits[j] = (lo, hi)

    hands = joints.get(HANDS_GROUP) or {}
    hand_joints = [str(j) for j in hands.get("joints") or []]
    hand_limits = {}
    for j in hand_joints:
        if j in all_j:
            raise ConfigError(f"joints.yaml : {j} est à la fois une main et une articulation d'un groupe")
        lim = (hands.get("limits") or {}).get(j)
        if lim is None:
            raise ConfigError(f"joints.yaml : limite manquante pour la main {j}")
        lo, hi = (float(x) for x in lim)
        if not 0.0 <= lo < hi <= 1.0:
            raise ConfigError(f"joints.yaml : plage de {j} invalide [{lo}, {hi}] (ouverture entre 0 et 1)")
        hand_limits[j] = (lo, hi)
    poses_file = poses

    calibration = None
    poses_from_cal = False
    cal_path = calibration_file or (active_calibration_path(root) if use_calibration else None)
    if cal_path:
        try:
            calibration = Calibration.load(cal_path)
        except CalibrationError as e:
            raise ConfigError(str(e)) from e
        missing = [j for j in all_j if j not in calibration.limits]
        if missing:
            raise ConfigError(f"Calibration {calibration.version} : articulations non calibrées {missing}")
        limits = {j: tuple(calibration.limits[j]) for j in all_j}
        cal_poses = calibration.poses or {}
        if all(g in cal_poses.get(n, {}) for n in ("rest", "start") for g in groups):
            poses = dict(poses, rest=cal_poses["rest"], start=cal_poses["start"])
            poses_from_cal = True

    # limites du repos et des segments repos <-> départ : limites d'usine mesurées si
    # connues, sinon limites de la configuration
    rest_limits = dict(limits)
    if calibration is not None:
        for j, m in (calibration.measured or {}).items():
            f = (m or {}).get("factory")
            if j in rest_limits and f:
                rest_limits[j] = (min(float(f[0]), limits[j][0]), max(float(f[1]), limits[j][1]))
    # mains : plage d'usine, que la calibration ne remplace pas
    limits.update(hand_limits)
    rest_limits.update(hand_limits)

    def _check(name, g, gp, lims, hint="", js=None):
        js = groups[g] if js is None else js
        missing = [j for j in js if j not in gp]
        if missing:
            raise ConfigError(f"poses : {name}/{g} incomplet : {missing}")
        for j in js:
            lo, hi = lims[j]
            if not lo <= float(gp[j]) <= hi:
                raise ConfigError(f"pose {name}/{j}={gp[j]} hors limites [{lo}, {hi}]" + hint)
        return {j: float(gp[j]) for j in js}

    hint = " (limites calibrées : capturer les poses pendant la calibration)" if calibration else ""
    pose_d = {"rest": {}, "start": {}}
    for g in groups:
        pose_d["start"][g] = _check("start", g, poses["start"].get(g) or {}, limits, hint)
        # le repos peut sortir de la zone de mouvement (contact), pas des limites d'usine
        pose_d["rest"][g] = _check("rest", g, poses["rest"].get(g) or {}, rest_limits, " (limites d'usine)")
    for n in ("rest", "start"):
        if hand_joints:
            # poses des mains : celles de la calibration si elle en porte, sinon poses.yaml
            src = (poses.get(n) or {}).get(HANDS_GROUP) or (poses_file.get(n) or {}).get(HANDS_GROUP) or {}
            pose_d[n][HANDS_GROUP] = _check(n, HANDS_GROUP, src, limits, js=hand_joints)

    emotions = [e["id"] for e in emos["emotions"]]
    if len(set(emotions)) != len(emotions) or not emotions:
        raise ConfigError("emotions.yaml : identifiants vides ou dupliqués")
    labels = {e["id"]: e.get("label", e["id"]) for e in emos["emotions"]}
    intentions = [i["id"] for i in emos.get("intentions", [])]
    if set(intentions) & set(emotions):
        raise ConfigError("Une intention ne doit pas porter le même identifiant qu'une émotion")

    cc = []
    manual = list(guard.get("coupled_constraints", []) or [])
    for c in manual + (list(calibration.coupled) if calibration else []):
        for j in c["terms"]:
            if j not in limits:
                raise ConfigError(f"Contrainte {c['name']} : articulation inconnue {j}")
        cc.append(CoupledConstraint(c["name"], {k: float(v) for k, v in c["terms"].items()},
                                    float(c["bound"]), c.get("description", "")))
    gcfg = GuardianConfig(
        limit_margin_rad=float(guard["limit_margin_rad"]),
        max_velocity_rad_s=float(guard["max_velocity_rad_s"]),
        max_acceleration_rad_s2=float(guard["max_acceleration_rad_s2"]),
        per_joint_velocity_rad_s={k: float(v) for k, v in (guard.get("per_joint_velocity_rad_s") or {}).items()},
        check_rate_hz=float(guard["check_rate_hz"]),
        min_duration_s=float(guard["min_duration_s"]),
        max_duration_s=float(guard["max_duration_s"]),
        start_tolerance_rad=float(guard["start_tolerance_rad"]),
        transition_velocity_rad_s=float(guard["transition_velocity_rad_s"]),
        coupled_constraints=cc,
        rest_velocity_rad_s=float(guard.get("rest_velocity_rad_s", 0.3)),
        per_joint_margin={k: float(v) for k, v in (guard.get("per_joint_margin") or {}).items()},
        per_joint_acceleration={k: float(v) for k, v in (guard.get("per_joint_acceleration") or {}).items()},
    )
    return Config(
        root=root, groups=groups, limits=limits,
        calibration_version=calibration.version if calibration else str(joints["calibration"]["version"]),
        calibrated=calibration is not None,
        poses=pose_d,
        poses_calibrated=poses_from_cal,
        emotions=emotions, emotion_labels=labels, intentions=intentions,
        guardian=gcfg, robot=robot, model=model,
        raw={"joints": joints, "poses": poses, "emotions": emos, "guardian": guard},
        calibration=calibration,
        rest_limits=rest_limits,
        hand_joints=hand_joints,
    )
