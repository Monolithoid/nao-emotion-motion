"""Stockage des prises (démonstrations).

Organisation sur disque :
    <data_dir>/takes/<take_id>/meta.json     métadonnées (JSON lisible)
    <data_dir>/takes/<take_id>/samples.npz   t (N,) en s ; q (N, J) en rad, MESURÉS

Les échantillons bruts ne sont jamais réécrits après enregistrement ; seules
les métadonnées (statut, annotations, segment utile) peuvent évoluer.

Une prise de bras (kind="arms") contient les articulations des deux bras
guidées à la main. Une prise de tête (kind="head") contient TOUTES les
articulations mesurées pendant l'exécution des bras : bras pilotés et tête
guidée. La trajectoire des bras vient soit d'une démonstration enregistrée
(`arm_source="take"`, `arm_take_id`), soit du modèle des bras
(`arm_source="generated"`, `arm_generation` : émotion, intensité, empreinte du
checkpoint, variation, graine). La consigne envoyée aux bras est conservée dans
samples.npz (`ref_t`, `ref_q`), à côté des mesures. `segment` désigne la
fenêtre d'exécution, sur laquelle bras et tête partagent la même phase.
`arm_gesture_id` identifie le geste des bras (même geste => même groupe pour le
découpage entraînement/validation).

Une prise de mains (kind="hands", passage 3) suit le même principe que la tête :
bras rejoués (prise ou geste généré), tête raide sur place, et mains PILOTÉES par
l'opérateur depuis l'interface (rien n'est guidé). LHand / RHand valent une
ouverture entre 0 (fermée) et 1 (ouverte), pas un angle. La consigne envoyée
(bras + mains, mêmes horodatages que les mesures) est conservée dans samples.npz
(`ref_*`) : c'est la consigne des mains, l'intention de l'opérateur, qui sert de
cible à l'apprentissage par défaut (`hands_model.target`).
"""
from __future__ import annotations

import datetime as _dt
import json
import secrets
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..events import EventContext
from ..trajectory import Trajectory, TrajectoryError, check_samples

FORMAT_VERSION = 1
KINDS = ("arms", "head", "hands")
FOLLOWER_KINDS = ("head", "hands")   # prises enregistrées sur une relecture des bras
STATUSES = ("pending", "accepted", "rejected")


class TakeError(ValueError):
    pass


def new_id(prefix: str) -> str:
    return f"{prefix}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"


@dataclass
class TakeMeta:
    take_id: str
    session_id: str
    kind: str
    emotion: str
    intensity: float
    joints: List[str]
    guided_joints: List[str]
    driven_joints: List[str]
    event: dict
    calibration_version: str
    config_fingerprint: str
    adapter: str
    sample_rate_hz: float
    segment: Tuple[float, float]
    head_initial: Optional[List[float]] = None   # [HeadYaw, HeadPitch] au début du segment
    hands_initial: Optional[List[float]] = None  # [LHand, RHand] mesurées au début du segment (passage 3)
    arm_take_id: Optional[str] = None
    arm_source: Optional[str] = None          # "take" | "generated" (prises de tête)
    arm_gesture_id: Optional[str] = None
    arm_generation: Optional[dict] = None
    intention: Optional[str] = None
    synthetic: bool = False
    status: str = "pending"
    annotations: Dict = field(default_factory=lambda: {"notes": "", "tags": []})
    unit: str = "rad"
    time_unit: str = "s"
    event_frame: str = "torso"
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    format_version: int = FORMAT_VERSION
    n_samples: int = 0
    duration_s: float = 0.0

    @property
    def event_ctx(self) -> EventContext:
        return EventContext.from_dict(self.event)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segment"] = list(self.segment)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TakeMeta":
        if d.get("format_version") != FORMAT_VERSION:
            raise TakeError(f"format_version {d.get('format_version')} non supportée")
        d = dict(d)
        d["segment"] = tuple(d["segment"])
        return cls(**d)


@dataclass
class Take:
    meta: TakeMeta
    traj: Trajectory  # mesures brutes

    def segment_trajectory(self, joints=None) -> Trajectory:
        t0, t1 = self.meta.segment
        tr = self.traj if joints is None else self.traj.select(joints)
        mask = (tr.t >= t0) & (tr.t <= t1)
        tt = np.unique(np.concatenate([[t0], tr.t[mask], [t1]]))
        return Trajectory(tr.joints, tt - t0, tr.sample(tt))


class TakeStore:
    def __init__(self, data_dir, config: Config):
        self.root = Path(data_dir)
        self.dir = self.root / "takes"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

    # -- validation ---------------------------------------------------------
    def validate(self, meta: TakeMeta, traj: Trajectory):
        c = self.cfg
        if meta.kind not in KINDS:
            raise TakeError(f"kind inconnu : {meta.kind}")
        if meta.emotion not in c.emotions:
            raise TakeError(f"Émotion inconnue : {meta.emotion}")
        if meta.intention is not None and meta.intention not in c.intentions:
            raise TakeError(f"Intention inconnue : {meta.intention}")
        if not (np.isfinite(meta.intensity) and 0.0 <= meta.intensity <= 1.0):
            raise TakeError("Intensité hors de [0, 1]")
        if meta.status not in STATUSES:
            raise TakeError(f"Statut inconnu : {meta.status}")
        if meta.unit != "rad" or meta.time_unit != "s" or meta.event_frame != "torso":
            raise TakeError("Unités/repère attendus : rad, s, torso")
        try:
            check_samples(traj.t, traj.q, traj.joints)
        except TrajectoryError as e:
            raise TakeError(str(e)) from e
        if traj.joints != meta.joints:
            raise TakeError("Articulations des métadonnées et des échantillons différentes")
        unknown = set(meta.joints) - set(c.controlled_joints)
        if unknown:
            raise TakeError(f"Articulations inconnues : {sorted(unknown)}")
        if set(meta.guided_joints) & set(meta.driven_joints):
            raise TakeError("Une articulation ne peut pas être à la fois guidée et pilotée")
        t0, t1 = meta.segment
        if not (traj.t[0] - 1e-9 <= t0 < t1 <= traj.t[-1] + 1e-9):
            raise TakeError("Segment utile hors de l'enregistrement")
        EventContext.from_dict(meta.event)  # lève si invalide
        if meta.kind == "arms":
            if set(c.arm_joints) - set(meta.joints):
                raise TakeError("Une prise de bras doit contenir toutes les articulations des bras")
            if meta.arm_take_id is not None:
                raise TakeError("Une prise de bras ne référence pas d'autre prise")
        else:
            if meta.kind == "head" and set(c.all_joints) - set(meta.joints):
                raise TakeError("Une prise de tête doit contenir bras (relecture mesurée) et tête")
            if meta.kind == "hands":
                if not c.hand_joints or set(c.arm_joints + c.hand_joints) - set(meta.joints):
                    raise TakeError("Une prise de mains doit contenir bras (relecture mesurée) et mains")
                if meta.guided_joints:
                    raise TakeError("Prise de mains : aucune articulation n'est guidée (mains pilotées)")
            src = meta.arm_source or "take"
            if src == "take":
                if not meta.arm_take_id or not self.exists(meta.arm_take_id):
                    raise TakeError("Prise de tête : arm_take_id manquant ou introuvable")
                if self.load_meta(meta.arm_take_id).kind != "arms":
                    raise TakeError("arm_take_id doit désigner une prise de bras")
            elif src == "generated":
                g = meta.arm_generation or {}
                if g.get("emotion") not in c.emotions or not g.get("arm_checkpoint_digest"):
                    raise TakeError("Prise de tête sur geste généré : arm_generation incomplet")
            else:
                raise TakeError(f"arm_source inconnu : {src}")
            if meta.kind == "head" and set(meta.guided_joints) != set(c.head_joints):
                raise TakeError("Prise de tête : seules les articulations de la tête sont guidées")

    # -- E/S ----------------------------------------------------------------
    def exists(self, take_id: str) -> bool:
        return (self.dir / take_id / "meta.json").exists()

    def group_id(self, meta: TakeMeta) -> str:
        """Identifiant du geste des bras (groupe de découpage entraînement/validation)."""
        if meta.kind == "arms":
            return meta.take_id
        return meta.arm_gesture_id or meta.arm_take_id or meta.take_id

    def save(self, meta: TakeMeta, traj: Trajectory, reference: Optional[Trajectory] = None) -> str:
        meta.n_samples = int(traj.t.shape[0])
        meta.duration_s = float(traj.t[-1] - traj.t[0])
        self.validate(meta, traj)
        d = self.dir / meta.take_id
        if d.exists():
            raise TakeError(f"La prise {meta.take_id} existe déjà")
        d.mkdir(parents=True)
        extra = {}
        if reference is not None:
            extra = {"ref_t": reference.t, "ref_q": reference.q, "ref_joints": np.array(reference.joints)}
        np.savez(d / "samples.npz", t=traj.t, q=traj.q, joints=np.array(traj.joints), **extra)
        self._write_meta(meta)
        return meta.take_id

    def _write_meta(self, meta: TakeMeta):
        p = self.dir / meta.take_id / "meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def load_meta(self, take_id: str) -> TakeMeta:
        return TakeMeta.from_dict(json.loads((self.dir / take_id / "meta.json").read_text(encoding="utf-8")))

    def load(self, take_id: str) -> Take:
        meta = self.load_meta(take_id)
        with np.load(self.dir / take_id / "samples.npz", allow_pickle=False) as z:
            joints = [str(j) for j in z["joints"]]
            traj = Trajectory(joints, z["t"], z["q"])
        if joints != meta.joints:
            raise TakeError(f"{take_id}: articulations incohérentes entre meta et échantillons")
        return Take(meta, traj)

    def update(self, take_id: str, *, status=None, notes=None, tags=None, segment=None, intention="__keep__"):
        meta = self.load_meta(take_id)
        if status is not None:
            if status not in STATUSES:
                raise TakeError(f"Statut inconnu : {status}")
            meta.status = status
        if notes is not None:
            meta.annotations["notes"] = str(notes)
        if tags is not None:
            meta.annotations["tags"] = [str(t) for t in tags]
        if intention != "__keep__":
            meta.intention = intention
        if segment is not None:
            meta.segment = (float(segment[0]), float(segment[1]))
        self.validate(meta, self.load(take_id).traj)
        self._write_meta(meta)
        return meta

    def list(self, kind=None, status=None, emotion=None) -> List[TakeMeta]:
        out = []
        for p in sorted(self.dir.glob("*/meta.json")):
            m = TakeMeta.from_dict(json.loads(p.read_text(encoding="utf-8")))
            if kind and m.kind != kind:
                continue
            if status and m.status != status:
                continue
            if emotion and m.emotion != emotion:
                continue
            out.append(m)
        return out

    def head_takes_for(self, arm_take_id: str, kind: str = "head") -> List[TakeMeta]:
        return [m for m in self.list(kind=kind) if m.arm_take_id == arm_take_id]

    def load_reference(self, take_id: str) -> Optional[Trajectory]:
        """Consigne envoyée pendant une prise de tête (bras) ou de mains (bras + mains). None si absente."""
        with np.load(self.dir / take_id / "samples.npz", allow_pickle=False) as z:
            if "ref_t" not in z.files:
                return None
            return Trajectory([str(j) for j in z["ref_joints"]], z["ref_t"], z["ref_q"])

    def stats(self, bins=(0.0, 0.34, 0.67, 1.0001)) -> dict:
        """Répartition des prises par type, émotion, tranche d'intensité et statut."""
        labels = [f"{bins[i]:.2f}-{min(bins[i+1], 1.0):.2f}" for i in range(len(bins) - 1)]
        table = {k: {e: {l: Counter() for l in labels} for e in self.cfg.emotions} for k in KINDS}
        contexts = defaultdict(Counter)
        for m in self.list():
            if m.emotion not in self.cfg.emotions:
                continue
            b = labels[int(np.clip(np.searchsorted(bins, m.intensity, side="right") - 1, 0, len(labels) - 1))]
            table[m.kind][m.emotion][b][m.status] += 1
            if m.kind == "head":
                ev = m.event_ctx
                contexts[self.group_id(m)]["avec_evenement" if ev.present else "sans_evenement"] += 1
        return {
            "bins": labels,
            "table": {k: {e: {b: dict(c) for b, c in v.items()} for e, v in t.items()} for k, t in table.items()},
            "head_contexts_per_arm_gesture": {k: dict(v) for k, v in contexts.items()},
        }
