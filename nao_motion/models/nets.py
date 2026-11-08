"""Petits réseaux PyTorch (MLP) et rechargement des checkpoints.

Modèle des bras : [émotion, intensité, événement] -> [points de contrôle (K×Ja), log durée].
Modèle de la tête : [émotion, intensité, bras planifiés, tête actuelle, événement]
                    -> points de contrôle de la tête (K×Jh), sur la MÊME phase que
                    les bras (même durée), ce qui permet d'apprendre anticipation
                    ou retard par rapport au geste des bras.
Modèle des mains : même forme que la tête, [..., ouverture actuelle des mains, ...]
                    -> ouverture des mains (K×2, 0 fermée .. 1 ouverte), sur la
                    phase des bras : la main s'ouvre au bon moment du geste.

Référence déterministe : une régression par moindres carrés produit la MOYENNE
des gestes portant la même étiquette. Si « joie » a été démontrée de deux
façons différentes, le modèle peut générer un geste intermédiaire qui n'a été
montré nulle part. Évolution prévue sans changer l'interface : `latent_dim > 0`
ajoute une entrée latente z (nulle par défaut) qui permettra un CVAE ou un
mélange de prototypes ; le premier prototype garde latent_dim = 0.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from .. import __version__
from ..config import Config
from ..events import EVENT_FEATURES

CKPT_FORMAT = "nao_motion.checkpoint"
CKPT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], dropout: float = 0.0, latent_dim: int = 0):
        super().__init__()
        self.in_dim, self.out_dim, self.latent_dim = in_dim, out_dim, latent_dim
        layers, d = [], in_dim + latent_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, z=None):
        if self.latent_dim:
            if z is None:
                z = torch.zeros(x.shape[0], self.latent_dim, dtype=x.dtype, device=x.device)
            x = torch.cat([x, z], dim=-1)
        return self.net(x)


def build(arch: dict) -> MLP:
    return MLP(arch["in_dim"], arch["out_dim"], arch["hidden"], arch.get("dropout", 0.0), arch.get("latent_dim", 0))


def save_checkpoint(path, model: MLP, *, component: str, arch: dict, meta: dict):
    payload = {
        "format": CKPT_FORMAT, "version": CKPT_VERSION, "component": component,
        "arch": json.loads(json.dumps(arch)), "state_dict": model.state_dict(),
        "meta": json.loads(json.dumps(meta)),  # types simples uniquement (chargement weights_only)
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "package_version": __version__, "torch_version": str(torch.__version__),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(payload, buf)
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


class LoadedModel:
    """Modèle rechargé + métadonnées, vérifié contre la configuration courante."""

    def __init__(self, path, cfg: Optional[Config] = None, component: Optional[str] = None):
        path = Path(path)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise CheckpointError(f"Lecture impossible de {path} : {e}") from e
        if ck.get("format") != CKPT_FORMAT or ck.get("version") != CKPT_VERSION:
            raise CheckpointError("Format de checkpoint inconnu")
        if component and ck["component"] != component:
            raise CheckpointError(f"Checkpoint {ck['component']} fourni, {component} attendu")
        self.path = path
        self.digest = file_digest(path)
        self.component = ck["component"]
        self.arch = ck["arch"]
        self.meta = ck["meta"]
        self.created_at = ck["created_at"]
        self.model = build(self.arch)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.warnings = []
        if cfg is not None:
            self.check_compat(cfg)

    def check_compat(self, cfg: Config):
        m = self.meta
        if m["emotions"] != cfg.emotions:
            raise CheckpointError(f"Liste d'émotions différente : modèle {m['emotions']} / config {cfg.emotions}")
        expected = {"arms": cfg.arm_joints, "head": cfg.head_joints, "hands": cfg.hand_joints}.get(self.component)
        if m["joints"] != expected:
            raise CheckpointError("Articulations du modèle différentes de la configuration")
        if self.component in ("head", "hands") and m["arm_joints"] != cfg.arm_joints:
            raise CheckpointError(f"Articulations des bras du modèle {self.component} différentes de la configuration")
        if self.component == "arms" and m.get("event_features", 0) != EVENT_FEATURES:
            raise CheckpointError("Modèle des bras entraîné sans l'événement (angle, distance) : "
                                  "réentraîner les bras, puis la tête")
        if self.component in ("head", "hands") and m.get("event_features", 4) != EVENT_FEATURES:
            raise CheckpointError("Encodage de l'événement différent (distance ajoutée) : réentraîner le modèle de la tête")
        if m["calibration_version"] != cfg.calibration_version:
            self.warnings.append(f"Calibration différente : modèle {m['calibration_version']}, "
                                 f"config {cfg.calibration_version}")

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
        return self.model(t).numpy().astype(np.float64)

    def export_numpy(self, path):
        """Export léger (numpy seul) pour une inférence sans PyTorch."""
        arrays = {}
        linears = [m for m in self.model.net if isinstance(m, nn.Linear)]
        for i, lin in enumerate(linears):
            arrays[f"W{i}"] = lin.weight.detach().numpy()
            arrays[f"b{i}"] = lin.bias.detach().numpy()
        header = {"component": self.component, "arch": self.arch, "meta": self.meta,
                  "source_digest": self.digest, "activation": "tanh"}
        np.savez(path, header=np.array(json.dumps(header)), **arrays)


class NumpyModel:
    """Inférence numpy pure d'un export `export_numpy` (même interface que LoadedModel)."""

    def __init__(self, path, cfg: Optional[Config] = None, component: Optional[str] = None):
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            n = len([k for k in z.files if k.startswith("W")])
            self.W = [z[f"W{i}"] for i in range(n)]
            self.b = [z[f"b{i}"] for i in range(n)]
        self.component = header["component"]
        if component and self.component != component:
            raise CheckpointError(f"Export {self.component} fourni, {component} attendu")
        self.arch, self.meta = header["arch"], header["meta"]
        self.digest = header["source_digest"]
        self.warnings = []
        if cfg is not None:
            LoadedModel.check_compat(self, cfg)

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if self.arch.get("latent_dim", 0):
            h = np.concatenate([h, np.zeros((h.shape[0], self.arch["latent_dim"]))], axis=1)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return h


def load_model(path, cfg=None, component=None):
    p = str(path)
    return NumpyModel(path, cfg, component) if p.endswith(".npz") else LoadedModel(path, cfg, component)
