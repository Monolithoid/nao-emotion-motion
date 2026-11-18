"""Petits réseaux PyTorch (MLP) et rechargement des checkpoints.

Trois modèles INDÉPENDANTS, de même entrée [émotion, intensité, événement (6)] :
  bras  -> [points de contrôle (K×Ja), log durée]
  tête  -> [points de contrôle (K×Jh), log durée]
  mains -> [ouverture des mains (K×2, 0 fermée .. 1 ouverte), log durée]
Chacun s'entraîne seul ; le moteur combine leurs sorties en un geste unique.

Référence déterministe : une régression par moindres carrés produit la MOYENNE
des gestes portant la même étiquette. Si « joie » a été démontrée de deux
façons différentes, le modèle peut générer un geste intermédiaire qui n'a été
montré nulle part.

Mode CVAE (model.yaml `cvae.enabled`, désactivé par défaut) : le MLP ci-dessus
devient le DÉCODEUR et reçoit en plus un code latent z (`latent_dim`). Un petit
ENCODEUR (sous-module `encoder`, utilisé seulement à l'entraînement et pour la
carte des z) lit la prise (consigne + points de contrôle + log durée) et donne
la loi de son z. À la génération, z est tiré d'une loi normale N(0, T²) (T =
température ; T = 0 donne le geste central z = 0). Sans CVAE (latent_dim = 0),
le réseau est exactement le MLP d'avant.
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


class Encoder(nn.Module):
    """CVAE encoder q(z | consigne, prise): [X, control points, log duration] -> (mu, logvar)."""

    def __init__(self, in_dim: int, latent_dim: int, hidden: List[int]):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, 2 * latent_dim))
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x, y, log_dur):
        h = self.net(torch.cat([x, y, log_dur.reshape(-1, 1)], dim=-1))
        mu, logvar = h[:, : self.latent_dim], h[:, self.latent_dim:]
        return mu, logvar.clamp(-10.0, 5.0)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], dropout: float = 0.0, latent_dim: int = 0,
                 encoder_hidden: Optional[List[int]] = None):
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
        # CVAE: encoder reading [X, Y (out_dim - 1 control points), log duration]
        self.encoder = Encoder(in_dim + out_dim, latent_dim, list(encoder_hidden)) \
            if latent_dim and encoder_hidden is not None else None

    def forward(self, x, z=None):
        if self.latent_dim:
            if z is None:
                z = torch.zeros(x.shape[0], self.latent_dim, dtype=x.dtype, device=x.device)
            x = torch.cat([x, z], dim=-1)
        return self.net(x)


def build(arch: dict) -> MLP:
    cv = arch.get("cvae") or {}
    return MLP(arch["in_dim"], arch["out_dim"], arch["hidden"], arch.get("dropout", 0.0), arch.get("latent_dim", 0),
               cv.get("encoder_hidden") if cv else None)


def is_cvae(model) -> bool:
    """True for a CVAE model (torch or numpy): it takes a latent z besides the consigne."""
    return bool((getattr(model, "arch", None) or {}).get("latent_dim", 0))


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
        name = {"arms": "des bras", "head": "de la tête", "hands": "des mains"}.get(self.component, self.component)
        if self.component in ("head", "hands") and not m.get("independent"):
            raise CheckpointError(f"Modèle {name} de l'ancien format (il dépendait du modèle des bras) : "
                                  "le réentraîner, il n'a plus besoin des bras")
        if m.get("event_features", 0) != EVENT_FEATURES:
            raise CheckpointError(f"Modèle {name} entraîné sans l'événement complet (angle, distance) : le réentraîner")
        k = (m.get("representation") or {}).get("n_control_points")
        if k and self.arch.get("out_dim") != int(k) * len(m["joints"]) + 1:
            raise CheckpointError(f"Modèle {name} incohérent : {self.arch.get('out_dim')} sorties pour {k} points "
                                  f"de contrôle x {len(m['joints'])} articulations")
        if m["calibration_version"] != cfg.calibration_version:
            self.warnings.append(f"Calibration différente : modèle {m['calibration_version']}, "
                                 f"config {cfg.calibration_version}")

    @torch.no_grad()
    def predict(self, x: np.ndarray, z: Optional[np.ndarray] = None) -> np.ndarray:
        t = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
        zt = None if z is None else torch.as_tensor(np.atleast_2d(z), dtype=torch.float32)
        return self.model(t, zt).numpy().astype(np.float64)

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

    def predict(self, x: np.ndarray, z: Optional[np.ndarray] = None) -> np.ndarray:
        h = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if self.arch.get("latent_dim", 0):
            zz = np.zeros((h.shape[0], self.arch["latent_dim"])) if z is None else \
                np.broadcast_to(np.atleast_2d(np.asarray(z, dtype=np.float64)), (h.shape[0], self.arch["latent_dim"]))
            h = np.concatenate([h, zz], axis=1)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return h


def load_model(path, cfg=None, component=None):
    p = str(path)
    return NumpyModel(path, cfg, component) if p.endswith(".npz") else LoadedModel(path, cfg, component)
