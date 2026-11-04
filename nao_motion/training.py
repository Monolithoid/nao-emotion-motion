"""Entraînement des modèles bras et tête, avec métriques et checkpoints.

Les métriques sont calculées sur des trajectoires denses (radians), en
entraînement et en validation, et comparées à une référence simple « moyenne
des prises de même émotion et d'intensité voisine ». Aucun nombre de prises
n'est imposé : le rapport indique simplement ce que les données permettent
de mesurer (ex. pas de validation possible avec un seul groupe).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config
from .data.representation import bspline_basis, decode
from .data.store import TakeStore
from .events import EventContext
from .guardian import Guardian
from .models.datasets import Examples, arm_examples, grouped_split, head_examples
from .models.features import Norm, head_input_dim
from .models.nets import LoadedModel, build, save_checkpoint


class TrainingError(RuntimeError):
    pass


def _run_dir(root, component):
    d = Path(root) / f"{component}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _fit(ex: Examples, tr, va, arch, K, J, tcfg, predict_duration: bool, log=None):
    torch.manual_seed(int(tcfg.get("seed", 0)))
    model = build(arch)
    B = torch.tensor(bspline_basis(K, np.linspace(0, 1, 50)), dtype=torch.float32)
    X = torch.tensor(ex.X, dtype=torch.float32)
    Y = torch.tensor(ex.Y, dtype=torch.float32).reshape(-1, K, J)
    Dl = torch.tensor(ex.log_dur, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    sw = float(tcfg.get("smoothness_weight", 0.0))
    dw = float(tcfg.get("duration_weight", 0.05))
    tr_t, va_t = torch.tensor(tr), torch.tensor(va)

    def loss_on(idx, train=True):
        model.train(train)
        out = model(X[idx])
        C = out[:, : K * J].reshape(-1, K, J)
        dense = torch.einsum("nk,bkj->bnj", B, C)
        target = torch.einsum("nk,bkj->bnj", B, Y[idx])
        loss = ((dense - target) ** 2).mean()
        if predict_duration:
            loss = loss + dw * ((out[:, -1] - Dl[idx]) ** 2).mean()
        if sw:
            loss = loss + sw * ((C[:, 2:] - 2 * C[:, 1:-1] + C[:, :-2]) ** 2).mean()
        return loss

    best, best_state, history = float("inf"), None, []
    epochs = int(tcfg["epochs"])
    for ep in range(epochs):
        opt.zero_grad()
        l = loss_on(tr_t)
        l.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                lv = float(loss_on(va_t, False)) if len(va) else float("nan")
            history.append({"epoch": ep, "train_loss": l.item(), "val_loss": lv})
            crit = lv if len(va) else float(l)
            if crit < best:
                best, best_state = crit, {k: v.clone() for k, v in model.state_dict().items()}
            if log:
                log(ep, l.item(), lv)
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def _dense_rmse(pred_n, true_n, norm: Norm, K):
    """RMSE en radians sur trajectoires denses (phase), globale et par articulation."""
    J = len(norm.joints)
    B = bspline_basis(K, np.linspace(0, 1, 100))
    P = np.einsum("nk,bkj->bnj", B, norm.inv(pred_n.reshape(-1, K, J)))
    T = np.einsum("nk,bkj->bnj", B, norm.inv(true_n.reshape(-1, K, J)))
    err = (P - T) ** 2
    return float(np.sqrt(err.mean())), dict(zip(norm.joints, np.sqrt(err.mean(axis=(0, 1))).round(4).tolist()))


def _baseline(ex: Examples, tr, va):
    """Prédiction de référence : moyenne des prises d'entraînement de même émotion et
    d'intensité voisine (±0.2), sinon même émotion, sinon moyenne globale."""
    preds = []
    for i in va:
        same = [j for j in tr if ex.emotions[j] == ex.emotions[i]]
        near = [j for j in same if abs(ex.intensities[j] - ex.intensities[i]) <= 0.2]
        pool = near or same or list(tr)
        preds.append(ex.Y[pool].mean(axis=0))
    return np.array(preds)


def _evaluate(model, ex, tr, va, norm, K, predict_duration):
    with torch.no_grad():
        out = model(torch.tensor(ex.X, dtype=torch.float32)).numpy().astype(np.float64)
    Yp = out[:, : ex.Y.shape[1]]
    m = {"n_examples": len(ex), "n_train": int(len(tr)), "n_val": int(len(va)),
         "n_train_groups": len({ex.groups[i] for i in tr}), "n_val_groups": len({ex.groups[i] for i in va})}
    m["train_rmse_rad"], m["train_rmse_per_joint"] = _dense_rmse(Yp[tr], ex.Y[tr], norm, K)
    if len(va):
        m["val_rmse_rad"], m["val_rmse_per_joint"] = _dense_rmse(Yp[va], ex.Y[va], norm, K)
        m["val_baseline_label_mean_rmse_rad"], _ = _dense_rmse(_baseline(ex, tr, va), ex.Y[va], norm, K)
        if predict_duration:
            m["val_duration_mae_s"] = float(np.abs(np.exp(out[va, -1]) - np.exp(ex.log_dur[va])).mean())
    else:
        m["warning"] = "Moins de deux groupes de prises : pas de validation possible, résultats non mesurés."
    if predict_duration:
        m["train_duration_mae_s"] = float(np.abs(np.exp(out[tr, -1]) - np.exp(ex.log_dur[tr])).mean())
    return m


def train_arms(store: TakeStore, cfg: Config, runs_dir, statuses=("accepted",), epochs: Optional[int] = None,
               log=None) -> Path:
    K = int(cfg.model["representation"]["n_control_points"])
    J = len(cfg.arm_joints)
    ex = arm_examples(store, cfg, statuses)
    if len(ex) == 0:
        raise TrainingError("Aucune prise de bras utilisable (statut 'accepted' requis par défaut)")
    tcfg = dict(cfg.model["training"])
    if epochs:
        tcfg["epochs"] = epochs
    tr, va, val_groups = grouped_split(ex.groups, float(tcfg["val_fraction"]), int(tcfg["seed"]))
    mc = cfg.model["arm_model"]
    arch = {"in_dim": ex.X.shape[1], "out_dim": K * J + 1, "hidden": list(mc["hidden"]),
            "dropout": float(mc.get("dropout", 0)), "latent_dim": 0}
    model, hist = _fit(ex, tr, va, arch, K, J, tcfg, True, log)
    norm = Norm.from_config(cfg, cfg.arm_joints)
    metrics = _evaluate(model, ex, tr, va, norm, K, True)
    run = _run_dir(runs_dir, "arms")
    meta = _meta(cfg, "arms", cfg.arm_joints, norm, K, ex, tr, va, tcfg)
    meta["synthetic_data"] = any(store.load_meta(t).synthetic for t in ex.take_ids)
    meta["metrics"] = metrics
    save_checkpoint(run / "model.pt", model, component="arms", arch=arch, meta=meta)
    loaded = LoadedModel(run / "model.pt", cfg, "arms")
    loaded.export_numpy(run / "model_numpy.npz")
    metrics["guardian"] = _guardian_sweep(cfg, arm_model=loaded)
    _write_report(run, metrics, hist, meta)
    (Path(runs_dir) / "latest_arms.txt").write_text(str(run / "model.pt"))
    return run


def train_head(store: TakeStore, cfg: Config, runs_dir, arm_checkpoint, statuses=("accepted",),
               epochs: Optional[int] = None, log=None) -> Path:
    """Entraîne la tête ; le modèle des bras est chargé et FIGÉ (aucun gradient)."""
    K = int(cfg.model["representation"]["n_control_points"])
    J = len(cfg.head_joints)
    arm = LoadedModel(arm_checkpoint, cfg, "arms")
    for p in arm.model.parameters():
        p.requires_grad_(False)
    source = cfg.model["head_model"].get("arm_input", "recorded")
    ex = head_examples(store, cfg, statuses, source, arm)
    if len(ex) == 0:
        raise TrainingError("Aucune prise de tête utilisable")
    tcfg = dict(cfg.model["training"])
    if epochs:
        tcfg["epochs"] = epochs
    tr, va, _ = grouped_split(ex.groups, float(tcfg["val_fraction"]), int(tcfg["seed"]))
    mc = cfg.model["head_model"]
    in_dim = head_input_dim(len(cfg.emotions), K, len(cfg.arm_joints), J)
    assert ex.X.shape[1] == in_dim
    arch = {"in_dim": in_dim, "out_dim": K * J, "hidden": list(mc["hidden"]),
            "dropout": float(mc.get("dropout", 0)), "latent_dim": 0}
    model, hist = _fit(ex, tr, va, arch, K, J, tcfg, False, log)
    norm = Norm.from_config(cfg, cfg.head_joints)
    metrics = _evaluate(model, ex, tr, va, norm, K, False)
    run = _run_dir(runs_dir, "head")
    meta = _meta(cfg, "head", cfg.head_joints, norm, K, ex, tr, va, tcfg)
    meta["synthetic_data"] = any(store.load_meta(t).synthetic for t in ex.take_ids)
    meta.update({"arm_joints": cfg.arm_joints, "arm_checkpoint": str(arm_checkpoint),
                 "arm_checkpoint_digest": arm.digest, "arm_input": source,
                 "inputs": ["emotion", "intensity", "arm_ctrl_norm", "arm_log_duration",
                            "head_current_norm", "event[present,sin_az,cos_az,elev]"]})
    meta["metrics"] = metrics
    save_checkpoint(run / "model.pt", model, component="head", arch=arch, meta=meta)
    loaded = LoadedModel(run / "model.pt", cfg, "head")
    loaded.export_numpy(run / "model_numpy.npz")
    metrics["guardian"] = _guardian_sweep(cfg, arm_model=arm, head_model=loaded)
    _write_report(run, metrics, hist, meta)
    (Path(runs_dir) / "latest_head.txt").write_text(str(run / "model.pt"))
    return run


def _meta(cfg, component, joints, norm, K, ex, tr, va, tcfg):
    return {"component": component, "joints": list(joints), "unit": "rad", "time_unit": "s",
            "emotions": cfg.emotions, "representation": {"type": "bspline_cubic_clamped_uniform",
                                                          "n_control_points": K, "time": "phase[0,1]+duration"},
            "norm": norm.to_dict(), "calibration_version": cfg.calibration_version,
            "config_fingerprint": cfg.fingerprint(),
            "train_take_ids": [ex.take_ids[i] for i in tr], "val_take_ids": [ex.take_ids[i] for i in va],
            "split": "par groupe (prise de bras source)", "training": tcfg,
            "synthetic_data": None}  # renseigné par l'appelant


def _guardian_sweep(cfg, arm_model, head_model=None, intensities=(0.2, 0.5, 0.8, 1.0)):
    """Taux de trajectoires générées acceptées par le guardian (geste seul), sur une grille."""
    from .engine import MotionEngine
    eng = MotionEngine(cfg, arm_model, head_model)
    g = Guardian(cfg)
    events = [EventContext.none(), EventContext(True, 0.6, 0.1, source="test")] if head_model else [None]
    total = ok = 0
    failures = {}
    for e in cfg.emotions:
        for i in intensities:
            for ev in events:
                gen = eng.generate(e, i, ev)
                traj = gen.combined if head_model else gen.arms
                ctx = None if head_model else dict(zip(cfg.head_joints, cfg.pose("start", cfg.head_joints)))
                rep = g.validate(traj, context_pose=ctx)
                total += 1
                ok += rep.ok
                for v in rep.violations:
                    failures[v.code] = failures.get(v.code, 0) + 1
    return {"accepted": ok, "total": total, "rate": ok / total if total else None, "violations_by_code": failures}


def _write_report(run, metrics, hist, meta):
    (run / "metrics.json").write_text(json.dumps({"metrics": metrics, "history": hist}, indent=2, ensure_ascii=False))
    (run / "split.json").write_text(json.dumps({"train": meta["train_take_ids"], "val": meta["val_take_ids"]},
                                               indent=2))


def latest(runs_dir, component) -> Optional[Path]:
    p = Path(runs_dir) / f"latest_{component}.txt"
    if p.exists():
        q = Path(p.read_text().strip())
        if q.exists():
            return q
    return None
