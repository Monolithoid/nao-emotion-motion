"""Entraînement des modèles bras, tête et mains (indépendants), avec métriques et checkpoints.

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
from .events import EVENT_FEATURES, EventContext
from .guardian import Guardian
from .models.datasets import (Examples, concat_examples, group_examples, group_joints, grouped_kfold, grouped_split,
                              mirrored_examples)
from .models.features import Norm, motion_input_dim
from .models.nets import LoadedModel, build, save_checkpoint


class TrainingError(RuntimeError):
    pass


def _run_dir(root, component):
    d = Path(root) / f"{component}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _fit(ex: Examples, tr, va, arch, K, J, tcfg, predict_duration: bool, log=None, keep_last=False):
    """Full-batch training. Keeps the state of the best validation epoch (checked every 10
    epochs), or the last state when `keep_last` (final model of a cross-validation, trained
    on every take for a number of epochs chosen by the folds)."""
    torch.manual_seed(int(tcfg.get("seed", 0)))
    model = build(arch)
    B = torch.tensor(bspline_basis(K, np.linspace(0, 1, 50)), dtype=torch.float32)
    X = torch.tensor(ex.X, dtype=torch.float32)
    Y = torch.tensor(ex.Y, dtype=torch.float32).reshape(-1, K, J)
    Dl = torch.tensor(ex.log_dur, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    sw = float(tcfg.get("smoothness_weight", 0.0))
    ew = float(tcfg.get("motion_energy_weight", 0.0))
    # target speeds (per take, per joint) and their overall scale, fixed: the energy term is
    # dimensionless (about 1 when the prediction does not move at all)
    tgt_speed = _speed_rms(torch.einsum("nk,bkj->bnj", B, Y))
    speed_scale = max(float(tgt_speed.mean().item()), 1e-6) if ew else 1.0
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
        if ew:
            # motion energy: each joint must move as much as in its take (RMS speed), whatever
            # the timing. Pointwise MSE alone rewards the MEAN of takes whose oscillations are
            # out of phase, which is flat (a wave becomes a still arm); this term does not care
            # about the phase, only about how much the joint moves.
            loss = loss + ew * ((_speed_rms(dense) - tgt_speed[idx]) ** 2).mean() / speed_scale ** 2
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
            if crit < best and not keep_last:
                best, best_state = crit, {k: v.clone() for k, v in model.state_dict().items()}
            if log:
                log(ep, l.item(), lv)
    if not keep_last:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def _speed_rms(dense):
    """RMS speed along the phase, per take and joint: (b, n, j) -> (b, j). Works on tensors."""
    d = (dense[:, 1:] - dense[:, :-1]) * (dense.shape[1] - 1)
    return ((d ** 2).mean(1) + 1e-8) ** 0.5


def _motion_kept(pred_n, true_n, norm: Norm, K):
    """How much movement the predictions keep compared with the takes: RMS speed (in phase,
    rad) of the predicted curves / of the takes, over the joints that move. 1 = as lively as
    the takes, 0 = still. The pointwise RMSE cannot show this: a flat curve through the middle
    of an oscillation has a modest RMSE."""
    J = len(norm.joints)
    B = bspline_basis(K, np.linspace(0, 1, 100))
    P = np.einsum("nk,bkj->bnj", B, norm.inv(np.asarray(pred_n).reshape(-1, K, J)))
    T = np.einsum("nk,bkj->bnj", B, norm.inv(np.asarray(true_n).reshape(-1, K, J)))
    sp = np.sqrt((np.diff(P, axis=1) ** 2).mean(axis=1)) * 99
    st = np.sqrt((np.diff(T, axis=1) ** 2).mean(axis=1)) * 99
    mv = st > 0.2  # joints moving at least ~0.2 rad per gesture length
    if not mv.any():
        return None
    return float(np.sqrt((sp[mv] ** 2).sum() / (st[mv] ** 2).sum()))


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
    m["train_motion_kept"] = _motion_kept(Yp[tr], ex.Y[tr], norm, K)
    if len(va):
        m["val_motion_kept"] = _motion_kept(Yp[va], ex.Y[va], norm, K)
        m["val_baseline_motion_kept"] = _motion_kept(_baseline(ex, tr, va), ex.Y[va], norm, K)
        m["val_rmse_rad"], m["val_rmse_per_joint"] = _dense_rmse(Yp[va], ex.Y[va], norm, K)
        m["val_baseline_label_mean_rmse_rad"], m["val_baseline_per_joint"] = _dense_rmse(_baseline(ex, tr, va), ex.Y[va], norm, K)
        if predict_duration:
            m["val_duration_mae_s"] = float(np.abs(np.exp(out[va, -1]) - np.exp(ex.log_dur[va])).mean())
    else:
        m["warning"] = "Moins de deux groupes de prises : pas de validation possible, résultats non mesurés."
    if predict_duration:
        m["train_duration_mae_s"] = float(np.abs(np.exp(out[tr, -1]) - np.exp(ex.log_dur[tr])).mean())
    return m


COMPONENTS = ("arms", "head", "hands")
_NO_TAKES = {"arms": "Aucune prise de bras utilisable (statut 'accepted' requis par défaut)",
             "head": "Aucune prise de tête utilisable (statut 'accepted' requis par défaut)",
             "hands": "Aucune prise de mains utilisable (statut 'accepted' requis par défaut)"}


# réglages modifiables pour un entraînement (interface / CLI), sans toucher à model.yaml
OVERRIDABLE = ("hidden", "dropout", "weight_decay", "lr", "smoothness_weight", "epochs", "mirror", "folds",
               "n_control_points", "motion_energy_weight")
MAX_FOLDS = 20

_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")


def check_overrides(o: Optional[dict]) -> dict:
    """Valide les réglages d'un entraînement ; lève TrainingError si une valeur est invalide."""
    out = {}
    for k, v in (o or {}).items():
        if v is None or v == "":
            continue
        if k not in OVERRIDABLE:
            raise TrainingError(f"Réglage inconnu : {k}")
        try:
            if k == "mirror":
                if isinstance(v, str) and v.strip().lower() in _TRUE + _FALSE:
                    v = v.strip().lower() in _TRUE
                ok = isinstance(v, bool)
            elif k == "hidden":
                v = [int(x) for x in (v.replace(",", " ").split() if isinstance(v, str) else v)]
                ok = 1 <= len(v) <= 4 and all(1 <= x <= 1024 for x in v)
            elif k == "epochs":
                v = int(v)
                ok = 1 <= v <= 100000
            elif k == "n_control_points":
                from .data.fidelity import check_k
                v = check_k(v)
                ok = True
            elif k == "folds":
                f = float(v)
                v = int(f)
                ok = f == v and 0 <= v <= MAX_FOLDS
            else:
                v = float(v)
                ok = np.isfinite(v) and v >= 0 and (k != "dropout" or v < 1) and (k != "lr" or v > 0)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise TrainingError(f"Valeur invalide pour {k} : {o[k]!r}")
        out[k] = v
    return out


def train_component(component: str, store: TakeStore, cfg: Config, runs_dir, statuses=("accepted",),
                    epochs: Optional[int] = None, log=None, overrides: Optional[dict] = None) -> Path:
    """Entraîne UN groupe (bras, tête ou mains), indépendamment des deux autres.

    Entrée : [émotion, intensité, événement] ; sortie : points de contrôle du groupe
    et log de sa durée. Aucun autre modèle n'est requis ni chargé.
    `overrides` : réglages propres à cet entraînement (taille du réseau, dropout,
    weight_decay, lr, lissage, époques, miroir, validation croisée) ; model.yaml n'est pas modifié.

    Mirror (training.mirror, default on): left/right mirrors of the TRAINING takes without
    an event are added in memory (data/mirror.py). Validation, the baseline and the
    reported train/val RMSE use the real takes only, so the metrics stay comparable.

    Cross-validation (training.folds = k >= 2): the takes are dealt into k parts by group;
    k models are trained, each validated on the part it never saw, so every take is validated
    exactly once. The reported validation is pooled over the k parts (plus mean and spread
    per part, in metrics["cv"]). The KEPT model is then trained on ALL the takes for the mean
    best epoch of the k parts (no take is held out any more). folds 0 or 1 = the former single
    split (val_fraction)."""
    ov = check_overrides(overrides)
    if component not in COMPONENTS:
        raise ValueError(component)
    # control points per joint: this training's setting, else model.yaml; stored in the model
    K = int(ov.get("n_control_points") or cfg.model["representation"]["n_control_points"])
    joints = group_joints(cfg, component)
    if not joints:
        raise TrainingError("Aucune main configurée (joints.yaml, section hands)")
    J = len(joints)
    mc = cfg.model.get(f"{component}_model") or cfg.model["head_model"]
    if component == "arms":
        mc = cfg.model["arm_model"]
    mc = dict(mc, **{k: ov[k] for k in ("hidden", "dropout") if k in ov})
    target = mc.get("target", "measured") if component == "hands" else "measured"
    ex = group_examples(store, cfg, component, statuses, target, n_ctrl=K)
    if len(ex) == 0:
        raise TrainingError(_NO_TAKES[component])
    tcfg = dict(cfg.model["training"])
    if epochs:
        tcfg["epochs"] = epochs
    tcfg.update({k: ov[k] for k in ("weight_decay", "lr", "smoothness_weight", "epochs", "motion_energy_weight")
                 if k in ov})
    use_mirror = bool(ov.get("mirror", tcfg.get("mirror", True)))
    folds_req = int(ov.get("folds", tcfg.get("folds", 1)) or 0)
    splits = grouped_kfold(ex.groups, folds_req, int(tcfg["seed"])) if folds_req >= 2 else []
    in_dim = motion_input_dim(len(cfg.emotions))
    assert ex.X.shape[1] == in_dim
    arch = {"in_dim": in_dim, "out_dim": K * J + 1, "hidden": list(mc["hidden"]),
            "dropout": float(mc.get("dropout", 0)), "latent_dim": 0}
    norm = Norm.from_config(cfg, joints)

    def fit(tr, va, tc, flog, keep_last=False):
        fit_ex, fit_tr, rep = ex, tr, {"enabled": use_mirror}
        if use_mirror:
            # mirrors of TRAINING takes only: a validation take never leaks into training
            mex, r = mirrored_examples(store, cfg, component, [ex.take_ids[i] for i in tr], target, n_ctrl=K)
            rep.update(r)
            fit_ex = concat_examples(ex, mex)
            fit_tr = np.concatenate([tr, np.arange(len(ex), len(ex) + len(mex))]).astype(int)
        model, hist = _fit(fit_ex, fit_tr, va, arch, K, J, tc, True, flog, keep_last)
        return model, hist, rep, list(fit_ex.take_ids[len(ex):])

    cv = None
    if splits:
        model, hist, metrics, mirror_rep, mirrored, tr, va, cv = _cross_validate(
            ex, splits, fit, norm, K, tcfg, log, folds_req)
    else:
        tr, va, _ = grouped_split(ex.groups, float(tcfg["val_fraction"]), int(tcfg["seed"]))
        model, hist, mirror_rep, mirrored = fit(tr, va, tcfg, log)
        metrics = _evaluate(model, ex, tr, va, norm, K, True)
        if len(va) and hist:
            metrics["best_epoch"] = min(hist, key=lambda h: h["val_loss"])["epoch"]
    # réglages effectivement utilisés, et époque du modèle retenu
    metrics["settings"] = {"hidden": arch["hidden"], "dropout": arch["dropout"],
                           "weight_decay": float(tcfg["weight_decay"]), "lr": float(tcfg["lr"]),
                           "smoothness_weight": float(tcfg.get("smoothness_weight", 0.0)),
                           "motion_energy_weight": float(tcfg.get("motion_energy_weight", 0.0)),
                           "epochs": int(tcfg["epochs"]), "mirror": use_mirror,
                           "folds": len(splits) if splits else 1, "n_control_points": K,
                           "overrides": sorted(ov)}
    metrics["mirror"] = mirror_rep
    # detail lost before any learning: takes vs their K-point curves (the model is compared
    # with those curves, so val/train RMSE never show this part)
    from .data.fidelity import reconstruction_rmse
    try:
        metrics["reconstruction_rmse_rad"] = reconstruction_rmse(store, cfg, component, K, ex.take_ids)
    except Exception as e:  # report only, never blocks a training
        metrics["reconstruction_error"] = str(e)
    curves = {}
    if cv:
        # loss curves of the folds and of the final model: in metrics.json, outside the metrics
        curves = {"fold_histories": cv.pop("fold_histories"), "final_history": cv.pop("final_history")}
        metrics["cv"] = cv
    run = _run_dir(runs_dir, component)
    meta = _meta(cfg, component, joints, norm, K, ex, tr, va, tcfg)
    meta["mirrored_take_ids"] = mirrored
    if cv:
        meta["split"] = f"validation croisée par groupe, {len(splits)} parts ; modèle final sur toutes les prises"
        meta["cv_val_take_ids"] = [f["val_take_ids"] for f in cv["per_fold"]]
    meta["synthetic_data"] = any(store.load_meta(t).synthetic for t in ex.take_ids)
    meta.update({"event_features": EVENT_FEATURES, "independent": True,
                 "inputs": ["emotion", "intensity", "event[present,sin_az,cos_az,elev,dist_known,proximity]"],
                 "outputs": [f"{component}_ctrl_norm", "log_duration"]})
    if component == "hands":
        meta.update({"unit": "opening[0=closed,1=open]", "target": target})
    meta["metrics"] = metrics
    save_checkpoint(run / "model.pt", model, component=component, arch=arch, meta=meta)
    loaded = LoadedModel(run / "model.pt", cfg, component)
    loaded.export_numpy(run / "model_numpy.npz")
    # guardian = playback safety, reported for information only: it never filters the takes,
    # never enters the loss and never picks the epoch
    metrics["guardian"] = _guardian_sweep(cfg, **{_ENGINE_ARG[component]: loaded})
    metrics["guardian_takes"] = _guardian_on_takes(cfg, store, component, sorted(set(ex.take_ids)))
    _write_report(run, metrics, hist, meta, curves)
    (Path(runs_dir) / f"latest_{component}.txt").write_text(run.name)
    return run


def _mean_history(hists):
    """Mean loss curve of the folds (same epochs in every fold), with the spread of the val loss."""
    out = []
    for rows in zip(*hists):
        tl = [r["train_loss"] for r in rows]
        vl = [r["val_loss"] for r in rows if np.isfinite(r["val_loss"])]
        out.append({"epoch": rows[0]["epoch"], "train_loss": float(np.mean(tl)),
                    "val_loss": float(np.mean(vl)) if vl else float("nan"),
                    "val_loss_min": float(np.min(vl)) if vl else float("nan"),
                    "val_loss_max": float(np.max(vl)) if vl else float("nan")})
    return out


def _cross_validate(ex, splits, fit, norm, K, tcfg, log, folds_req):
    """k-fold cross-validation, then the kept model trained on every take (see train_component)."""
    k = len(splits)
    KJ = ex.Y.shape[1]
    oof = np.zeros_like(ex.Y)
    oof_base = np.zeros_like(ex.Y)
    oof_dur = np.zeros(len(ex))
    rows, hists, train_pj = [], [], []

    def stage_log(label):
        if not log:
            return None
        return lambda ep, l, lv: log(ep, l, lv, stage=label)

    for f, (tr, va) in enumerate(splits):
        model, hist, rep, _ = fit(tr, va, tcfg, stage_log(f"part {f + 1}/{k}"))
        with torch.no_grad():
            out = model(torch.tensor(ex.X, dtype=torch.float32)).numpy().astype(np.float64)
        oof[va] = out[va, :KJ]
        oof_dur[va] = out[va, -1]
        base = _baseline(ex, tr, va)
        oof_base[va] = base
        v, _ = _dense_rmse(out[va, :KJ], ex.Y[va], norm, K)
        b, _ = _dense_rmse(base, ex.Y[va], norm, K)
        t, tpj = _dense_rmse(out[tr, :KJ], ex.Y[tr], norm, K)
        train_pj.append(tpj)
        rows.append({"fold": f + 1, "n_train": int(len(tr)), "n_val": int(len(va)),
                     "n_mirrored": int(rep.get("added", 0) or 0),
                     "val_rmse_rad": v, "val_baseline_rmse_rad": b, "train_rmse_rad": t,
                     "ratio": v / b if b else None,
                     "best_epoch": min(hist, key=lambda h: h["val_loss"])["epoch"],
                     "val_take_ids": [ex.take_ids[i] for i in va]})
        hists.append(hist)
    best = [r["best_epoch"] for r in rows]
    final_epochs = int(round(float(np.mean(best)))) + 1  # best_epoch e = state after e+1 steps
    # kept model: every take, no validation, stopped at the mean best epoch of the folds
    model, final_hist, mirror_rep, mirrored = fit(np.arange(len(ex)), np.array([], dtype=int),
                                                  dict(tcfg, epochs=final_epochs), stage_log("modèle final"),
                                                  keep_last=True)
    alli = np.arange(len(ex))
    m = {"n_examples": len(ex), "n_train": len(ex), "n_val": len(ex),
         "n_train_groups": len(set(ex.groups)), "n_val_groups": len(set(ex.groups))}
    # pooled out-of-fold validation: each take predicted by the model that never saw it
    m["val_rmse_rad"], m["val_rmse_per_joint"] = _dense_rmse(oof, ex.Y, norm, K)
    m["val_baseline_label_mean_rmse_rad"], m["val_baseline_per_joint"] = _dense_rmse(oof_base, ex.Y, norm, K)
    m["val_motion_kept"] = _motion_kept(oof, ex.Y, norm, K)
    m["val_baseline_motion_kept"] = _motion_kept(oof_base, ex.Y, norm, K)
    m["val_duration_mae_s"] = float(np.abs(np.exp(oof_dur) - np.exp(ex.log_dur)).mean())
    # train error: mean over the folds (comparable with the validation above)
    m["train_rmse_rad"] = float(np.mean([r["train_rmse_rad"] for r in rows]))
    m["train_rmse_per_joint"] = {j: round(float(np.mean([p[j] for p in train_pj])), 4) for j in norm.joints}
    with torch.no_grad():
        out = model(torch.tensor(ex.X, dtype=torch.float32)).numpy().astype(np.float64)
    m["final_train_rmse_rad"], _ = _dense_rmse(out[:, :KJ], ex.Y, norm, K)
    m["train_motion_kept"] = _motion_kept(out[:, :KJ], ex.Y, norm, K)
    m["train_duration_mae_s"] = float(np.abs(np.exp(out[:, -1]) - np.exp(ex.log_dur)).mean())
    m["best_epoch"] = final_epochs - 1
    vals = [r["val_rmse_rad"] for r in rows]
    bases = [r["val_baseline_rmse_rad"] for r in rows]
    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    cv = {"folds": k, "requested": int(folds_req), "seed": int(tcfg["seed"]),
          "val_rmse_mean": float(np.mean(vals)), "val_rmse_std": float(np.std(vals)),
          "baseline_mean": float(np.mean(bases)), "baseline_std": float(np.std(bases)),
          "ratio_mean": float(np.mean(ratios)) if ratios else None,
          "ratio_std": float(np.std(ratios)) if ratios else None,
          "best_epoch_mean": float(np.mean(best)), "best_epoch_std": float(np.std(best)),
          "final_epochs": final_epochs,
          # no validation for the final model: train loss only (NaN is not valid JSON for the browser)
          "final_history": [{"epoch": h["epoch"], "train_loss": h["train_loss"]} for h in final_hist], "fold_histories": hists,
          "kept": "final model trained on all the takes for the mean best epoch of the folds",
          "per_fold": rows}
    if k < folds_req:
        cv["warning"] = f"{folds_req} parts demandées, {k} possibles (une part par groupe de prises au plus)"
    return model, _mean_history(hists), m, mirror_rep, mirrored, alli, np.array([], dtype=int), cv


_ENGINE_ARG = {"arms": "arm_model", "head": "head_model", "hands": "hands_model"}


def train_arms(store: TakeStore, cfg: Config, runs_dir, statuses=("accepted",), epochs: Optional[int] = None,
               log=None) -> Path:
    return train_component("arms", store, cfg, runs_dir, statuses, epochs, log)


def train_head(store: TakeStore, cfg: Config, runs_dir, statuses=("accepted",), epochs: Optional[int] = None,
               log=None) -> Path:
    return train_component("head", store, cfg, runs_dir, statuses, epochs, log)


def train_hands(store: TakeStore, cfg: Config, runs_dir, statuses=("accepted",), epochs: Optional[int] = None,
                log=None) -> Path:
    return train_component("hands", store, cfg, runs_dir, statuses, epochs, log)


def _meta(cfg, component, joints, norm, K, ex, tr, va, tcfg):
    return {"component": component, "joints": list(joints), "unit": "rad", "time_unit": "s",
            "emotions": cfg.emotions, "representation": {"type": "bspline_cubic_clamped_uniform",
                                                          "n_control_points": K, "time": "phase[0,1]+duration"},
            "norm": norm.to_dict(), "calibration_version": cfg.calibration_version,
            "config_fingerprint": cfg.fingerprint(),
            "train_take_ids": [ex.take_ids[i] for i in tr], "val_take_ids": [ex.take_ids[i] for i in va],
            "split": "par groupe (prise ; anciennes prises tête/mains : geste des bras rejoué)", "training": tcfg,
            "synthetic_data": None}  # renseigné par l'appelant


def _guardian_sweep(cfg, arm_model=None, head_model=None, intensities=(0.2, 0.5, 0.8, 1.0), hands_model=None):
    """Information only (playback safety, not a training criterion): share of generated gestures
    the guardian accepts on a grid, as played (made playable first: slowed down, brought back
    inside the limits) and raw. Groups without a model stay at the start pose."""
    from .engine import MotionEngine
    from .guardian.playable import make_playable
    eng = MotionEngine(cfg, arm_model, head_model, hands_model=hands_model)
    g = Guardian(cfg)
    raw_ok = 0
    raw_failures = {}
    scales = []
    events = [EventContext.none(), EventContext(True, 0.6, 0.1, source="test"),
              EventContext(True, -0.4, 0.0, source="test", distance_m=0.6)]
    total = ok = 0
    failures = {}
    for e in cfg.emotions:
        for i in intensities:
            for ev in events:
                gen = eng.generate(e, i, ev)
                raw = g.validate(gen.combined)
                raw_ok += raw.ok
                for v in raw.violations:
                    raw_failures[v.code] = raw_failures.get(v.code, 0) + 1
                a = make_playable(cfg, gen.combined)
                scales.append(a.time_scale)
                rep = g.validate(a.traj)
                total += 1
                ok += rep.ok
                for v in rep.violations:
                    failures[v.code] = failures.get(v.code, 0) + 1
    return {"accepted": ok, "total": total, "rate": ok / total if total else None, "violations_by_code": failures,
            "raw_accepted": raw_ok, "raw_violations_by_code": raw_failures,
            "median_slow_down": round(float(np.median(scales)), 2) if scales else None}


def _guardian_on_takes(cfg, store, component, take_ids):
    """Diagnostic: share of the RECORDED takes (gestures done by hand on the robot) that the guardian
    accepts, raw and made playable. A low raw rate means the guardian is stricter than the
    gestures themselves (calibrated zone too narrow, speed limits below the real gestures): any
    model that imitates them well will be refused too. Information only."""
    from .control.recording import replay_trajectory
    from .guardian.playable import make_playable
    joints = group_joints(cfg, component)
    others = [j for j in cfg.all_joints if j not in joints]
    ctx = dict(zip(others, cfg.pose("start", others))) if others else {}
    g = Guardian(cfg)
    n = raw_ok = ok = 0
    raw_failures, joints_out, excess = {}, {}, []
    for tid in take_ids:
        try:
            traj = replay_trajectory(store.load(tid), joints)
        except Exception:
            continue
        n += 1
        raw = g.validate(traj, context_pose=ctx, check_duration=False)
        raw_ok += raw.ok
        # how far the take goes beyond the zone (limits / coupled envelopes, margins included):
        # takes that defined the zone (calibration extended with them) touch its edge by a few
        # hundredths of a radian, which is expected
        ex = [abs(v.value - v.limit) for v in raw.violations
              if v.code in ("limit", "coupled") and v.value is not None and v.limit is not None]
        excess.append(max(ex) if ex else 0.0)
        for v in raw.violations:
            raw_failures[v.code] = raw_failures.get(v.code, 0) + 1
            if v.code == "limit" and v.joint:
                joints_out[v.joint] = joints_out.get(v.joint, 0) + 1
        ok += g.validate(make_playable(cfg, traj, ctx).traj, context_pose=ctx).ok
    return {"total": n, "raw_accepted": raw_ok, "accepted_playable": ok, "raw_violations_by_code": raw_failures,
            "takes_outside_limits_by_joint": joints_out, "calibration": cfg.calibration_version,
            "median_excess_rad": round(float(np.median(excess)), 3) if excess else None,
            "max_excess_rad": round(float(np.max(excess)), 3) if excess else None}


def _write_report(run, metrics, hist, meta, curves=None):
    (run / "metrics.json").write_text(json.dumps({"metrics": metrics, "history": hist, **(curves or {})},
                                                 indent=2, ensure_ascii=False))
    split = {"train": meta["train_take_ids"], "val": meta["val_take_ids"],
             "train_mirrored": meta.get("mirrored_take_ids", [])}
    if meta.get("cv_val_take_ids"):
        split["cv_val"] = meta["cv_val_take_ids"]  # validation takes of each fold
    (run / "split.json").write_text(json.dumps(split, indent=2))


def latest(runs_dir, component) -> Optional[Path]:
    p = Path(runs_dir) / f"latest_{component}.txt"
    if p.exists():
        name = Path(p.read_text().strip())
        # nom du run relatif au dossier runs/ (indépendant du dossier courant)
        run_name = name.parent.name if name.name == "model.pt" else name.name
        for q in (Path(runs_dir) / run_name / "model.pt", name):
            if q.exists():
                return q
    return None
