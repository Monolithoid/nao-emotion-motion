"""Cross-validation at training (training.folds): k models validated on the part they never saw,
every take validated once, kept model retrained on all the takes for the mean best epoch."""
import json
from pathlib import Path

import numpy as np
import pytest

from nao_motion.models.datasets import grouped_kfold
from nao_motion.models.nets import LoadedModel
from nao_motion.training import TrainingError, check_overrides
from nao_motion.ui.server import Api


def test_grouped_kfold_validates_every_group_once():
    groups = [f"g{i // 2}" for i in range(23)]  # 12 groups, two examples each (last one alone)
    folds = grouped_kfold(groups, 5, seed=0)
    assert len(folds) == 5
    seen = []
    for tr, va in folds:
        assert not {groups[i] for i in tr} & {groups[i] for i in va}  # a group never on both sides
        assert sorted(np.concatenate([tr, va]).tolist()) == list(range(len(groups)))
        seen += va.tolist()
    assert sorted(seen) == list(range(len(groups)))  # each example validated exactly once
    sizes = [len({groups[i] for i in va}) for _, va in folds]
    assert max(sizes) - min(sizes) <= 1
    assert grouped_kfold(groups, 5, seed=0)[0][1].tolist() == folds[0][1].tolist()  # reproducible
    assert len(grouped_kfold(["a", "b", "c"], 5, 0)) == 3  # capped at the number of groups
    assert grouped_kfold(["a", "a"], 5, 0) == [] and grouped_kfold(groups, 1, 0) == []


def test_folds_override_validation():
    assert check_overrides({"folds": "5"}) == {"folds": 5}
    assert check_overrides({"folds": 0}) == {"folds": 0}
    for bad in ({"folds": -1}, {"folds": 2.5}, {"folds": 21}, {"folds": "x"}):
        with pytest.raises(TrainingError):
            check_overrides(bad)


def test_cross_validation_training(trained, cfg):
    wb = trained["wb"]
    job = wb.train("head", epochs=60, background=False, overrides={"folds": 3})
    assert job["state"] == "done", job["error"]
    assert any(p.get("stage") == "modèle final" for p in job["progress"])
    run = Path(job["run"])
    text = (run / "metrics.json").read_text()
    assert "NaN" not in text  # read by the browser: NaN is not valid JSON
    d = json.loads(text)
    m, cv = d["metrics"], d["metrics"]["cv"]
    assert cv["folds"] == 3 and len(cv["per_fold"]) == 3 and m["settings"]["folds"] == 3
    # every take validated once, by a model that did not train on it
    val_ids = [t for f in cv["per_fold"] for t in f["val_take_ids"]]
    assert len(val_ids) == m["n_examples"] == m["n_val"] == m["n_train"]
    assert len(set(val_ids)) == len(val_ids)
    for f in cv["per_fold"]:
        assert f["n_train"] + f["n_val"] == m["n_examples"] and 0 <= f["best_epoch"] < 60
    assert np.isclose(cv["val_rmse_mean"], np.mean([f["val_rmse_rad"] for f in cv["per_fold"]]))
    assert m["val_rmse_rad"] > 0 and m["val_baseline_label_mean_rmse_rad"] > 0 and cv["val_rmse_std"] >= 0
    # kept model: all the takes, mean best epoch of the folds
    assert cv["final_epochs"] == int(round(np.mean([f["best_epoch"] for f in cv["per_fold"]]))) + 1
    assert m["best_epoch"] == cv["final_epochs"] - 1
    assert len(d["final_history"]) >= 1 and d["final_history"][-1]["epoch"] == cv["final_epochs"] - 1
    assert len(d["fold_histories"]) == 3 and "val_loss_min" in d["history"][0]
    head = LoadedModel(run / "model.pt", cfg, "head")
    assert head.meta["val_take_ids"] == [] and len(head.meta["train_take_ids"]) == m["n_examples"]
    assert json.loads((run / "split.json").read_text())["cv_val"] == [f["val_take_ids"] for f in cv["per_fold"]]
    # the curves of the folds are served to the training tab
    h = Api(wb).get_train_history({"run": run.name})
    assert len(h["fold_histories"]) == 3 and h["final_history"]


def test_single_split_still_available(trained):
    wb = trained["wb"]
    job = wb.train("head", epochs=30, background=False, overrides={"folds": 1})
    assert job["state"] == "done", job["error"]
    m = json.loads((Path(job["run"]) / "metrics.json").read_text())["metrics"]
    assert "cv" not in m and m["settings"]["folds"] == 1
    assert 0 < m["n_val"] < m["n_examples"] and m["n_train"] + m["n_val"] == m["n_examples"]


def test_folds_field_in_network_tab():
    static = Path(__file__).resolve().parents[1] / "nao_motion" / "ui" / "static"
    html, js = (static / "index.html").read_text(), (static / "mlp.js").read_text()
    assert 'id="tr-folds"' in html and 'folds: "tr-folds"' in js and "foldsChart" in js
