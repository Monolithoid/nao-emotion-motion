import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from nao_motion.config import load_config
from nao_motion.engine import MotionEngine
from nao_motion.events import EventContext
from nao_motion.models.datasets import grouped_split, head_examples
from nao_motion.models.features import arm_input, motion_input_dim
from nao_motion.training import latest as training_latest
from nao_motion.models.nets import CheckpointError, LoadedModel, NumpyModel, load_model


def test_checkpoint_reload_identical_predictions(trained, cfg):
    p = Path(trained["arm_run"]) / "model.pt"
    m1, m2 = LoadedModel(p, cfg, "arms"), LoadedModel(p, cfg, "arms")
    x = arm_input(cfg.emotions, "joie", 0.6)
    np.testing.assert_array_equal(m1.predict(x), m2.predict(x))
    meta = m1.meta
    for k in ("joints", "emotions", "norm", "representation", "calibration_version", "train_take_ids",
              "val_take_ids", "config_fingerprint", "unit", "metrics"):
        assert k in meta
    assert meta["synthetic_data"] is True


def test_numpy_export_matches_torch(trained, cfg):
    run = Path(trained["head_run"])
    mt = LoadedModel(run / "model.pt", cfg, "head")
    mn = NumpyModel(run / "model_numpy.npz", cfg, "head")
    x = np.random.default_rng(0).normal(size=(5, mt.arch["in_dim"])).astype(np.float32)
    np.testing.assert_allclose(mt.predict(x), mn.predict(x), atol=1e-5)


def test_checkpoint_refuses_incompatible_config(trained, cfg, tmp_path):
    d = tmp_path / "c"
    shutil.copytree(cfg.root, d)
    e = yaml.safe_load((d / "emotions.yaml").read_text())
    e["emotions"].append({"id": "fierte", "label": "fierté"})
    (d / "emotions.yaml").write_text(yaml.safe_dump(e, allow_unicode=True))
    other = load_config(d)
    with pytest.raises(CheckpointError):
        load_model(Path(trained["arm_run"]) / "model.pt", other, "arms")
    with pytest.raises(CheckpointError):
        load_model(Path(trained["arm_run"]) / "model.pt", cfg, "head")


def test_head_model_is_independent_of_arms(trained, cfg, tmp_path):
    """Le modèle de la tête n'a que [émotion, intensité, événement] en entrée et prédit sa durée."""
    head = LoadedModel(Path(trained["head_run"]) / "model.pt", cfg, "head")
    assert head.meta["independent"] is True
    assert "arm_checkpoint" not in head.meta and "arm_joints" not in head.meta
    assert head.arch["in_dim"] == motion_input_dim(len(cfg.emotions))
    assert head.arch["out_dim"] == 8 * len(cfg.head_joints) + 1
    # un ancien modèle tête (dépendant des bras) est refusé avec un message clair
    ck = torch.load(Path(trained["head_run"]) / "model.pt", map_location="cpu", weights_only=True)
    del ck["meta"]["independent"]
    ck["meta"]["arm_joints"] = cfg.arm_joints
    old = tmp_path / "old_head.pt"
    torch.save(ck, old)
    with pytest.raises(CheckpointError, match="ancien format"):
        LoadedModel(old, cfg, "head")


def test_head_trains_without_any_arm_model(trained, cfg, tmp_path):
    """Tête entraînée seule : aucune prise ni aucun modèle des bras dans le dossier."""
    import shutil
    from .conftest import make_wb
    src = trained["wb"]
    wb = make_wb(tmp_path, cfg)
    for m in src.store.list(kind="head"):
        shutil.copytree(src.store.dir / m.take_id, wb.store.dir / m.take_id)
    assert wb.store.list(kind="arms") == [] and training_latest(wb.runs_dir, "arms") is None
    job = wb.train("head", epochs=30, background=False)
    assert job["state"] == "done", job["error"]
    eng = wb.get_engine()
    assert eng.arm_model is None and eng.head_model is not None
    g = eng.generate("joie", 0.7, EventContext(True, 0.5, 0.0))
    # bras maintenus en position de départ, tête générée
    np.testing.assert_allclose(g.arms.q, np.tile(cfg.pose("start", cfg.arm_joints), (len(g.arms.t), 1)))
    assert any("bras maintenus" in n for n in g.notes)
    assert g.head.q.std(axis=0).max() > 0


def test_no_leakage_between_train_and_val(trained, cfg):
    wb = trained["wb"]
    ex = head_examples(wb.store, cfg)
    tr, va, _ = grouped_split(ex.groups, 0.3, 0)
    assert len(va) > 0
    assert not ({ex.groups[i] for i in tr} & {ex.groups[i] for i in va})
    head = LoadedModel(Path(trained["head_run"]) / "model.pt", cfg, "head")
    assert not set(head.meta["train_take_ids"]) & set(head.meta["val_take_ids"])


def test_engine_generates_valid_shapes_and_interpolates(trained, cfg):
    eng = MotionEngine.from_paths(cfg, Path(trained["arm_run"]) / "model.pt", Path(trained["head_run"]) / "model.pt")
    g = eng.generate("joie", 0.7, EventContext(True, 0.5, 0.0))
    assert g.combined.joints == cfg.arm_joints + cfg.head_joints
    assert np.array_equal(g.arms.t, g.head.t)  # base temporelle commune pour la fusion
    # chaque groupe garde sa durée apprise ; le geste dure autant que le plus long
    assert set(g.durations) == {"arms", "head"}
    assert g.duration == pytest.approx(max(g.durations.values()), abs=1e-3)
    short = min(g.durations, key=g.durations.get)
    tr = getattr(g, short)
    after = tr.t >= g.durations[short] + 0.01
    if after.any():  # le groupe le plus court tient sa pose finale
        np.testing.assert_allclose(tr.q[after], np.tile(tr.q[after][0], (after.sum(), 1)), atol=1e-9)
    a = eng.generate("joie", 0.7).arm_ctrl
    b = eng.generate("tristesse", 0.7).arm_ctrl
    mid = eng.generate({"joie": 0.5, "tristesse": 0.5}, 0.7).arm_ctrl
    assert not np.allclose(mid, a) and not np.allclose(mid, b)
    v1 = eng.generate("joie", 0.7, variation=0.05, seed=1).arm_ctrl
    v2 = eng.generate("joie", 0.7, variation=0.05, seed=1).arm_ctrl
    v3 = eng.generate("joie", 0.7, variation=0.05, seed=2).arm_ctrl
    np.testing.assert_array_equal(v1, v2)
    assert not np.allclose(v1, v3)
    with pytest.raises(ValueError):
        eng.generate("joie", 1.5)
    with pytest.raises(ValueError):
        eng.generate("inconnue", 0.5)
    # l'absence d'événement et un événement devant donnent des entrées différentes
    h_none = eng.generate("joie", 0.7, EventContext.none()).head_ctrl
    h_front = eng.generate("joie", 0.7, EventContext(True, 0.0, 0.0)).head_ctrl
    assert not np.allclose(h_none, h_front)


def test_engine_stretch_timing_and_no_model(trained, cfg):
    eng = MotionEngine.from_paths(cfg, Path(trained["arm_run"]) / "model.pt", Path(trained["head_run"]) / "model.pt")
    eng.timing = "stretch"
    g = eng.generate("joie", 0.7)
    # étirés : aucun groupe ne tient sa pose avant la fin
    for tr in (g.arms, g.head):
        assert np.abs(np.diff(tr.q[-6:-1], axis=0)).max() > 0 or np.ptp(tr.q, axis=0).max() < 1e-6
    with pytest.raises(CheckpointError):
        MotionEngine(cfg).generate("joie", 0.5)


def test_training_triggers_no_motion(trained):
    wb = trained["wb"]
    n = len(wb.adapter.command_log)
    q = dict(wb.adapter.q)
    job = wb.train("arms", epochs=5, background=False)
    assert job["state"] == "done"
    assert len(wb.adapter.command_log) == n and wb.adapter.q == q


def test_metrics_report(trained):
    m = json.loads((Path(trained["arm_run"]) / "metrics.json").read_text())["metrics"]
    for k in ("train_rmse_rad", "val_rmse_rad", "val_baseline_label_mean_rmse_rad", "guardian", "n_val_groups"):
        assert k in m


def test_head_collection_without_arms(trained, cfg, tmp_path):
    """Passage tête : seule la tête est relâchée, les bras restent raides sur place,
    aucune commande n'est envoyée ; ni prise de bras ni modèle requis."""
    from .conftest import make_wb
    wb = make_wb(tmp_path, cfg)
    wb.ctrl.hold(cfg.all_joints)
    q_arms = wb.adapter.read_angles(cfg.arm_joints)
    n = len(wb.adapter.command_log)
    wb.event = EventContext(True, 0.5, 0.1, source="test", distance_m=1.0)
    info = wb.prepare_head_take("surprise", 0.8)
    assert info["kind"] == "head" and "arm_source" not in info
    assert all(wb.adapter.roles[j] == "guided" for j in cfg.head_joints)
    assert all(wb.adapter.roles[j] == "driven" for j in cfg.arm_joints)
    wb.start_recording(countdown_s=0.2, background=False)
    np.testing.assert_allclose(wb.adapter.read_angles(cfg.arm_joints), q_arms, atol=1e-6)  # bras immobiles
    assert not any(set(c[1]) & set(cfg.head_joints) for c in wb.adapter.command_log[n:])  # tête jamais commandée
    hid = wb.save_recording(status="accepted")
    m = wb.store.load_meta(hid)
    assert m.kind == "head" and m.joints == cfg.head_joints and m.arm_take_id is None and m.arm_source is None
    assert m.emotion == "surprise" and m.event["present"] and m.head_initial is not None
    assert m.segment[1] - m.segment[0] > 0.5  # segment proposé là où la tête bouge
    ex = head_examples(wb.store, cfg)
    assert ex.take_ids == [hid] and ex.X.shape[1] == motion_input_dim(len(cfg.emotions))


def test_legacy_head_takes_on_replayed_arms_still_train(trained, cfg, tmp_path):
    """Les prises de tête déjà validées, enregistrées sur des bras rejoués (ancien format),
    restent lisibles et servent à l'entraînement : seules les articulations de la tête sont lues."""
    from nao_motion.data.store import TakeMeta, new_id
    from nao_motion.trajectory import Trajectory
    from .conftest import make_wb
    from .test_data import arm_take
    wb = make_wb(tmp_path, cfg)
    am, atr = arm_take(cfg)
    am.status = "accepted"
    wb.store.save(am, atr)
    j = cfg.arm_joints + cfg.head_joints
    t = np.linspace(0, 2, 101)
    q = np.tile(cfg.pose("start", j), (101, 1))
    q[:, -2] += 0.4 * np.sin(np.pi * t / 2) ** 2  # HeadYaw
    meta = TakeMeta(take_id=new_id("head"), session_id="s", kind="head", emotion="joie", intensity=0.5, joints=j,
                    guided_joints=cfg.head_joints, driven_joints=cfg.arm_joints, event=EventContext.none().to_dict(),
                    calibration_version=cfg.calibration_version, config_fingerprint="x", adapter="test",
                    sample_rate_hz=50, segment=(0.0, 2.0), arm_take_id=am.take_id, arm_source="take",
                    arm_gesture_id=am.take_id, status="accepted")
    wb.store.save(meta, Trajectory(j, t, q))
    assert wb.store.load(meta.take_id).meta.arm_take_id == am.take_id
    ex = head_examples(wb.store, cfg)
    assert ex.take_ids == [meta.take_id]
    assert ex.Y.shape[1] == 8 * len(cfg.head_joints)
    job = wb.train("head", epochs=20, background=False)
    assert job["state"] == "done", job["error"]


def test_head_model_with_old_event_encoding_is_refused(trained, cfg, tmp_path):
    """Un modèle de tête entraîné sans la distance (4 entrées d'événement) doit être réentraîné."""
    run = Path(trained["head_run"])
    head = LoadedModel(run / "model.pt", cfg, "head")
    assert head.meta["event_features"] == 6
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    ck["meta"]["event_features"] = 4
    old = tmp_path / "old.pt"
    torch.save(ck, old)
    with pytest.raises(CheckpointError, match="distance"):
        LoadedModel(old, cfg, "head")


def test_distance_changes_generated_head(trained, cfg):
    eng = MotionEngine.from_paths(cfg, Path(trained["arm_run"]) / "model.pt", Path(trained["head_run"]) / "model.pt")
    near = eng.generate("peur", 0.9, EventContext(True, 0.5, 0.0, distance_m=0.3), None, 0.0, 0)
    far = eng.generate("peur", 0.9, EventContext(True, 0.5, 0.0, distance_m=3.0), None, 0.0, 0)
    assert not np.allclose(near.head_ctrl, far.head_ctrl)


def test_arm_model_receives_event(trained, cfg):
    """Le geste des bras dépend de l'angle et de la distance (ex. pointer vers l'événement)."""
    eng = MotionEngine.from_paths(cfg, Path(trained["arm_run"]) / "model.pt")
    assert eng.arm_model.meta["event_features"] == 6
    left, _ = eng.plan_arms("surprise", 0.9, event=EventContext(True, 0.7, 0.2, distance_m=0.5))
    right, _ = eng.plan_arms("surprise", 0.9, event=EventContext(True, -0.6, 0.0, distance_m=0.5))
    far, _ = eng.plan_arms("surprise", 0.9, event=EventContext(True, 0.7, 0.2, distance_m=3.0))
    none, _ = eng.plan_arms("surprise", 0.9)
    for other in (right, far, none):
        assert not np.allclose(left, other)


def test_arm_model_without_event_is_refused(trained, cfg, tmp_path):
    """Un modèle des bras entraîné avant l'ajout de l'événement doit être réentraîné."""
    ck = torch.load(Path(trained["arm_run"]) / "model.pt", map_location="cpu", weights_only=True)
    del ck["meta"]["event_features"]
    old = tmp_path / "old_arms.pt"
    torch.save(ck, old)
    with pytest.raises(CheckpointError, match="le réentraîner"):
        LoadedModel(old, cfg, "arms")


def test_training_overrides_change_network_without_touching_config(trained, cfg):
    """Réglages d'un entraînement (interface) : appliqués au réseau et consignés, model.yaml intact."""
    from nao_motion.training import TrainingError, check_overrides
    wb = trained["wb"]
    before = json.dumps(cfg.model, sort_keys=True)
    job = wb.train("head", epochs=40, background=False,
                   overrides={"hidden": "16", "dropout": "0.1", "weight_decay": 0.01, "smoothness_weight": ""})
    assert job["state"] == "done", job["error"]
    head = LoadedModel(Path(job["run"]) / "model.pt", cfg, "head")
    assert head.arch["hidden"] == [16] and head.arch["dropout"] == 0.1
    m = json.loads((Path(job["run"]) / "metrics.json").read_text())["metrics"]
    assert m["settings"]["weight_decay"] == 0.01 and m["settings"]["hidden"] == [16]
    assert 0 <= m["best_epoch"] < 40
    assert json.dumps(cfg.model, sort_keys=True) == before
    for bad in ({"hidden": "0"}, {"dropout": 1.5}, {"lr": 0}, {"inconnu": 1}, {"hidden": "a,b"}):
        with pytest.raises(TrainingError):
            check_overrides(bad)
