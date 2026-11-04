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
from nao_motion.models.features import arm_input
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


def test_head_checkpoint_references_frozen_arm_model(trained, cfg):
    arm_p = Path(trained["arm_run"]) / "model.pt"
    head = LoadedModel(Path(trained["head_run"]) / "model.pt", cfg, "head")
    arm = LoadedModel(arm_p, cfg, "arms")
    assert head.meta["arm_checkpoint_digest"] == arm.digest  # modèle des bras inchangé par l'entraînement tête


def test_no_leakage_between_train_and_val(trained, cfg):
    wb = trained["wb"]
    ex = head_examples(wb.store, cfg)
    tr, va, _ = grouped_split(ex.groups, 0.3, 0)
    assert len(va) > 0
    assert not ({ex.groups[i] for i in tr} & {ex.groups[i] for i in va})
    head = LoadedModel(Path(trained["head_run"]) / "model.pt", cfg, "head")
    train_ids, val_ids = set(head.meta["train_take_ids"]), set(head.meta["val_take_ids"])
    arm_of = {m.take_id: m.arm_take_id for m in wb.store.list(kind="head")}
    assert not ({arm_of[t] for t in train_ids} & {arm_of[t] for t in val_ids})


def test_engine_generates_valid_shapes_and_interpolates(trained, cfg):
    eng = MotionEngine.from_paths(cfg, Path(trained["arm_run"]) / "model.pt", Path(trained["head_run"]) / "model.pt")
    g = eng.generate("joie", 0.7, EventContext(True, 0.5, 0.0))
    assert g.combined.joints == cfg.arm_joints + cfg.head_joints
    assert abs(g.arms.duration - g.head.duration) < 1e-9  # base temporelle commune
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


def test_head_collection_on_generated_arm_gesture(trained, cfg):
    """Le modèle des bras joue l'émotion ; on enregistre (émotion/intensité, bras réalisés, tête)."""
    from nao_motion.data.store import TakeError
    wb = trained["wb"]
    arm_path = Path(trained["arm_run"]) / "model.pt"
    wb.engine = MotionEngine.from_paths(cfg, arm_path)  # modèle des bras de référence (figé)
    wb.ctrl.acknowledge() if wb.ctrl.state.value != "idle" else None
    wb.ctrl.hold(cfg.all_joints)
    wb.ctrl.arm(confirm=True)
    # le modèle de test est peu entraîné : on retient un geste que le guardian accepte
    # (un geste refusé n'est jamais exécuté, ce qui est testé ailleurs)
    from nao_motion.data.representation import decode
    gesture = None
    for emo in cfg.emotions:
        for inten in (0.3, 0.6, 1.0):
            n, d = wb.engine.plan_arms(emo, inten)
            tr = decode(wb.engine.arm_norm.inv(n), d, cfg.arm_joints, 50)
            if wb.guardian.validate(tr, context_pose=dict(zip(cfg.head_joints, cfg.pose("start", cfg.head_joints)))).ok:
                gesture = {"emotion": emo, "intensity": inten, "seed": 3}
                break
        if gesture:
            break
    assert gesture is not None
    ids = []
    for ev in (EventContext.none(), EventContext(True, -0.5, 0.0, source="test")):
        wb.event = ev
        info = wb.prepare_head_take(generated=gesture)
        assert info["arm_source"] == "generated"
        wb.start_recording(countdown_s=0.2, background=False)
        ids.append(wb.save_recording(status="accepted"))
        wb.ctrl.hold(cfg.head_joints)
    wb.ctrl.disarm()
    m1, m2 = (wb.store.load_meta(i) for i in ids)
    arm_digest = LoadedModel(arm_path, cfg, "arms").digest
    assert m1.arm_generation["arm_checkpoint_digest"] == arm_digest
    assert m1.arm_take_id is None and m1.emotion == gesture["emotion"]
    assert wb.store.group_id(m1) == wb.store.group_id(m2)  # même geste => même groupe
    ref = wb.store.load_reference(ids[0])
    take = wb.store.load(ids[0])
    tt = np.linspace(take.meta.segment[0], take.meta.segment[1], 40)
    meas = take.traj.select(cfg.arm_joints).sample(tt)
    assert np.abs(meas - ref.sample(tt)).max() < 0.05  # bras réalisés ≈ consigne générée
    ex = head_examples(wb.store, cfg)
    assert ids[0] in ex.take_ids
    m1.arm_generation = {"emotion": gesture["emotion"]}
    m1.take_id = m1.take_id + "-copie"
    with pytest.raises(TakeError):
        wb.store.save(m1, take.traj)


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
    assert not np.allclose(near.combined.q, far.combined.q)
