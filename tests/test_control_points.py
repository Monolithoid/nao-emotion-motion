"""Number of B-spline control points: fidelity on the takes, per-training setting, per-model K."""
import numpy as np
import pytest

from nao_motion.data.fidelity import fidelity
from nao_motion.data.store import TakeMeta, new_id
from nao_motion.engine import MotionEngine, model_k
from nao_motion.events import EventContext
from nao_motion.models.datasets import group_examples
from nao_motion.models.nets import load_model
from nao_motion.trajectory import Trajectory
from nao_motion.training import TrainingError, check_overrides

from .conftest import make_wb


def _save_arm_take(wb, cfg, q_fn, emotion="joie", dur=3.0):
    j = cfg.arm_joints + cfg.head_joints
    t = np.linspace(0, dur, int(dur * 50) + 1)
    q = np.tile(cfg.pose("start", j), (len(t), 1))
    q[:, 0] += q_fn(t)
    meta = TakeMeta(take_id=new_id("arms"), session_id="s", kind="arms", emotion=emotion, intensity=0.5, joints=j,
                    guided_joints=cfg.arm_joints, driven_joints=cfg.head_joints, event=EventContext.none().to_dict(),
                    calibration_version=cfg.calibration_version, config_fingerprint="x", adapter="test",
                    sample_rate_hz=50, segment=(0.0, dur), status="accepted")
    wb.store.save(meta, Trajectory(j, t, q))
    return meta.take_id


def test_fidelity_shows_what_few_points_flatten(tmp_path, cfg):
    wb = make_wb(tmp_path, cfg)
    env = lambda t: np.sin(np.pi * t / 3.0) ** 2
    _save_arm_take(wb, cfg, lambda t: -0.4 * env(t) * np.sin(2 * np.pi * 2.0 * t))  # waving, 2 Hz
    _save_arm_take(wb, cfg, lambda t: -0.3 * env(t))                                   # slow raise
    r = fidelity(wb.store, cfg, "arms", [8, 16, 32])
    assert r["n_takes"] == 2 and r["real_takes"]
    rows = {row["n_control_points"]: row for row in r["rows"]}
    assert rows[8]["rmse_rad"] > 3 * rows[16]["rmse_rad"] > 0
    assert rows[32]["rmse_rad"] < rows[16]["rmse_rad"]
    assert rows[8]["amplitude_kept_min"] < 0.3 < 0.9 < rows[16]["amplitude_kept_min"]
    assert rows[8]["worst_take_rmse_rad"] >= rows[8]["rmse_rad"]
    with pytest.raises(ValueError):
        fidelity(wb.store, cfg, "arms", [3])


def test_control_points_override_is_checked():
    assert check_overrides({"n_control_points": "16"}) == {"n_control_points": 16}
    for bad in ("3", "65", "12.5", "x"):
        with pytest.raises(TrainingError):
            check_overrides({"n_control_points": bad})


def test_training_with_its_own_control_points(tmp_path, cfg):
    wb = make_wb(tmp_path, cfg)
    for k, emo in enumerate(cfg.emotions[:3] * 2):
        _save_arm_take(wb, cfg, lambda t, a=0.1 + 0.05 * k: -a * np.sin(np.pi * t / 3.0) ** 2, emo)
    default_k = int(cfg.model["representation"]["n_control_points"])
    k = 6 if default_k != 6 else 7
    assert group_examples(wb.store, cfg, "arms", n_ctrl=k).Y.shape[1] == k * len(cfg.arm_joints)
    job = wb.train("arms", epochs=20, background=False, overrides={"n_control_points": k, "folds": 0})
    assert job["state"] == "done", job["error"]
    from nao_motion import training
    run = training.latest(wb.runs_dir, "arms")
    m = load_model(run, cfg, "arms")
    assert model_k(m, default_k) == k and m.arch["out_dim"] == k * len(cfg.arm_joints) + 1
    metrics = m.meta["metrics"]
    assert metrics["settings"]["n_control_points"] == k
    assert metrics["reconstruction_rmse_rad"] is not None
    # the model keeps its own K even though model.yaml says otherwise
    gen = MotionEngine(cfg, arm_model=m).generate(cfg.emotions[0], 0.5)
    assert gen.arm_ctrl.shape == (k, len(cfg.arm_joints))
    assert gen.head_ctrl.shape[0] == default_k  # group without a model: start pose held
