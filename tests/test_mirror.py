"""Left/right mirror of takes (training-time augmentation, data/mirror.py)."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from nao_motion.data.mirror import (check_mirror, factory_limits, mirror_joint, mirror_trajectory)
from nao_motion.trajectory import Trajectory


def test_nao_sign_conventions():
    assert mirror_joint("LShoulderPitch") == ("RShoulderPitch", 1.0)
    for j in ("ShoulderRoll", "ElbowYaw", "ElbowRoll", "WristYaw"):
        assert mirror_joint("L" + j) == ("R" + j, -1.0)
        assert mirror_joint("R" + j) == ("L" + j, -1.0)
    assert mirror_joint("LHand") == ("RHand", 1.0)
    assert mirror_joint("HeadYaw") == ("HeadYaw", -1.0)
    assert mirror_joint("HeadPitch") == ("HeadPitch", 1.0)


def _arm_traj(cfg, rng):
    joints = cfg.arm_joints
    start = np.array(cfg.pose("start", joints))
    t = np.linspace(0, 1.5, 40)
    q = start + 0.05 * np.sin(np.outer(t, rng.uniform(1, 3, len(joints))))
    return Trajectory(list(joints), t, q)


def test_mirror_is_an_involution_and_swaps_sides(cfg):
    tr = _arm_traj(cfg, np.random.default_rng(0))
    m = mirror_trajectory(tr)
    assert m.joints == tr.joints
    np.testing.assert_allclose(mirror_trajectory(m).q, tr.q)
    k = tr.joints.index
    np.testing.assert_allclose(m.q[:, k("RShoulderRoll")], -tr.q[:, k("LShoulderRoll")])
    np.testing.assert_allclose(m.q[:, k("RElbowRoll")], -tr.q[:, k("LElbowRoll")])
    np.testing.assert_allclose(m.q[:, k("RShoulderPitch")], tr.q[:, k("LShoulderPitch")])
    # a left ElbowRoll (negative) becomes a valid right ElbowRoll (positive)
    assert (m.q[:, k("RElbowRoll")] > 0).all()


def test_mirror_kept_inside_symmetric_zone(cfg):
    tr = _arm_traj(cfg, np.random.default_rng(1))
    fac, _ = factory_limits(cfg, tr.joints)
    assert check_mirror(cfg, tr, mirror_trajectory(tr), fac) is None


def test_mirror_dropped_when_other_side_is_narrower(cfg):
    """L/R zones are not symmetric: a left gesture the right side cannot do is dropped."""
    tr = _arm_traj(cfg, np.random.default_rng(2))
    k = tr.joints.index("LShoulderPitch")
    tr.q[:, k] = np.linspace(0.0, 1.2, len(tr.t))   # left arm goes far down
    lim = dict(cfg.limits)
    lim["LShoulderPitch"] = (-1.8, 1.3)
    lim["RShoulderPitch"] = (-1.8, 0.5)              # right side calibrated narrower
    narrow = dataclasses.replace(cfg, limits=lim)
    fac = {j: (-3.0, 3.0) for j in tr.joints}
    reason = check_mirror(narrow, tr, mirror_trajectory(tr), fac)
    assert reason and "RShoulderPitch" in reason and "calibrated zone" in reason
    # the original already outside on its own side: the mirror may go as far, not further
    lim["LShoulderPitch"] = (-1.8, 0.5)
    assert check_mirror(dataclasses.replace(cfg, limits=lim), tr, mirror_trajectory(tr), fac) is None


def test_mirror_dropped_outside_factory_limits(cfg):
    tr = _arm_traj(cfg, np.random.default_rng(3))
    fac = {j: (-3.0, 3.0) for j in tr.joints}
    fac["RShoulderRoll"] = (-0.01, 0.01)
    assert "factory" in check_mirror(cfg, tr, mirror_trajectory(tr), fac)


def test_mirrored_examples_only_without_event(trained, cfg):
    from nao_motion.models.datasets import mirrored_examples
    st = trained["wb"].store
    head = [m.take_id for m in st.list(kind="head") if m.status == "accepted"]
    with_ev = [t for t in head if st.load_meta(t).event_ctx.present]
    assert with_ev  # the fixture records head takes with an event
    ex, rep = mirrored_examples(st, cfg, "head", head)
    assert rep["skipped_with_event"] == len(with_ev)
    assert rep["eligible"] == len(head) - len(with_ev)
    assert rep["added"] + rep["dropped"] == rep["eligible"] == len(ex) + rep["dropped"]
    assert all(t.endswith("~mirror") for t in ex.take_ids)
    assert not ex.X[:, len(cfg.emotions) + 1].any()  # event "present" feature is 0
    # stored takes untouched
    assert not any(st.exists(t) for t in ex.take_ids)


def test_training_report_mirror_and_honest_validation(trained):
    run = Path(trained["head_run"])
    m = json.loads((run / "metrics.json").read_text())["metrics"]
    mr = m["mirror"]
    assert mr["enabled"] and m["settings"]["mirror"]
    assert mr["added"] + mr["dropped"] == mr["eligible"]
    split = json.loads((run / "split.json").read_text())
    assert m["n_examples"] == len(split["train"]) + len(split["val"])  # metrics on real takes only
    assert len(split["train_mirrored"]) == mr["added"] > 0
    # only training takes are mirrored: no validation take leaks into training
    assert {t.replace("~mirror", "") for t in split["train_mirrored"]} <= set(split["train"])


def test_mirror_can_be_turned_off(trained):
    wb = trained["wb"]
    j = wb.train("head", epochs=20, background=False, overrides={"mirror": False})
    assert j["state"] == "done", j["error"]
    m = json.loads((Path(j["run"]) / "metrics.json").read_text())["metrics"]
    assert m["mirror"] == {"enabled": False} and m["settings"]["mirror"] is False


def test_mirror_override_parsing():
    from nao_motion.training import TrainingError, check_overrides
    assert check_overrides({"mirror": "off"}) == {"mirror": False}
    assert check_overrides({"mirror": True}) == {"mirror": True}
    with pytest.raises(TrainingError):
        check_overrides({"mirror": "maybe"})
