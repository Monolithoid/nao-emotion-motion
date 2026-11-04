import math
import shutil

import numpy as np
import pytest
import yaml

from nao_motion.config import ConfigError, load_config
from nao_motion.events import EventContext, EventError, head_to_torso
from nao_motion.guardian import Guardian
from nao_motion.trajectory import Trajectory


def test_config_loads_without_legs(cfg):
    assert not any(j.startswith(("LHip", "LKnee", "LAnkle")) for j in cfg.all_joints)
    assert set(cfg.intentions).isdisjoint(cfg.emotions)
    assert cfg.calibrated is False  # valeurs de remplacement signalées


def test_rest_and_start_poses_pass_guardian(cfg):
    g = Guardian(cfg)
    for name in ("rest", "start"):
        q = np.array(cfg.pose(name, cfg.all_joints))
        assert g.validate(Trajectory(cfg.all_joints, [0, 1], [q, q]), check_duration=False).ok


def test_config_rejects_leg_joint(tmp_path, cfg):
    d = tmp_path / "c"
    shutil.copytree(cfg.root, d)
    j = yaml.safe_load((d / "joints.yaml").read_text())
    j["groups"]["head"].append("LKneePitch")
    j["limits"]["LKneePitch"] = [-1, 1]
    (d / "joints.yaml").write_text(yaml.safe_dump(j))
    with pytest.raises(ConfigError):
        load_config(d)


def test_absent_event_differs_from_front():
    absent = EventContext.none().features()
    front = EventContext(True, 0.0, 0.0).features()
    assert absent[0] == 0 and front[0] == 1
    assert not np.allclose(absent, front)


def test_event_validation():
    with pytest.raises(EventError):
        EventContext(True, None, 0.0)
    with pytest.raises(EventError):
        EventContext(True, float("nan"), 0.0)
    with pytest.raises(EventError):
        EventContext.from_source(present=True, azimuth=10, elevation=0, frame="camera_inconnue", unit="deg")
    with pytest.raises(EventError):
        EventContext.from_source(present=True, azimuth=10, elevation=0, frame="head", unit="deg")


def test_degrees_and_head_frame_conversion():
    e = EventContext.from_source(present=True, azimuth=90, elevation=0, unit="deg")
    assert math.isclose(e.azimuth_rad, math.pi / 2)
    az, el = head_to_torso(0.0, 0.0, 0.5, 0.0)  # tête tournée de 0.5 rad à gauche
    assert math.isclose(az, 0.5, abs_tol=1e-9) and math.isclose(el, 0.0, abs_tol=1e-9)
    az, el = head_to_torso(0.0, 0.0, 0.0, 0.3)  # HeadPitch > 0 : tête baissée
    assert math.isclose(el, -0.3, abs_tol=1e-9)
    e2 = EventContext.from_source(present=True, azimuth=0, elevation=0, frame="head", head_yaw_rad=0.4,
                                  head_pitch_rad=0.0)
    assert math.isclose(e2.azimuth_rad, 0.4, abs_tol=1e-9) and e2.frame == "torso"


def test_event_roundtrip():
    e = EventContext(True, 0.3, -0.1, source="camera")
    assert EventContext.from_dict(e.to_dict()).features().tolist() == e.features().tolist()
    assert EventContext.from_dict(EventContext.none().to_dict()).present is False
