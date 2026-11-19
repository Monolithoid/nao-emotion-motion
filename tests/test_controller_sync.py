import numpy as np
import pytest

from nao_motion.adapters.base import ConnectionLost, RoleError
from nao_motion.control.controller import ControllerError, State
from nao_motion.control.recording import HeadRecording
from nao_motion.events import EventContext
from nao_motion.trajectory import Trajectory

from .test_data import arm_take


def gesture(cfg, amp=0.15, dur=2.0):
    j = cfg.all_joints
    t = np.linspace(0, dur, int(dur * 50) + 1)
    return Trajectory(j, t, np.array(cfg.pose("start", j)) + amp * np.sin(np.pi * t / dur)[:, None] ** 2)


def test_no_motion_on_connect_or_construction(wb, cfg):
    q0 = dict(wb.adapter.q)
    wb.adapter.sleep(1.0)
    assert wb.adapter.command_log == []
    assert wb.adapter.q == q0
    assert wb.ctrl.armed is False and wb.ctrl.state == State.IDLE


def test_execution_requires_arming(wb, cfg):
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    assert plan.report.ok
    with pytest.raises(ControllerError):
        wb.ctrl.execute(plan)
    assert wb.adapter.command_log == []
    with pytest.raises(ControllerError):
        wb.ctrl.arm(confirm=False)


def test_full_cycle_returns_to_rest(wb, cfg):
    wb.ctrl.arm(confirm=True)
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    names = [s[0] for s in plan.segments]
    assert names == ["repos_vers_depart", "depart_vers_geste", "geste", "geste_vers_depart", "depart_vers_repos"]
    assert wb.ctrl.execute(plan) == "done"
    q = wb.adapter.read_angles(cfg.all_joints)
    assert np.abs(q - np.array(cfg.pose("rest", cfg.all_joints))).max() < 0.05


def test_plan_refused_if_not_at_rest(wb, cfg):
    for j in cfg.arm_joints:
        wb.adapter.q[j] += 0.3
    with pytest.raises(ControllerError):
        wb.ctrl.plan_cycle(gesture(cfg))
    assert wb.ctrl.plan_cycle(gesture(cfg), from_pose="current").report.ok


def test_stop_holds_without_return_to_rest(wb, cfg):
    wb.ctrl.arm(confirm=True)
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    n_ticks = []

    def on_tick(t, q):
        n_ticks.append(t)
        if t > 1.5:
            wb.ctrl.stop()

    assert wb.ctrl.execute(plan, on_tick=on_tick) == "stopped"
    assert wb.ctrl.state == State.HOLDING and wb.ctrl.armed is False
    n_cmd = len(wb.adapter.command_log)
    held = wb.adapter.read_angles(cfg.all_joints)
    wb.adapter.sleep(2.0)
    assert len(wb.adapter.command_log) == n_cmd  # no command after the stop
    np.testing.assert_allclose(wb.adapter.read_angles(cfg.all_joints), held, atol=1e-9)
    assert np.abs(held - np.array(cfg.pose("rest", cfg.all_joints))).max() > 0.05  # not back at rest


def test_connection_loss_faults_without_further_commands(wb, cfg):
    wb.ctrl.arm(confirm=True)
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    wb.adapter.fail_after_s = wb.adapter.now() - wb.adapter._connect_time + 1.0
    with pytest.raises(ConnectionLost):
        wb.ctrl.execute(plan)
    assert wb.ctrl.state == State.FAULT and not wb.ctrl.armed
    n = len(wb.adapter.command_log)
    with pytest.raises(ControllerError):
        wb.ctrl.arm(confirm=True)
    assert len(wb.adapter.command_log) == n


def test_real_robot_refused_when_uncalibrated(wb, cfg):
    wb.adapter.is_real = True
    with pytest.raises(ControllerError):
        wb.ctrl.arm(confirm=True)


def test_guided_joint_cannot_be_commanded(wb, cfg):
    wb.adapter.set_roles(guided=cfg.head_joints, driven=cfg.arm_joints)
    with pytest.raises(RoleError):
        wb.adapter.send_angles(cfg.head_joints, [0.0, 0.0], 0.1)
    with pytest.raises(RoleError):
        wb.adapter.set_roles(guided=["HeadYaw"], driven=["HeadYaw"])
    with pytest.raises(RoleError):
        wb.adapter.read_angles(["LKneePitch"])


def test_head_pass_releases_only_head_without_motion(wb, cfg):
    """Head pass: no arm gesture required; the head is relaxed, the arms stay
    stiff at their measured posture, nothing is sent to the motors."""
    wb.ctrl.hold(cfg.all_joints)
    wb.event = EventContext(True, 0.5, 0.0, source="test")
    q_arms = wb.adapter.read_angles(cfg.arm_joints)
    n = len(wb.adapter.command_log)
    wb.prepare_head_take("joy", 0.6)
    rec = wb.recording
    assert isinstance(rec, HeadRecording)
    assert all(wb.adapter.roles[j] == "guided" for j in cfg.head_joints)
    assert all(wb.adapter.roles[j] == "driven" for j in cfg.arm_joints)
    wb.start_recording(countdown_s=0.2, background=False)
    assert rec.result.complete and rec.result.traj.joints == cfg.head_joints
    # arms only held at their measured posture: no target moves them
    for _, joints, angles in wb.adapter.command_log[n:]:
        assert not set(joints) & set(cfg.head_joints)
        np.testing.assert_allclose(angles[:len(cfg.arm_joints)], q_arms, atol=1e-6)
    np.testing.assert_allclose(wb.adapter.read_angles(cfg.arm_joints), q_arms, atol=1e-6)
    hm = wb.store.load_meta(wb.save_recording(status="accepted"))
    assert hm.kind == "head" and hm.arm_take_id is None and hm.event["present"]
    assert set(hm.guided_joints) == set(cfg.head_joints)


def test_refused_plan_is_never_executed(wb, cfg):
    from nao_motion.guardian import GuardianRejection
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    plan.report.ok = False  # simulates a guardian refusal
    wb.plans["p"] = {"gen": None, "plan": plan}
    wb.ctrl.arm(confirm=True)
    with pytest.raises(GuardianRejection):
        wb.execute_plan("p", background=False)
    with pytest.raises(GuardianRejection):
        wb.ctrl.execute(plan)
    assert wb.adapter.command_log == []


def test_rest_outside_motion_zone_is_allowed(tmp_path):
    """Rest outside the movement zone (arms resting on the knees: contact). Only the
    rest <-> start segments leave it, checked against the factory limits and the
    reduced velocity; the gesture stays subject to the full guardian."""
    import copy
    from nao_motion.config import load_config
    from .conftest import make_wb
    cfg = copy.deepcopy(load_config())
    j = "LShoulderPitch"
    lo, hi = cfg.limits[j]
    cfg.rest_limits[j] = (lo, hi + 0.4)           # "factory" wider than the calibrated zone
    cfg.poses["rest"]["left_arm"][j] = hi + 0.2   # rest outside the zone, within factory limits
    wb = make_wb(tmp_path, cfg)
    wb.adapter.q[j] = wb.adapter.target[j] = hi + 0.2
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    assert plan.report.ok, plan.report.to_dict()
    assert [s[0] for s in plan.segments][0] == "repos_vers_depart"
    # rest <-> start segment too fast: refused
    cfg.guardian.rest_velocity_rad_s = 0.01
    rep = wb.ctrl.guardian.validate_rest_transition(wb.ctrl._transition(
        np.array(cfg.pose("rest", cfg.all_joints)), np.array(cfg.pose("start", cfg.all_joints)), cfg.all_joints))
    assert not rep.ok and any(v.code == "rest_velocity" for v in rep.violations)
    cfg.guardian.rest_velocity_rad_s = 0.3
    # rest beyond the factory limits: refused
    cfg.poses["rest"]["left_arm"][j] = hi + 0.5
    wb.adapter.q[j] = wb.adapter.target[j] = hi + 0.5
    plan = wb.ctrl.plan_cycle(gesture(cfg))
    assert not plan.report.ok and any(v.code == "rest_limit" for v in plan.report.violations)
    # the gesture itself is still checked by the full guardian
    cfg.poses["rest"]["left_arm"][j] = hi + 0.2
    wb.adapter.q[j] = wb.adapter.target[j] = hi + 0.2
    plan = wb.ctrl.plan_cycle(gesture(cfg, amp=1.5))
    assert not plan.report.ok and any(v.code == "limit" for v in plan.report.violations)
