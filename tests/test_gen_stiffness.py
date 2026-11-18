"""Generation tab: stiffness of the motors around Play.

Play stiffens the joints of the cycle just before its first command and relaxes them once the
cycle is over; the operator can relax at any time, even mid-gesture (the play ends there, no
fault); key 3 goes back to rest then relaxes every motor; key 4 stiffens in place."""
import time

import numpy as np
import pytest

from nao_motion.control.controller import ControllerError, State
from nao_motion.ui.server import Api

from .conftest import make_wb
from .test_controller_sync import gesture


def _plan(wb, cfg, pid="p-test"):
    wb.plans[pid] = {"played": gesture(cfg), "gen": None, "plan": None}
    return pid


def _put_in_start_pose(wb, cfg):
    for j, v in zip(cfg.all_joints, cfg.pose("start", cfg.all_joints)):
        wb.adapter.q[j] = wb.adapter.target[j] = v


def test_play_stiffens_then_relaxes_the_cycle_joints(wb, cfg):
    pid = _plan(wb, cfg)
    wb.ctrl.arm(confirm=True)
    r = wb.execute_plan(pid, background=False, start="auto", end="rest", relax_end=True)
    assert r["started"] and wb.play_info["state"] == "done" and wb.play_info["relaxed_end"]
    assert wb.adapter.command_log, "nothing was sent"
    for j in cfg.all_joints:
        assert wb.adapter.stiffness[j] == 0.0 and wb.adapter.roles[j] == "guided"
    assert wb.ctrl.state == State.IDLE and wb.ctrl.armed


def test_play_without_relax_at_the_end_keeps_the_motors_stiff(wb, cfg):
    pid = _plan(wb, cfg)
    wb.ctrl.arm(confirm=True)
    wb.execute_plan(pid, background=False, start="auto", end="rest", relax_end=False)
    assert wb.play_info["state"] == "done" and not wb.play_info.get("relaxed_end")
    assert all(wb.adapter.roles[j] == "driven" and wb.adapter.stiffness[j] > 0 for j in cfg.all_joints)


def test_play_starts_from_relaxed_arms(wb, cfg):
    """Motors relaxed before Play (e.g. by the end of the previous play): stiffened in place first."""
    arms = cfg.groups["left_arm"] + cfg.groups["right_arm"]
    wb.adapter.relax(arms)
    pid = _plan(wb, cfg)
    wb.ctrl.arm(confirm=True)
    wb.execute_plan(pid, background=False, start="auto", end="rest", relax_end=False)
    assert wb.play_info["state"] == "done"
    assert all(wb.adapter.roles[j] == "driven" for j in arms)


def test_relax_during_the_play_ends_it_without_fault(tmp_path, cfg):
    wb = make_wb(tmp_path, cfg, virtual_time=False)
    api = Api(wb)  # built once, before the play, as by the interface
    pid = _plan(wb, cfg)
    wb.ctrl.arm(confirm=True)
    wb.execute_plan(pid, background=True, start="auto", end="rest", relax_end=True)
    t0 = time.monotonic()
    while wb.ctrl.state != State.EXECUTING and time.monotonic() - t0 < 2:
        time.sleep(0.01)
    time.sleep(0.3)
    assert wb.ctrl.state == State.EXECUTING
    api.post_relax({"groups": ["head", "left_arm", "right_arm", "hands"], "confirm": True})
    t0 = time.monotonic()
    while wb.play_info.get("ended_at") is None and time.monotonic() - t0 < 2:
        time.sleep(0.01)
    assert wb.play_info["state"] == "relaxed", (wb.play_info.get("error"), wb.ctrl.events[-5:])
    assert wb.ctrl.state == State.IDLE and wb.ctrl.armed and wb.ctrl.last_error is None
    for j in cfg.controlled_joints:
        assert wb.adapter.stiffness[j] == 0.0 and wb.adapter.roles[j] == "guided"
    n = len(wb.adapter.command_log)
    time.sleep(0.2)
    assert len(wb.adapter.command_log) == n, "commands still sent after the relax"


def test_relax_requires_confirmation(wb, cfg):
    with pytest.raises(ControllerError):
        wb.ctrl.relax(cfg.all_joints)


def test_rest_then_relax_from_relaxed_arms_in_the_start_pose(wb, cfg):
    _put_in_start_pose(wb, cfg)
    wb.adapter.relax(cfg.groups["left_arm"] + cfg.groups["right_arm"])
    wb.ctrl.arm(confirm=True)
    r = Api(wb).post_rest_relax({})
    assert r["started"]
    t0 = time.monotonic()
    while wb.play_info.get("ended_at") is None and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert wb.play_info["kind"] == "rest" and wb.play_info["state"] == "done", wb.play_info
    rest = np.array(cfg.pose("rest", cfg.all_joints))
    np.testing.assert_allclose(wb.adapter.read_angles(cfg.all_joints), rest, atol=0.05)
    for j in cfg.controlled_joints:
        assert wb.adapter.stiffness[j] == 0.0 and wb.adapter.roles[j] == "guided"


def test_rest_then_relax_needs_armed_motors(wb, cfg):
    _put_in_start_pose(wb, cfg)
    with pytest.raises(ControllerError, match="désarmé"):
        wb.rest_and_relax(background=False)
    assert wb.adapter.command_log == []


def test_stiffen_in_place_does_not_move_and_is_refused_mid_motion(tmp_path, cfg):
    wb = make_wb(tmp_path, cfg, virtual_time=False)
    wb.adapter.relax(cfg.all_joints)
    q0 = wb.adapter.read_angles(cfg.all_joints)
    Api(wb).post_hold({})
    assert all(wb.adapter.roles[j] == "driven" for j in cfg.controlled_joints)
    np.testing.assert_allclose(wb.adapter.read_angles(cfg.all_joints), q0, atol=1e-6)
    pid = _plan(wb, cfg)
    wb.ctrl.arm(confirm=True)
    wb.execute_plan(pid, background=True, start="auto", end="rest")
    t0 = time.monotonic()
    while wb.ctrl.state != State.EXECUTING and time.monotonic() - t0 < 2:
        time.sleep(0.01)
    with pytest.raises(ControllerError, match="Mouvement en cours"):
        wb.ctrl.hold(cfg.all_joints)
    wb.stop()
