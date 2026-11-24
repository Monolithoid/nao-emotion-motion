"""Exported generator, generate(..., duration_s=D): several learned gestures blended into one
movement cut to D seconds, the tempo fitted within a margin that depends on the intensity."""
import json

import numpy as np
import pytest

from .test_gesture_export import exported, exported_fast  # noqa: F401  (module fixtures)


def _pose(c, gen):
    joints = gen._joint_order()
    return dict(zip(joints, c.pose("start", joints)))


def _span(gen, g):
    """Spacing of the possible cuts: one knot span of the finest group of the longest draw."""
    K = max(m.K for m in gen.models.values())
    return max(g.sequence["draw_durations_s"]) / (K - 3)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_count_forces_the_number_of_draws(exported, n):
    c, _, _, _, gen = exported
    g = gen.generate(c.emotions[0], 0.6, current_pose=_pose(c, gen), count=n, seed=1)
    assert g.ok, g.report.summary()
    assert g.sequence["draws"] == n and len(g.sequence["draw_seeds"]) == n
    # no duration: the whole learned material, at the learned tempo
    assert g.sequence["time_scale"] == 1.0
    seg = dict((name, (a, b)) for name, a, b in g.segments)
    assert seg["gesture"][1] - seg["gesture"][0] == pytest.approx(sum(g.sequence["draw_durations_s"]), abs=0.05)


@pytest.mark.parametrize("D, n", [(1.0, 1), (2.5, 2), (4.0, 3)])
def test_duration_draws_until_it_is_covered(exported, D, n):
    """Draws are added until their learned durations exceed D (the fake gestures last ~1.5 s)."""
    c, _, _, _, gen = exported
    g = gen.generate(c.emotions[0], 0.5, current_pose=_pose(c, gen), duration_s=D, seed=2)
    assert g.ok, g.report.summary()
    assert g.sequence["draws"] == n
    assert sum(g.sequence["draw_durations_s"][:-1]) <= D < sum(g.sequence["draw_durations_s"])
    seg = dict((name, (a, b)) for name, a, b in g.segments)
    assert seg["gesture"][1] - seg["gesture"][0] == pytest.approx(D, abs=0.03)
    assert [s[0] for s in g.segments] == ["to_start", "start_to_gesture", "gesture", "gesture_to_start"]


@pytest.mark.parametrize("intensity", [0.0, 0.3, 0.8])
def test_final_duration_within_the_margin(exported, intensity):
    c, _, _, _, gen = exported
    D = 5.37
    g = gen.generate(c.emotions[1], intensity, duration_s=D, seed=3)
    lo, hi = g.sequence["scale_range"]
    assert lo == pytest.approx(max(0.7, 1 - (1 - intensity))) and hi == pytest.approx(1 + (1 - intensity))
    assert lo - 1e-9 <= g.sequence["time_scale"] <= hi + 1e-9
    # the cut lands within one knot span of D; the margin covers the rest
    assert abs(g.duration - D) <= max(0.03, _span(gen, g) * (1 - min(1.0, hi - 1)))
    assert g.ok, g.report.summary()


def test_full_intensity_keeps_the_learned_tempo(exported):
    c, _, _, _, gen = exported
    for D in (3.1, 7.27):
        g = gen.generate(c.emotions[0], 1.0, current_pose=_pose(c, gen), duration_s=D, seed=4)
        assert g.sequence["scale_range"] == [1.0, 1.0] and g.sequence["time_scale"] == 1.0
        assert abs(g.sequence["duration_s"] - D) <= _span(gen, g) + 1e-6
        assert g.ok
    off = gen.generate(c.emotions[0], 0.0, duration_s=3.1, seed=4, duration_fit="off")
    assert off.sequence["time_scale"] == 1.0 and off.sequence["duration_fit"] == "off"


def test_same_seed_same_movement(exported):
    c, _, _, _, gen = exported
    a = gen.generate(c.emotions[2], 0.4, duration_s=4.0, seed=9, variation=0.05)
    b = gen.generate(c.emotions[2], 0.4, duration_s=4.0, seed=9, variation=0.05)
    d = gen.generate(c.emotions[2], 0.4, duration_s=4.0, seed=10, variation=0.05)
    assert a.positions == b.positions and a.times == b.times and a.sequence == b.sequence
    assert a.positions != d.positions
    # the first draw is the gesture generate() gives for that seed alone
    assert a.sequence["draw_seeds"][0] == 9
    # without a seed, one is drawn and reported: the sequence can be generated again
    u = gen.generate(c.emotions[2], 0.4, duration_s=4.0, variation=0.05)
    again = gen.generate(c.emotions[2], 0.4, duration_s=4.0, variation=0.05, seed=u.sequence["seed"])
    assert again.positions == u.positions


def test_continuous_and_ends_at_rest_on_the_start_pose(exported):
    c, _, _, _, gen = exported
    g = gen.generate(c.emotions[0], 0.3, duration_s=6.0, seed=5)
    assert g.sequence["draws"] >= 4
    t, q = np.array(g.times), np.array(g.positions)
    v = np.diff(q, axis=0) / np.diff(t)[:, None]
    limits = np.array([gen.guardian.velocity_limit(j) for j in g.joints])
    assert (np.abs(v) <= limits * 1.001).all()                  # no jump anywhere, junctions included
    assert np.abs(np.diff(v, axis=0)).max() < 0.05              # smooth velocity (C2 spline)
    start = np.array([gen.guardian.pose("start")[j] for j in g.joints])
    np.testing.assert_allclose(q[-1], start, atol=1e-9)         # ends on the start pose...
    assert np.abs(v[-1]).max() < 1e-3                           # ...at zero velocity
    assert g.ok


def test_no_duration_gives_the_former_output(exported):
    c, _, _, _, gen = exported
    g = gen.generate(c.emotions[0], 0.7, seed=3, current_pose=_pose(c, gen))
    assert g.sequence is None and "sequence" not in g.to_dict()
    assert "duration_s" not in g.inputs and "count" not in g.inputs
    raw = gen.generate(c.emotions[0], 0.7, seed=3, adapt=False)
    assert raw.positions == gen.gesture(c.emotions[0], 0.7, seed=3)[2]


def test_too_fast_gestures_are_made_playable_and_fitted(exported_fast):
    """Gestures above the velocity limits: the safety slow-down lengthens them, the cut is chosen
    on the PLAYED duration; up to the maximum duration of the guardian."""
    c, _, _, _, gen = exported_fast
    for D, i in ((3.0, 0.0), (3.0, 1.0), (14.0, 0.2)):
        g = gen.generate(c.emotions[0], i, current_pose=_pose(c, gen), duration_s=D, seed=2)
        assert g.ok, (D, i, g.report.summary())
        assert g.adaptation["changed"]
        lo, hi = g.sequence["scale_range"]
        assert lo - 1e-9 <= g.sequence["time_scale"] <= hi + 1e-9
        assert abs(g.sequence["duration_s"] - D) <= 0.35 * D


def test_tightening_only_if_the_guardian_accepts(exported_fast):
    """Gestures already slowed down to the limits are never played faster."""
    c, _, _, _, gen = exported_fast
    g = gen.generate(c.emotions[0], 0.0, duration_s=0.5, seed=2)
    assert g.sequence["time_scale"] >= g.sequence["scale_range"][0]
    assert gen.guardian.check_trajectory(g.times, g.joints, g.positions, check_duration=False).ok


def test_capped_at_the_maximum_duration(exported):
    c, _, _, _, gen = exported
    g = gen.generate(c.emotions[0], 0.5, duration_s=40.0, seed=1)
    assert g.sequence["target_s"] == gen.guardian.max_duration_s
    assert g.duration <= gen.guardian.max_duration_s + 1e-9
    assert any("above the guardian maximum" in n for n in g.notes)
    assert g.ok


def test_invalid_sequence_inputs(exported):
    c, _, _, lib, gen = exported
    for kw in ({"duration_s": 0}, {"duration_s": float("nan")}, {"count": 0},
               {"duration_s": 2, "duration_fit": "x"}, {"duration_s": 2, "margin_k": -1}):
        with pytest.raises(lib.GestureError):
            gen.generate(c.emotions[0], 0.5, **kw)


def test_capability_and_settings_are_exported(exported):
    c, _, out, lib, gen = exported
    assert "duration_s" in gen.info["capabilities"]
    data = json.loads(open(out["data"], encoding="utf-8").read())
    sq = c.model["sequence"]
    assert data["sequence"] == {"duration_fit": sq["duration_fit"], "margin_k": sq["duration_margin_k"],
                                "scale_min": sq["scale_min"], "scale_max": sq["scale_max"]}
    data.pop("sequence")  # files exported before this setting: defaults
    assert lib.GestureGenerator(data).sequence_settings == lib.SEQUENCE_DEFAULTS


def test_command_line_duration(exported, capsys):
    c, _, out, lib, _ = exported
    assert lib._main([out["data"], c.emotions[0], "0.5", "--duration=3.2", "--seed=1"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["sequence"]["draws"] == 3 and abs(res["duration_s"] - 3.2) < 0.05
    assert lib._main([out["data"], c.emotions[0], "0.5", "--count=2"]) == 0
    assert lib._main([out["data"], c.emotions[0], "0.5", "--bogus=2"]) == 2
