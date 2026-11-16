"""Générateur exporté (nao_gesture.py + nao_gesture.json) : mêmes gestes et mêmes verdicts que le projet."""
import ast
import importlib.util
import json
import shutil

import numpy as np
import pytest

from nao_motion.calibration import activate
from nao_motion.config import load_config
from nao_motion.demo import simulate_calibration
from nao_motion.engine import MotionEngine
from nao_motion.events import EVENT_FEATURES, EventContext
from nao_motion.gesture_export import ExportError, build_library_source, export_generator
from nao_motion.guardian import Guardian

from .conftest import make_wb


def _fake_npz(path, cfg, comp, seed, hidden=(16, 16)):
    """Export numpy au format de `LoadedModel.export_numpy` sans PyTorch : couches cachées
    aléatoires, dernière couche réglée pour un geste doux autour du départ (~1,5 s) qui
    varie un peu avec l'émotion et l'événement."""
    joints = {"arms": cfg.arm_joints, "head": cfg.head_joints, "hands": cfg.hand_joints}[comp]
    K, J = int(cfg.model["representation"]["n_control_points"]), len(joints)
    dims = [len(cfg.emotions) + 1 + EVENT_FEATURES, *hidden, K * J + 1]
    rng = np.random.default_rng(seed)
    arrays = {}
    for i in range(len(dims) - 1):
        arrays[f"W{i}"] = rng.normal(0, 0.3, (dims[i + 1], dims[i])).astype(np.float32)
        arrays[f"b{i}"] = rng.normal(0, 0.1, dims[i + 1]).astype(np.float32)
    last = len(dims) - 2
    bump = np.sin(np.pi * np.arange(K) / (K - 1))[:, None] * rng.uniform(-0.06, 0.06, J)[None, :]
    arrays[f"W{last}"] = (rng.normal(0, 0.004, (dims[-1], dims[-2]))
                          * np.append(np.abs(bump.reshape(-1)) > 0, 0)[:, None]).astype(np.float32)
    arrays[f"b{last}"] = np.append(bump.reshape(-1), np.log(1.5)).astype(np.float32)
    meta = {"emotions": cfg.emotions, "joints": list(joints), "independent": True,
            "event_features": EVENT_FEATURES, "calibration_version": cfg.calibration_version,
            "train_take_ids": ["a", "b"], "val_take_ids": ["c"], "synthetic_data": True}
    header = {"component": comp, "arch": {"in_dim": dims[0], "out_dim": dims[-1], "hidden": list(hidden),
                                          "latent_dim": 0}, "meta": meta, "source_digest": f"fake{seed}",
              "activation": "tanh"}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, header=np.array(json.dumps(header)), **arrays)
    return path


def _load_lib(path):
    spec = importlib.util.spec_from_file_location("nao_gesture_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def exported(tmp_path_factory, cfg):
    tmp = tmp_path_factory.mktemp("gesture")
    d = tmp / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    wb = make_wb(tmp / "w", load_config(d))
    path = simulate_calibration(wb).save(d / "calibration" / "gesture-test.yaml")
    activate(d, path, allow_simulated=True)
    c = load_config(d)
    models = {comp: _fake_npz(tmp / "runs" / f"r_{comp}" / "model_numpy.npz", c, comp, k)
              for k, comp in enumerate(("arms", "head", "hands"))}
    out = export_generator(c, tmp / "out", models=models)
    lib = _load_lib(out["library"])
    return c, models, out, lib, lib.GestureGenerator.load(out["data"])


CASES = [("joie", 0.7, None),
         ("colere", 1.0, EventContext(True, 0.5, 0.2, source="t", distance_m=0.8)),
         ({"joie": 0.6, "tristesse": 0.4}, 0.3, EventContext(True, -1.2, -0.1, source="t")),
         ("peur", 0.0, EventContext(True, 2.9, 0.4, source="t", distance_m=3.0))]


def _event_dict(ev):
    if ev is None or not ev.present:
        return None
    d = {"azimuth_rad": ev.azimuth_rad, "elevation_rad": ev.elevation_rad}
    if ev.distance_m is not None:
        d["distance_m"] = ev.distance_m
    return d


def test_only_two_files(exported):
    _, _, out, _, _ = exported
    assert sorted(p.name for p in __import__("pathlib").Path(out["library"]).parent.iterdir()
                  if p.suffix) == \
        ["nao_gesture.json", "nao_gesture.py"]
    assert out["missing"] == [] and out["calibrated"]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_same_gesture_as_engine(exported, case):
    c, models, _, _, gen = exported
    emo, inten, ev = CASES[case]
    eng = MotionEngine.from_paths(c, models["arms"], models["head"], models["hands"])
    ref = eng.generate(emo, inten, ev)
    joints, tt, rows, extra = gen.gesture(emo, inten, _event_dict(ev))
    assert joints == ref.combined.joints
    np.testing.assert_allclose(tt, ref.combined.t, atol=1e-12)
    np.testing.assert_allclose(np.array(rows), ref.combined.q, atol=1e-9)
    assert extra["durations"] == ref.durations
    g = gen.generate(emo, inten, _event_dict(ev))
    assert g.ok == Guardian(c).validate(ref.combined).ok
    assert g.ok, [v.to_dict() for v in g.report.violations][:8]


@pytest.mark.parametrize("from_rest", [False, True])
def test_same_cycle_as_controller(exported, tmp_path, from_rest):
    c, models, _, _, gen = exported
    wb = make_wb(tmp_path, c)
    eng = MotionEngine.from_paths(c, models["arms"], models["head"], models["hands"])
    ev = EventContext(True, 0.4, 0.1, source="t", distance_m=1.0)
    ref = eng.generate(c.emotions[0], 0.8, ev)
    joints = ref.combined.joints
    pose = dict(zip(joints, c.pose("rest" if from_rest else "start", joints)))
    if not from_rest:
        pose = {j: v + 0.05 * (k % 3 - 1) * (not c.is_hand(j)) for k, (j, v) in enumerate(pose.items())}
    plan = wb.ctrl.plan_cycle(ref.combined, from_pose="rest" if from_rest else "current", current=pose,
                              to_pose="rest" if from_rest else "start")
    g = gen.generate(c.emotions[0], 0.8, _event_dict(ev), current_pose=pose, from_rest=from_rest,
                     return_to="rest" if from_rest else "start")
    assert [s[0] for s in g.segments] == [s[0] for s in plan.segments]
    np.testing.assert_allclose(g.times, plan.traj.t, atol=1e-9)
    np.testing.assert_allclose(np.array(g.positions), plan.traj.q, atol=1e-7)
    assert g.ok == plan.report.ok
    assert g.ok, [v.to_dict() for v in g.report.violations][:8]
    names, angles, times = g.angle_interpolation()
    assert names == joints and len(angles) == len(joints) and times[0][0] > 0
    assert abs(times[0][-1] - g.duration) < 1e-9 and angles[0][-1] == pytest.approx(g.positions[-1][0])
    names, angles, times = g.angle_interpolation(rate_hz=10)
    assert len(times[0]) <= g.duration * 10 + 2 and times[0][-1] == pytest.approx(g.duration)


def test_refusals(exported):
    c, _, _, lib, gen = exported
    with pytest.raises(lib.GestureError):
        gen.generate("inconnue", 0.5)
    with pytest.raises(lib.GestureError):
        gen.generate(c.emotions[0], 1.5)
    with pytest.raises(lib.GestureError):
        gen.generate(c.emotions[0], 0.5, {"azimuth_deg": 10})
    joints = gen.gesture(c.emotions[0], 0.5)[0]
    far = {j: 0.0 for j in joints}
    far["LShoulderRoll"] = 1.3  # loin du repos
    g = gen.generate(c.emotions[0], 0.5, current_pose=far, from_rest=True)
    assert not g.ok and "not_at_rest" in {v.code for v in g.report.violations}
    with pytest.raises(lib.GuardianError):
        g.angle_interpolation()


def test_event_degrees_and_head_frame(exported):
    _, _, _, lib, _ = exported
    a = lib.event_features({"azimuth_deg": 30, "elevation_deg": -10, "distance_m": 1.0})
    b = EventContext.from_source(present=True, azimuth=30, elevation=-10, unit="deg", distance=1.0).features()
    np.testing.assert_allclose(a, b, atol=1e-6)
    h = lib.event_features({"azimuth_rad": 0.2, "elevation_rad": 0.1, "frame": "head",
                            "head_yaw_rad": 0.5, "head_pitch_rad": -0.2})
    r = EventContext.from_source(present=True, azimuth=0.2, elevation=0.1, frame="head", head_yaw_rad=0.5,
                                 head_pitch_rad=-0.2).features()
    np.testing.assert_allclose(h, r, atol=1e-6)
    assert lib.event_features(None) == [0.0] * EVENT_FEATURES


def test_command_line(exported):
    c, _, out, lib, _ = exported
    assert lib._main([out["data"], c.emotions[0], "0.5", "20", "0", "1.5"]) == 0
    assert lib._main([out["data"]]) == 2


def test_export_requires_a_model(exported, tmp_path):
    c = exported[0]
    with pytest.raises(ExportError):
        export_generator(c, tmp_path, models={"arms": None, "head": None, "hands": None})


def test_library_is_python27_compatible():
    """Pas de Python 2 dans l'environnement de test : contrôle statique de la syntaxe et des imports."""
    src = build_library_source()
    tree = ast.parse(src)
    assert src.startswith("# -*- coding: utf-8 -*-")
    fut = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    assert fut and {a.name for a in fut[0].names} >= {"division", "print_function", "unicode_literals"}
    allowed = {"__future__", "bisect", "io", "json", "math", "random", "struct", "sys"}
    for n in ast.walk(tree):
        assert not isinstance(n, (ast.JoinedStr, ast.AnnAssign, ast.AsyncFunctionDef, ast.Await,
                                  ast.YieldFrom, ast.Nonlocal, ast.NamedExpr)), type(n).__name__
        if isinstance(n, ast.FunctionDef):
            assert n.returns is None and not n.args.kwonlyargs and not n.args.posonlyargs
            assert all(a.annotation is None for a in n.args.args)
        if isinstance(n, ast.Import):
            assert {a.name for a in n.names} <= allowed
        if isinstance(n, ast.ImportFrom):
            assert n.module in allowed
        if isinstance(n, ast.Starred):
            raise AssertionError("décompression * hors appel : à éviter (Python 2)")
        if isinstance(n, ast.Call):
            assert not (isinstance(n.func, ast.Name) and n.func.id in ("open",)), "io.open requis en Python 2"
        if isinstance(n, ast.ClassDef):
            assert n.bases, f"{n.name} : hériter explicitement (object) en Python 2"


def test_variation_is_seeded(exported):
    c, _, _, _, gen = exported
    a = gen.gesture(c.emotions[0], 0.5, variation=0.05, seed=3)[2]
    b = gen.gesture(c.emotions[0], 0.5, variation=0.05, seed=3)[2]
    d = gen.gesture(c.emotions[0], 0.5, variation=0.05, seed=4)[2]
    base = gen.gesture(c.emotions[0], 0.5)[2]
    assert a == b and a != d and a != base
    assert a[0] == pytest.approx(base[0]) and a[-1] == pytest.approx(base[-1])  # extrémités inchangées


def test_rest_transition_check_matches_project(exported):
    """check_rest_transition (guardian autonome) : même verdict que validate_rest_transition."""
    from nao_motion.control.transitions import min_jerk
    c, _, _, _, gen = exported
    joints = c.arm_joints
    rest, start = c.pose("rest", joints), c.pose("start", joints)
    g = Guardian(c)
    for v in (0.2, 5.0):
        tr = min_jerk(rest, start, joints, v, 50.0, min_duration=0.05)
        ref = g.validate_rest_transition(tr)
        rep = gen.guardian.check_rest_transition(tr.t.tolist(), joints, tr.q.tolist())
        assert rep.ok == ref.ok
        assert sorted({x.code for x in rep.violations}) == sorted({x.code for x in ref.violations})
