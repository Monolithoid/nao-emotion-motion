"""Guardian exporté (guardian.json + nao_guardian.py) : mêmes verdicts que le guardian du projet."""
import ast
import json
import shutil

import numpy as np
import pytest

from nao_motion.calibration import activate
from nao_motion.config import load_config
from nao_motion.demo import simulate_calibration
from nao_motion.guardian import Guardian
from nao_motion.guardian.export import CHECKER, ExportError, export_guardian
from nao_motion.standalone import nao_guardian as ng
from nao_motion.trajectory import Trajectory

from .conftest import make_wb


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory, cfg):
    tmp = tmp_path_factory.mktemp("export")
    d = tmp / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    wb = make_wb(tmp / "w", load_config(d))
    path = simulate_calibration(wb).save(d / "calibration" / "export-test.yaml")
    activate(d, path)
    c = load_config(d)
    out = export_guardian(c, tmp / "out")
    return c, out, ng.Guardian.load(out["guardian"])


def test_export_requires_calibration(tmp_path, cfg):
    d = tmp_path / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    base = load_config(d)
    with pytest.raises(ExportError):
        export_guardian(base, tmp_path / "o")
    out = export_guardian(base, tmp_path / "o", allow_uncalibrated=True)
    assert out["calibrated"] is False
    with pytest.raises(ng.GuardianError, match="remplacement"):
        ng.Guardian.load(out["guardian"])
    assert ng.Guardian.load(out["guardian"], allow_uncalibrated=True).limits


def test_export_content(calibrated):
    c, out, g = calibrated
    d = json.loads(open(out["guardian"], encoding="utf-8").read())
    assert d["source"]["calibrated"] and d["source"]["calibration_version"] == c.calibration_version
    assert set(d["limits"]) == set(c.controlled_joints)  # bras, tête et mains (plage d'usine)
    origins = {x["origin"] for x in d["coupled_constraints"]}
    assert origins == {"manual", "calibration"}
    assert set(d["poses"]) == {"rest", "start"} and set(d["poses"]["rest"]) == set(c.controlled_joints)
    assert open(out["checker"], encoding="utf-8").read() == CHECKER.read_text(encoding="utf-8")
    assert g.calibrated and g.joints == sorted(c.controlled_joints)


def _random_trajectories(c, rng, n=150):
    joints_sets = [c.all_joints, c.arm_joints, c.head_joints, c.groups["left_arm"]]
    for k in range(n):
        js = list(joints_sets[k % len(joints_sets)])
        m = int(rng.integers(2, 40))
        dur = float(rng.choice([0.1, 0.8, 2.0, 6.0, 20.0]))
        t = np.sort(rng.uniform(0, dur, m))
        t = np.unique(t)
        if len(t) < 2:
            continue
        lo = np.array([c.limits[j][0] for j in js])
        hi = np.array([c.limits[j][1] for j in js])
        spread = float(rng.choice([0.2, 0.6, 1.05]))  # parfois au-delà des limites
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        q = mid + half * spread * np.sin(rng.uniform(0, 2 * np.pi, len(js)) + t[:, None] * rng.uniform(0.2, 3, len(js)))
        yield Trajectory(js, t, q)


def _same(a, b):
    assert a.ok == b.ok
    ka = sorted((v.code, v.joint or "") for v in a.violations)
    kb = sorted((v.code, v.joint or "") for v in b.violations)
    assert ka == kb


def test_same_verdicts_as_project_guardian(calibrated):
    c, _, g = calibrated
    ref = Guardian(c)
    rng = np.random.default_rng(1)
    ctx = dict(zip(c.head_joints, c.pose("start", c.head_joints)))
    n_ok = n_bad = 0
    for tr in _random_trajectories(c, rng):
        state = dict(zip(tr.joints, (tr.q[0] + rng.normal(0, 0.04, len(tr.joints))).tolist()))
        a = ref.validate(tr, context_pose=ctx, start_state=state)
        b = g.check_trajectory(tr.t.tolist(), tr.joints, tr.q.tolist(), context=ctx, start_state=state)
        _same(a, b)
        assert a.stats["n_checked_points"] == b.stats["n_checked_points"]
        n_ok += a.ok
        n_bad += not a.ok
    assert n_ok > 5 and n_bad > 50  # les deux cas sont réellement exercés


def test_rest_to_start_cycle_accepted(calibrated):
    c, _, g = calibrated
    rest, start = g.pose("rest"), g.pose("start")
    assert g.check_pose(rest).ok and g.check_pose(start).ok
    d = g.min_duration(rest, start)
    assert g.check_move(rest, start, max(d, 1.0)).ok
    too_short = g.check_move(rest, start, 0.1)
    assert not too_short.ok
    ref = Guardian(c)
    tr = Trajectory(list(start), [0.0, 0.1], [[rest[j] for j in start], [start[j] for j in start]])
    _same(ref.validate(tr), too_short)
    # grand déplacement trop rapide : refus pour vitesse
    lo, hi = g.allowed_range("HeadYaw")
    a, b = dict(start, HeadYaw=lo), dict(start, HeadYaw=hi)
    fast = g.check_move(a, b, 0.5)
    assert "velocity" in {v.code for v in fast.violations}
    assert g.check_move(a, b, g.min_duration(a, b) + 0.5).ok


def test_pose_checks(calibrated):
    c, _, g = calibrated
    lo, hi = g.allowed_range("HeadYaw")
    assert g.check_pose({"HeadYaw": (lo + hi) / 2}).ok
    assert [v.code for v in g.check_pose({"HeadYaw": hi + 0.01}).violations] == ["limit"]
    assert [v.code for v in g.check_pose({"LKneePitch": 0.0}).violations] == ["legs"]
    assert [v.code for v in g.check_pose({"Foo": 0.0}).violations] == ["unknown_joint"]
    assert [v.code for v in g.check_pose({"HeadYaw": float("nan")}).violations] == ["format"]
    # contrainte couplée manuelle : bras croisés devant le torse
    pose = {"LShoulderRoll": -0.2, "RShoulderRoll": 0.2}
    rep = g.check_pose(pose)
    assert any(v.code == "coupled" for v in rep.violations) or not rep.ok


def test_malformed_inputs_rejected(calibrated):
    _, _, g = calibrated
    for t, js, q in [([0], ["HeadYaw"], [[0]]), ([0, 0], ["HeadYaw"], [[0], [0]]),
                     ([0, 1], ["HeadYaw", "HeadYaw"], [[0, 0], [0, 0]]), ([0, 1], ["HeadYaw"], [[0]]),
                     ([0, 1], [3], [[0], [0]]), ([0, "x"], ["HeadYaw"], [[0], [0]])]:
        rep = g.check_trajectory(t, js, q)
        assert not rep.ok and rep.violations[0].code == "format"
    assert not g.check_trajectory_dict({"joints": ["HeadYaw"], "t": [0, 1], "q": [[0], [0]], "unit": "deg"}).ok
    with pytest.raises(ng.GuardianError):
        ng.Guardian({"kind": "autre"})


def test_command_line(calibrated, tmp_path):
    c, out, _ = calibrated
    rest, start = (dict(zip(c.all_joints, c.pose(n, c.all_joints))) for n in ("rest", "start"))
    good = Trajectory(c.all_joints, [0, 2.0], [[rest[j] for j in c.all_joints], [start[j] for j in c.all_joints]])
    p = tmp_path / "geste.json"
    p.write_text(json.dumps({"trajectory": good.to_dict()}))
    assert ng.main([out["guardian"], str(p)]) == 0
    bad = Trajectory(["HeadYaw"], [0, 1], [[0], [9]])
    p.write_text(json.dumps(bad.to_dict()))
    assert ng.main([out["guardian"], str(p)]) == 1
    assert ng.main([out["guardian"]]) == 2


def test_checker_is_python27_compatible():
    """Pas de Python 2 dans l'environnement de test : contrôle statique de la syntaxe et des imports."""
    src = CHECKER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert src.startswith("# -*- coding: utf-8 -*-")
    fut = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    assert fut and {a.name for a in fut[0].names} >= {"division", "print_function", "unicode_literals"}
    allowed = {"__future__", "bisect", "io", "json", "math", "sys"}
    for n in ast.walk(tree):
        assert not isinstance(n, (ast.JoinedStr, ast.AnnAssign, ast.AsyncFunctionDef, ast.Await,
                                  ast.YieldFrom, ast.Nonlocal, ast.NamedExpr)), type(n).__name__
        if isinstance(n, (ast.FunctionDef,)):
            assert n.returns is None and not n.args.kwonlyargs and not n.args.posonlyargs
            assert all(a.annotation is None for a in n.args.args)
        if isinstance(n, ast.Import):
            assert {a.name for a in n.names} <= allowed
        if isinstance(n, ast.ImportFrom):
            assert n.module in allowed
        if isinstance(n, ast.Starred):
            pass
        if isinstance(n, ast.Call):
            assert not (isinstance(n.func, ast.Name) and n.func.id in ("open",)), "io.open requis en Python 2"
        if isinstance(n, ast.ClassDef):
            assert n.bases, f"{n.name} : hériter explicitement (object) en Python 2"
