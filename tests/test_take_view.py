import numpy as np
import pytest

from nao_motion.data.representation import fit_control_points, greville, reconstruct, resample_phase
from nao_motion.trajectory import Trajectory
from nao_motion.ui.server import Api


def _wiggle(n=400, dur=3.0):
    t = np.linspace(0, dur, n)
    q = np.stack([0.4 * np.sin(2 * np.pi * 2.5 * t / dur * 3), 0.2 * t], axis=-1)
    return Trajectory(["A", "B"], t, q)


def test_reconstruct_matches_training_fit_and_more_points_fit_better():
    tr = _wiggle()
    t, raw, fit = reconstruct(tr, 8, 100, (0.5, 2.5))
    assert t[0] == pytest.approx(0.5) and t[-1] == pytest.approx(2.5) and raw.shape == fit.shape == (100, 2)
    qu, _ = resample_phase(tr.t, tr.q, 100, (0.5, 2.5))
    assert np.allclose(raw, qu)
    ctrl = fit_control_points(qu, 8)
    assert np.allclose(fit[0], ctrl[0]) and np.allclose(fit[-1], ctrl[-1])  # clamped ends
    err = [np.sqrt(np.mean((f - r) ** 2)) for _, r, f in (reconstruct(tr, k, 100, (0.5, 2.5)) for k in (6, 8, 16, 24))]
    assert err == sorted(err, reverse=True) and err[-1] < err[0] / 4
    # a linear joint is reproduced exactly whatever the number of points
    assert np.allclose(fit[:, 1], raw[:, 1], atol=1e-6)


def test_greville_spans_the_phase():
    g = greville(8)
    assert g[0] == 0 and g[-1] == 1 and np.all(np.diff(g) > 0) and len(g) == 8


def test_take_view_endpoint(trained):
    api = Api(trained["wb"])
    tid = trained["arm_ids"][0]
    v = api.get_take_view(tid, {})
    wb = trained["wb"]
    assert v["group"] == "arms" and v["joints"] == list(wb.cfg.arm_joints)
    f = v["fit"]
    assert f["n_ctrl"] == f["default_n_ctrl"] and len(f["ctrl"]) == f["n_ctrl"] and len(f["ctrl_t"]) == f["n_ctrl"]
    assert len(f["t"]) == len(f["q"]) == f["n_samples"] and set(f["rmse_per_joint"]) == set(v["joints"])
    assert [c["n_ctrl"] for c in f["compare"]] == sorted(c["n_ctrl"] for c in f["compare"])
    assert v["role"] in ("train", "validation", "train + mirror", "train (cross-validated)")
    v16 = api.get_take_view(tid, {"n": "16"})
    assert v16["fit"]["n_ctrl"] == 16 and v16["fit"]["rmse_rad"] <= f["rmse_rad"] + 1e-9
    with pytest.raises(ValueError):
        api.get_take_view(tid, {"n": "2"})
    vm = api.get_take_view(tid, {"model": "1"})
    assert "error" in vm["model"] or (len(vm["model"]["t"]) == len(vm["model"]["q"]) and vm["model"]["rmse_rad"] >= 0)
    h = api.get_take_view(trained["head_ids"][0], {})
    assert h["group"] == "head" and h["joints"] == list(wb.cfg.head_joints)
