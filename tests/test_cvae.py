"""CVAE mode: z input to the MLP decoder, encoder reading the takes, z map and sweep."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nao_motion.engine import MotionEngine
from nao_motion.events import EventContext
from nao_motion.models.nets import LoadedModel, NumpyModel, is_cvae
from nao_motion.training import TrainingError, check_overrides, cvae_settings
from nao_motion.ui.server import Api


@pytest.fixture(scope="module")
def cvae_wb(trained, cfg, tmp_path_factory):
    """Same takes as `trained`, own runs folder (the session models stay plain MLPs)."""
    from nao_motion.workbench import Workbench
    tmp = tmp_path_factory.mktemp("cvae")
    wb = Workbench(trained["wb"].store.root, tmp / "runs", cfg=cfg, adapter_kind="sim", virtual_time=True)
    wb.ctrl.connect()
    job = wb.train("arms", epochs=80, background=False,
                   overrides={"cvae": True, "latent_dim": 3, "kl_warmup": 30, "encoder_hidden": "16", "folds": 2})
    assert job["state"] == "done", job["error"]
    return wb, Path(job["run"])


def test_cvae_off_by_default_keeps_the_plain_mlp(trained, cfg):
    assert cvae_settings(cfg)["enabled"] is False
    m = LoadedModel(Path(trained["arm_run"]) / "model.pt", cfg, "arms")
    assert m.arch["latent_dim"] == 0 and "cvae" not in m.arch and m.model.encoder is None
    assert not any(k.startswith("encoder") for k in m.model.state_dict())
    assert not is_cvae(m) and m.meta["metrics"]["settings"]["cvae"] is False


def test_cvae_overrides_checked():
    o = check_overrides({"cvae": "true", "latent_dim": "3", "beta": "0.02", "kl_warmup": 10, "encoder_hidden": "16,16"})
    assert o == {"cvae": True, "latent_dim": 3, "beta": 0.02, "kl_warmup": 10, "encoder_hidden": [16, 16]}
    for bad in ({"latent_dim": 0}, {"latent_dim": 2.5}, {"beta": -1}, {"encoder_hidden": "0"}, {"kl_warmup": -3}):
        with pytest.raises(TrainingError):
            check_overrides(bad)


def test_cvae_training_metrics(cvae_wb, cfg):
    wb, run = cvae_wb
    d = json.loads((run / "metrics.json").read_text())
    m = d["metrics"]
    s = m["settings"]
    assert s["cvae"] == {"latent_dim": 3, "beta": 0.003, "kl_warmup": 30, "encoder_hidden": [16]}
    assert s["motion_energy_weight"] == 0.0  # energy term off in CVAE mode
    for k in ("recon_rmse_rad", "recon_motion_kept", "draws_motion_kept", "draws_spread_rad", "val_recon_rmse_rad",
              "z_active", "z_var_per_dim", "z_kl_per_dim"):
        assert k in m, k
    assert m["z_dim"] == 3 and len(m["z_var_per_dim"]) == 3 and 0 <= m["z_active"] <= 3
    # the best epoch is looked for after the KL warm-up
    assert all(f["best_epoch"] >= 30 for f in m["cv"]["per_fold"])
    assert "NaN" not in (run / "metrics.json").read_text()


def test_cvae_checkpoint_numpy_and_engine(cvae_wb, cfg):
    wb, run = cvae_wb
    mt = LoadedModel(run / "model.pt", cfg, "arms")
    mn = NumpyModel(run / "model_numpy.npz", cfg, "arms")
    assert is_cvae(mt) and is_cvae(mn) and mt.model.encoder is not None
    x = np.random.default_rng(0).normal(size=(4, mt.arch["in_dim"])).astype(np.float32)
    z = np.random.default_rng(1).normal(size=(4, 3))
    np.testing.assert_allclose(mt.predict(x, z), mn.predict(x, z), atol=1e-5)
    np.testing.assert_allclose(mt.predict(x), mn.predict(x, np.zeros(3)), atol=1e-5)
    lat = mt.meta["latent"]
    assert lat["dim"] == 3 and len(lat["mu"]) == len(lat["take_ids"]) == len(lat["pca"]["xy"])
    eng = MotionEngine(cfg, mt)
    central = [eng.generate("joy", 0.7, temperature=0.0).arms.q for _ in range(2)]
    np.testing.assert_array_equal(*central)
    a = eng.generate("joy", 0.7, temperature=1.0, seed=1)
    b = eng.generate("joy", 0.7, temperature=1.0, seed=2)
    assert a.inputs["latent"]["arms"] != b.inputs["latent"]["arms"]
    assert a.arms.q.shape != b.arms.q.shape or not np.allclose(a.arms.q, b.arms.q)
    np.testing.assert_array_equal(a.arms.q, eng.generate("joy", 0.7, temperature=1.0, seed=1).arms.q)
    g = eng.generate("joy", 0.7, latent={"arms": [0.5, -1.0, 0.0]})
    assert g.inputs["latent"]["arms"] == [0.5, -1.0, 0.0]
    with pytest.raises(ValueError):
        eng.generate("joy", 0.7, latent={"arms": [1.0]})


def test_cvae_map_sweep_and_generate_api(cvae_wb, cfg):
    wb, run = cvae_wb
    api = Api(wb)
    assert api.get_config({})["cvae"]["enabled"] is False
    lat = api.get_latent({"component": "arms"})
    assert lat["available"] and lat["dim"] == 3 and len(lat["events"]) == len(lat["take_ids"])
    assert api.get_latent({"component": "head"})["available"] is False  # no head model here
    sw = api.post_latent_sweep({"component": "arms", "dim": 1, "emotion": "anger", "intensity": 0.8,
                                "use_event": False, "steps": 4})
    C = np.array(sw["curves"])
    assert C.shape == (4, 100, len(cfg.arm_joints)) and len(sw["durations"]) == 4 and sw["values"][0] == -2.0
    i = 0
    out = api.post_generate({"emotion": lat["emotions"][i], "intensity": lat["intensities"][i],
                             "latent": {"arms": lat["mu"][i]}, "event": lat["events"][i]})
    assert out["generated"]["inputs"]["latent"]["arms"] == [round(v, 4) for v in lat["mu"][i]]
    out = api.post_generate({"emotion": "joy", "intensity": 0.5, "temperature": 0})
    assert out["generated"]["inputs"]["latent"]["arms"] == [0.0, 0.0, 0.0]
    # Takes tab: the model curve of a take uses its own z (deterministic), never a random draw
    tid = lat["take_ids"][0]
    v1 = api.get_take_view(tid, {"model": "1"})["model"]
    v2 = api.get_take_view(tid, {"model": "1"})["model"]
    assert "error" not in v1 and v1["q"] == v2["q"]


def test_cvae_fields_in_ui():
    static = Path(__file__).resolve().parents[1] / "nao_motion" / "ui" / "static"
    html, js, lat = (static / "index.html").read_text(), (static / "mlp.js").read_text(), (static / "latent.js").read_text()
    for i in ("tr-cvae", "tr-zdim", "tr-beta", "tr-warmup", "tr-enc", "g-temp", "lat-map", "lat-sweep"):
        assert f'id="{i}"' in html, i
    assert "zChart" in js and 'src="latent.js"' in html and "latent_sweep" in lat
