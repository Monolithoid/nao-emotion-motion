"""Exported generator with CVAE models: z drawn at each generation, like the project engine."""
import json
import shutil

import numpy as np
import pytest

from nao_motion.calibration import activate
from nao_motion.config import load_config
from nao_motion.demo import simulate_calibration
from nao_motion.engine import MotionEngine
from nao_motion.gesture_export import export_generator

from .conftest import make_wb
from .test_gesture_export import _fake_npz, _load_lib

LATENT = 3


@pytest.fixture(scope="module")
def exported(tmp_path_factory, cfg):
    tmp = tmp_path_factory.mktemp("gesture_cvae")
    d = tmp / "configs"
    shutil.copytree(cfg.root, d)
    (d / "calibration" / "ACTIVE").unlink(missing_ok=True)
    wb = make_wb(tmp / "w", load_config(d))
    path = simulate_calibration(wb).save(d / "calibration" / "gesture-test.yaml")
    activate(d, path, allow_simulated=True)
    c = load_config(d)
    models = {comp: _fake_npz(tmp / "runs" / f"r_{comp}" / "model_numpy.npz", c, comp, k, latent_dim=LATENT)
              for k, comp in enumerate(("arms", "head", "hands"))}
    out = export_generator(c, tmp / "out", models=models)
    lib = _load_lib(out["library"])
    return c, models, out, lib, lib.GestureGenerator.load(out["data"])


def test_bundle_carries_the_default_temperature(exported):
    c, _, out, _, gen = exported
    data = json.loads(open(out["data"]).read())
    assert data["latent"]["temperature"] == float(c.model["cvae"]["temperature"])
    assert gen.latent_temperature == float(c.model["cvae"]["temperature"])
    assert gen.latent_dims == {"arms": LATENT, "head": LATENT, "hands": LATENT}


def test_given_latent_matches_engine(exported):
    c, models, _, _, gen = exported
    z = {"arms": [0.5, -1.0, 0.2], "head": [-0.3, 0.8, 0.0], "hands": [1.0, 0.0, -0.5]}
    eng = MotionEngine.from_paths(c, models["arms"], models["head"], models["hands"])
    ref = eng.generate("joy", 0.7, latent=z)
    joints, tt, rows, extra = gen.gesture("joy", 0.7, latent=z)
    assert joints == ref.combined.joints
    np.testing.assert_allclose(np.array(rows), ref.combined.q, atol=1e-9)
    assert extra["latent"] == z
    central = gen.gesture("joy", 0.7, temperature=0)
    np.testing.assert_allclose(np.array(central[2]),
                               eng.generate("joy", 0.7, temperature=0.0).combined.q, atol=1e-9)
    assert not np.allclose(np.array(central[2]), np.array(rows))


def test_draws_differ_and_seed_reproduces(exported):
    *_, gen = exported
    a = gen.generate("anger", 0.8, seed=1)
    b = gen.generate("anger", 0.8, seed=2)
    assert a.inputs["latent"]["arms"] != b.inputs["latent"]["arms"]
    assert a.inputs["temperature"] == gen.latent_temperature
    assert a.positions != b.positions
    again = gen.generate("anger", 0.8, seed=1)
    assert again.inputs["latent"] == a.inputs["latent"] and again.positions == a.positions
    unseeded = [gen.gesture("anger", 0.8)[3]["latent"]["arms"] for _ in range(3)]
    assert len(set(map(tuple, unseeded))) == 3
    assert gen.gesture("anger", 0.8, temperature=0)[3]["latent"]["arms"] == [0.0] * LATENT
    with pytest.raises(Exception):
        gen.gesture("anger", 0.8, temperature=-1)
    with pytest.raises(Exception):
        gen.gesture("anger", 0.8, latent={"arms": [1.0]})


def test_file_without_latent_setting_keeps_the_central_gesture(exported):
    _, _, out, lib, _ = exported
    data = json.loads(open(out["data"]).read())
    del data["latent"]
    old = lib.GestureGenerator(data)
    assert old.latent_temperature == 0.0
    assert old.gesture("joy", 0.7)[3]["latent"]["arms"] == [0.0] * LATENT
