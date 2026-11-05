import numpy as np
import pytest

from nao_motion.config import load_config
from nao_motion.events import EventContext


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_wb(tmp_path, cfg, **kw):
    from nao_motion.workbench import Workbench
    kw.setdefault("virtual_time", True)
    wb = Workbench(tmp_path / "data", tmp_path / "runs", cfg=cfg, adapter_kind="sim", **kw)
    wb.ctrl.connect()
    return wb


@pytest.fixture
def wb(tmp_path, cfg):
    return make_wb(tmp_path, cfg)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, cfg):
    """Petit jeu synthétique + modèles bras et tête entraînés (rapide)."""
    from nao_motion.demo import collect_synthetic
    tmp = tmp_path_factory.mktemp("trained")
    wb = make_wb(tmp, cfg, sensor_noise_rad=0.002)
    arm_ids, head_ids = collect_synthetic(
        wb, reps=2, intensities=(0.3, 1.0), head_per_arm=6,
        head_contexts=[EventContext.none(), EventContext(True, 0.6, 0.1, source="test", distance_m=0.4),
                       EventContext(True, 0.6, 0.1, source="test", distance_m=2.5)], log=lambda *a: None)
    ja = wb.train("arms", epochs=800, background=False)
    assert ja["state"] == "done", ja["error"]
    jh = wb.train("head", epochs=60, background=False)
    assert jh["state"] == "done", jh["error"]
    return {"wb": wb, "arm_ids": arm_ids, "head_ids": head_ids, "arm_run": ja["run"], "head_run": jh["run"]}
