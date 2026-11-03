import importlib.util
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from nao_motion.adapters.base import AdapterError
from nao_motion.adapters.bridge_client import BridgeAdapter

BRIDGE = Path(__file__).resolve().parents[1] / "bridge" / "naoqi_bridge.py"


@pytest.fixture
def bridge():
    spec = importlib.util.spec_from_file_location("naoqi_bridge", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    backend = mod.FakeBackend()
    br = mod.Bridge(backend, watchdog_s=0.3)
    ready = threading.Event()
    threading.Thread(target=mod.serve, args=(br, "127.0.0.1", 0, ready), daemon=True).start()
    assert ready.wait(5)
    return br, backend, ready.port


def test_bridge_protocol_and_roles(bridge, cfg):
    br, backend, port = bridge
    a = BridgeAdapter(cfg, host="127.0.0.1", port=port)
    a.connect()
    assert a.is_real is False and a.info["protocol"] == 1
    assert backend.q == {} and backend.stiff == {}  # aucun mouvement ni raideur à la connexion
    backend.q.update({"HeadYaw": 0.1, "HeadPitch": -0.1})
    np.testing.assert_allclose(a.read_angles(["HeadYaw", "HeadPitch"]), [0.1, -0.1])
    a.set_roles(driven=["HeadYaw", "HeadPitch"])
    a.send_angles(["HeadYaw"], [0.2], 0.1)
    assert backend.q["HeadYaw"] == 0.2
    with pytest.raises(AdapterError):
        a._call("set_angles", joints=["LKneePitch"], angles=[0.0], speed=0.1)
    a.disconnect()


def test_bridge_watchdog_stops_when_client_silent(bridge, cfg):
    br, backend, port = bridge
    a = BridgeAdapter(cfg, host="127.0.0.1", port=port)
    a.connect()
    a.set_roles(driven=["HeadYaw"])
    a.send_angles(["HeadYaw"], [0.2], 0.1)
    a._alive = False  # le battement s'interrompt (PC planté / réseau coupé), socket encore ouverte
    time.sleep(1.0)
    assert br.watchdog_trips >= 1 and backend.killed >= 1
