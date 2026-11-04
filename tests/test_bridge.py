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


def test_robot_address_given_by_interface(cfg, tmp_path):
    """Le pont démarre sans adresse ; l'interface fournit IP ou nom DNS, sans aucun mouvement."""
    spec = importlib.util.spec_from_file_location("naoqi_bridge", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    created = []

    def factory(host, port):
        created.append((host, port))
        return mod.FakeBackend()

    br = mod.Bridge(None, 0.3, factory)
    ready = threading.Event()
    threading.Thread(target=mod.serve, args=(br, "127.0.0.1", 0, ready), daemon=True).start()
    assert ready.wait(5)
    from nao_motion.workbench import Workbench
    wb = Workbench(tmp_path / "d", tmp_path / "r", cfg=cfg, adapter_kind="sim", virtual_time=True)
    a = BridgeAdapter(cfg, host="127.0.0.1", port=ready.port)
    a.connect()
    assert a.info["robot_connected"] is False
    with pytest.raises(AdapterError, match="adresse"):
        a.read_angles(["HeadYaw"])
    a.disconnect()
    with pytest.raises(ValueError):
        wb.connect_robot("pas une adresse!", 9559, "127.0.0.1", ready.port)
    info = wb.connect_robot("nao.local", 9559, "127.0.0.1", ready.port)
    assert created == [("nao.local", 9559)] and info["robot_address"] == "nao.local:9559"
    st = wb.status()
    assert st["adapter"] == "bridge" and st["robot_address"] == "nao.local:9559" and not st["armed"]
    assert br.motion_active is False and created[0] and wb.adapter.roles["HeadYaw"] == "idle"
    wb.connect_robot("192.168.1.10", 9559, "127.0.0.1", ready.port)
    assert created[-1] == ("192.168.1.10", 9559)
    wb.use_simulation()
    assert wb.status()["adapter"] == "sim"
