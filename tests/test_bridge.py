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
    assert backend.q == {} and backend.stiff == {}  # no movement or stiffness on connection
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
    a._alive = False  # the heartbeat stops (PC crashed / network down), socket still open
    time.sleep(1.0)
    assert br.watchdog_trips >= 1 and backend.killed >= 1


def test_robot_address_given_by_interface(cfg, tmp_path):
    """The bridge starts without an address; the interface provides an IP or DNS name, without any movement."""
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
        wb.connect_robot("pas une adresse!", 9559, "127.0.0.1", ready.port, bridge_on_robot=False)
    info = wb.connect_robot("nao.local", 9559, "127.0.0.1", ready.port, bridge_on_robot=False)
    assert created == [("nao.local", 9559)] and info["robot_address"] == "nao.local:9559"
    st = wb.status()
    assert st["adapter"] == "bridge" and st["robot_address"] == "nao.local:9559" and not st["armed"]
    assert br.motion_active is False and created[0] and wb.adapter.roles["HeadYaw"] == "idle"
    wb.connect_robot("192.168.1.10", 9559, "127.0.0.1", ready.port, bridge_on_robot=False)
    assert created[-1] == ("192.168.1.10", 9559)
    wb.use_simulation()
    assert wb.status()["adapter"] == "sim"


def _load_bridge_module():
    spec = importlib.util.spec_from_file_location("naoqi_bridge", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bridge_on_robot_mode(cfg, tmp_path):
    """Bridge started on the NAO (--on-robot): already linked to the local NAOqi; the interface reaches
    the bridge at the robot's address and does not ask for a NAOqi connection again."""
    mod = _load_bridge_module()
    created = []

    def factory(host, port):
        created.append((host, port))
        return mod.FakeBackend()

    br = mod.Bridge(None, 0.3, factory)
    br.handle({"cmd": "connect_robot", "host": "127.0.0.1", "port": 9559})  # like --on-robot
    ready = threading.Event()
    threading.Thread(target=mod.serve, args=(br, "127.0.0.1", 0, ready), daemon=True).start()
    assert ready.wait(5)
    from nao_motion.workbench import Workbench
    wb = Workbench(tmp_path / "d", tmp_path / "r", cfg=cfg, adapter_kind="sim", virtual_time=True)
    assert cfg.robot["bridge"]["location"] == "robot"
    # the "robot address" is 127.0.0.1 here because the fake bridge runs locally
    wb.connect_robot("127.0.0.1", 9559, None, ready.port)
    assert created == [("127.0.0.1", 9559)]  # aucune reconnexion NAOqi
    assert wb.status()["adapter"] == "bridge" and not wb.status()["armed"] and br.motion_active is False
    wb.use_simulation()


def test_backend_prefers_naoqi_module(monkeypatch):
    """On the robot (NAOqi 2.1), the `naoqi` module is used before `qi`."""
    import sys
    import types
    mod = _load_bridge_module()
    calls = []
    fake = types.ModuleType("naoqi")
    fake.ALProxy = lambda name, ip, port: calls.append((name, ip, port)) or object()
    monkeypatch.setitem(sys.modules, "naoqi", fake)
    be = mod.NaoqiBackend("127.0.0.1", 9559)
    assert be.api == "naoqi" and calls[0] == ("ALMotion", "127.0.0.1", 9559)
    monkeypatch.setitem(sys.modules, "naoqi", None)
    monkeypatch.setitem(sys.modules, "qi", None)
    with pytest.raises(RuntimeError, match="on-robot"):
        mod.NaoqiBackend("127.0.0.1", 9559)


def test_bridge_is_python27_compatible():
    """The bridge must run on the NAO's Python 2.7.3: static check (no Python 2 here)."""
    import ast
    src = BRIDGE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert src.startswith("# -*- coding: utf-8 -*-")
    allowed = {"__future__", "argparse", "json", "re", "socket", "sys", "threading", "time", "os", "naoqi", "qi"}
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
        if isinstance(n, ast.ClassDef):
            assert n.bases, n.name


def test_joint_names_are_native_str_on_python2(monkeypatch):
    """NAOqi 2.1 (ALProxy, Python 2) refuses unicode for getLimits(String):
    the bridge converts JSON strings to str. Observed on the robot on 2026-11-04."""
    import types
    mod = _load_bridge_module()
    assert mod.native_str("HeadYaw") == "HeadYaw"  # Python 3: unchanged
    monkeypatch.setattr(mod, "sys", types.SimpleNamespace(version_info=(2, 7, 3)))
    assert mod.native_str(u"HeadYaw") == b"HeadYaw"
    assert mod.native_str(1.5) == 1.5


def test_reconnect_same_bridge_keeps_calibration(cfg, tmp_path):
    """Drop during a calibration: the dead socket is closed (the bridge, which serves only
    one client, is freed) and "Connecter au robot" on the same bridge reconnects in
    place without losing the open calibration or its explorations."""
    from nao_motion.adapters.base import ConnectionLost
    from nao_motion.workbench import Workbench
    mod = _load_bridge_module()
    br = mod.Bridge(None, 0.3, lambda h, p: mod.FakeBackend())
    br.handle({"cmd": "connect_robot", "host": "127.0.0.1", "port": 9559})
    ready = threading.Event()
    threading.Thread(target=mod.serve, args=(br, "127.0.0.1", 0, ready), daemon=True).start()
    assert ready.wait(5)
    wb = Workbench(tmp_path / "d", tmp_path / "r", cfg=cfg, adapter_kind="sim", virtual_time=True)
    wb.connect_robot("127.0.0.1", 9559, None, ready.port)
    ad = wb.adapter
    wb.open_calibration()
    wb.calibration.session.sweeps["head"] = {"joints": cfg.groups["head"], "q": np.zeros((60, 2))}
    ad._sock.shutdown(__import__("socket").SHUT_RDWR)  # simulated network drop
    with pytest.raises(ConnectionLost):
        ad.read_angles(["HeadYaw"])
    assert ad._sock is None  # dead socket closed
    wb.connect_robot("127.0.0.1", 9559, None, ready.port)
    assert wb.adapter is ad and wb.calibration is not None and "head" in wb.calibration.session.sweeps
    assert ad.read_angles(["HeadYaw"]) == [0.0]
    with pytest.raises(Exception, match="Calibration ouverte"):
        wb.connect_robot("127.0.0.2", 9559, None, ready.port)
    wb.close_calibration()
    wb.use_simulation()
