"""Bridge drop during a recording (no real robot).

The Wi-Fi drop is simulated by a TCP relay placed between the interface and the bridge:
"cutting" makes the bytes vanish in both directions WITHOUT closing the sockets,
like a lost radio link (neither the PC nor the robot receives a close).
"""
import importlib.util
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from nao_motion.adapters.base import ConnectionLost
from nao_motion.adapters.bridge_client import BridgeAdapter
from nao_motion.control.recording import RecState

BRIDGE = Path(__file__).resolve().parents[1] / "bridge" / "naoqi_bridge.py"
TIMEOUT = 0.5  # bridge response timeout, reduced for the test (3 s by default)


def _bridge_module():
    spec = importlib.util.spec_from_file_location("naoqi_bridge", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _start_bridge(mod, **kw):
    backend = mod.FakeBackend()
    backend.q.update({"HeadYaw": 0.0, "HeadPitch": 0.0})
    br = mod.Bridge(backend, watchdog_s=0.3)
    ready = threading.Event()
    threading.Thread(target=mod.serve, args=(br, "127.0.0.1", 0, ready), kwargs=kw, daemon=True).start()
    assert ready.wait(5)
    return br, backend, ready.port


class Relay:
    """Cuttable TCP relay: models the PC <-> robot Wi-Fi link."""

    def __init__(self, upstream_port):
        self.up = upstream_port
        self.cut = False
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self.links = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            c, _ = self.srv.accept()
            if self.cut:  # network down: the connection never really completes
                self.links.append((c, None, None))
                continue
            u = socket.create_connection(("127.0.0.1", self.up))
            link = {"dead": False}
            self.links.append((c, u, link))
            for a, b in ((c, u), (u, c)):
                threading.Thread(target=self._pump, args=(a, b, link), daemon=True).start()

    def _pump(self, a, b, link):
        while True:
            try:
                data = a.recv(4096)
            except OSError:
                return
            if not data:
                return
            if self.cut or link["dead"]:
                link["dead"] = True  # a cut link never comes back
                continue  # bytes lost, sockets left open
            try:
                b.sendall(data)
            except OSError:
                return

    def drop(self):
        """Cut: nothing gets through any more, neither on open links nor on new ones."""
        self.cut = True
        for _, _, link in self.links:
            if link is not None:
                link["dead"] = True

    def restore(self):
        self.cut = False


@pytest.fixture
def short_timeout(cfg, monkeypatch):
    monkeypatch.setitem(cfg.robot.setdefault("bridge", {}), "timeout_s", TIMEOUT)
    return cfg


def _wait(cond, timeout=8.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _bridge_wb(tmp_path, cfg, port):
    from nao_motion.workbench import Workbench
    wb = Workbench(tmp_path / "d", tmp_path / "r", cfg=cfg, adapter_kind="sim", virtual_time=True)
    wb.connect_robot("127.0.0.1", 9559, None, port)
    assert wb.adapter.name == "bridge" and not wb.sim_demonstration
    return wb


@pytest.mark.parametrize("kind", ["arms", "head"])
def test_wifi_cut_during_recording_keeps_partial_take_and_reconnects(kind, short_timeout, tmp_path):
    cfg = short_timeout
    mod = _bridge_module()
    br, backend, bport = _start_bridge(mod)
    relay = Relay(bport)
    wb = _bridge_wb(tmp_path, cfg, relay.port)
    wb.prepare_take(kind, cfg.emotions[0], 0.5)
    wb.start_recording(countdown_s=0.0, max_duration_s=60.0)
    rec = wb.recording
    assert _wait(lambda: rec.state == RecState.RECORDING and rec.elapsed > 0.3)

    relay.drop()  # coupure Wi-Fi en plein enregistrement
    t_cut = time.monotonic()
    # the interface answers during the drop (no call blocks the server)
    for _ in range(5):
        t0 = time.monotonic()
        st = wb.status()
        wb.hands_state()
        assert time.monotonic() - t0 < 0.2
    assert _wait(lambda: not wb.rec_thread.is_alive(), timeout=4 * TIMEOUT + 3)
    assert time.monotonic() - t_cut < 3 * TIMEOUT + 2  # bounded delay, no blocking

    # take kept as INCOMPLETE, consistent and visible state
    assert rec.state == RecState.DONE and rec.result is not None and rec.result.complete is False
    assert "pont" in rec.error.lower() and rec.result.message.startswith("Incomplète")
    st = wb.status()
    assert st["recording"]["state"] == "done" and st["recording"]["complete"] is False
    assert st["connected"] is False and st["bridge"]["lost"] and st["controller"] == "fault"
    assert not st["armed"] and not wb._busy()

    # "Reconnecter" to the same bridge without restarting the server: the bridge, which still thinks
    # it serves the old connection (no close received), makes way for the new client
    relay.restore()
    wb.connect_robot("127.0.0.1", 9559, None, relay.port)
    st = wb.status()
    assert st["connected"] and st["bridge"]["lost"] is None and st["controller"] == "idle"
    np.testing.assert_allclose(wb.adapter.read_angles(["HeadYaw"]), [0.0])

    # the partial take saves normally (marked incomplete)
    take_id = wb.save_recording()
    meta = wb.store.load(take_id).meta
    assert "incomplete" in meta.annotations["tags"] and meta.kind == kind
    assert len(wb.store.load(take_id).traj.t) >= 2
    # and a new take can be prepared
    wb.prepare_take(kind, cfg.emotions[0], 0.5)
    wb.adapter.disconnect()


def test_cut_before_any_sample_aborts_cleanly(short_timeout, tmp_path):
    cfg = short_timeout
    mod = _bridge_module()
    br, backend, bport = _start_bridge(mod)
    relay = Relay(bport)
    wb = _bridge_wb(tmp_path, cfg, relay.port)
    wb.prepare_take("arms", cfg.emotions[0], 0.5)
    relay.drop()
    wb.start_recording(countdown_s=5.0, max_duration_s=60.0)  # drop during the countdown
    assert _wait(lambda: not wb.rec_thread.is_alive(), timeout=4 * TIMEOUT + 3)
    assert wb.recording.state == RecState.ABORTED and wb.recording.result is None
    assert wb.recording.error and not wb._busy()
    relay.restore()
    wb.connect_robot("127.0.0.1", 9559, None, relay.port)
    assert wb.adapter.is_connected()
    wb.adapter.disconnect()


def test_stop_during_cut_does_not_hang(short_timeout, tmp_path):
    """"Arrêter" during the drop: bounded response, take kept."""
    cfg = short_timeout
    mod = _bridge_module()
    br, backend, bport = _start_bridge(mod)
    relay = Relay(bport)
    wb = _bridge_wb(tmp_path, cfg, relay.port)
    wb.prepare_take("arms", cfg.emotions[0], 0.5)
    wb.start_recording(countdown_s=0.0, max_duration_s=60.0)
    assert _wait(lambda: wb.recording.state == RecState.RECORDING and wb.recording.elapsed > 0.2)
    relay.drop()
    t0 = time.monotonic()
    wb.stop()
    wb.stop_recording()
    assert time.monotonic() - t0 < 3 * TIMEOUT + 3
    assert wb.recording.state == RecState.DONE and wb.recording.result is not None
    assert not wb._busy()


def test_calls_never_wait_forever_on_a_silent_bridge(cfg):
    """Bridge that accepts then stops answering: every call fails in bounded time, including
    those that were waiting for the lock (recording loop, heartbeat, web requests)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    held = []

    def accept_and_answer_once():
        c, _ = srv.accept()
        held.append(c)
        f = c.makefile("rwb")
        f.readline()
        f.write(b'{"ok": true, "result": {"protocol": 1, "real": false, "robot_connected": true}}\n')
        f.flush()  # then nothing: link lost

    threading.Thread(target=accept_and_answer_once, daemon=True).start()
    a = BridgeAdapter(cfg, host="127.0.0.1", port=port, timeout_s=TIMEOUT)
    a.heartbeat_s = 10.0  # no heartbeat during this test
    a.connect()
    errors = []

    def call():
        try:
            a._call("read_angles", joints=["HeadYaw"])
        except Exception as e:
            errors.append(e)

    t0 = time.monotonic()
    ths = [threading.Thread(target=call) for _ in range(4)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(5)
    assert not any(th.is_alive() for th in ths)
    assert time.monotonic() - t0 < 2 * TIMEOUT + 2
    assert len(errors) == 4 and all(isinstance(e, ConnectionLost) for e in errors)
    assert not a.is_connected() and a.lost_reason
    srv.close()


def test_bridge_frees_itself_from_a_silent_client():
    """Robot side: a silent client (PC gone without closing) is disconnected after idle_s,
    and a new client is served even if the old one never closed."""
    mod = _bridge_module()
    br, backend, port = _start_bridge(mod, idle_s=0.5)
    old = socket.create_connection(("127.0.0.1", port))
    old.sendall(b'{"cmd": "set_angles", "joints": ["HeadYaw"], "angles": [0.1], "speed": 0.1}\n')
    assert old.makefile("rb").readline()
    assert br.motion_active
    # new client while the old one is still open: it replaces it
    new = socket.create_connection(("127.0.0.1", port), timeout=2)
    f = new.makefile("rwb")
    f.write(b'{"cmd": "hello"}\n')
    f.flush()
    assert b'"ok": true' in f.readline()
    assert backend.killed >= 1 and not br.motion_active  # the old one was driving: stop and hold
    old.settimeout(2)
    assert old.recv(10) == b""  # old connection closed by the bridge
    # silent client: closed after idle_s
    new.settimeout(3)
    t0 = time.monotonic()
    assert new.recv(10) == b""
    assert time.monotonic() - t0 < 2.5
    old.close()
    new.close()
