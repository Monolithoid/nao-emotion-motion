import pytest

from nao_motion.ui.server import Api


def test_llm_request_never_executes(trained):
    wb = trained["wb"]
    api = Api(wb)
    n = len(wb.adapter.command_log)
    out = api.post_request({"emotion": "joie", "intensity": 0.6})
    assert "plan_id" in out and "Proposition" in out["note"]
    assert len(wb.adapter.command_log) == n
    with pytest.raises(ValueError):
        api.post_execute({"plan_id": out["plan_id"]})  # sans confirmation
    assert len(wb.adapter.command_log) == n


def test_event_endpoint_and_stats(trained):
    api = Api(trained["wb"])
    ev = api.post_event({"present": True, "azimuth": -20, "elevation": 10, "unit": "deg", "frame": "torso"})
    assert ev["present"] and ev["frame"] == "torso" and ev["unit"] == "rad"
    assert api.post_event({"present": False})["present"] is False
    st = api.get_stats({})
    assert set(st["table"]) == {"arms", "head"}
    take_id = trained["arm_ids"][0]
    d = api.get_take(take_id)
    assert d["replay"]["guardian"]["ok"] and len(d["head_takes"]) >= 1


def test_static_nao_figure_served():
    """L'illustration du NAO (image + script de mise en évidence) est servie avec le bon type."""
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from nao_motion.ui.server import make_handler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(object()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        with urllib.request.urlopen(base + "/nao.png") as r:
            assert r.headers["Content-Type"] == "image/png"
            assert r.read(8) == b"\x89PNG\r\n\x1a\n"
        with urllib.request.urlopen(base + "/") as r:
            html = r.read().decode()
        assert 'id="nao-fig"' in html and "nao-figure.js" in html
    finally:
        srv.shutdown()
