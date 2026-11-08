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
    assert set(st["table"]) == {"arms", "head", "hands"}
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
        with urllib.request.urlopen(base + "/favicon.ico") as r:
            assert r.headers["Content-Type"] == "image/x-icon"
            assert r.read(4) == b"\x00\x00\x01\x00"
        with urllib.request.urlopen(base + "/favicon.svg") as r:
            assert r.headers["Content-Type"].startswith("image/svg+xml")
            assert b"<svg" in r.read()
        with urllib.request.urlopen(base + "/") as r:
            html = r.read().decode()
        assert 'id="nao-fig"' in html and "nao-figure.js" in html
        # raccourcis clavier : script chargé, barre d'aide présente, script servi
        assert 'id="shortcuts"' in html and 'src="shortcuts.js"' in html
        with urllib.request.urlopen(base + "/shortcuts.js") as r:
            assert "javascript" in r.headers["Content-Type"] and b"ArrowRight" in (js := r.read())
            assert b'"a-accept"' in js and b'"h-accept"' in js and b"Enter" in js
    finally:
        srv.shutdown()


def test_hands_endpoints(tmp_path, cfg):
    """Onglet Mains : configuration, lecture de l'ouverture, commande (armement requis)."""
    from .conftest import make_wb
    wb = make_wb(tmp_path, cfg)
    api = Api(wb)
    assert api.get_config({})["hands"] == ["LHand", "RHand"]
    st = api.get_hands({})
    assert st["measured"] == {"LHand": 0.3, "RHand": 0.3} and st["limits"]["LHand"] == [0.0, 1.0]
    with pytest.raises(Exception):
        api.post_hands({"targets": {"LHand": 1.0}})  # contrôleur désarmé : rien n'est envoyé
    assert wb.adapter.command_log == []
    api.post_arm({"armed": True, "confirm": True})
    assert api.post_hands({"targets": {"LHand": 1.0, "RHand": 0.0}}).get("started")
    wb._hands_thread and wb._hands_thread.join(5)
    assert api.get_hands({})["measured"] == {"LHand": 1.0, "RHand": 0.0}
    api.post_hold({})  # « Raidir tout sur place » couvre aussi les mains
    assert all(wb.adapter.roles[j] == "driven" for j in cfg.controlled_joints)


def test_hands_tab_served():
    from nao_motion.ui.server import STATIC
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'data-tab="hands"' in html and 'src="hands.js"' in html
    for j in ("LHand", "RHand"):
        assert f'id="m-{j}"' in html
    assert '"m-prepare"' in (STATIC / "shortcuts.js").read_text(encoding="utf-8")


def test_random_draw_shortcut():
    from nao_motion.ui.server import STATIC
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "shortcuts.js").read_text(encoding="utf-8")
    for btn in ("a-random", "h-random", "m-random", "g-random"):
        assert f'id="{btn}"' in html and f'"{btn}"' in js
    assert "Digit0" in js and "Numpad0" in js
