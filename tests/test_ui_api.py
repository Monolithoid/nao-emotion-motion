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
