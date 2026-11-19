from .base import AdapterError, ConnectionLost, RobotAdapter, RoleError  # noqa: F401


def make_adapter(cfg, kind=None, **kw):
    """Creates the requested adapter. None moves the robot on creation or connection."""
    kind = kind or cfg.robot.get("adapter", "sim")
    if kind == "sim":
        from .sim import SimAdapter
        return SimAdapter(cfg, **kw)
    if kind == "bridge":
        from .bridge_client import BridgeAdapter
        return BridgeAdapter(cfg, **kw)
    raise ValueError(f"Adaptateur inconnu : {kind}")
