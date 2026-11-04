from .base import AdapterError, ConnectionLost, RobotAdapter, RoleError  # noqa: F401


def make_adapter(cfg, kind=None, **kw):
    """Crée l'adaptateur demandé. Aucun ne bouge le robot à la création ni à la connexion."""
    kind = kind or cfg.robot.get("adapter", "sim")
    if kind == "sim":
        from .sim import SimAdapter
        return SimAdapter(cfg, **kw)
    if kind == "bridge":
        from .bridge_client import BridgeAdapter
        return BridgeAdapter(cfg, **kw)
    raise ValueError(f"Adaptateur inconnu : {kind}")
