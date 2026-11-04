"""Validation d'une adresse de robot saisie dans l'interface (IP ou nom DNS)."""
import ipaddress
import re

_LABEL = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def validate_host(host: str) -> str:
    h = (host or "").strip()
    if not h:
        raise ValueError("Adresse du robot vide : indiquer une IP ou un nom DNS (ex. nao.local)")
    try:
        return str(ipaddress.ip_address(h))
    except ValueError:
        pass
    name = h[:-1] if h.endswith(".") else h
    if len(name) > 253 or not all(_LABEL.match(p) for p in name.split(".")):
        raise ValueError(f"Adresse invalide : {host!r} (IP ou nom DNS attendu, ex. 192.168.1.10 ou nao.local)")
    return h
