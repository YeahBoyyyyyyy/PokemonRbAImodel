"""Filtres de coups légaux à l'inférence (statut redondant, etc.)."""

from __future__ import annotations

from typing import Dict, Optional

# Statuts normalisés (alignés sur STATUS_LIST du modèle)
STATUS_ALIASES: Dict[str, str] = {
    "": "none",
    "none": "none",
    "par": "par",
    "paralyzed": "par",
    "brn": "brn",
    "burned": "brn",
    "burn": "brn",
    "slp": "slp",
    "sleep": "slp",
    "asleep": "slp",
    "frz": "frz",
    "freeze": "frz",
    "frozen": "frz",
    "psn": "psn",
    "poison": "psn",
    "poisoned": "psn",
    "tox": "tox",
    "toxic": "tox",
    "badlypoisoned": "tox",
    "fnt": "none",
}

# Coup (token) -> statut qu'il applique (si déjà présent chez la cible, coup inutile)
MOVE_APPLIES_STATUS: Dict[str, str] = {
    "thunderwave": "par",
    "stunspore": "par",
    "glare": "par",
    "nuzzle": "par",
    "willowisp": "brn",
    "toxic": "tox",
    "poisonpowder": "psn",
    "poisongas": "psn",
    "spore": "slp",
    "sleeppowder": "slp",
    "hypnosis": "slp",
    "lovelykiss": "slp",
    "darkvoid": "slp",
}

MAJOR_STATUS = frozenset({"par", "brn", "slp", "frz", "psn", "tox"})

# Types défenseurs immunisés à certains statuts (Gen 9)
_DEFENDER_TYPE_STATUS_IMMUNITY: Dict[str, frozenset[str]] = {
    "steel": frozenset({"psn", "tox"}),
    "poison": frozenset({"psn", "tox"}),
}


def normalize_status(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    key = str(raw).lower().strip()
    return STATUS_ALIASES.get(key, key if key in STATUS_ALIASES.values() else None)


def is_status_inflictor_move(token: str) -> bool:
    return bool(token and token in MOVE_APPLIES_STATUS)


def status_inflictor_blocked_by_types(
    applies: Optional[str],
    defender_types: list,
) -> bool:
    """True si la cible est immunisée au statut (ex. Acier vs poison)."""
    if not applies or not defender_types:
        return False
    for raw in defender_types:
        key = str(raw).lower().strip() if raw is not None else ""
        if hasattr(raw, "name"):
            key = str(raw.name).lower()
        elif hasattr(raw, "type"):
            key = str(raw.type).lower()
        immune = _DEFENDER_TYPE_STATUS_IMMUNITY.get(key)
        if immune and applies in immune:
            return True
    return False


def status_move_blocked(
    applies: Optional[str],
    current: Optional[str],
    *,
    block_if_any_major_status: bool = True,
) -> bool:
    """True = ne pas jouer ce coup (cible a déjà un statut majeur)."""
    if not applies:
        return False
    if not current or current not in MAJOR_STATUS:
        return False
    if block_if_any_major_status:
        return True
    return applies == current
