"""Defensive-switch heuristics (×2/×4 vulnerabilities + bulk balance).

Uses only type-effectiveness lookups (``materials.type_effectiveness``) and
the bundled Gen-9 pokedex (``pokedex_9G_complete.pokemon_data_gen9``) for
base stats. No damage-calc dependency.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

_COMMON_DIR = Path(__file__).resolve().parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from materials import type_effectiveness  # noqa: E402

PokemonLike = Union[object, Dict[str, object]]

# Lazy pokedex import (avoids paying the cost if no caller actually needs it).
_POKEDEX = None


def _normalize_token(value: Optional[object]) -> str:
    if value is None:
        return ""
    raw = getattr(value, "name", value)
    return re.sub(r"[^a-z0-9]", "", str(raw).lower().strip())


def _get_pokedex_entry(species: str) -> Optional[dict]:
    global _POKEDEX
    if not species:
        return None
    if _POKEDEX is None:
        try:
            from pokedex_9G_complete import pokemon_data_gen9  # type: ignore
            from materials import get_name  # type: ignore

            _POKEDEX = {"data": pokemon_data_gen9, "get_name": get_name}
        except Exception:
            _POKEDEX = {"data": {}, "get_name": None}
    data = _POKEDEX["data"]
    get_name = _POKEDEX.get("get_name")
    if not data:
        return None
    if get_name is not None:
        ref = type("S", (), {"species": species, "name": species})()
        key = get_name(ref)
        if key and key in data:
            return data[key]
    # Fallback: try exact and normalized lookups.
    if species in data:
        return data[species]
    norm = _normalize_token(species)
    for key, entry in data.items():
        if _normalize_token(key) == norm:
            return entry
    return None


def _types_from_pokemon(mon: PokemonLike) -> List[object]:
    """Get a Pokémon's effective types (Tera type if active)."""
    types = getattr(mon, "types", None) or []
    if types:
        return list(types)
    t1 = getattr(mon, "type_1", None)
    t2 = getattr(mon, "type_2", None)
    out = [t for t in (t1, t2) if t is not None]
    if out:
        return out
    if isinstance(mon, dict):
        entry = mon
        if entry.get("tera_active") and entry.get("tera_type"):
            return [str(entry.get("tera_type"))]
        species = str(entry.get("species") or "")
        info = _get_pokedex_entry(species)
        if info:
            return [str(t).lower() for t in info.get("types", [])]
    return []


def defensive_type_multipliers(
    defender: PokemonLike, attacker: PokemonLike
) -> Tuple[float, float]:
    """Return ``(product, single_max)`` of incoming type multipliers."""
    def_types = _types_from_pokemon(defender)
    att_types = _types_from_pokemon(attacker)
    if not def_types or not att_types:
        return 1.0, 1.0
    per_type: List[float] = [
        float(type_effectiveness(atk, def_types)) for atk in att_types
    ]
    product = 1.0
    for v in per_type:
        product *= v
    return product, max(per_type) if per_type else 1.0


def _pokedex_defensive_stats(species: str) -> Tuple[float, float, float]:
    entry = _get_pokedex_entry(species)
    if not entry:
        return 80.0, 80.0, 80.0
    stats = entry.get("stats", {})
    return (
        float(stats.get("HP", 80)),
        float(stats.get("Defense", 80)),
        float(stats.get("Sp. Def", 80)),
    )


def _move_physical_special_split(move: object) -> Optional[str]:
    """Read a move's damage category from poke-env metadata."""
    category = getattr(move, "category", None)
    if category is None:
        return None
    name = getattr(category, "name", category)
    text = str(name).strip().capitalize()
    if text in ("Physical", "Special"):
        return text
    return None


def opponent_attack_physical_weight(attacker: PokemonLike) -> float:
    """Return ``[0, 1]``: 1 = mostly physical, 0 = mostly special."""
    phys = 0
    spec = 0
    moves = getattr(attacker, "moves", None) or {}
    if moves:
        for move in moves.values():
            base_power = int(getattr(move, "base_power", 0) or 0)
            if base_power <= 0:
                continue
            cat = _move_physical_special_split(move)
            if cat == "Physical":
                phys += 1
            elif cat == "Special":
                spec += 1
    elif isinstance(attacker, dict):
        # No live access to move metadata -> assume neutral.
        return 0.5
    total = phys + spec
    if total == 0:
        return 0.5
    return phys / total


def defensive_bulk_score(defender: PokemonLike, attacker: PokemonLike) -> float:
    """Higher is better against the attacker's likely damage category."""
    if isinstance(defender, dict):
        species = str(defender.get("species") or "")
    else:
        species = str(getattr(defender, "species", None) or getattr(defender, "name", ""))
    hp, def_, spd = _pokedex_defensive_stats(species)
    pw = opponent_attack_physical_weight(attacker)
    weighted = (1.0 - pw) * spd + pw * def_
    return (hp / 100.0) * weighted


def needs_defensive_switch(
    active: PokemonLike,
    enemy: PokemonLike,
    *,
    hp_fraction: float = 1.0,
    product_taken_threshold: float = 4.0,
    max_hit_threshold: float = 2.0,
    hp_max_hit: float = 0.5,
) -> bool:
    """Decide whether the active mon should consider switching out."""
    product, max_hit = defensive_type_multipliers(active, enemy)
    if product >= product_taken_threshold:
        return True
    if max_hit >= max_hit_threshold and product >= max_hit_threshold:
        return hp_fraction < hp_max_hit
    if max_hit >= 4.0:
        return True
    return False


def switch_improvement_ok(
    current: PokemonLike,
    backup: PokemonLike,
    enemy: PokemonLike,
    *,
    max_backup_product: float = 1.0,
    max_backup_single: float = 1.0,
    min_product_ratio: float = 0.5,
) -> bool:
    """Require the backup to be strictly better both type- and bulk-wise."""
    cur_p, cur_m = defensive_type_multipliers(current, enemy)
    bak_p, bak_m = defensive_type_multipliers(backup, enemy)

    if bak_p > max_backup_product or bak_m > max_backup_single:
        return False
    if bak_p >= cur_p * min_product_ratio:
        return False

    cur_bulk = defensive_bulk_score(current, enemy)
    bak_bulk = defensive_bulk_score(backup, enemy)
    if bak_bulk < cur_bulk * 0.85:
        return False
    return True


def pick_best_defensive_switch(
    candidates: Sequence[PokemonLike],
    current: PokemonLike,
    enemy: PokemonLike,
) -> Optional[PokemonLike]:
    best: Optional[PokemonLike] = None
    best_key = float("inf")
    for cand in candidates:
        if not switch_improvement_ok(current, cand, enemy):
            continue
        prod, mx = defensive_type_multipliers(cand, enemy)
        bulk = defensive_bulk_score(cand, enemy)
        key = prod * 10.0 + mx + 500.0 / max(bulk, 1.0)
        if key < best_key:
            best_key = key
            best = cand
    return best
