"""Convert a poke-env battle snapshot into training-style state dicts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RB_EXTRACTORS = _PROJECT_ROOT / "random_battle" / "data_extractors"
for _path in (_PROJECT_ROOT, _RB_EXTRACTORS):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from poke_env.battle import AbstractBattle, Field, SideCondition
from poke_env.battle.pokemon import Pokemon

from rb_team_slots import MAX_TEAM_SIZE, normalize_species_key
from random_battle.players.rb_move_filters import normalize_status

UNKNOWN_HP = -1.0

HAZARD_KEYS = frozenset({"spikes", "stealthrock", "toxicspikes", "stickyweb"})

FIELD_TO_TERRAIN = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
}

WEATHER_ALIASES = {
    "desolateland": "sunnyday",
    "primordialsea": "raindance",
    "deltastream": "none",
    "snowscape": "snow",
}


def _normalize_token(value: Optional[object]) -> str:
    if value is None:
        return ""
    raw = getattr(value, "name", value)
    lowered = str(raw).lower().strip()
    import re

    return re.sub(r"[^a-z0-9]", "", lowered)


def species_name(pokemon: Optional[Pokemon]) -> str:
    if pokemon is None:
        return "unknown"
    raw = getattr(pokemon, "species", None) or getattr(pokemon, "name", None)
    if raw is None:
        return "unknown"
    name = getattr(raw, "name", raw)
    return str(name).strip() or "unknown"


def _status_name(pokemon: Pokemon) -> Optional[str]:
    status = getattr(pokemon, "status", None)
    if status is None:
        return None
    name = getattr(status, "name", status)
    return normalize_status(str(name).lower() if name else None)


def _boosts_dict(pokemon: Pokemon) -> Dict[str, int]:
    boosts = getattr(pokemon, "boosts", None) or {}
    out: Dict[str, int] = {}
    for key in ("atk", "def", "spa", "spd", "spe"):
        val = boosts.get(key, 0) if isinstance(boosts, dict) else 0
        out[key] = int(val)
    return out


def _volatiles(pokemon: Pokemon) -> List[str]:
    effects = getattr(pokemon, "effects", None) or {}
    tokens: List[str] = []
    for effect in effects:
        token = _normalize_token(getattr(effect, "name", effect))
        if token:
            tokens.append(token)
    return tokens


def _moves_seen(pokemon: Pokemon, *, is_own: bool) -> List[str]:
    moves = getattr(pokemon, "moves", None) or {}
    if not moves:
        return []
    if not is_own and not pokemon.active and not pokemon.revealed:
        return []
    seen: List[str] = []
    for move in moves.values():
        move_id = getattr(move, "id", None) or getattr(move, "name", move)
        token = _normalize_token(move_id)
        if token:
            seen.append(token)
    return seen


def _hp_percent(pokemon: Pokemon, *, is_own: bool, revealed: bool) -> float:
    if not is_own and not revealed:
        return UNKNOWN_HP
    frac = getattr(pokemon, "current_hp_fraction", None)
    if frac is None:
        return UNKNOWN_HP if not is_own else 1.0
    return float(frac)


def _empty_pokemon_dict(species: str, *, preview_only: bool) -> Dict[str, object]:
    return {
        "species": species or "unknown",
        "is_active": False,
        "hp_percent": UNKNOWN_HP if preview_only else 1.0,
        "status": None,
        "fainted": False,
        "stats_boosts": {},
        "volatile_conditions": [],
        "moves_seen": [],
        "item": None,
        "ability": None,
        "tera_type": None,
        "tera_active": False,
        "revealed": not preview_only,
    }


def _mask_pokemon_dict(
    species: str,
    poke_data: Dict[str, object],
    *,
    is_own: bool,
    is_active: bool,
) -> Dict[str, object]:
    if is_own:
        return {
            "species": species,
            "is_active": is_active,
            "hp_percent": float(poke_data.get("hp_percent", 1.0)),
            "status": poke_data.get("status"),
            "fainted": bool(poke_data.get("fainted", False)),
            "stats_boosts": dict(poke_data.get("boosts", poke_data.get("stats_boosts", {})) or {}),
            "volatile_conditions": list(poke_data.get("volatiles", poke_data.get("volatile_conditions", [])) or []),
            "moves_seen": list(poke_data.get("moves_seen", []) or []),
            "item": poke_data.get("item"),
            "ability": poke_data.get("ability"),
            "tera_type": poke_data.get("tera_type"),
            "tera_active": bool(poke_data.get("tera_active", False)),
            "revealed": True,
        }

    preview = bool(poke_data.get("preview_species", False))
    revealed = bool(poke_data.get("revealed", False))
    if not preview and not revealed:
        return _empty_pokemon_dict("unknown", preview_only=True)
    if preview and not revealed:
        return {
            "species": species,
            "is_active": is_active,
            "hp_percent": UNKNOWN_HP,
            "status": None,
            "fainted": bool(poke_data.get("fainted", False)) if poke_data.get("fainted") else False,
            "stats_boosts": {},
            "volatile_conditions": [],
            "moves_seen": [],
            "item": None,
            "ability": None,
            "tera_type": None,
            "tera_active": False,
            "revealed": False,
        }
    return {
        "species": species,
        "is_active": is_active,
        "hp_percent": float(poke_data.get("hp_percent", UNKNOWN_HP)),
        "status": poke_data.get("status"),
        "fainted": bool(poke_data.get("fainted", False)),
        "stats_boosts": dict(poke_data.get("boosts", poke_data.get("stats_boosts", {})) or {}),
        "volatile_conditions": list(poke_data.get("volatiles", poke_data.get("volatile_conditions", [])) or []),
        "moves_seen": list(poke_data.get("moves_seen", []) or []),
        "item": poke_data.get("item"),
        "ability": poke_data.get("ability"),
        "tera_type": poke_data.get("tera_type"),
        "tera_active": bool(poke_data.get("tera_active", False)),
        "revealed": True,
    }


def _pokemon_raw(
    pokemon: Pokemon,
    *,
    is_own: bool,
    is_active: bool,
    preview_species: bool,
) -> Dict[str, object]:
    revealed = is_own or bool(pokemon.revealed) or is_active
    return {
        "hp_percent": _hp_percent(pokemon, is_own=is_own, revealed=revealed),
        "status": _status_name(pokemon) if revealed else None,
        "fainted": bool(getattr(pokemon, "fainted", False)),
        "boosts": _boosts_dict(pokemon) if revealed else {},
        "volatiles": _volatiles(pokemon) if revealed else [],
        "moves_seen": _moves_seen(pokemon, is_own=is_own),
        "item": _normalize_token(pokemon.item) or None if revealed and pokemon.item else None,
        "ability": _normalize_token(pokemon.ability) or None if revealed and pokemon.ability else None,
        "tera_type": _normalize_token(pokemon.tera_type) or None if revealed else None,
        "tera_active": bool(getattr(pokemon, "is_terastallized", False)) if revealed else False,
        "preview_species": preview_species,
        "revealed": revealed,
    }


def _split_side_conditions(
    side_conditions: Optional[Dict[SideCondition, int]],
) -> Tuple[Dict[str, bool], Dict[str, int]]:
    side: Dict[str, bool] = {}
    hazards: Dict[str, int] = {}
    if not side_conditions:
        return side, hazards
    for condition, layers in side_conditions.items():
        key = _normalize_token(getattr(condition, "name", condition))
        if not key:
            continue
        if key in HAZARD_KEYS:
            hazards[key] = int(layers) if layers else 1
        else:
            side[key] = True
    return side, hazards


def _weather_token(battle: AbstractBattle) -> Optional[str]:
    weather = getattr(battle, "weather", None)
    if not weather:
        return None
    token = _normalize_token(getattr(weather, "name", weather))
    return WEATHER_ALIASES.get(token, token) or None


def _terrain_token(battle: AbstractBattle) -> Optional[str]:
    fields = getattr(battle, "fields", None) or {}
    for field in fields:
        mapped = FIELD_TO_TERRAIN.get(field)
        if mapped:
            return mapped
    return None


def _ensure_preview_order(
    battle: AbstractBattle,
    cache: Dict[str, List[str]],
    *,
    opponent: bool,
) -> List[str]:
    tag = battle.battle_tag
    if tag in cache:
        return cache[tag]

    order: List[str] = []
    preview = (
        battle.teampreview_opponent_team
        if opponent
        else battle.teampreview_team
    )
    if preview:
        for mon in preview:
            order.append(species_name(mon))
    else:
        team = battle.opponent_team if opponent else battle.team
        for mon in team.values():
            order.append(species_name(mon))

    cache[tag] = order[:MAX_TEAM_SIZE]
    return cache[tag]


def _team_states(
    team: Dict[str, Pokemon],
    preview_order: List[str],
    *,
    is_own: bool,
    active_species: Optional[str],
) -> List[Dict[str, object]]:
    by_species = {species_name(mon): mon for mon in team.values()}
    active_norm = normalize_species_key(active_species or "")
    result: List[Dict[str, object]] = []

    for species in preview_order[:MAX_TEAM_SIZE]:
        mon = by_species.get(species)
        is_active = normalize_species_key(species) == active_norm
        if mon is None:
            poke_data = {
                "preview_species": True,
                "revealed": is_own,
            }
        else:
            poke_data = _pokemon_raw(
                mon,
                is_own=is_own,
                is_active=is_active,
                preview_species=True,
            )
        result.append(
            _mask_pokemon_dict(
                species,
                poke_data,
                is_own=is_own,
                is_active=is_active,
            )
        )

    while len(result) < MAX_TEAM_SIZE:
        result.append(
            _mask_pokemon_dict(
                "",
                {"preview_species": not is_own, "revealed": False},
                is_own=is_own,
                is_active=False,
            )
        )
    return result[:MAX_TEAM_SIZE]


def battle_to_state_dict(
    battle: AbstractBattle,
    my_preview_cache: Dict[str, List[str]],
    opp_preview_cache: Dict[str, List[str]],
    *,
    my_last_move: Optional[str] = None,
    opp_last_move: Optional[str] = None,
) -> Dict[str, object]:
    my_order = _ensure_preview_order(battle, my_preview_cache, opponent=False)
    opp_order = _ensure_preview_order(battle, opp_preview_cache, opponent=True)

    my_active = species_name(battle.active_pokemon) if battle.active_pokemon else None
    opp_active = (
        species_name(battle.opponent_active_pokemon)
        if battle.opponent_active_pokemon
        else None
    )

    my_side, my_hazards = _split_side_conditions(battle.side_conditions)
    opp_side, opp_hazards = _split_side_conditions(battle.opponent_side_conditions)

    return {
        "turn": int(getattr(battle, "turn", 0) or 0),
        "my_team": _team_states(
            battle.team,
            my_order,
            is_own=True,
            active_species=my_active,
        ),
        "opp_team": _team_states(
            battle.opponent_team,
            opp_order,
            is_own=False,
            active_species=opp_active,
        ),
        "my_side_conditions": my_side,
        "opp_side_conditions": opp_side,
        "my_hazards": my_hazards,
        "opp_hazards": opp_hazards,
        "weather": _weather_token(battle),
        "terrain": _terrain_token(battle),
        "my_last_move": my_last_move,
        "opp_last_move": opp_last_move,
        "player": "me",
        "my_team_slot_order": list(my_order[:MAX_TEAM_SIZE]),
    }
