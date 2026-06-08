"""
Visibility rules for Random Battle training examples.

At decision time the model only sees what that player would know:
- Own team: all 6 preview species in fixed slot order (HP/moves when known)
- Opponent: species from team preview; details only after reveal.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

from extract_pokechamp_training_data import GameState, PokemonState

from rb_team_slots import normalize_species_key

UNKNOWN_HP = -1.0
MAX_TEAM_SIZE = 6


def _default_team_entry() -> Dict:
    return {
        "hp_percent": UNKNOWN_HP,
        "status": None,
        "fainted": False,
        "boosts": {},
        "volatiles": [],
        "moves_seen": [],
        "item": None,
        "ability": None,
        "tera_type": None,
        "tera_active": False,
        "preview_species": True,
        "revealed": False,
    }


def _empty_pokemon_state(species: str, *, preview_only: bool) -> PokemonState:
    return PokemonState(
        species=species or "unknown",
        is_active=False,
        hp_percent=UNKNOWN_HP if preview_only else 1.0,
        status=None,
        fainted=False,
        stats_boosts={},
        volatile_conditions=[],
        moves_seen=[],
        item=None,
        ability=None,
        tera_type=None,
        tera_active=False,
        revealed=not preview_only,
    )


def mask_pokemon_for_actor(
    species: str,
    poke_data: Dict,
    *,
    is_own: bool,
    is_active: bool,
) -> PokemonState:
    if is_own:
        return PokemonState(
            species=species,
            is_active=is_active,
            hp_percent=float(poke_data.get("hp_percent", 1.0)),
            status=poke_data.get("status"),
            fainted=bool(poke_data.get("fainted", False)),
            stats_boosts=poke_data.get("boosts", poke_data.get("stats_boosts", {})).copy(),
            volatile_conditions=poke_data.get("volatiles", poke_data.get("volatile_conditions", [])).copy(),
            moves_seen=poke_data.get("moves_seen", []).copy(),
            item=poke_data.get("item"),
            ability=poke_data.get("ability"),
            tera_type=poke_data.get("tera_type"),
            tera_active=bool(poke_data.get("tera_active", False)),
            revealed=True,
        )

    preview = bool(poke_data.get("preview_species", False))
    revealed = bool(poke_data.get("revealed", False))

    if not preview and not revealed:
        return _empty_pokemon_state("unknown", preview_only=True)

    if preview and not revealed:
        return PokemonState(
            species=species,
            is_active=is_active,
            hp_percent=UNKNOWN_HP,
            status=None,
            fainted=bool(poke_data.get("fainted", False)) if poke_data.get("fainted") else False,
            stats_boosts={},
            volatile_conditions=[],
            moves_seen=[],
            item=None,
            ability=None,
            tera_type=None,
            tera_active=False,
            revealed=False,
        )

    return PokemonState(
        species=species,
        is_active=is_active,
        hp_percent=float(poke_data.get("hp_percent", UNKNOWN_HP)),
        status=poke_data.get("status"),
        fainted=bool(poke_data.get("fainted", False)),
        stats_boosts=poke_data.get("boosts", poke_data.get("stats_boosts", {})).copy(),
        volatile_conditions=poke_data.get("volatiles", poke_data.get("volatile_conditions", [])).copy(),
        moves_seen=poke_data.get("moves_seen", []).copy(),
        item=poke_data.get("item"),
        ability=poke_data.get("ability"),
        tera_type=poke_data.get("tera_type"),
        tera_active=bool(poke_data.get("tera_active", False)),
        revealed=True,
    )


def _team_in_preview_order(
    team: Dict[str, Dict],
    preview_order: List[str],
    *,
    is_own: bool,
    active: Optional[str],
) -> List[PokemonState]:
    result: List[PokemonState] = []
    order = preview_order[:MAX_TEAM_SIZE]
    for species in order:
        poke_data = team.get(species, _default_team_entry())
        if is_own and species not in team:
            poke_data = {**_default_team_entry(), "preview_species": True, "revealed": False}
        result.append(
            mask_pokemon_for_actor(
                species,
                poke_data,
                is_own=is_own,
                is_active=(species == active),
            )
        )
    while len(result) < MAX_TEAM_SIZE:
        result.append(_empty_pokemon_state("", preview_only=not is_own))
    return result[:MAX_TEAM_SIZE]


def build_visible_game_state(
    raw_state: Dict,
    *,
    player: str,
    preview_orders: Optional[Dict[str, List[str]]] = None,
) -> GameState:
    opponent = "p2" if player == "p1" else "p1"
    player_state = raw_state[player]
    opp_state = raw_state[opponent]

    orders = preview_orders or {}
    my_order = orders.get(player) or player_state.get("preview_order") or list(player_state["team"].keys())
    opp_order = orders.get(opponent) or opp_state.get("preview_order") or list(opp_state["team"].keys())

    my_team = _team_in_preview_order(
        player_state["team"],
        my_order,
        is_own=True,
        active=player_state.get("active"),
    )
    opp_team = _team_in_preview_order(
        opp_state["team"],
        opp_order,
        is_own=False,
        active=opp_state.get("active"),
    )

    return GameState(
        turn=raw_state.get("turn", 0),
        my_team=my_team,
        opp_team=opp_team,
        my_side_conditions=player_state.get("side_conditions", {}).copy(),
        opp_side_conditions=opp_state.get("side_conditions", {}).copy(),
        my_hazards=player_state.get("hazards", {}).copy(),
        opp_hazards=opp_state.get("hazards", {}).copy(),
        weather=raw_state.get("weather"),
        terrain=raw_state.get("terrain"),
        my_last_move=player_state.get("last_move"),
        opp_last_move=opp_state.get("last_move"),
        player=player,
        my_team_slot_order=list(my_order[:MAX_TEAM_SIZE]),
    )


def strip_future_move_from_active(
    game_state: GameState, action_type: str, action_target: str
) -> GameState:
    if action_type != "move":
        return game_state

    state = deepcopy(game_state)
    target = normalize_species_key(str(action_target or ""))
    for poke in state.my_team:
        if poke.is_active:
            poke.moves_seen = [m for m in poke.moves_seen if m != target]
            break
    return state
