"""Position heuristic for engine search and policy score nudges.

Scores a battle snapshot or poke-env battle from HP balance, alive counts,
hazard control (chip on their side, clean on ours), and active HP (safe setup).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

HAZARD_KEYS = frozenset({"spikes", "stealthrock", "toxicspikes", "stickyweb"})
HAZARD_LAYER_CAPS: Dict[str, int] = {
    "spikes": 3,
    "toxicspikes": 2,
    "stealthrock": 1,
    "stickyweb": 1,
}
HAZARD_SETUP_MOVES = frozenset(
    {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
)
HAZARD_CLEAR_MOVES = frozenset({"defog", "rapidspin", "tidyup", "courtchange"})
RECOVERY_MOVES = frozenset(
    {
        "wish", "roost", "recover", "softboiled", "slackoff", "healorder",
        "moonlight", "synthesis", "rest", "strengthsap",
    }
)

EngineSnapshot = Dict[str, Any]


def _norm(token: Optional[str]) -> str:
    if not token:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(token).lower())


def _hazard_layers_from_side_conds(side_conds: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not side_conds:
        return out
    for raw_id, entry in side_conds.items():
        tag = _norm(raw_id)
        if tag not in HAZARD_KEYS:
            continue
        layers = 1
        if isinstance(entry, dict):
            layers = int(entry.get("layers") or 1)
        out[tag] = max(layers, out.get(tag, 0))
    return out


def hazard_pressure(hazards: Mapping[str, int]) -> float:
    """Normalized 0..1 hazard presence on one side of the field."""
    if not hazards:
        return 0.0
    total = 0.0
    for key, cap in HAZARD_LAYER_CAPS.items():
        layers = min(int(hazards.get(key, 0)), cap)
        total += layers / cap
    return total / len(HAZARD_LAYER_CAPS)


def _hazards_from_poke_env_side(
    side_conditions: Optional[Mapping[Any, int]],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not side_conditions:
        return out
    _map = {
        "STEALTHROCK": "stealthrock",
        "STEALTH_ROCK": "stealthrock",
        "SPIKES": "spikes",
        "TOXICSPIKES": "toxicspikes",
        "TOXIC_SPIKES": "toxicspikes",
        "STICKYWEB": "stickyweb",
        "STICKY_WEB": "stickyweb",
    }
    for cond, layers in side_conditions.items():
        key = _map.get(
            str(getattr(cond, "name", cond)).upper().replace(" ", "").replace("-", ""),
            _norm(getattr(cond, "name", cond)),
        )
        if key in HAZARD_KEYS:
            out[key] = max(int(layers or 1), out.get(key, 0))
    return out


def _mon_hp_fraction(mon: Mapping[str, Any]) -> Tuple[float, bool]:
    """Return (hp_fraction, alive) for one engine/request pokemon dict."""
    if mon.get("fainted"):
        return 0.0, False
    hp = mon.get("hp")
    maxhp = mon.get("maxhp") or mon.get("baseMaxhp")
    if hp is not None and maxhp:
        try:
            mx = int(maxhp)
            cur = int(hp)
            if mx <= 0 or cur <= 0:
                return 0.0, False
            return max(0.0, min(1.0, cur / mx)), True
        except (TypeError, ValueError):
            pass
    cond = str(mon.get("condition") or "")
    if cond.endswith(" fnt") or cond.startswith("0 "):
        return 0.0, False
    main = cond.split(" ")[0] if cond else ""
    if "/" in main:
        try:
            cur_s, max_s = main.split("/", 1)
            mx = int(max_s)
            cur = int(cur_s)
            if mx <= 0 or cur <= 0:
                return 0.0, False
            return max(0.0, min(1.0, cur / mx)), True
        except ValueError:
            pass
    if cond:
        return 1.0, True
    return 1.0, True


def _team_stats_from_side(side: Mapping[str, Any]) -> Tuple[float, int, float]:
    """Mean HP fraction, alive count, active mon HP for one engine side."""
    pokemon = list(side.get("pokemon") or [])
    if not pokemon:
        return 0.0, 0, 1.0
    hp_sum = 0.0
    alive = 0
    active_hp = 1.0
    for mon in pokemon:
        frac, is_alive = _mon_hp_fraction(mon)
        if is_alive:
            alive += 1
            hp_sum += frac
            if mon.get("active") or mon.get("isActive"):
                active_hp = frac
    denom = max(len(pokemon), 1)
    return hp_sum / denom, alive, active_hp


def _merge_side(state_side: Mapping[str, Any], req_side: Mapping[str, Any]) -> Dict[str, Any]:
    """Combine engine state (hp/maxhp) with request view (condition, active)."""
    state_mons = list(state_side.get("pokemon") or [])
    req_mons = list(req_side.get("pokemon") or [])
    if not state_mons and not req_mons:
        return dict(state_side)
    merged_mons: List[Dict[str, Any]] = []
    n = max(len(state_mons), len(req_mons))
    for i in range(n):
        mon: Dict[str, Any] = {}
        if i < len(state_mons) and isinstance(state_mons[i], dict):
            mon.update(state_mons[i])
        if i < len(req_mons) and isinstance(req_mons[i], dict):
            req = req_mons[i]
            if req.get("condition"):
                mon["condition"] = req["condition"]
            if req.get("active"):
                mon["active"] = True
        merged_mons.append(mon)
    out = dict(state_side)
    out["pokemon"] = merged_mons
    if not out.get("sideConditions") and req_side.get("sideConditions"):
        out["sideConditions"] = req_side.get("sideConditions")
    return out


def _species_from_side_mon(mon: Mapping[str, Any]) -> str:
    spec = mon.get("species")
    if isinstance(spec, str):
        m = re.match(r"\[Species:([^\]]+)\]", spec, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
        if spec.strip():
            return spec.strip().lower()
    details = mon.get("details") or ""
    if isinstance(details, str) and details.strip():
        return details.split(",", 1)[0].strip().lower()
    return ""


def _types_for_species(species: str) -> List[str]:
    if not species:
        return []
    try:
        from defensive_switch import _get_pokedex_entry

        info = _get_pokedex_entry(species)
        if info:
            return [str(t).lower() for t in info.get("types", [])]
    except Exception:
        pass
    return []


def _active_mon(side: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for mon in side.get("pokemon") or []:
        if isinstance(mon, Mapping) and (mon.get("active") or mon.get("isActive")):
            return mon
    return None


def _mon_dict_for_bulk(mon: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if mon is None:
        return {}
    species = _species_from_side_mon(mon)
    out: Dict[str, Any] = {"species": species}
    moves = mon.get("moves")
    if moves:
        out["moves"] = moves
    return out


def _active_mon_from_snap(
    snap: EngineSnapshot, side_key: str
) -> Optional[Dict[str, Any]]:
    """Active mon with moves attached from engine request (for bulk weighting)."""
    p1_side, p2_side = _sides_from_snap(snap)
    side = p1_side if side_key == "p1" else p2_side
    active = _active_mon(side or {})
    if active is None:
        return None
    merged = dict(active)
    req = ((snap.get("requests") or {}).get(side_key) or {})
    active_req = (req.get("active") or [None])[0]
    if isinstance(active_req, dict) and active_req.get("moves"):
        merged["moves"] = active_req["moves"]
    return merged


def active_matchup_term(
    my_active: Optional[Mapping[str, Any]],
    opp_active: Optional[Mapping[str, Any]],
) -> float:
    """0..1 matchup: type offense/defense + pokedex bulk (HP/Def/SpD)."""
    if my_active is None or opp_active is None:
        return 0.5
    my_species = _species_from_side_mon(my_active)
    opp_species = _species_from_side_mon(opp_active)
    my_types = _types_for_species(my_species)
    opp_types = _types_for_species(opp_species)
    try:
        from materials import type_effectiveness
        from defensive_switch import (
            defensive_bulk_score,
            defensive_type_multipliers,
            normalize_bulk_score,
        )
    except Exception:
        return 0.5

    my_ref = _mon_dict_for_bulk(my_active)
    opp_ref = _mon_dict_for_bulk(opp_active)

    if my_types and opp_types:
        off = max(float(type_effectiveness(t, opp_types)) for t in my_types)
        _prod, taken_max = defensive_type_multipliers(my_ref, opp_ref)
        off_s = min(1.0, max(0.0, (off - 0.25) / 1.75))
        def_s = min(1.0, max(0.0, (2.0 - float(taken_max)) / 1.75))
    else:
        off_s = 0.5
        def_s = 0.5

    bulk_s = normalize_bulk_score(defensive_bulk_score(my_ref, opp_ref))
    return 0.30 * off_s + 0.25 * def_s + 0.45 * bulk_s


def _sides_from_snap(snap: EngineSnapshot) -> Tuple[Optional[Mapping], Optional[Mapping]]:
    """P1 and P2 side dicts, merging serialized state with compact requests."""
    state = snap.get("state") or {}
    sides = state.get("sides") or []
    requests = snap.get("requests") or {}
    p1_req = ((requests.get("p1") or {}).get("side") or {})
    p2_req = ((requests.get("p2") or {}).get("side") or {})
    if len(sides) >= 2 and (sides[0].get("pokemon") or sides[1].get("pokemon")):
        return (
            _merge_side(sides[0], p1_req),
            _merge_side(sides[1], p2_req),
        )
    if p1_req.get("pokemon") and p2_req.get("pokemon"):
        return p1_req, p2_req
    return None, None


def position_score(
    *,
    my_hp_total: float,
    opp_hp_total: float,
    my_alive: int,
    opp_alive: int,
    my_hazards: Mapping[str, int],
    opp_hazards: Mapping[str, int],
    active_hp: float = 1.0,
    opp_active_hp: float = 1.0,
    matchup: float = 0.5,
    w_hp: float = 0.38,
    w_alive: float = 0.24,
    w_chip: float = 0.10,
    w_hazard: float = 0.16,
    w_setup: float = 0.11,
    w_matchup: float = 0.06,
) -> float:
    """P1 win heuristic in [0, 1], 0.5 = even."""
    my_hp = max(0.0, float(my_hp_total))
    opp_hp = max(0.0, float(opp_hp_total))
    hp_ratio = my_hp / (my_hp + opp_hp + 1e-6)

    alive = max(0, int(my_alive))
    opp_a = max(0, int(opp_alive))
    alive_ratio = alive / (alive + opp_a + 1e-6)

    my_hz = hazard_pressure(my_hazards)
    opp_hz = hazard_pressure(opp_hazards)
    hazard_term = 0.5 + 0.5 * (opp_hz - my_hz)

    setup_term = max(0.0, min(1.0, float(active_hp)))
    chip_term = 1.0 - max(0.0, min(1.0, float(opp_active_hp)))

    matchup_term = max(0.0, min(1.0, float(matchup)))

    score = (
        w_hp * hp_ratio
        + w_alive * alive_ratio
        + w_chip * chip_term
        + w_hazard * hazard_term
        + w_setup * setup_term
        + w_matchup * matchup_term
    )
    return max(0.0, min(1.0, score))


def _terminal_outcome_score(
    snap: EngineSnapshot,
    *,
    p1_name: str = "Bot",
) -> Optional[float]:
    """Hard 0/1/0.5 when the battle ended; else None."""
    if not snap.get("ended"):
        return None
    winner = snap.get("winner")
    if winner == p1_name:
        return 1.0
    if winner:
        return 0.0
    return 0.5


def score_from_engine_snap(
    snap: EngineSnapshot,
    *,
    p1_name: str = "Bot",
) -> float:
    """Absolute position heuristic from an engine snapshot (P1 perspective)."""
    ended = _terminal_outcome_score(snap, p1_name=p1_name)
    if ended is not None:
        return ended

    p1_side, p2_side = _sides_from_snap(snap)
    if p1_side is None or p2_side is None:
        return 0.5

    p1_hp, p1_alive, active_hp = _team_stats_from_side(p1_side)
    p2_hp, p2_alive, opp_active_hp = _team_stats_from_side(p2_side)

    my_hazards = _hazard_layers_from_side_conds(p1_side.get("sideConditions") or {})
    opp_hazards = _hazard_layers_from_side_conds(p2_side.get("sideConditions") or {})
    matchup = active_matchup_term(
        _active_mon_from_snap(snap, "p1"),
        _active_mon_from_snap(snap, "p2"),
    )

    return position_score(
        my_hp_total=p1_hp,
        opp_hp_total=p2_hp,
        my_alive=p1_alive,
        opp_alive=p2_alive,
        my_hazards=my_hazards,
        opp_hazards=opp_hazards,
        active_hp=active_hp,
        opp_active_hp=opp_active_hp,
        matchup=matchup,
    )


def relative_score_from_engine_snaps(
    leaf: EngineSnapshot,
    root: EngineSnapshot,
    *,
    p1_name: str = "Bot",
    scale: float = 27.0,
) -> float:
    """Score a leaf vs the search root: 0.5 + scale * (pos(leaf) - pos(root)).

    Over a short lookahead (e.g. 3 plies) absolute position scores barely move;
    comparing to the root amplifies which candidate line improved the game.
    """
    ended = _terminal_outcome_score(leaf, p1_name=p1_name)
    if ended is not None:
        return ended
    leaf_s = score_from_engine_snap(leaf, p1_name=p1_name)
    root_s = score_from_engine_snap(root, p1_name=p1_name)
    delta = leaf_s - root_s
    return max(0.0, min(1.0, 0.5 + scale * delta))


def score_from_state_dict(state: Dict[str, Any]) -> float:
    """Heuristic from a poke_env_to_state dict."""

    def _side_stats(team: list) -> Tuple[float, int, float]:
        if not team:
            return 0.0, 0, 1.0
        hp_sum = 0.0
        alive = 0
        active_hp = 1.0
        for mon in team:
            if not mon:
                continue
            hp = float(mon.get("hp_percent", 0.0) or 0.0)
            if hp < 0:
                hp = 0.0
            if mon.get("fainted") or hp <= 0:
                continue
            alive += 1
            hp_sum += hp
            if mon.get("is_active"):
                active_hp = hp
        denom = max(len(team), 1)
        return hp_sum / denom, alive, active_hp

    my_team = state.get("my_team") or []
    opp_team = state.get("opp_team") or []
    my_hp, my_alive, active_hp = _side_stats(my_team)
    opp_hp, opp_alive, opp_active_hp = _side_stats(opp_team)
    my_hazards = dict(state.get("my_hazards") or {})
    opp_hazards = dict(state.get("opp_hazards") or {})

    my_active = next((m for m in my_team if m and m.get("is_active")), None)
    opp_active = next(
        (
            m
            for m in opp_team
            if m
            and m.get("is_active")
            and (m.get("revealed") or float(m.get("hp_percent", -1) or -1) >= 0)
        ),
        None,
    )
    if opp_active is not None and opp_active.get("moves_seen"):
        opp_ref = dict(opp_active)
        seen = list(opp_active.get("moves_seen") or [])
        opp_ref["moves"] = [{"id": m} for m in seen if m]
        matchup = active_matchup_term(my_active, opp_ref)
    else:
        matchup = active_matchup_term(my_active, opp_active)

    return position_score(
        my_hp_total=my_hp,
        opp_hp_total=opp_hp,
        my_alive=my_alive,
        opp_alive=opp_alive,
        my_hazards=my_hazards,
        opp_hazards=opp_hazards,
        active_hp=active_hp,
        opp_active_hp=opp_active_hp,
        matchup=matchup,
    )


def _team_alive_count(battle: Any, *, opponent: bool) -> int:
    team = (
        getattr(battle, "opponent_team", None)
        if opponent
        else getattr(battle, "team", None)
    ) or {}
    return sum(
        1 for p in team.values() if p is not None and not getattr(p, "fainted", False)
    )


def late_game_hazard_penalty(battle: Any) -> float:
    """Penalty magnitude for setting hazards when few switches remain."""
    opp_alive = _team_alive_count(battle, opponent=True)
    my_alive = _team_alive_count(battle, opponent=False)
    total_alive = opp_alive + my_alive
    turn = int(getattr(battle, "turn", 0) or 0)

    if opp_alive <= 1:
        return 0.35
    if opp_alive == 2 and total_alive <= 4:
        return 0.22
    if total_alive <= 3:
        return 0.18
    if turn >= 28 and opp_alive <= 3:
        return 0.12
    return 0.0


def move_policy_bonus(move_token: str, battle: Any) -> float:
    """Small additive nudge on top of the multi-head policy score."""
    token = _norm(move_token)
    if not token or battle is None:
        return 0.0

    try:
        from combat_helpers import can_guaranteed_ko_on_hit, is_clean_damaging_move
    except ImportError:
        can_guaranteed_ko_on_hit = None  # type: ignore
        is_clean_damaging_move = None  # type: ignore

    my_hazards = _hazards_from_poke_env_side(getattr(battle, "side_conditions", None))
    opp_hazards = _hazards_from_poke_env_side(
        getattr(battle, "opponent_side_conditions", None)
    )

    active = getattr(battle, "active_pokemon", None)
    active_hp = 1.0
    if active is not None:
        frac = getattr(active, "current_hp_fraction", None)
        if frac is not None:
            active_hp = float(frac)

    bonus = 0.0

    if token in HAZARD_SETUP_MOVES:
        cap = HAZARD_LAYER_CAPS.get(token, 1)
        layers = int(opp_hazards.get(token, 0))
        if layers >= cap:
            bonus -= 0.10
        elif layers > 0:
            bonus -= 0.04 * (layers / cap)
        endgame = late_game_hazard_penalty(battle)
        if endgame > 0.0:
            bonus -= endgame

    if token in HAZARD_CLEAR_MOVES:
        if my_hazards:
            bonus += 0.08 * hazard_pressure(my_hazards)
        if token == "defog" and opp_hazards:
            # Defog also clears our hazards — small penalty if we had opp hazards set.
            bonus -= 0.03 * hazard_pressure(opp_hazards)

    # Delayed / redundant healing.
    if token in RECOVERY_MOVES:
        if active_hp >= 0.85:
            bonus -= 0.30
        elif active_hp >= 0.65:
            bonus -= 0.18
        if token == "wish" and active_hp >= 0.45:
            bonus -= 0.12

    # Setup when healthy; risky when low.
    setup_tokens = {
        "swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance",
        "irondefense", "bulkup", "coil", "growth", "shellsmash", "agility",
        "rockpolish", "autotomize", "geomancy", "victorydance", "noretreat",
        "workup", "tailglow", "clangoroussoul",
    }
    if token == "bellydrum" and active is not None:
        try:
            from tactical_rules import eiscue_belly_drum_done

            if eiscue_belly_drum_done(active):
                bonus -= 0.40
        except Exception:
            boosts = getattr(active, "boosts", None) or {}
            if int(boosts.get("atk", 0) or 0) >= 6:
                bonus -= 0.40

    if token in setup_tokens:
        if active_hp >= 0.75:
            bonus += 0.05
        elif active_hp < 0.45:
            bonus -= 0.06
        enemy = getattr(battle, "opponent_active_pokemon", None)
        if enemy is not None and active is not None and active_hp >= 0.60:
            try:
                from materials import type_effectiveness

                my_types = list(getattr(active, "types", None) or [])
                if not my_types:
                    t1, t2 = getattr(active, "type_1", None), getattr(active, "type_2", None)
                    my_types = [t for t in (t1, t2) if t is not None]
                enemy_types = list(getattr(enemy, "types", None) or [])
                if not enemy_types:
                    t1, t2 = getattr(enemy, "type_1", None), getattr(enemy, "type_2", None)
                    enemy_types = [t for t in (t1, t2) if t is not None]
                off = 1.0
                for atk in my_types:
                    off *= float(type_effectiveness(atk, enemy_types))
                if off >= 2.0:
                    bonus += 0.12
            except Exception:
                pass

    # Finish with a clean guaranteed KO when possible.
    if can_guaranteed_ko_on_hit is not None and is_clean_damaging_move is not None:
        enemy = getattr(battle, "opponent_active_pokemon", None)
        active = getattr(battle, "active_pokemon", None)
        if enemy is not None and active is not None:
            for move in getattr(battle, "available_moves", None) or []:
                mid = _norm(getattr(move, "id", "") or "")
                if mid != token:
                    continue
                if is_clean_damaging_move(move) and can_guaranteed_ko_on_hit(
                    move, active, enemy
                ):
                    bonus += 0.35
                break

    return bonus
