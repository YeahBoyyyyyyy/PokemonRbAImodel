"""Battle tactics: species scripts, move filters, switch hints."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

_COMMON_DIR = Path(__file__).resolve().parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from defensive_switch import (  # noqa: E402
    estimate_pokemon_speed,
    opponent_attack_physical_weight,
    pick_best_defensive_switch,
    switch_improvement_ok,
)
from combat_helpers import (  # noqa: E402
    can_guaranteed_ko_on_hit,
    estimate_enemy_worst_damage,
    find_clean_guaranteed_ko_move,
    is_clean_damaging_move,
)

_PROTECT_TOKENS = frozenset(
    {
        "protect", "detect", "spikyshield", "kingsshield", "banefulbunker",
        "burningbulwark", "obstruct", "silktrap", "maxguard",
    }
)
_SETUP_TOKENS = frozenset(
    {
        "swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance",
        "bulkup", "irondefense", "coil", "growth", "shellsmash", "agility",
        "rockpolish", "autotomize", "geomancy", "victorydance", "noretreat",
        "workup", "tailglow", "clangoroussoul", "bellydrum", "curse",
    }
)
_POISON_TYPES = frozenset({"poison", "steel"})


def _norm(token: Optional[str]) -> str:
    if not token:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(token).lower())


def _species_key(mon: Any) -> str:
    if mon is None:
        return ""
    raw = getattr(mon, "species", None) or getattr(mon, "name", "")
    return _norm(str(getattr(raw, "name", raw)))


def _types(mon: Any) -> List[str]:
    types = getattr(mon, "types", None) or []
    out = [str(getattr(t, "name", t)).lower() for t in types if t]
    if out:
        return out
    for attr in ("type_1", "type_2"):
        t = getattr(mon, attr, None)
        if t is not None:
            out.append(str(getattr(t, "name", t)).lower())
    return out


def _is_poison_type(mon: Any) -> bool:
    return any(t in _POISON_TYPES for t in _types(mon))


def _has_volatile(mon: Any, token: str) -> bool:
    effects = getattr(mon, "effects", None) or {}
    for effect in effects:
        name = _norm(getattr(effect, "name", effect))
        if name == token:
            return True
    vols = getattr(mon, "volatile_conditions", None)
    if isinstance(vols, list):
        return token in {_norm(v) for v in vols}
    return False


def _item_token(mon: Any) -> str:
    item = getattr(mon, "item", None)
    if not item:
        return ""
    return _norm(str(getattr(item, "name", item)))


def opponent_confirmed_no_item(mon: Any) -> bool:
    """True when the foe is known to hold no item (Poltergeist would fail)."""
    if mon is None:
        return False
    item = getattr(mon, "item", None)
    if item is None:
        return True
    tok = _item_token(mon)
    if tok in ("unknownitem", "unknown_item"):
        return False
    return not tok


def _boosts(mon: Any) -> Dict[str, int]:
    raw = getattr(mon, "boosts", None) or {}
    if not isinstance(raw, dict):
        return {}
    return {k: int(v) for k, v in raw.items()}


def opponent_can_setup(enemy: Any) -> bool:
    """True if the foe has revealed setup options."""
    if enemy is None:
        return False
    moves = getattr(enemy, "moves", None) or {}
    for move in moves.values():
        mid = _norm(getattr(move, "id", "") or "")
        if mid in _SETUP_TOKENS:
            return True
        self_boost = getattr(move, "self_boost", None) or {}
        if isinstance(self_boost, dict) and any(int(v) > 0 for v in self_boost.values()):
            return True
    return False


def enemy_is_setup(enemy: Any, *, min_positive: int = 2) -> bool:
    if enemy is None:
        return False
    boosts = _boosts(enemy)
    return sum(v for v in boosts.values() if v > 0) >= min_positive


def enemy_stat_drops_severe(mon: Any, *, threshold: int = -2) -> bool:
    boosts = _boosts(mon)
    return sum(v for v in boosts.values() if v < 0) <= threshold


def my_side_toxic_spikes(battle: Any) -> int:
    conds = getattr(battle, "side_conditions", None) or {}
    for cond, layers in conds.items():
        if _norm(getattr(cond, "name", cond)) == "toxicspikes":
            return int(layers or 1)
    return 0


def extract_last_opponent_move(battle: Any) -> Optional[str]:
    """Parse the latest opponent move from poke-env turn observations."""
    role = getattr(battle, "player_role", None)
    if not role:
        return None
    opp_prefix = "p2" if str(role).lower().startswith("p1") else "p1"
    observations = getattr(battle, "observations", None) or {}
    turn = int(getattr(battle, "turn", 0) or 0)
    last: Optional[str] = None
    for t in range(turn, max(0, turn - 2) - 1, -1):
        obs = observations.get(t)
        if obs is None:
            continue
        for event in getattr(obs, "events", []) or []:
            if len(event) < 4 or event[1] != "move":
                continue
            actor = str(event[2] or "")
            if actor.startswith(opp_prefix):
                last = _norm(event[3])
    return last


def gigaton_hammer_on_cooldown(battle: Any, *, last_opp_move: Optional[str]) -> bool:
    """Team-wide GH cooldown: used by the foe last turn."""
    token = _norm(last_opp_move)
    if token != "gigatonhammer":
        return False
    return int(getattr(battle, "turn", 0) or 0) >= 1


def is_encored(mon: Any) -> bool:
    """True when the volatile Encore locks this Pokémon to one move."""
    return mon is not None and _has_volatile(mon, "encore")


def encored_forced_move(battle: Any, available_moves: Sequence[Any]) -> Optional[Any]:
    """Return the only legal move while Encore is active, if any."""
    my = getattr(battle, "active_pokemon", None)
    if my is None or not is_encored(my):
        return None
    moves = list(available_moves or [])
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]

    enabled_ids: List[str] = []
    last_req = getattr(my, "_last_request", None) or {}
    for entry in last_req.get("moves", []) or []:
        if entry.get("disabled"):
            continue
        mid = entry.get("id") or entry.get("move")
        if mid:
            enabled_ids.append(_norm(str(mid)))
    if enabled_ids:
        for move in moves:
            if _norm(getattr(move, "id", "") or "") in enabled_ids:
                return move
    return moves[0]


def filter_moves_tactical(
    moves: Sequence[Any],
    *,
    battle: Any,
    last_my_move: Optional[str],
    protect_streak: int = 0,
) -> List[Any]:
    """Drop protect/substitute spam and other anti-patterns."""
    my = getattr(battle, "active_pokemon", None)
    if my is not None and is_encored(my):
        return list(moves)
    enemy = getattr(battle, "opponent_active_pokemon", None)
    foe_no_item = opponent_confirmed_no_item(enemy)
    out: List[Any] = []
    last = _norm(last_my_move)
    try:
        from random_battle.players.rb_move_filters import sleep_clause_blocks_move
    except ImportError:
        sleep_clause_blocks_move = None  # type: ignore

    for move in moves:
        token = _norm(getattr(move, "id", "") or "")
        if token == "poltergeist" and foe_no_item:
            continue
        if sleep_clause_blocks_move is not None and sleep_clause_blocks_move(
            battle, token
        ):
            continue
        if token in _PROTECT_TOKENS:
            if last in _PROTECT_TOKENS or protect_streak >= 1:
                continue
            if protect_streak >= 2:
                continue
        if token == "substitute" and my is not None and _has_volatile(my, "substitute"):
            continue
        out.append(move)
    if out:
        return out
    fallback = [
        m
        for m in moves
        if _norm(getattr(m, "id", "") or "") != "poltergeist" or not foe_no_item
    ]
    return fallback if fallback else list(moves)


def opponent_encored_move(state: Dict[str, object]) -> Optional[str]:
    """If the foe is Encore-locked, they must repeat their last move."""
    opp = None
    for mon in state.get("opp_team") or []:
        if mon and mon.get("is_active"):
            opp = mon
            break
    if opp is None:
        return None
    vols = {_norm(v) for v in (opp.get("volatile_conditions") or [])}
    if "encore" not in vols:
        return None
    last = _norm(str(state.get("opp_last_move") or ""))
    return last or None


def move_policy_tactical_bonus(
    move_token: str,
    *,
    battle: Any,
    last_my_move: Optional[str],
    last_opp_move: Optional[str],
) -> float:
    token = _norm(move_token)
    bonus = 0.0
    enemy = getattr(battle, "opponent_active_pokemon", None)
    my = getattr(battle, "active_pokemon", None)

    if enemy is not None and is_encored(enemy) and last_opp_move:
        if token == _norm(last_opp_move):
            bonus += 0.55
        else:
            bonus -= 0.35

    if token in _PROTECT_TOKENS and _norm(last_my_move) in _PROTECT_TOKENS:
        bonus -= 0.40

    if token == "substitute" and my is not None and _has_volatile(my, "substitute"):
        bonus -= 0.35

    if token == "poltergeist" and opponent_confirmed_no_item(enemy):
        bonus -= 0.90

    try:
        from random_battle.players.rb_move_filters import sleep_clause_blocks_move

        if sleep_clause_blocks_move(battle, token):
            bonus -= 0.90
    except ImportError:
        pass

    if token == "suckerpunch" and enemy is not None and opponent_can_setup(enemy):
        if not enemy_is_setup(enemy, min_positive=1):
            bonus -= 0.20

    if enemy_is_setup(enemy) and my is not None:
        for move in getattr(battle, "available_moves", None) or []:
            if _norm(getattr(move, "id", "")) != token:
                continue
            if is_clean_damaging_move(move) and can_guaranteed_ko_on_hit(move, my, enemy):
                bonus += 0.30
            elif (getattr(move, "base_power", 0) or 0) > 0:
                bonus += 0.12
            break

    if _norm(last_opp_move) == "gigatonhammer" and token == "gigatonhammer":
        bonus -= 0.50

    return bonus


def pick_poison_absorber(
    switches: Sequence[Any],
    current: Any,
    enemy: Any,
) -> Optional[Any]:
    """Best poison type to absorb toxic spikes on switch-in."""
    candidates = [
        s for s in switches if _is_poison_type(s) and not getattr(s, "fainted", False)
    ]
    if not candidates:
        return None
    return pick_best_defensive_switch(candidates, current, enemy) or candidates[0]


def pick_eiscue_vs_physical(
    switches: Sequence[Any],
    enemy: Any,
    *,
    physical_threshold: float = 0.85,
) -> Optional[Any]:
    if enemy is None:
        return None
    if opponent_attack_physical_weight(enemy) < physical_threshold:
        return None
    for mon in switches:
        sp = _species_key(mon)
        if "eiscue" not in sp or getattr(mon, "fainted", False):
            continue
        return mon
    return None


def pick_scarf_preserving_switch(
    switches: Sequence[Any],
    current: Any,
    enemy: Any,
) -> Optional[Any]:
    """Prefer staying in if current holds Choice Scarf and matchup is not awful."""
    if current is None or "scarf" not in _item_token(current):
        return None
    if enemy is None:
        return None
    from materials import type_effectiveness

    taken = 1.0
    for atk in _types(enemy):
        for def_t in _types(current):
            taken *= float(type_effectiveness(atk, [def_t]))
    if taken < 4.0:
        return current
    return None


def gliscor_action(
    moves: Sequence[Any],
    *,
    battle: Any,
    last_my_move: Optional[str],
) -> Optional[Any]:
    """Protect 1/2 if foe can't setup else Toxic; Substitute when faster."""
    my = getattr(battle, "active_pokemon", None)
    enemy = getattr(battle, "opponent_active_pokemon", None)
    if my is None or "gliscor" not in _species_key(my):
        return None

    by_token: Dict[str, Any] = {}
    for m in moves:
        t = _norm(getattr(m, "id", ""))
        if t:
            by_token[t] = m

    if enemy is not None and estimate_pokemon_speed(my) > estimate_pokemon_speed(enemy):
        if "substitute" in by_token and not _has_volatile(my, "substitute"):
            return by_token["substitute"]

    turn = int(getattr(battle, "turn", 0) or 0)
    last = _norm(last_my_move)
    if last not in _PROTECT_TOKENS and turn % 2 == 1:
        if "protect" in by_token and not opponent_can_setup(enemy):
            return by_token["protect"]
    if "toxic" in by_token and opponent_can_setup(enemy):
        return by_token["toxic"]
    if "protect" in by_token and last not in _PROTECT_TOKENS and not opponent_can_setup(enemy):
        return by_token["protect"]
    return None


def eiscue_belly_drum_done(mon: Any, *, already_used: bool = False) -> bool:
    """True if this Eiscue must not Belly Drum again (+6 Atk or already used)."""
    if mon is None or "eiscue" not in _species_key(mon):
        return False
    if already_used:
        return True
    return int(_boosts(mon).get("atk", 0) or 0) >= 6


def strip_eiscue_belly_drum(
    moves: Sequence[Any],
    battle: Any,
    *,
    already_used: bool = False,
) -> List[Any]:
    """Remove Belly Drum from legal moves once Eiscue has already drummed."""
    my = getattr(battle, "active_pokemon", None)
    if not eiscue_belly_drum_done(my, already_used=already_used):
        return list(moves)
    filtered = [m for m in moves if _norm(getattr(m, "id", "")) != "bellydrum"]
    return filtered if filtered else list(moves)


def eiscue_belly_drum_action(
    moves: Sequence[Any],
    *,
    battle: Any,
    already_used: bool = False,
) -> Optional[Any]:
    my = getattr(battle, "active_pokemon", None)
    enemy = getattr(battle, "opponent_active_pokemon", None)
    if my is None or "eiscue" not in _species_key(my):
        return None
    if eiscue_belly_drum_done(my, already_used=already_used):
        return None
    if enemy is None or opponent_attack_physical_weight(enemy) < 0.85:
        return None
    for m in moves:
        if _norm(getattr(m, "id", "")) == "bellydrum":
            return m
    return None


def my_is_setup_mon(mon: Any, *, min_positive_boosts: int = 1) -> bool:
    """True when the active mon has invested in setup boosts."""
    boosts = _boosts(mon)
    return sum(v for v in boosts.values() if v > 0) >= min_positive_boosts


def should_defensive_tera_for_ko(
    move: Any,
    my: Any,
    enemy: Any,
    *,
    can_tera: bool,
) -> bool:
    """Tera only if it turns the chosen move into a guaranteed clean OHKO."""
    if not can_tera or move is None or my is None or enemy is None:
        return False
    tera_type = _norm(str(getattr(my, "tera_type", None) or ""))
    if not tera_type or getattr(my, "tera_type_used", False):
        return False
    if getattr(my, "is_terastallized", False):
        return False
    if can_guaranteed_ko_on_hit(move, my, enemy):
        return False
    return can_guaranteed_ko_on_hit(move, my, enemy, my_tera_type=tera_type)


def should_defensive_tera_survive_setup(
    my: Any,
    enemy: Any,
    *,
    can_tera: bool,
    max_chip_fraction: float = 0.45,
    min_hp_fraction: float = 0.35,
) -> bool:
    """Tera defensively to survive a hit and keep a boosted mon (low chip after Tera).

    Triggers when without Tera the worst enemy hit likely KOs, but with Tera the
    same hit leaves the mon alive and costs at most ``max_chip_fraction`` HP.
    """
    if not can_tera or my is None or enemy is None:
        return False
    if not my_is_setup_mon(my):
        return False
    tera_type = _norm(str(getattr(my, "tera_type", None) or ""))
    if not tera_type or getattr(my, "tera_type_used", False):
        return False
    if getattr(my, "is_terastallized", False):
        return False

    hp = float(getattr(my, "current_hp_fraction", None) or 1.0)
    if hp < min_hp_fraction:
        return False

    dmg_before = estimate_enemy_worst_damage(enemy, my)
    dmg_after = estimate_enemy_worst_damage(enemy, my, my_tera_type=tera_type)

    if dmg_before < hp * 0.85:
        return False
    if dmg_after >= hp:
        return False
    if dmg_after > max_chip_fraction:
        return False
    return dmg_after < dmg_before * 0.55


def filter_opponent_branch_moves(
    branches: List[tuple],
    *,
    gigaton_cooldown: bool,
) -> List[tuple]:
    if not gigaton_cooldown:
        return branches
    return [
        b for b in branches if _norm(b[1] if len(b) > 1 else "") != "gigatonhammer"
    ] or branches
