"""Lightweight type-based combat heuristics.

Provides fast, deterministic per-move *scores* used by the hybrid / model
players for snap decisions (best attack, Tera, KO check) without needing
to spawn the Node simulator. Real damage is produced by
``EngineSimulator`` (``pkmn_engine_simulator.py``) when ``--use_engine``
is enabled.

The scoring is intentionally simple:

    score = base_power * type_effectiveness * STAB

Returned values are exposed as a ``DamageEstimate`` (compat shim) where
``avg_frac`` / ``max_frac`` are normalised into a [0, 1.5] range — close
enough to "fraction of the defender's HP" for thresholding (KO check),
without claiming numerical accuracy.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

_COMMON_DIR = Path(__file__).resolve().parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from materials import type_effectiveness  # noqa: E402

from defensive_switch import estimate_pokemon_speed  # noqa: E402

# Empirical scaling so that 80 BP * 1x effectiveness * 1.5 STAB ~= 0.6 (60% HP).
_SCORE_TO_FRAC = 0.005

# Common RB moves that lower the user's stats (fallback when metadata is sparse).
_SELF_STAT_DROP_MOVES = frozenset(
    {
        "closecombat", "superpower", "overheat", "dracometeor", "leafstorm",
        "psychoboost", "fleurcannon", "hammerarm", "vcreate", "makeitrain",
        "armorcannon", "headlongrush", "wavecrash",
        "highjumpkick", "doubleedge", "woodhammer", "bravebird", "flareblitz",
    }
)


@dataclass(frozen=True)
class DamageEstimate:
    """Score wrapper exposed to callers expecting an HP-fraction estimate.

    The numbers are approximate: ``avg_frac`` is a [0, 1.5] heuristic
    obtained from ``base_power * type_effectiveness * STAB``, not a
    physically accurate damage prediction.
    """

    avg_frac: float
    min_frac: float
    max_frac: float
    ko_chance: float


def _normalize_type_name(value: object) -> str:
    if value is None:
        return ""
    raw = getattr(value, "name", value)
    return str(raw).lower().strip()


def _pokemon_types_from_poke(mon: object) -> List[str]:
    types = getattr(mon, "types", None) or []
    out: List[str] = []
    for t in types:
        name = _normalize_type_name(t)
        if name:
            out.append(name)
    return out


def _move_token(move: object) -> str:
    mid = getattr(move, "id", None) or getattr(move, "name", None) or ""
    return re.sub(r"[^a-z0-9]", "", str(mid).lower().strip())


def move_lowers_own_stats(move: object) -> bool:
    """True when the move is known to drop the user's stats."""
    token = _move_token(move)
    if token in _SELF_STAT_DROP_MOVES:
        return True
    self_boost = getattr(move, "self_boost", None) or {}
    if isinstance(self_boost, dict) and any(int(v) < 0 for v in self_boost.values()):
        return True
    self_target = getattr(move, "target", None)
    target_name = getattr(self_target, "name", self_target)
    if str(target_name).lower() in ("self", "ally", "allies"):
        boost = getattr(move, "boosts", None) or {}
        if isinstance(boost, dict) and any(int(v) < 0 for v in boost.values()):
            return True
    return False


def move_effective_accuracy(move: object) -> float:
    """100 = always hits; lower values are percent accuracy."""
    acc = getattr(move, "accuracy", None)
    if acc is True:
        return 100.0
    if acc is None:
        attrs = _move_attributes(move)
        return 100.0 if attrs is not None else 0.0
    try:
        return float(acc)
    except (TypeError, ValueError):
        return 100.0


def is_clean_damaging_move(move: object) -> bool:
    """Damaging move with full accuracy and no self stat drop."""
    if _move_attributes(move) is None:
        return False
    if move_lowers_own_stats(move):
        return False
    return move_effective_accuracy(move) >= 100.0


def can_guaranteed_ko_on_hit(
    move: object,
    my_mon: object,
    enemy_mon: object,
    *,
    my_tera_type: Optional[str] = None,
) -> bool:
    """True if the worst damage roll still KOs on hit."""
    if enemy_mon is None or my_mon is None:
        return False
    est = estimate_my_damage_on_enemy(
        move, my_mon, enemy_mon, my_tera_type=my_tera_type
    )
    if est is None:
        return False
    enemy_hp = float(getattr(enemy_mon, "current_hp_fraction", None) or 1.0)
    return est.min_frac >= enemy_hp


def find_clean_guaranteed_ko_move(
    moves: Iterable[object],
    my_mon: object,
    enemy_mon: Optional[object],
    *,
    my_tera_type: Optional[str] = None,
) -> Optional[object]:
    """Best fully-accurate, no-stat-drop move that guaranteed KOs on hit."""
    if enemy_mon is None or my_mon is None:
        return None
    best_move: Optional[object] = None
    best_key = (-1.0, -1.0)
    for move in moves:
        if not is_clean_damaging_move(move):
            continue
        if not can_guaranteed_ko_on_hit(
            move, my_mon, enemy_mon, my_tera_type=my_tera_type
        ):
            continue
        attrs = _move_attributes(move)
        if attrs is None:
            continue
        mv_type, base_power, _category = attrs
        my_types = (
            [_normalize_type_name(my_tera_type)]
            if my_tera_type
            else _pokemon_types_from_poke(my_mon)
        )
        enemy_types = _pokemon_types_from_poke(enemy_mon)
        raw = _score_attack(mv_type, base_power, my_types, enemy_types)
        key = (move_effective_accuracy(move), raw)
        if key > best_key:
            best_key = key
            best_move = move
    return best_move


def _damaging_moves_from_mon(mon: object) -> List[object]:
    moves = getattr(mon, "moves", None) or {}
    if isinstance(moves, dict):
        return [
            mv
            for mv in moves.values()
            if mv is not None and _move_attributes(mv) is not None
        ]
    return []


def ko_potential_score(
    moves: Iterable[object],
    attacker: object,
    defender: object,
) -> float:
    """Higher = more likely to KO on switch-in (3.0 = clean guaranteed OHKO)."""
    if defender is None or attacker is None:
        return 0.0
    enemy_hp = float(getattr(defender, "current_hp_fraction", None) or 1.0)
    best = 0.0
    for move in moves:
        if is_clean_damaging_move(move) and can_guaranteed_ko_on_hit(
            move, attacker, defender
        ):
            return 3.0
        est = estimate_my_damage_on_enemy(move, attacker, defender)
        if est is None:
            continue
        if est.min_frac >= enemy_hp:
            best = max(best, 2.6)
        elif est.max_frac >= enemy_hp:
            best = max(best, 2.0 + 0.35 * est.ko_chance)
        elif est.max_frac >= enemy_hp * 0.72:
            best = max(best, 1.35 + 0.45 * est.ko_chance)
        else:
            best = max(best, est.ko_chance * 0.55)
    return best


def score_fast_revenge_switch(
    bench: object,
    enemy_mon: object,
    *,
    active_mon: Optional[object] = None,
    require_faster_than_active: bool = False,
) -> float:
    """Score a bench mon: outspeeds foe + KO potential (+ bonus vs active speed)."""
    if enemy_mon is None or getattr(bench, "fainted", False):
        return -1.0
    spe_bench = estimate_pokemon_speed(bench)
    spe_enemy = estimate_pokemon_speed(enemy_mon)
    if spe_bench <= spe_enemy:
        return -1.0
    if require_faster_than_active and active_mon is not None:
        if spe_bench <= estimate_pokemon_speed(active_mon):
            return -1.0
    ko = ko_potential_score(_damaging_moves_from_mon(bench), bench, enemy_mon)
    if ko < 0.75:
        return -1.0
    speed_margin = (spe_bench - spe_enemy) / max(spe_enemy, 1.0)
    score = ko * (1.0 + 0.4 * min(1.5, speed_margin))
    if active_mon is not None:
        spe_active = estimate_pokemon_speed(active_mon)
        if spe_bench > spe_active:
            active_margin = (spe_bench - spe_active) / max(spe_active, 1.0)
            score *= 1.0 + 0.2 * min(1.0, active_margin)
    return score


def pick_best_fast_revenge_switch(
    switches: Iterable[object],
    enemy_mon: object,
    *,
    active_mon: Optional[object] = None,
    require_faster_than_active: bool = False,
    min_score: float = 1.0,
) -> Optional[object]:
    """Best bench Pokémon that is faster and can likely KO."""
    best: Optional[object] = None
    best_score = float(min_score)
    for mon in switches:
        if getattr(mon, "fainted", False):
            continue
        sc = score_fast_revenge_switch(
            mon,
            enemy_mon,
            active_mon=active_mon,
            require_faster_than_active=require_faster_than_active,
        )
        if sc > best_score:
            best_score = sc
            best = mon
    return best


def is_fast_clean_revenge_switch(bench: object, enemy_mon: object) -> bool:
    """Bench mon outspeeds the foe and has a guaranteed clean OHKO on switch-in."""
    return score_fast_revenge_switch(bench, enemy_mon) >= 2.9


def is_fast_revenge_switch(
    bench: object,
    enemy_mon: object,
    *,
    active_mon: Optional[object] = None,
    min_score: float = 1.15,
) -> bool:
    """Faster than foe with meaningful KO potential on switch-in."""
    return (
        score_fast_revenge_switch(bench, enemy_mon, active_mon=active_mon)
        >= min_score
    )


def find_fastest_revenge_killer(
    switches: Iterable[object],
    enemy_mon: object,
    *,
    active_mon: Optional[object] = None,
    require_faster_than_active: bool = False,
) -> Optional[object]:
    """Bench Pokémon that outspeeds the foe and has the best revenge-KO score."""
    return pick_best_fast_revenge_switch(
        switches,
        enemy_mon,
        active_mon=active_mon,
        require_faster_than_active=require_faster_than_active,
        min_score=1.0,
    )


def _move_attributes(move: object) -> Optional[tuple]:
    """Return ``(move_type, base_power, category)`` from a poke-env Move.

    Returns ``None`` for status moves or moves without metadata.
    """
    if move is None:
        return None
    base_power = int(getattr(move, "base_power", 0) or 0)
    if base_power <= 0:
        return None
    category_obj = getattr(move, "category", None)
    category = _normalize_type_name(category_obj).capitalize() or "Status"
    if category not in ("Physical", "Special"):
        return None
    mv_type = _normalize_type_name(getattr(move, "type", None))
    if not mv_type:
        return None
    return mv_type, base_power, category


def _score_to_estimate(score: float, defender_hp: float) -> DamageEstimate:
    avg = min(1.5, max(0.0, score * _SCORE_TO_FRAC))
    # Reasonable spread mimicking the 0.85-1.00 damage roll.
    minf = avg * 0.85
    maxf = avg * 1.0
    if defender_hp <= 0:
        ko = 1.0
    elif maxf >= defender_hp:
        ko = min(1.0, maxf / max(defender_hp, 0.01))
    else:
        ko = 0.0
    return DamageEstimate(avg_frac=avg, min_frac=minf, max_frac=maxf, ko_chance=ko)


def _score_attack(
    move_type: str,
    base_power: int,
    attacker_types: List[str],
    defender_types: List[str],
) -> float:
    eff = float(type_effectiveness(move_type, defender_types))
    stab = 1.5 if move_type in attacker_types else 1.0
    return float(base_power) * eff * stab


# ---------------------------------------------------------------------------
# Public API (preserved for back-compat with the hybrid/model players)
# ---------------------------------------------------------------------------


def estimate_my_damage_on_enemy(
    move: object,
    my_mon: object,
    enemy_mon: object,
    *,
    my_tera_type: Optional[str] = None,
) -> Optional[DamageEstimate]:
    attrs = _move_attributes(move)
    if attrs is None:
        return None
    mv_type, base_power, _category = attrs
    if my_tera_type:
        attacker_types = [_normalize_type_name(my_tera_type)]
    else:
        attacker_types = _pokemon_types_from_poke(my_mon)
    defender_types = _pokemon_types_from_poke(enemy_mon)
    score = _score_attack(mv_type, base_power, attacker_types, defender_types)
    enemy_hp = float(getattr(enemy_mon, "current_hp_fraction", None) or 1.0)
    return _score_to_estimate(score, enemy_hp)


def estimate_enemy_worst_damage(
    enemy_mon: object,
    my_mon: object,
    *,
    my_tera_type: Optional[str] = None,
) -> float:
    """Return the worst expected HP fraction the enemy can take from us.

    Uses the enemy's revealed damaging moves if any; otherwise falls back to
    STAB attacks on both physical and special sides at 80 BP.
    """
    if my_tera_type:
        my_types = [_normalize_type_name(my_tera_type)]
    else:
        my_types = _pokemon_types_from_poke(my_mon)
    enemy_types = _pokemon_types_from_poke(enemy_mon)

    worst = 0.0
    seen_damaging = False
    for mv in (getattr(enemy_mon, "moves", None) or {}).values():
        attrs = _move_attributes(mv)
        if attrs is None:
            continue
        seen_damaging = True
        mv_type, base_power, _category = attrs
        score = _score_attack(mv_type, base_power, enemy_types, my_types)
        worst = max(worst, score * _SCORE_TO_FRAC)

    if seen_damaging:
        return min(1.5, worst)

    for t in enemy_types:
        score = _score_attack(t, 80, enemy_types, my_types)
        worst = max(worst, score * _SCORE_TO_FRAC)
    return min(1.5, worst)


def should_terastallize(
    move: object,
    my_mon: object,
    enemy_mon: Optional[object],
) -> bool:
    """Tera decision purely from type effectiveness (no damage calc).

    Offensive trigger: the chosen move gets a fresh STAB after Tera and lands
    at least neutral on the target.
    Defensive trigger: a current ×2/×4 weakness on enemy's seen moves drops
    to ≤ ×1 after the Tera type takes over.
    """
    if my_mon is None or enemy_mon is None:
        return False
    tera_type = _normalize_type_name(getattr(my_mon, "tera_type", None))
    if not tera_type:
        return False

    move_type = _normalize_type_name(getattr(move, "type", None))
    my_types = _pokemon_types_from_poke(my_mon)
    enemy_types = _pokemon_types_from_poke(enemy_mon)

    if move_type and move_type == tera_type and move_type not in my_types:
        eff = float(type_effectiveness(move_type, enemy_types))
        if eff >= 1.0:
            return True

    worst_before, worst_after = 1.0, 1.0
    for mv in (getattr(enemy_mon, "moves", None) or {}).values():
        attrs = _move_attributes(mv)
        if attrs is None:
            continue
        atk_type, _bp, _cat = attrs
        worst_before = max(worst_before, float(type_effectiveness(atk_type, my_types)))
        worst_after = max(worst_after, float(type_effectiveness(atk_type, [tera_type])))
    if worst_before >= 2.0 and worst_after <= 1.0:
        return True
    return False


def best_damaging_move(
    moves: Iterable[object],
    my_mon: object,
    enemy_mon: Optional[object],
) -> Optional[object]:
    """Return the move that maximises ``base_power * effectiveness * STAB``.

    Boosted further when a KO looks likely (max roll covers the enemy HP).
    Returns ``None`` if no damaging move is available.
    """
    if enemy_mon is None:
        return None
    enemy_hp = float(getattr(enemy_mon, "current_hp_fraction", None) or 1.0)
    my_types = _pokemon_types_from_poke(my_mon)
    enemy_types = _pokemon_types_from_poke(enemy_mon)

    best_move: Optional[object] = None
    best_score = -1.0
    for move in moves:
        attrs = _move_attributes(move)
        if attrs is None:
            continue
        mv_type, base_power, _category = attrs
        raw = _score_attack(mv_type, base_power, my_types, enemy_types)
        frac = raw * _SCORE_TO_FRAC
        score = frac
        if frac >= enemy_hp:
            score += 1.0  # likely KO bonus
        if frac * 0.85 >= enemy_hp:
            score += 0.5  # min-roll KO bonus
        if score > best_score:
            best_score = score
            best_move = move
    return best_move
