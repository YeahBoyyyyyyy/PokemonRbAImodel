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

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

_COMMON_DIR = Path(__file__).resolve().parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from materials import type_effectiveness  # noqa: E402

# Empirical scaling so that 80 BP * 1x effectiveness * 1.5 STAB ~= 0.6 (60% HP).
_SCORE_TO_FRAC = 0.005


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
