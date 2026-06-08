"""
Simulation d'un tour (profondeur 1) : notre action puis branches adverses.

Cette simulation produit un ``state_dict`` mis à jour avec uniquement les
transitions structurelles (last_move, moves_seen, turn, switch actif).
Aucune mise à jour de HP : pour une simulation exacte des dégâts, utiliser
le moteur Showdown via ``common/pkmn_engine_simulator.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from common.set_dex_prior import aggregate_move_probabilities, enumerate_likely_moves
from random_battle.models.IA_multihead_predictor import normalize_token

TurnAggregation = Literal["min", "mean", "weighted_mean"]
OpponentAction = Tuple[str, str]  # ("move", token) | ("switch", species)


@dataclass(frozen=True)
class TurnBranch:
    action: OpponentAction
    weight: float


def _active_mon(state: Dict[str, object], *, opponent: bool) -> Optional[Dict[str, object]]:
    key = "opp_team" if opponent else "my_team"
    for mon in state.get(key) or []:
        if mon and mon.get("is_active"):
            return mon
    return None


def opponent_moves_for_state(
    state: Dict[str, object],
    *,
    set_dex: Optional[Dict[str, object]],
    vocab: Dict[str, int],
    top_k: int = 6,
    min_prob: float = 0.03,
    mismatch_penalty: float = 0.15,
) -> List[str]:
    opp = _active_mon(state, opponent=True)
    if not opp:
        return []
    species = str(opp.get("species") or "")
    if not species or species == "unknown":
        seen = [normalize_token(m) for m in (opp.get("moves_seen") or []) if normalize_token(m)]
        return seen[:top_k]
    return enumerate_likely_moves(
        set_dex,
        species,
        list(opp.get("moves_seen") or []),
        vocab,
        top_k=top_k,
        min_prob=min_prob,
        mismatch_penalty=mismatch_penalty,
    )


def enumerate_opponent_switches(
    state: Dict[str, object],
    *,
    top_k: int = 2,
) -> List[str]:
    """Espèces du banc adverse révélé, non KO."""
    active = _active_mon(state, opponent=True)
    active_species = str((active or {}).get("species") or "")
    candidates: List[str] = []
    for mon in state.get("opp_team") or []:
        if not mon or mon.get("is_active"):
            continue
        if mon.get("fainted"):
            continue
        species = str(mon.get("species") or "")
        if not species or species == "unknown":
            continue
        if not mon.get("revealed") and float(mon.get("hp_percent", -1)) < 0:
            continue
        if species == active_species:
            continue
        if species not in candidates:
            candidates.append(species)
    return candidates[:top_k]


def enumerate_my_switches(state: Dict[str, object], *, top_k: int = 3) -> List[str]:
    active = _active_mon(state, opponent=False)
    active_species = str((active or {}).get("species") or "")
    out: List[str] = []
    for mon in state.get("my_team") or []:
        if not mon or mon.get("is_active") or mon.get("fainted"):
            continue
        species = str(mon.get("species") or "")
        if species and species != active_species and species not in out:
            out.append(species)
    return out[:top_k]


def state_after_my_move(
    state: Dict[str, object],
    move_token: str,
    *,
    my_last_move: Optional[str] = None,
) -> Dict[str, object]:
    """Update ``my_last_move`` after our move (no damage applied)."""
    next_state = copy.deepcopy(state)
    token = normalize_token(my_last_move or move_token)
    next_state["my_last_move"] = token
    return next_state


def state_after_opponent_move(
    state: Dict[str, object],
    move_token: str,
    *,
    increment_turn: bool = True,
) -> Dict[str, object]:
    """Update ``opp_last_move`` + ``moves_seen`` (no damage applied)."""
    next_state = copy.deepcopy(state)
    token = normalize_token(move_token)
    next_state["opp_last_move"] = token

    opp_team: List[Dict[str, object]] = []
    for mon in next_state.get("opp_team") or []:
        entry = dict(mon)
        if entry.get("is_active") and token:
            seen = list(entry.get("moves_seen") or [])
            if token not in seen:
                seen.append(token)
            entry["moves_seen"] = seen
            if not entry.get("revealed"):
                entry["revealed"] = True
        opp_team.append(entry)
    next_state["opp_team"] = opp_team

    if increment_turn:
        next_state["turn"] = int(next_state.get("turn", 0) or 0) + 1
    return next_state


def _set_active_species(
    state: Dict[str, object], team_key: str, species: str
) -> Dict[str, object]:
    next_state = copy.deepcopy(state)
    norm_target = normalize_token(species)
    team: List[Dict[str, object]] = []
    for mon in next_state.get(team_key) or []:
        entry = dict(mon)
        entry["is_active"] = normalize_token(str(entry.get("species") or "")) == norm_target
        team.append(entry)
    next_state[team_key] = team
    return next_state


def state_after_my_switch(state: Dict[str, object], species: str) -> Dict[str, object]:
    next_state = _set_active_species(state, "my_team", species)
    next_state["my_last_move"] = None
    return next_state


def state_after_opponent_switch(
    state: Dict[str, object], species: str, *, increment_turn: bool = True
) -> Dict[str, object]:
    next_state = _set_active_species(state, "opp_team", species)
    next_state["opp_last_move"] = None
    if increment_turn:
        next_state["turn"] = int(next_state.get("turn", 0) or 0) + 1
    return next_state


def opponent_turn_branches(
    state: Dict[str, object],
    *,
    set_dex: Optional[Dict[str, object]],
    vocab: Dict[str, int],
    opp_top_k: int = 6,
    opp_min_prob: float = 0.03,
    opp_switch_top_k: int = 2,
    mismatch_penalty: float = 0.15,
    switch_weight: float = 0.25,
) -> List[TurnBranch]:
    moves = opponent_moves_for_state(
        state,
        set_dex=set_dex,
        vocab=vocab,
        top_k=opp_top_k,
        min_prob=opp_min_prob,
        mismatch_penalty=mismatch_penalty,
    )
    opp = _active_mon(state, opponent=True) or {}
    probs_map = (
        aggregate_move_probabilities(
            set_dex,
            str(opp.get("species") or ""),
            list(opp.get("moves_seen") or []),
            mismatch_penalty,
        )
        if set_dex
        else {}
    )

    branches: List[TurnBranch] = []
    for mv in moves:
        w = max(probs_map.get(mv, 1.0 / max(len(moves), 1)), 1e-6)
        branches.append(TurnBranch(action=("move", mv), weight=w))

    for species in enumerate_opponent_switches(state, top_k=opp_switch_top_k):
        branches.append(TurnBranch(action=("switch", species), weight=switch_weight))

    return branches


def apply_opponent_branch(
    state: Dict[str, object],
    branch: TurnBranch,
) -> Dict[str, object]:
    kind, target = branch.action
    if kind == "move":
        return state_after_opponent_move(state, target)
    return state_after_opponent_switch(state, target)


def aggregate_turn_values(
    values: Sequence[float],
    weights: Sequence[float],
    mode: TurnAggregation,
) -> float:
    if not values:
        return 0.5
    if mode == "min":
        return float(min(values))
    if mode == "weighted_mean":
        wsum = sum(weights) or 1.0
        return float(sum(v * w for v, w in zip(values, weights)) / wsum)
    return float(sum(values) / len(values))


def win_prob_after_one_ply_turn(
    predict_state,
    state: Dict[str, object],
    my_move_token: str,
    *,
    set_dex: Optional[Dict[str, object]],
    vocab: Dict[str, int],
    opp_top_k: int = 6,
    opp_min_prob: float = 0.03,
    opp_switch_top_k: int = 2,
    mismatch_penalty: float = 0.15,
    aggregation: TurnAggregation = "min",
) -> float:
    """Our move -> opponent branches (moves + switches) -> aggregated P(win)."""
    after_my = state_after_my_move(state, my_move_token)
    branches = opponent_turn_branches(
        after_my,
        set_dex=set_dex,
        vocab=vocab,
        opp_top_k=opp_top_k,
        opp_min_prob=opp_min_prob,
        opp_switch_top_k=opp_switch_top_k,
        mismatch_penalty=mismatch_penalty,
    )
    if not branches:
        return float(predict_state(after_my))

    values: List[float] = []
    weights: List[float] = []
    for br in branches:
        after = apply_opponent_branch(after_my, br)
        values.append(float(predict_state(after)))
        weights.append(br.weight)
    return aggregate_turn_values(values, weights, aggregation)


def win_prob_after_my_switch_one_ply(
    predict_state,
    state: Dict[str, object],
    switch_species: str,
    *,
    set_dex: Optional[Dict[str, object]],
    vocab: Dict[str, int],
    opp_top_k: int = 6,
    opp_min_prob: float = 0.03,
    opp_switch_top_k: int = 2,
    mismatch_penalty: float = 0.15,
    aggregation: TurnAggregation = "min",
) -> float:
    after_switch = state_after_my_switch(state, switch_species)
    branches = opponent_turn_branches(
        after_switch,
        set_dex=set_dex,
        vocab=vocab,
        opp_top_k=opp_top_k,
        opp_min_prob=opp_min_prob,
        opp_switch_top_k=opp_switch_top_k,
        mismatch_penalty=mismatch_penalty,
    )
    if not branches:
        return float(predict_state(after_switch))

    values: List[float] = []
    weights: List[float] = []
    for br in branches:
        after = apply_opponent_branch(after_switch, br)
        values.append(float(predict_state(after)))
        weights.append(br.weight)
    return aggregate_turn_values(values, weights, aggregation)
