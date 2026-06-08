"""
Évaluation P(victoire) pour un état RB + recherche arborescente profondeur 1.

Label entraîné : le joueur `state["player"]` finit par gagner le combat (pas une vraie
value function Monte-Carlo), mais utile pour comparer des coups via états successeurs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.project_paths import setup_import_paths
from random_battle.config import MOVE_VOCAB_PATH, RB_SET_DEX_PATH, WINRATE_MODEL_DIR

setup_import_paths(shared_data=True, rb_data=True)

from common.set_dex_prior import load_set_dex
from random_battle.models.IA_multihead_predictor import Config, FeatureBuilder
from random_battle.players.turn_simulation import (
    TurnAggregation,
    opponent_moves_for_state,
    state_after_my_move,
    state_after_opponent_move,
    win_prob_after_one_ply_turn,
    win_prob_after_my_switch_one_ply,
)


@dataclass(frozen=True)
class ActionCandidate:
    """Action légale + état après coup (perspective du joueur actif)."""

    action_type: str  # "move" | "switch"
    action_target: str
    state_after: Dict[str, object]

    @property
    def label(self) -> str:
        return f"{self.action_type}:{self.action_target}"


@dataclass
class ScoredAction:
    candidate: ActionCandidate
    win_prob: float


class WinRateEvaluator:
    """Charge model.keras et prédit win_prob pour un state dict RB."""

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
        set_dex_path: Optional[Path] = None,
        mismatch_penalty: float = 0.15,
        slot_mode: str = "exact",
    ) -> None:
        model_path = Path(model_path or (WINRATE_MODEL_DIR / "model.keras"))
        vocab_path = Path(vocab_path or MOVE_VOCAB_PATH)
        set_dex_path = Path(set_dex_path or RB_SET_DEX_PATH)

        if not model_path.is_file():
            raise FileNotFoundError(f"Win-rate model not found: {model_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Move vocab not found: {vocab_path}")

        self.model = tf.keras.models.load_model(model_path)
        vocab = {k: int(v) for k, v in json.loads(vocab_path.read_text(encoding="utf-8")).items()}
        set_dex = load_set_dex(str(set_dex_path)) if set_dex_path.is_file() else None
        self.builder = FeatureBuilder(
            Config(),
            vocab,
            set_dex,
            mismatch_penalty=mismatch_penalty,
            slot_mode=slot_mode,
        )

    def _example_from_state(self, state: Dict[str, object]) -> Dict[str, object]:
        return {
            "state": state,
            "action_type": "move",
            "action_target": "",
            "winner": False,
            "is_voluntary": True,
        }

    def predict_state(self, state: Dict[str, object]) -> float:
        """P(victoire) pour state['player'] (0..1)."""
        built = self.builder.build_example(self._example_from_state(state))
        if built is None:
            return 0.5
        features, _, _ = built
        batch = {k: np.expand_dims(v, 0) for k, v in features.items()}
        out = self.model.predict(batch, verbose=0)
        return float(out["win_prob"][0][0])

    def score_candidates(
        self, candidates: Sequence[ActionCandidate]
    ) -> List[ScoredAction]:
        scored = [
            ScoredAction(candidate=c, win_prob=self.predict_state(c.state_after))
            for c in candidates
        ]
        scored.sort(key=lambda s: s.win_prob, reverse=True)
        return scored

    def best_action_depth1(
        self, candidates: Sequence[ActionCandidate]
    ) -> Optional[ScoredAction]:
        scored = self.score_candidates(candidates)
        return scored[0] if scored else None

    def predict_after_my_move(self, state: Dict[str, object], move_token: str) -> float:
        return self.predict_state(state_after_my_move(state, move_token))

    def predict_after_one_ply_turn(
        self,
        state: Dict[str, object],
        move_token: str,
        *,
        opp_top_k: int = 6,
        opp_min_prob: float = 0.03,
        opp_switch_top_k: int = 2,
        aggregation: TurnAggregation = "min",
    ) -> float:
        return win_prob_after_one_ply_turn(
            self.predict_state,
            state,
            move_token,
            set_dex=self.builder.set_dex,
            vocab=self.builder.vocab,
            opp_top_k=opp_top_k,
            opp_min_prob=opp_min_prob,
            opp_switch_top_k=opp_switch_top_k,
            mismatch_penalty=self.builder.mismatch_penalty,
            aggregation=aggregation,
        )

    def predict_after_my_switch_one_ply(
        self,
        state: Dict[str, object],
        switch_species: str,
        *,
        opp_top_k: int = 6,
        opp_min_prob: float = 0.03,
        opp_switch_top_k: int = 2,
        aggregation: TurnAggregation = "min",
    ) -> float:
        return win_prob_after_my_switch_one_ply(
            self.predict_state,
            state,
            switch_species,
            set_dex=self.builder.set_dex,
            vocab=self.builder.vocab,
            opp_top_k=opp_top_k,
            opp_min_prob=opp_min_prob,
            opp_switch_top_k=opp_switch_top_k,
            mismatch_penalty=self.builder.mismatch_penalty,
            aggregation=aggregation,
        )

    def list_opponent_branch_moves(
        self, state: Dict[str, object], *, opp_top_k: int = 6, opp_min_prob: float = 0.03
    ) -> List[str]:
        return opponent_moves_for_state(
            state,
            set_dex=self.builder.set_dex,
            vocab=self.builder.vocab,
            top_k=opp_top_k,
            min_prob=opp_min_prob,
            mismatch_penalty=self.builder.mismatch_penalty,
        )


def state_after_move(
    state: Dict[str, object],
    move_token: str,
    *,
    my_last_move: Optional[str] = None,
) -> Dict[str, object]:
    """Alias rétrocompatible → state_after_my_move."""
    return state_after_my_move(state, move_token, my_last_move=my_last_move)


def state_after_switch(
    state: Dict[str, object],
    slot_index: int,
    *,
    my_team_slot_order: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Active le Pokémon du slot preview (0-5) côté joueur."""
    import copy

    order = my_team_slot_order or list(state.get("my_team_slot_order") or [])
    if not order or slot_index < 0 or slot_index >= len(order):
        return copy.deepcopy(state)

    target = order[slot_index]
    next_state = copy.deepcopy(state)
    my_team = []
    for mon in next_state.get("my_team") or []:
        m = dict(mon)
        m["is_active"] = (m.get("species") == target)
        my_team.append(m)
    next_state["my_team"] = my_team
    next_state["my_last_move"] = None
    return next_state
