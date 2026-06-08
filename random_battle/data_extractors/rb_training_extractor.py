"""
Random Battle training data extractor with partial observability.

Uses pre-action snapshots and team preview slot order for switch labels.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_RB_EXTRACTORS = _SCRIPT_DIR
_SHARED_EXTRACTORS = _PROJECT_ROOT / "shared" / "data_extractors"
for _path in (_PROJECT_ROOT, _RB_EXTRACTORS, _SHARED_EXTRACTORS):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from extract_pokechamp_training_data import (  # noqa: E402
    TrainingDataExtractor,
    TrainingExample,
)
from rb_team_slots import (  # noqa: E402
    MAX_TEAM_SIZE,
    normalize_species_key,
    normalize_switch_target,
    parse_team_preview_order,
)
from rb_visibility import (  # noqa: E402
    build_visible_game_state,
    strip_future_move_from_active,
)

PreActionSnapshot = Tuple[int, str, str, str, Dict]


class RandomBattleTrainingExtractor(TrainingDataExtractor):
    """Extract RB examples with per-action visibility masking."""

    def reset_state(self) -> None:
        super().reset_state()
        self.team_preview_order: Dict[str, List[str]] = {"p1": [], "p2": []}
        self.running_preview_order: Dict[str, List[str]] = {"p1": [], "p2": []}

    def extract_from_log(
        self,
        log_text: str,
        inputlog: Optional[str] = None,
        prefer_inputlog: bool = True,
    ) -> List[TrainingExample]:
        self.reset_state()
        if not log_text:
            return []

        log_lines = log_text.strip().split("\n")
        self._extract_winner(log_lines)
        self.team_preview_order = parse_team_preview_order(log_lines)
        self.running_preview_order = {
            "p1": list(self.team_preview_order["p1"]),
            "p2": list(self.team_preview_order["p2"]),
        }

        snapshots: List[PreActionSnapshot] = []
        self._parse_log_to_states(log_lines, pre_action_snapshots=snapshots)
        self._attach_preview_orders_to_snapshots(snapshots)

        actions = self._load_actions(log_lines, inputlog, prefer_inputlog)
        return self._create_examples_from_snapshots(snapshots, actions)

    def extract_from_pokechamp_battle(self, battle: Dict) -> List[TrainingExample]:
        log_text = battle.get("text", "")
        inputlog = battle.get("inputlog")
        return self.extract_from_log(
            str(log_text),
            inputlog=str(inputlog) if inputlog else None,
            prefer_inputlog=bool(inputlog),
        )

    def _register_species_in_order(self, player: str, species: Optional[str]) -> None:
        if not species or player not in self.running_preview_order:
            return
        order = self.running_preview_order[player]
        norm = normalize_species_key(species)
        if not norm:
            return
        if any(normalize_species_key(s) == norm for s in order):
            return
        if len(order) < MAX_TEAM_SIZE:
            order.append(species)

    def _attach_preview_orders_to_snapshots(self, snapshots: List[PreActionSnapshot]) -> None:
        for _turn, player, _atype, _target, raw_state in snapshots:
            for pid, order in self.team_preview_order.items():
                if order:
                    raw_state[pid]["preview_order"] = list(order)
            active = raw_state.get(player, {}).get("active")
            self._register_species_in_order(player, active)

    def _load_actions(
        self,
        log_lines: List[str],
        inputlog: Optional[str],
        prefer_inputlog: bool,
    ) -> List[Tuple[int, str, str, str]]:
        if prefer_inputlog and inputlog:
            from extract_action_chunks import parse_inputlog_actions

            actions = parse_inputlog_actions(inputlog)
            if actions:
                return actions
        return self._infer_actions_from_log(log_lines)

    def _create_examples_from_snapshots(
        self,
        snapshots: List[PreActionSnapshot],
        log_actions: List[Tuple[int, str, str, str]],
    ) -> List[TrainingExample]:
        examples: List[TrainingExample] = []
        forced_switches = self._detect_forced_switches(snapshots)

        log_index = 0
        for turn, player, action_type, action_target, raw_state in snapshots:
            if log_index < len(log_actions):
                log_turn, log_player, log_type, log_target = log_actions[log_index]
                if (log_turn, log_player, log_type) == (turn, player, action_type):
                    action_target = log_target
                    log_index += 1

            player_state = raw_state.get(player, {})
            opponent = "p2" if player == "p1" else "p1"
            opp_state = raw_state.get(opponent, {})
            if not player_state.get("active") or not opp_state.get("active"):
                continue

            if action_type == "switch":
                self._register_species_in_order(player, action_target)

            preview_order = list(self.running_preview_order.get(player, []))

            if action_type == "switch":
                slot_str = normalize_switch_target(
                    action_target,
                    preview_order,
                    active_species=player_state.get("active"),
                )
                if slot_str is None:
                    continue
                action_target = slot_str

            is_voluntary = (turn, player) not in forced_switches
            visible = build_visible_game_state(
                raw_state,
                player=player,
                preview_orders={
                    "p1": list(self.running_preview_order["p1"]),
                    "p2": list(self.running_preview_order["p2"]),
                },
            )
            visible = strip_future_move_from_active(visible, action_type, action_target)

            examples.append(
                TrainingExample(
                    state=visible,
                    action_type=action_type,
                    action_target=action_target,
                    is_voluntary=is_voluntary,
                    winner=(self.winner == player),
                )
            )

        return examples

    def _detect_forced_switches(
        self, snapshots: List[PreActionSnapshot]
    ) -> Set[Tuple[int, str]]:
        forced: Set[Tuple[int, str]] = set()
        for turn, player, action_type, _, raw_state in snapshots:
            if action_type != "switch":
                continue
            player_state = raw_state.get(player, {})
            active = player_state.get("active")
            if not active:
                continue
            active_data = player_state.get("team", {}).get(active, {})
            if active_data.get("fainted") or active_data.get("hp_percent", 1.0) <= 0:
                forced.add((turn, player))
        return forced
