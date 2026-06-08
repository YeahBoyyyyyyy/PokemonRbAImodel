"""High-level interface over the Node sim bridge.

Combines ``showdown_set_builder`` (poke-env battle -> Showdown sets) and
``pkmn_bridge.PkmnBridge`` (Showdown engine call) so that the rest of the
codebase only needs to talk to a single object.

Typical usage from the AI:

    sim = EngineSimulator(set_dex=SET_DEX)
    snap = sim.setup_from_battle(battle)
    # Apply our chosen attack and the opponent's predicted attack.
    snap2 = sim.apply_choices(
        snap,
        my_choice=sim.choice_for_move(snap, "p1", "thunderbolt"),
        opp_choice=sim.choice_for_move(snap, "p2", "softboiled"),
    )
    win = sim.estimate_win_probability(snap2)

The same EngineSimulator instance can be reused for many simulations,
each on a forked snapshot (the bridge keeps no per-call state).
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pkmn_bridge import PkmnBridge, teampreview_choice
from showdown_set_builder import (
    PLACEHOLDER_SET,
    SetBuildConfig,
    build_battle_teams,
    build_battle_teams_variants,
)
from battle_state_heuristic import score_from_engine_snap
from engine_state_overlay import apply_state_overlay, build_state_overlay

# ---------------------------------------------------------------------------
# Snapshot wrapper (thin)
# ---------------------------------------------------------------------------

EngineSnapshot = Dict[str, Any]


def _slug(value: Optional[object]) -> str:
    if value is None:
        return ""
    raw = getattr(value, "name", value)
    return re.sub(r"[^a-z0-9]", "", str(raw).lower().strip())


# ---------------------------------------------------------------------------
# Helpers to read the bridge response
# ---------------------------------------------------------------------------


def _side_pokemon(snap: EngineSnapshot, side: str) -> List[Dict[str, Any]]:
    request = ((snap.get("requests") or {}).get(side) or {})
    side_data = request.get("side") or {}
    return list(side_data.get("pokemon", []))


def _active_pokemon(snap: EngineSnapshot, side: str) -> Optional[Dict[str, Any]]:
    for poke in _side_pokemon(snap, side):
        if poke.get("active"):
            return poke
    return None


def _parse_condition(condition: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Parse Showdown 'cur/max status' string. Returns (cur, max, status_token).

    Examples: '196/196', '0 fnt', '342/539 brn', '74/100' (percent only).
    """
    if not condition:
        return None, None, None
    cleaned = condition.strip()
    if not cleaned:
        return None, None, None
    if cleaned.endswith(" fnt"):
        return 0, None, "fnt"
    status: Optional[str] = None
    parts = cleaned.split(" ")
    main = parts[0]
    if len(parts) > 1:
        status = parts[1].strip() or None
    if "/" in main:
        try:
            cur_s, max_s = main.split("/", 1)
            return int(cur_s), int(max_s), status
        except ValueError:
            return None, None, status
    try:
        return int(main), None, status
    except ValueError:
        return None, None, status


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------


@dataclass
class EngineSimulatorConfig:
    format_id: str = "gen9customgame"
    p1_name: str = "Bot"
    p2_name: str = "Opponent"


class EngineSimulator:
    """High-level wrapper around the Node bridge and the set builder."""

    def __init__(
        self,
        bridge: Optional[PkmnBridge] = None,
        *,
        set_dex: Optional[Dict[str, Any]] = None,
        config: Optional[EngineSimulatorConfig] = None,
        set_build_config: Optional[SetBuildConfig] = None,
    ) -> None:
        self.bridge = bridge or PkmnBridge()
        self._owns_bridge = bridge is None
        self.set_dex = set_dex
        self.cfg = config or EngineSimulatorConfig()
        self.set_build_config = set_build_config or SetBuildConfig()

    def close(self) -> None:
        if self._owns_bridge:
            self.bridge.close()

    def __enter__(self) -> "EngineSimulator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Setup: battle -> initial snapshot (team preview resolved)
    # ------------------------------------------------------------------

    def setup_from_battle(
        self,
        battle: Any,
        *,
        seed: Optional[Sequence[int]] = None,
        opponent_config: Optional[SetBuildConfig] = None,
        propagate_state: bool = True,
    ) -> EngineSnapshot:
        """Initialize a battle from a poke-env Battle and return the snapshot
        ready to act (team preview is auto-resolved).

        When ``propagate_state`` is True (default), the live HP / status /
        boosts / hazards / weather / terrain are overlayed on the fresh
        engine state so the simulation starts from the *actual* in-game
        situation, not from a brand-new battle.
        """
        p1_team, p2_team = build_battle_teams(
            battle,
            set_dex=self.set_dex,
            config=opponent_config or self.set_build_config,
        )
        snap = self.setup_from_sets(p1_team=p1_team, p2_team=p2_team, seed=seed)
        if propagate_state:
            try:
                overlay = build_state_overlay(battle)
                snap = self._overlay_state(snap, overlay)
            except Exception:
                # If the overlay fails we'd rather simulate from a fresh
                # state than crash the whole AI.
                pass
        return snap

    def _overlay_state(
        self,
        snap: EngineSnapshot,
        overlay: Dict[str, Any],
    ) -> EngineSnapshot:
        """Apply a state overlay and re-fetch requests so the engine view
        is consistent with the patched state.
        """
        state = snap.get("state")
        if not isinstance(state, dict):
            return snap
        apply_state_overlay(state, overlay)
        # Re-query the engine so requests / ended / winner match the
        # mutated state. This also catches inconsistencies early.
        refreshed = self.bridge.requests(state)
        # Preserve the (mutated) state we just sent; ``requests`` returns
        # a freshly serialized state of the same battle.
        return refreshed

    def setup_from_battle_variants(
        self,
        battle: Any,
        *,
        n_variants: int,
        seed: Optional[Sequence[int]] = None,
        base_seed: int = 0,
        opponent_config: Optional[SetBuildConfig] = None,
        propagate_state: bool = True,
    ) -> List[EngineSnapshot]:
        """Initialize ``n_variants`` snapshots, each with a different
        plausible opponent build. Variant 0 always matches
        :py:meth:`setup_from_battle` for stability.
        """
        variants = build_battle_teams_variants(
            battle,
            n_variants=n_variants,
            set_dex=self.set_dex,
            base_config=opponent_config or self.set_build_config,
            base_seed=base_seed,
        )
        snapshots: List[EngineSnapshot] = []
        for p1_team, p2_team in variants:
            snap = self.setup_from_sets(p1_team=p1_team, p2_team=p2_team, seed=seed)
            if propagate_state:
                try:
                    overlay = build_state_overlay(battle)
                    snap = self._overlay_state(snap, overlay)
                except Exception:
                    pass
            snapshots.append(snap)
        return snapshots

    def setup_from_sets(
        self,
        *,
        p1_team: Sequence[Dict[str, Any]],
        p2_team: Sequence[Dict[str, Any]],
        seed: Optional[Sequence[int]] = None,
    ) -> EngineSnapshot:
        """Initialize from raw Showdown sets and auto-resolve team preview."""
        snap = self.bridge.init(
            p1_team=list(p1_team),
            p2_team=list(p2_team),
            format_id=self.cfg.format_id,
            p1_name=self.cfg.p1_name,
            p2_name=self.cfg.p2_name,
            seed=seed,
        )
        # If we're in team preview, send the default ordering for both sides.
        p1_req = (snap.get("requests") or {}).get("p1") or {}
        p2_req = (snap.get("requests") or {}).get("p2") or {}
        if p1_req.get("teamPreview") or p2_req.get("teamPreview"):
            snap = self.bridge.step(
                snap["state"],
                p1_choice=teampreview_choice(len(p1_team)),
                p2_choice=teampreview_choice(len(p2_team)),
            )
        return snap

    # ------------------------------------------------------------------
    # Step + fork
    # ------------------------------------------------------------------

    @staticmethod
    def fork(snap: EngineSnapshot) -> EngineSnapshot:
        """Deep-copy a snapshot so the original can be reused for branching."""
        return deepcopy(snap)

    def apply_choices(
        self,
        snap: EngineSnapshot,
        *,
        my_choice: Optional[str],
        opp_choice: Optional[str],
        seed: Optional[Sequence[int]] = None,
        include_log: bool = False,
        auto_force_switch: bool = True,
        max_force_switch_iter: int = 4,
    ) -> EngineSnapshot:
        """Apply a (p1, p2) choice pair and (optionally) resolve any follow-up
        ``forceSwitch`` requests (post-pivot or post-KO) by picking the next
        legal switch on each side.
        """
        snap = self.bridge.step(
            snap["state"],
            p1_choice=my_choice,
            p2_choice=opp_choice,
            seed=seed,
            include_log=include_log,
        )
        if auto_force_switch:
            snap = self._resolve_force_switches(
                snap, max_iter=max_force_switch_iter, include_log=include_log
            )
        return snap

    def _resolve_force_switches(
        self,
        snap: EngineSnapshot,
        *,
        max_iter: int,
        include_log: bool,
    ) -> EngineSnapshot:
        """When p1 or p2 owes a switch (pivot or KO), pick one automatically.

        Iterates a few times in case both sides need to switch in sequence.
        """
        for _ in range(max(max_iter, 1)):
            if snap.get("ended"):
                return snap
            requests = snap.get("requests") or {}
            p1_req = requests.get("p1") or {}
            p2_req = requests.get("p2") or {}
            p1_needs = bool(p1_req.get("forceSwitch"))
            p2_needs = bool(p2_req.get("forceSwitch"))
            if not p1_needs and not p2_needs:
                return snap
            p1_choice = self._first_legal_switch(snap, "p1") if p1_needs else None
            p2_choice = self._first_legal_switch(snap, "p2") if p2_needs else None
            # If a side owes a switch but no legal slot exists, fall back to
            # "default" — the engine will pick the first legal option.
            if p1_needs and p1_choice is None:
                p1_choice = "default"
            if p2_needs and p2_choice is None:
                p2_choice = "default"
            snap = self.bridge.step(
                snap["state"],
                p1_choice=p1_choice,
                p2_choice=p2_choice,
                include_log=include_log,
            )
        return snap

    @staticmethod
    def _first_legal_switch(snap: EngineSnapshot, side: str) -> Optional[str]:
        """Return ``"switch N"`` for the first non-active, non-fainted slot."""
        for idx, poke in enumerate(_side_pokemon(snap, side), start=1):
            if poke.get("active"):
                continue
            cur, _, status = _parse_condition(poke.get("condition"))
            if status == "fnt" or cur == 0:
                continue
            return f"switch {idx}"
        return None

    # ------------------------------------------------------------------
    # Choice helpers (resolve move/switch tokens to Showdown choice strings)
    # ------------------------------------------------------------------

    def choice_for_move(
        self,
        snap: EngineSnapshot,
        side: str,
        move_token: str,
        *,
        terastallize: bool = False,
        fallback: str = "default",
    ) -> str:
        """Resolve a move id to a Showdown choice string like 'move 1'.

        If the move isn't in the active Pokemon's moveset, returns the
        ``fallback`` choice (default keeps the engine permissive).
        """
        idx = self._find_move_index(snap, side, move_token)
        if idx is None:
            return fallback
        suffix = " terastallize" if terastallize else ""
        return f"move {idx}{suffix}"

    def choice_for_switch(
        self,
        snap: EngineSnapshot,
        side: str,
        species_token: str,
        *,
        fallback: str = "default",
    ) -> str:
        idx = self._find_switch_index(snap, side, species_token)
        if idx is None:
            return fallback
        return f"switch {idx}"

    def _find_move_index(self, snap: EngineSnapshot, side: str, move_token: str) -> Optional[int]:
        target = _slug(move_token)
        if not target:
            return None
        request = ((snap.get("requests") or {}).get(side) or {})
        active = request.get("active")
        if not active:
            return None
        moves = active[0].get("moves", []) if active else []
        for idx, move in enumerate(moves, start=1):
            if _slug(move.get("id") or move.get("move")) == target:
                # Skip disabled moves (Showdown still includes them).
                if move.get("disabled"):
                    continue
                return idx
        return None

    def _find_switch_index(self, snap: EngineSnapshot, side: str, species_token: str) -> Optional[int]:
        target = _slug(species_token)
        if not target:
            return None
        for idx, poke in enumerate(_side_pokemon(snap, side), start=1):
            details = poke.get("details") or ""
            species = details.split(",", 1)[0]
            if _slug(species) == target and not poke.get("active"):
                if _parse_condition(poke.get("condition"))[2] == "fnt":
                    continue
                return idx
        return None

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    @staticmethod
    def active_hp_fraction(snap: EngineSnapshot, side: str) -> float:
        active = _active_pokemon(snap, side)
        if not active:
            return 0.0
        cur, mx, _status = _parse_condition(active.get("condition"))
        if cur is None or not mx:
            return 0.0 if cur == 0 else 1.0
        return max(0.0, min(1.0, cur / mx))

    @staticmethod
    def side_alive_count(snap: EngineSnapshot, side: str) -> int:
        return sum(
            1 for poke in _side_pokemon(snap, side)
            if _parse_condition(poke.get("condition"))[2] != "fnt"
        )

    @staticmethod
    def side_total_hp_fraction(snap: EngineSnapshot, side: str) -> float:
        team = _side_pokemon(snap, side)
        if not team:
            return 0.0
        total = 0.0
        for poke in team:
            cur, mx, status = _parse_condition(poke.get("condition"))
            if status == "fnt" or cur == 0:
                continue
            if cur is None or not mx:
                total += 1.0
            else:
                total += max(0.0, min(1.0, cur / mx))
        return total / len(team)

    @staticmethod
    def winner(snap: EngineSnapshot) -> Optional[str]:
        w = snap.get("winner")
        return w if isinstance(w, str) and w else None

    @staticmethod
    def ended(snap: EngineSnapshot) -> bool:
        return bool(snap.get("ended"))

    def estimate_win_probability(self, snap: EngineSnapshot) -> float:
        """Position heuristic: HP, alive, hazards, active HP (setup safety)."""
        return score_from_engine_snap(snap, p1_name=self.cfg.p1_name)


# ---------------------------------------------------------------------------
# Convenience: dump a snapshot to a JSON file (debug)
# ---------------------------------------------------------------------------


def dump_snapshot(snap: EngineSnapshot, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
