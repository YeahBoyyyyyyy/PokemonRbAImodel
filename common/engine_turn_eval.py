"""Engine-based 1-ply turn evaluation.

This module bridges the high-level :class:`EngineSimulator` and the AI's
decision loop. Given a current poke-env battle, a *single* action we want
to evaluate, and a list of plausible opponent responses, it runs the
actual Showdown simulator once per opponent branch and aggregates a
scalar score per action.

Terminal heuristic scores compare the leaf position to the search root
(``0.5 + scale * (pos_leaf - pos_root)``) so short lookaheads can
discriminate moves; internal pruning still uses absolute position scores.
Plug a model-based evaluator via ``model_evaluator`` if needed.

Typical usage from ``RbSearchPlayer``::

    eval = EngineTurnEvaluator(simulator)
    win_prob_move = eval.evaluate_my_move(
        battle, my_move="thunderbolt",
        opp_branches=[("move", "softboiled", 0.4),
                      ("move", "seismictoss", 0.4),
                      ("switch", "tyranitar", 0.2)],
        terastallize=False,
    )
    win_prob_switch = eval.evaluate_my_switch(
        battle, my_switch="garchomp",
        opp_branches=[...],
    )
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pkmn_engine_simulator import EngineSimulator, EngineSnapshot
from engine_snapshot_to_state import patch_state_from_engine

# (kind, target, weight). kind in {"move", "switch"}.
OpponentBranch = Tuple[str, str, float]


@dataclass
class TurnEvalResult:
    """Output of a single 1-ply evaluation."""

    win_prob: float
    branch_scores: List[Tuple[OpponentBranch, float]]
    setup_failed: bool = False


class EngineTurnEvaluator:
    """Wraps an EngineSimulator for 1-ply turn evaluations.

    Set ``n_opponent_worlds > 1`` to hedge against opponent set uncertainty:
    the evaluator will build N plausible opponent teams (varying role / item /
    ability / tera type / unknown moves for unrevealed slots) and average
    the win-probability across all of them. Variant 0 always uses the top-
    match build, so the first world is identical to a single-world eval.
    """

    def __init__(
        self,
        simulator: EngineSimulator,
        *,
        aggregation: str = "min",
        n_opponent_worlds: int = 1,
        world_aggregation: str = "mean",
        model_evaluator: Optional[Callable[[Dict[str, Any]], float]] = None,
        base_state_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        search_depth: int = 1,
        depth2_opp_top_k: int = 3,
        depth2_my_top_k: int = 3,
        deep_opp_move_cap: int = 4,
        opp_switch_base: int = 1,
        opp_switch_max: int = 3,
        opp_switch_adv_lo: float = 0.45,
        opp_switch_adv_hi: float = 0.62,
        prune_my_move_delta: float = 0.15,
        win_cutoff: float = 0.92,
        loss_cutoff: float = 0.08,
        relative_heuristic_scale: float = 25.0,
        min_switch_remaining_depth: int = 2,
        verbose: bool = False,
        progress_every_s: float = 5.0,
    ) -> None:
        """Args
        ----
        simulator: backing EngineSimulator.
        aggregation: how to combine the per-branch scores (min / mean /
            weighted_mean / max).
        n_opponent_worlds: number of plausible opponent builds to evaluate
            (1 = no sampling).
        world_aggregation: how to combine the same-branch score across
            worlds (mean / min / max).
        model_evaluator: optional callable ``state_dict -> win_prob``. When
            provided AND ``base_state_fn`` is set, the engine snapshot is
            patched back into a state-dict and the model is used as the
            terminal value function (richer than HP+alive heuristic).
        base_state_fn: callable ``battle -> base state-dict``. Required
            when ``model_evaluator`` is set.
        search_depth: number of full turns simulated before scoring.
            1 = 1-ply (our move -> opp reply -> score). 2 / 3 = expectimax
            over the next turn(s): at each of OUR nodes we take the best
            follow-up (max), at each OPPONENT node we aggregate their likely
            replies. Deeper plies only explore our follow-up *moves* (not
            further switches). Root switch candidates use at least
            ``min_switch_remaining_depth`` full turns of lookahead after the
            pivot (default 2) so revenge-kill lines stay visible even when
            ``search_depth`` is low. Depth >= 2 is much slower; use small
            ``n_opponent_worlds`` and the pruning knobs below.
        min_switch_remaining_depth: minimum full turns simulated after a
            root switch before scoring (>= 2 recommended for offensive pivots).
        depth2_opp_top_k: legacy, kept for compatibility (deep opponent move
            branches are now capped by ``deep_opp_move_cap``).
        depth2_my_top_k: number of OUR follow-up moves explored at each deep
            node (after a cheap heuristic probe ranks them).
        deep_opp_move_cap: cap on opponent move branches at deep nodes
            (a mon has <= 4 moves, so 4 = all of them).
        opp_switch_base / opp_switch_max: number of opponent switch branches
            considered at a deep node, scaled by how disadvantaged the
            opponent is (more switches when they're losing the matchup, since
            a winning opponent rarely pivots out).
        opp_switch_adv_lo / opp_switch_adv_hi: P1-winprob thresholds for the
            adaptive switch scaling. Below ``lo`` the opponent is ahead -> 0
            switches; above ``hi`` they're clearly behind -> ``opp_switch_max``.
        prune_my_move_delta: at a deep OUR node, skip follow-up moves whose
            probe score is more than this below the best probe.
        win_cutoff / loss_cutoff: positions already decided (heuristic
            win-prob >= win_cutoff or <= loss_cutoff) are scored immediately
            without deepening, and the OUR-node search stops early once a
            move reaches ``win_cutoff``.
        relative_heuristic_scale: terminal heuristic is
            ``0.5 + scale * (pos_leaf - pos_root)``; pruning still uses
            absolute position scores.
        """
        self.sim = simulator
        self.aggregation = aggregation
        self.n_opponent_worlds = max(1, int(n_opponent_worlds))
        self.world_aggregation = world_aggregation
        self.model_evaluator = model_evaluator
        self.base_state_fn = base_state_fn
        self.search_depth = max(1, int(search_depth))
        self.depth2_opp_top_k = max(1, int(depth2_opp_top_k))
        self.deep_my_top_k = max(1, int(depth2_my_top_k))
        self.deep_opp_move_cap = max(1, int(deep_opp_move_cap))
        self.opp_switch_base = max(0, int(opp_switch_base))
        self.opp_switch_max = max(self.opp_switch_base, int(opp_switch_max))
        self.opp_switch_adv_lo = float(opp_switch_adv_lo)
        self.opp_switch_adv_hi = float(opp_switch_adv_hi)
        self.prune_my_move_delta = float(prune_my_move_delta)
        self.win_cutoff = float(win_cutoff)
        self.loss_cutoff = float(loss_cutoff)
        self.relative_heuristic_scale = float(relative_heuristic_scale)
        self.min_switch_remaining_depth = max(0, int(min_switch_remaining_depth))
        self.verbose = bool(verbose)
        self.progress_every_s = float(progress_every_s)
        # Progress counters (reset per top-level action evaluation).
        self._sims = 0
        self._t_start = 0.0
        self._t_last_beat = 0.0

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(scores: Sequence[float], weights: Sequence[float], mode: str) -> float:
        if not scores:
            return 0.5
        if mode == "min":
            return float(min(scores))
        if mode == "max":
            return float(max(scores))
        if mode == "weighted_mean":
            total_w = sum(weights) or 1.0
            return float(sum(s * w for s, w in zip(scores, weights)) / total_w)
        return float(sum(scores) / len(scores))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_my_move(
        self,
        battle: Any,
        *,
        my_move: str,
        opp_branches: Sequence[OpponentBranch],
        terastallize: bool = False,
        seed: Optional[Sequence[int]] = None,
    ) -> TurnEvalResult:
        """Score a single move against several opponent branches."""
        return self._evaluate_my_action(
            battle,
            my_kind="move",
            my_target=my_move,
            opp_branches=opp_branches,
            terastallize=terastallize,
            seed=seed,
        )

    def evaluate_my_switch(
        self,
        battle: Any,
        *,
        my_switch: str,
        opp_branches: Sequence[OpponentBranch],
        seed: Optional[Sequence[int]] = None,
    ) -> TurnEvalResult:
        """Score a single switch against several opponent branches."""
        return self._evaluate_my_action(
            battle,
            my_kind="switch",
            my_target=my_switch,
            opp_branches=opp_branches,
            terastallize=False,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_my_action(
        self,
        battle: Any,
        *,
        my_kind: str,
        my_target: str,
        opp_branches: Sequence[OpponentBranch],
        terastallize: bool = False,
        seed: Optional[Sequence[int]] = None,
    ) -> TurnEvalResult:
        # Always use a deterministic seed so different branches of the same
        # decision share identical RNG (damage rolls, secondary effects, ...).
        # Without this, two attacks would each get their own random rolls and
        # the comparison would be noisy.
        if seed is None:
            seed = [1, 2, 3, 4]

        # Reset progress counters for this candidate evaluation.
        self._sims = 0
        self._t_start = time.time()
        self._t_last_beat = self._t_start
        if self.verbose:
            self._log(
                f"[engine] eval {my_kind} {my_target!r} "
                f"depth={self.search_depth} worlds={self.n_opponent_worlds} "
                f"branches={len(opp_branches)} ..."
            )

        # Build N plausible opponent builds; the first one matches the
        # legacy single-world behaviour.
        try:
            base_snaps = self.sim.setup_from_battle_variants(
                battle, n_variants=self.n_opponent_worlds, seed=seed
            )
        except Exception:
            return TurnEvalResult(win_prob=0.5, branch_scores=[], setup_failed=True)
        if not base_snaps:
            return TurnEvalResult(win_prob=0.5, branch_scores=[], setup_failed=True)

        # If the caller couldn't build opponent branches (e.g. a freshly
        # revealed mon with no observed moves and no set_dex match), don't
        # collapse to a single "default opp" sim — derive the opponent's real
        # legal replies straight from the sampled engine world. This keeps the
        # lookahead alive instead of silently degenerating to 1-ply.
        opp_branches = list(opp_branches)
        if not opp_branches:
            opp_branches = self._adaptive_opp_branches(base_snaps[0])
            if self.verbose:
                self._log(
                    f"[engine]   (no caller branches -> engine fallback: "
                    f"{len(opp_branches)} branches)"
                )

        # Precompute the base state-dict for model scoring or verbose diag.
        base_state: Optional[Dict[str, Any]] = None
        if self.base_state_fn is not None:
            try:
                base_state = self.base_state_fn(battle)
            except Exception:
                base_state = None
        if self.verbose and base_state is not None:
            if self.model_evaluator is not None:
                try:
                    raw = float(self.model_evaluator(base_state))
                    self._log(f"[engine]   diag: model(base_state)={raw:.4f}")
                except Exception as exc:
                    self._log(f"[engine]   diag: model(base_state) FAILED: {exc}")
            else:
                try:
                    from battle_state_heuristic import score_from_state_dict

                    raw = float(score_from_state_dict(base_state))
                    self._log(f"[engine]   diag: heuristic(base_state)={raw:.4f}")
                except Exception as exc:
                    self._log(f"[engine]   diag: heuristic FAILED: {exc}")

        # Full turns to simulate after our action (move or switch).
        remaining_depth = max(0, self.search_depth - 1)
        if my_kind == "switch":
            remaining_depth = max(
                remaining_depth, self.min_switch_remaining_depth
            )

        per_branch_world_scores: List[List[float]] = [[] for _ in opp_branches]
        weights = [max(w, 1e-6) for _, _, w in opp_branches]
        # Per-world aggregated score (for diagnostic / "world_aggregation").
        per_world_scores: List[float] = []

        for base_snap in base_snaps:
            # Resolve our side's choice once per world (moveset is the same
            # but the choice indices depend on the engine's moveset order).
            if my_kind == "move":
                my_choice = self.sim.choice_for_move(
                    base_snap, "p1", my_target, terastallize=terastallize
                )
            else:
                my_choice = self.sim.choice_for_switch(base_snap, "p1", my_target)

            world_scores: List[float] = []
            for branch_idx, branch in enumerate(opp_branches):
                opp_choice = self._opp_choice(base_snap, branch)
                try:
                    next_snap = self._apply(base_snap, my_choice, opp_choice, seed)
                    score = self._value_after_turn(
                        next_snap,
                        base_state,
                        seed,
                        remaining_depth,
                        root_snap=base_snap,
                    )
                except Exception as exc:
                    if self.verbose:
                        self._log(f"[engine]   branch score failed: {exc}")
                    score = 0.5
                per_branch_world_scores[branch_idx].append(score)
                world_scores.append(score)

            if world_scores:
                per_world_scores.append(
                    self._aggregate(world_scores, weights, self.aggregation)
                )

        if not opp_branches:
            # No branches given: average a "default opp" against each world.
            default_scores: List[float] = []
            for base_snap in base_snaps:
                if my_kind == "move":
                    my_choice = self.sim.choice_for_move(
                        base_snap, "p1", my_target, terastallize=terastallize
                    )
                else:
                    my_choice = self.sim.choice_for_switch(base_snap, "p1", my_target)
                try:
                    next_snap = self._apply(base_snap, my_choice, "default", seed)
                    default_scores.append(
                        self._score_snap(next_snap, base_state, root_snap=base_snap)
                    )
                except Exception:
                    default_scores.append(0.5)
            final_score = sum(default_scores) / max(1, len(default_scores))
            self._log_done(my_kind, my_target, final_score)
            return TurnEvalResult(win_prob=final_score, branch_scores=[])

        # Aggregate across worlds for each branch first (mean), then across
        # branches according to ``self.aggregation``. This way a low-prob
        # opponent attack that happens to KO us in 1/N worlds doesn't
        # collapse the entire score to 0.
        branch_means: List[float] = []
        per_branch: List[Tuple[OpponentBranch, float]] = []
        for idx, branch in enumerate(opp_branches):
            scores = per_branch_world_scores[idx] or [0.5]
            if self.world_aggregation == "min":
                m = float(min(scores))
            elif self.world_aggregation == "max":
                m = float(max(scores))
            else:
                m = float(sum(scores) / len(scores))
            branch_means.append(m)
            per_branch.append((branch, m))

        agg = self._aggregate(branch_means, weights, self.aggregation)
        self._log_done(my_kind, my_target, agg)
        return TurnEvalResult(win_prob=agg, branch_scores=per_branch)

    # ------------------------------------------------------------------
    # Progress logging (verbose) + sim counter / heartbeat
    # ------------------------------------------------------------------

    @staticmethod
    def _log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def _apply(
        self,
        snap: EngineSnapshot,
        my_choice: Optional[str],
        opp_choice: Optional[str],
        seed: Optional[Sequence[int]],
    ) -> EngineSnapshot:
        """Wrap ``sim.apply_choices`` to count sims and emit a heartbeat.

        The heartbeat prints from *inside* the (possibly deep) search so a
        single slow decision still shows it's progressing rather than hung.
        """
        result = self.sim.apply_choices(
            self.sim.fork(snap),
            my_choice=my_choice,
            opp_choice=opp_choice,
            seed=seed,
        )
        self._sims += 1
        if self.verbose:
            now = time.time()
            if now - self._t_last_beat >= self.progress_every_s:
                self._t_last_beat = now
                self._log(
                    f"[engine]   ... {self._sims} sims, "
                    f"{now - self._t_start:.1f}s elapsed"
                )
        return result

    def _log_done(self, my_kind: str, my_target: str, score: float) -> None:
        if not self.verbose:
            return
        self._log(
            f"[engine] done {my_kind} {my_target!r}: wp={score:.4f} "
            f"({self._sims} sims, {time.time() - self._t_start:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Leaf value / deeper plies (expectimax)
    # ------------------------------------------------------------------

    def _score_snap(
        self,
        snap_after: Dict[str, Any],
        base_state: Optional[Dict[str, Any]],
        *,
        root_snap: Optional[EngineSnapshot] = None,
    ) -> float:
        """Terminal value of a snapshot (model if available, else heuristic)."""
        if base_state is not None and self.model_evaluator is not None:
            try:
                state_after = patch_state_from_engine(base_state, snap_after)
                return float(self.model_evaluator(state_after))
            except Exception:
                pass
        if root_snap is not None:
            from battle_state_heuristic import relative_score_from_engine_snaps

            return float(
                relative_score_from_engine_snaps(
                    snap_after,
                    root_snap,
                    p1_name=self.sim.cfg.p1_name,
                    scale=self.relative_heuristic_scale,
                )
            )
        return float(self.sim.estimate_win_probability(snap_after))

    def _opp_choice(self, snap: EngineSnapshot, branch: OpponentBranch) -> str:
        kind, target, _w = branch
        if kind == "move":
            return self.sim.choice_for_move(snap, "p2", target)
        if kind == "switch":
            return self.sim.choice_for_switch(snap, "p2", target)
        return "default"

    # ---- Recursive expectimax (depth >= 2) -------------------------------

    def _value_after_turn(
        self,
        snap: EngineSnapshot,
        base_state: Optional[Dict[str, Any]],
        seed: Optional[Sequence[int]],
        depth: int,
        *,
        root_snap: EngineSnapshot,
    ) -> float:
        """Value of a snapshot with ``depth`` full turns left to simulate.

        ``depth == 0`` (or game over / decided position) -> terminal score.
        Otherwise it's OUR turn (MAX node): try our best follow-up moves and
        return the max. Pruning:
          * decided positions (heuristic win-prob extreme) are not deepened;
          * follow-up moves far below the best probe are skipped;
          * the search stops as soon as a move reaches ``win_cutoff``.
        """
        if depth <= 0 or self.sim.ended(snap):
            return self._score_snap(snap, base_state, root_snap=root_snap)

        # Cheap heuristic gate: don't deepen an already-decided position.
        heur = float(self.sim.estimate_win_probability(snap))
        if heur >= self.win_cutoff or heur <= self.loss_cutoff:
            return self._score_snap(snap, base_state, root_snap=root_snap)

        my_moves = self._legal_my_moves(snap)
        if not my_moves:
            return self._score_snap(snap, base_state, root_snap=root_snap)

        ranked = self._probe_rank_my_moves(snap, my_moves, base_state, seed)
        candidates = self._select_candidates(ranked)

        best: Optional[float] = None
        for mv in candidates:
            v = self._opp_expect(snap, mv, base_state, seed, depth, root_snap=root_snap)
            best = v if best is None else max(best, v)
            if best >= self.win_cutoff:
                break  # found a winning line; max can't improve meaningfully
        return (
            best
            if best is not None
            else self._score_snap(snap, base_state, root_snap=root_snap)
        )

    def _opp_expect(
        self,
        snap: EngineSnapshot,
        my_move: str,
        base_state: Optional[Dict[str, Any]],
        seed: Optional[Sequence[int]],
        depth: int,
        *,
        root_snap: EngineSnapshot,
    ) -> float:
        """Expectation over the opponent's replies to our ``my_move``.

        Opponent branches are rebuilt from THIS snapshot (so a deeper ply
        uses the opponent's *current* active mon), with attack branches
        capped and switch branches scaled by how disadvantaged they are.
        """
        my_choice = self.sim.choice_for_move(snap, "p1", my_move)
        branches = self._adaptive_opp_branches(snap)
        values: List[float] = []
        weights: List[float] = []
        for branch in branches:
            opp_choice = self._opp_choice(snap, branch)
            try:
                nxt = self._apply(snap, my_choice, opp_choice, seed)
                v = self._value_after_turn(
                    nxt, base_state, seed, depth - 1, root_snap=root_snap
                )
            except Exception:
                v = 0.5
            values.append(v)
            weights.append(max(branch[2], 1e-6))
        if not values:
            return self._score_snap(snap, base_state, root_snap=root_snap)
        return self._aggregate(values, weights, self.aggregation)

    def _select_candidates(self, ranked: List[Tuple[float, str]]) -> List[str]:
        """Top-k of our follow-up moves, after delta-pruning weak ones."""
        if not ranked:
            return []
        best_probe = ranked[0][0]
        kept = [
            mv
            for (score, mv) in ranked
            if score >= best_probe - self.prune_my_move_delta
        ]
        return kept[: self.deep_my_top_k]

    def _probe_rank_my_moves(
        self,
        snap: EngineSnapshot,
        my_moves: Sequence[str],
        base_state: Optional[Dict[str, Any]],
        seed: Optional[Sequence[int]],
    ) -> List[Tuple[float, str]]:
        """Order our follow-up moves by a cheap 1-step heuristic probe.

        The probe uses the fast HP+alive heuristic (not the model) since it
        only needs to *rank* moves, not score them precisely. Each move is
        played against the opponent's single most-likely reply.
        """
        moves = list(my_moves)
        if len(moves) <= 1:
            return [(1.0, moves[0])] if moves else []
        probe_branch = self._top_opp_branch(snap)
        opp_choice = (
            self._opp_choice(snap, probe_branch) if probe_branch else "default"
        )
        scored: List[Tuple[float, str]] = []
        for mv in moves:
            my_choice = self.sim.choice_for_move(snap, "p1", mv)
            try:
                nxt = self._apply(snap, my_choice, opp_choice, seed)
                scored.append((float(self.sim.estimate_win_probability(nxt)), mv))
            except Exception:
                scored.append((0.0, mv))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return scored

    # ---- Adaptive opponent branches at a deep node -----------------------

    def _adaptive_opp_branches(self, snap: EngineSnapshot) -> List[OpponentBranch]:
        """Opponent replies to consider at a deep node.

        * Attacks: the opponent active's legal moves, capped at
          ``deep_opp_move_cap`` (a mon has <= 4 moves).
        * Switches: scaled by how disadvantaged the opponent is. We use P1's
          heuristic win-prob as the advantage signal: a losing opponent
          pivots more, a winning one stays in.
        """
        moves = self._legal_opp_moves(snap)[: self.deep_opp_move_cap]
        branches: List[OpponentBranch] = [("move", m, 1.0) for m in moves]

        n_switch, switch_weight = self._opp_switch_plan(snap)
        if n_switch > 0:
            for sp in self._legal_opp_switches(snap)[:n_switch]:
                branches.append(("switch", sp, switch_weight))

        if not branches:
            branches = [("move", "default", 1.0)]
        return branches

    def _opp_switch_plan(self, snap: EngineSnapshot) -> Tuple[int, float]:
        """(#switch branches, per-branch weight) based on opponent disadvantage.

        ``estimate_win_probability`` is from P1's perspective, so a high value
        means the opponent is disadvantaged and more likely to switch out.
        """
        p1_wp = float(self.sim.estimate_win_probability(snap))
        lo, hi = self.opp_switch_adv_lo, self.opp_switch_adv_hi
        if p1_wp <= lo:
            return 0, 0.0  # opponent is ahead -> won't pivot
        # Linear ramp from base (at lo) to max (at hi).
        if hi <= lo:
            frac = 1.0
        else:
            frac = min(1.0, (p1_wp - lo) / (hi - lo))
        n = self.opp_switch_base + round(
            (self.opp_switch_max - self.opp_switch_base) * frac
        )
        n = int(max(self.opp_switch_base, min(self.opp_switch_max, n)))
        # Weight grows with disadvantage: ~0.15 near lo, ~0.5 near/above hi.
        weight = 0.15 + 0.35 * frac
        return n, weight

    def _top_opp_branch(self, snap: EngineSnapshot) -> Optional[OpponentBranch]:
        moves = self._legal_opp_moves(snap)
        if moves:
            return ("move", moves[0], 1.0)
        switches = self._legal_opp_switches(snap)
        if switches:
            return ("switch", switches[0], 1.0)
        return None

    # ---- Snapshot readers -------------------------------------------------

    @staticmethod
    def _side_pokemon(snap: EngineSnapshot, side: str) -> List[Dict[str, Any]]:
        request = ((snap.get("requests") or {}).get(side) or {})
        side_data = request.get("side") or {}
        return list(side_data.get("pokemon", []))

    def _legal_my_moves(self, snap: EngineSnapshot) -> List[str]:
        return self._legal_moves(snap, "p1")

    def _legal_opp_moves(self, snap: EngineSnapshot) -> List[str]:
        return self._legal_moves(snap, "p2")

    @staticmethod
    def _legal_moves(snap: EngineSnapshot, side: str) -> List[str]:
        """Non-disabled move ids for ``side``'s active Pokemon."""
        request = ((snap.get("requests") or {}).get(side) or {})
        if request.get("forceSwitch") or request.get("wait"):
            return []
        active = request.get("active")
        if not active:
            return []
        moves = active[0].get("moves", []) if active else []
        out: List[str] = []
        for move in moves:
            if move.get("disabled"):
                continue
            mid = move.get("id") or move.get("move")
            if mid:
                out.append(str(mid))
        return out

    def _legal_opp_switches(self, snap: EngineSnapshot) -> List[str]:
        """Species of ``p2``'s non-active, non-fainted bench mons."""
        out: List[str] = []
        for poke in self._side_pokemon(snap, "p2"):
            if poke.get("active"):
                continue
            condition = str(poke.get("condition") or "")
            if condition.endswith(" fnt") or condition.startswith("0 "):
                continue
            details = str(poke.get("details") or "")
            species = details.split(",", 1)[0].strip()
            if species and species not in out:
                out.append(species)
        return out

    # ------------------------------------------------------------------
    # Helpers for the caller to build opp_branches from set_dex priors
    # ------------------------------------------------------------------

    @staticmethod
    def branches_from_priors(
        move_probs: dict,  # {move_id: probability}
        switch_species: Sequence[str],
        *,
        top_k_moves: int = 4,
        switch_weight: float = 0.2,
    ) -> List[OpponentBranch]:
        """Build a typical (move, ..., switch, ...) branch list.

        The caller has usually already computed move_probs via
        ``set_dex_prior.aggregate_move_probabilities``.
        """
        ranked = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k_moves]
        out: List[OpponentBranch] = []
        for move_id, p in ranked:
            out.append(("move", move_id, float(p)))
        for sp in switch_species:
            out.append(("switch", sp, switch_weight))
        return out
