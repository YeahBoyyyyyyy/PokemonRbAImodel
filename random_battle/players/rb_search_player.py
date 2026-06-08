"""RbHybridPlayer + win-rate pour départager coups et switchs."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.battle_state_heuristic import move_policy_bonus
from common.defensive_switch import needs_defensive_switch, switch_improvement_ok
from common.set_dex_prior import aggregate_move_probabilities
from random_battle.data_extractors.rb_team_slots import normalize_species_key
from random_battle.models.IA_multihead_predictor import normalize_token
from random_battle.players.poke_env_to_state import species_name
from random_battle.players.rb_hybrid_player import RbHybridPlayer
from random_battle.players.turn_simulation import (
    enumerate_opponent_switches,
    opponent_moves_for_state,
)
from random_battle.players.win_rate_evaluator import WinRateEvaluator, state_after_move


class RbSearchPlayer(RbHybridPlayer):
    """
    Hybride + tiebreak win-rate sur attaques et switchs.
    Quand les deux sont légaux : compare P(victoire) après un tour simulé.
    """

    def __init__(
        self,
        *,
        winrate_model_path: Optional[Path] = None,
        use_winrate_search: bool = True,
        tiebreak_top_k: int = 5,
        tiebreak_switch_top_k: int = 4,
        tiebreak_score_ratio: float = 0.35,
        tiebreak_move_score_ratio: float = 0.12,
        tiebreak_move_min_k: int = 2,
        tiebreak_min_winrate_gap: float = 0.02,
        switch_vs_move_margin: float = 0.06,
        # When the active is NOT in defensive danger, an offensive switch
        # (e.g. bringing a revenge-killer) must beat the best move by this
        # larger margin to be played — avoids switch-spam while still
        # allowing clearly-better pivots the model wouldn't pick on its own.
        offensive_switch_margin: float = 0.10,
        switch_model_penalty: float = 0.65,
        use_one_ply_turn: bool = True,
        opp_branch_top_k: int = 6,
        opp_branch_min_prob: float = 0.03,
        opp_switch_top_k: int = 2,
        turn_eval_aggregation: str = "min",
        # --- Engine-based 1-ply simulation (real Showdown via Node) ---
        use_engine: bool = False,
        engine_aggregation: str = "weighted_mean",
        engine_n_worlds: int = 1,
        engine_world_aggregation: str = "mean",
        # Post-turn engine leaf scoring. Default: position heuristic (HP,
        # hazards, setup). Set True to use the win-rate model (often miscalibrated).
        engine_use_model: bool = False,
        # Lookahead depth for engine move eval. 1 = 1-ply, 2/3 = expectimax
        # over the next turn(s) with pruning (see EngineTurnEvaluator).
        engine_depth: int = 1,
        engine_depth2_opp_top_k: int = 3,
        engine_depth2_my_top_k: int = 3,
        engine_deep_opp_move_cap: int = 4,
        engine_opp_switch_base: int = 1,
        engine_opp_switch_max: int = 3,
        engine_prune_delta: float = 0.15,
        engine_win_cutoff: float = 0.92,
        engine_verbose: bool = False,
        # --- Decision logging (post-mortem analysis) ---
        decision_log_path: Optional[Path] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_winrate_search = use_winrate_search
        self.tiebreak_top_k = max(1, int(tiebreak_top_k))
        self.tiebreak_switch_top_k = max(1, int(tiebreak_switch_top_k))
        self.tiebreak_score_ratio = float(tiebreak_score_ratio)
        self.tiebreak_move_score_ratio = float(tiebreak_move_score_ratio)
        self.tiebreak_move_min_k = max(1, int(tiebreak_move_min_k))
        self.tiebreak_min_winrate_gap = float(tiebreak_min_winrate_gap)
        self.switch_vs_move_margin = float(switch_vs_move_margin)
        self.offensive_switch_margin = float(offensive_switch_margin)
        self.switch_model_penalty = float(switch_model_penalty)
        self.use_one_ply_turn = bool(use_one_ply_turn)
        self.opp_branch_top_k = max(1, int(opp_branch_top_k))
        self.opp_branch_min_prob = float(opp_branch_min_prob)
        self.opp_switch_top_k = max(0, int(opp_switch_top_k))
        if turn_eval_aggregation not in ("min", "mean", "weighted_mean"):
            turn_eval_aggregation = "min"
        self.turn_eval_aggregation = turn_eval_aggregation
        self._winrate: Optional[WinRateEvaluator] = None
        if use_winrate_search:
            self._winrate = WinRateEvaluator(model_path=winrate_model_path)

        # --- Engine integration (lazy-init) ---
        self.use_engine = bool(use_engine)
        self.engine_aggregation = (
            engine_aggregation
            if engine_aggregation in ("min", "mean", "weighted_mean", "max")
            else "weighted_mean"
        )
        self.engine_n_worlds = max(1, int(engine_n_worlds))
        self.engine_world_aggregation = (
            engine_world_aggregation
            if engine_world_aggregation in ("min", "mean", "max")
            else "mean"
        )
        self.engine_use_model = bool(engine_use_model)
        self.engine_depth = max(1, int(engine_depth))
        self.engine_depth2_opp_top_k = max(1, int(engine_depth2_opp_top_k))
        self.engine_depth2_my_top_k = max(1, int(engine_depth2_my_top_k))
        self.engine_deep_opp_move_cap = max(1, int(engine_deep_opp_move_cap))
        self.engine_opp_switch_base = max(0, int(engine_opp_switch_base))
        self.engine_opp_switch_max = max(
            self.engine_opp_switch_base, int(engine_opp_switch_max)
        )
        self.engine_prune_delta = float(engine_prune_delta)
        self.engine_win_cutoff = float(engine_win_cutoff)
        self.engine_verbose = bool(engine_verbose)
        self._engine_sim = None
        self._engine_eval = None

        # --- Decision log ---
        self.decision_log_path: Optional[Path] = (
            Path(decision_log_path) if decision_log_path else None
        )
        if self.decision_log_path is not None:
            self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate so each run starts fresh.
            self.decision_log_path.write_text("", encoding="utf-8")

    def _should_consider_switch(self, battle: AbstractBattle) -> bool:
        my = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        if my is None or enemy is None:
            return False
        hp = float(my.current_hp_fraction if my.current_hp_fraction is not None else 1.0)
        return needs_defensive_switch(
            my,
            enemy,
            hp_fraction=hp,
            product_taken_threshold=self.switch_critical_adv,
            max_hit_threshold=self.switch_enemy_adv,
            hp_max_hit=self.switch_hp_max_hit,
        )

    def _filter_defensive_switches(
        self, battle: AbstractBattle, switches: List[Pokemon]
    ) -> List[Pokemon]:
        my = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        if my is None or enemy is None:
            return []
        return [
            s
            for s in switches
            if switch_improvement_ok(
                my, s, enemy, min_product_ratio=self.switch_min_improvement
            )
        ]

    def _switch_candidates(
        self,
        battle: AbstractBattle,
        switches: List[Pokemon],
        defensively_needed: bool,
    ) -> List[Pokemon]:
        """Bench mons worth engine-evaluating as switch options this turn.

        Always includes the best *offensive* matchups vs the current enemy
        (so revenge-killers get scored even when the model wouldn't pick
        them), plus the defensive switches when the active is in danger.
        Capped at ``tiebreak_switch_top_k`` to bound engine cost.
        """
        if not switches:
            return []
        enemy = battle.opponent_active_pokemon
        cap = self.tiebreak_switch_top_k
        picked: List[Pokemon] = []
        # Defensive switches first when we actually need to pivot out.
        if defensively_needed:
            for s in self._filter_defensive_switches(battle, switches):
                if s not in picked:
                    picked.append(s)
        # Then the strongest offensive matchups (type product vs enemy).
        if enemy is not None:
            by_offense = sorted(
                switches,
                key=lambda s: self._offensive_matchup(s, enemy),
                reverse=True,
            )
        else:
            by_offense = list(switches)
        for s in by_offense:
            if len(picked) >= cap:
                break
            if s not in picked:
                picked.append(s)
        return picked[:cap]

    def _winrate_kwargs(self) -> dict:
        return dict(
            opp_top_k=self.opp_branch_top_k,
            opp_min_prob=self.opp_branch_min_prob,
            opp_switch_top_k=self.opp_switch_top_k,
            aggregation=self.turn_eval_aggregation,
        )

    def _compute_move_scores(
        self,
        move_probs: np.ndarray,
        available_moves: List[Move],
        *,
        battle: Optional[AbstractBattle] = None,
    ) -> List[tuple[float, Move]]:
        """Model scores + small hazard/setup nudges from battle_state_heuristic."""
        scores = super()._compute_move_scores(
            move_probs, available_moves, battle=battle
        )
        if battle is None:
            return scores
        boosted: List[tuple[float, Move]] = []
        for score, move in scores:
            token = normalize_token(getattr(move, "id", ""))
            bonus = move_policy_bonus(token, battle)
            boosted.append((max(0.0, score + bonus), move))
        boosted.sort(key=lambda pair: pair[0], reverse=True)
        return boosted

    def _tiebreak_shortlist_scores(
        self,
        scored: List[tuple[float, object]],
        *,
        top_k: int,
        min_k: int = 1,
        score_ratio: Optional[float] = None,
    ) -> List[object]:
        if not scored:
            return []
        ratio = (
            self.tiebreak_score_ratio if score_ratio is None else float(score_ratio)
        )
        best_score = scored[0][0]
        threshold = best_score * ratio if best_score > 0 else 0.0
        shortlist: List[object] = []
        for score, item in scored:
            if len(shortlist) >= top_k:
                break
            if score >= threshold:
                shortlist.append(item)
        # Keep at least min_k top-scored items (e.g. alternate between two attacks).
        effective_min = min(max(1, int(min_k)), top_k, len(scored))
        if len(shortlist) < effective_min:
            for _score, item in scored:
                if item in shortlist:
                    continue
                shortlist.append(item)
                if len(shortlist) >= effective_min:
                    break
        return shortlist

    # ------------------------------------------------------------------
    # Engine plumbing
    # ------------------------------------------------------------------

    def _ensure_engine(self) -> bool:
        """Lazily create the EngineSimulator and EngineTurnEvaluator.

        Returns True on success, False if instantiation failed (in which case
        we fall back to the approximate evaluator).
        """
        if self._engine_eval is not None:
            return True
        if not self.use_engine:
            return False
        try:
            from common.engine_turn_eval import EngineTurnEvaluator
            from common.pkmn_engine_simulator import EngineSimulator

            set_dex = self._winrate.builder.set_dex if self._winrate else None
            self._engine_sim = EngineSimulator(set_dex=set_dex)
            model_evaluator = None
            base_state_fn = None
            if self.engine_use_model and self._winrate is not None:
                model_evaluator = self._winrate.predict_state
                base_state_fn = self._state_dict
            self._engine_eval = EngineTurnEvaluator(
                self._engine_sim,
                aggregation=self.engine_aggregation,
                n_opponent_worlds=self.engine_n_worlds,
                world_aggregation=self.engine_world_aggregation,
                model_evaluator=model_evaluator,
                base_state_fn=base_state_fn,
                search_depth=self.engine_depth,
                depth2_opp_top_k=self.engine_depth2_opp_top_k,
                depth2_my_top_k=self.engine_depth2_my_top_k,
                deep_opp_move_cap=self.engine_deep_opp_move_cap,
                opp_switch_base=self.engine_opp_switch_base,
                opp_switch_max=self.engine_opp_switch_max,
                prune_my_move_delta=self.engine_prune_delta,
                win_cutoff=self.engine_win_cutoff,
                verbose=self.engine_verbose,
            )
            return True
        except Exception as exc:
            print(f"[RbSearchPlayer] engine init failed, fallback to model eval: {exc}")
            self.use_engine = False
            return False

    # ------------------------------------------------------------------
    # Decision log (post-mortem analysis)
    # ------------------------------------------------------------------

    def _log_decision(
        self,
        battle: AbstractBattle,
        *,
        kind: str,
        candidates: List[Dict[str, Any]],
        chosen_label: Optional[str],
    ) -> None:
        if self.decision_log_path is None:
            return
        try:
            my_active = battle.active_pokemon
            opp_active = battle.opponent_active_pokemon
            my_hp_total = sum(
                float(p.current_hp_fraction or 0.0)
                for p in battle.team.values()
                if p and not p.fainted
            )
            opp_hp_total = sum(
                float(p.current_hp_fraction or 0.0)
                for p in battle.opponent_team.values()
                if p and not p.fainted
            )
            entry = {
                "battle_tag": battle.battle_tag,
                "turn": int(getattr(battle, "turn", 0) or 0),
                "kind": kind,  # "move" | "switch"
                "my_active": species_name(my_active) if my_active else None,
                "opp_active": species_name(opp_active) if opp_active else None,
                "my_hp_total": round(my_hp_total, 2),
                "opp_hp_total": round(opp_hp_total, 2),
                "my_team_alive": sum(
                    1 for p in battle.team.values() if p and not p.fainted
                ),
                "opp_team_alive": sum(
                    1 for p in battle.opponent_team.values() if p and not p.fainted
                ),
                "candidates": candidates,  # list of {label, winprob, model_score, chosen}
                "chosen": chosen_label,
            }
            with self.decision_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Never let logging break a battle.
            pass

    def _engine_close(self) -> None:
        try:
            if self._engine_sim is not None:
                self._engine_sim.close()
        except Exception:
            pass
        self._engine_sim = None
        self._engine_eval = None

    def __del__(self) -> None:
        self._engine_close()

    def _engine_opp_branches(
        self, state: Dict[str, object]
    ) -> List[Tuple[str, str, float]]:
        """Build opponent branches (moves + switches) from set_dex priors."""
        if self._winrate is None:
            return []
        builder = self._winrate.builder
        # Opponent moves with normalized probabilities.
        moves = opponent_moves_for_state(
            state,
            set_dex=builder.set_dex,
            vocab=builder.vocab,
            top_k=self.opp_branch_top_k,
            min_prob=self.opp_branch_min_prob,
            mismatch_penalty=builder.mismatch_penalty,
        )
        opp_active = None
        for mon in state.get("opp_team") or []:
            if mon and mon.get("is_active"):
                opp_active = mon
                break
        species = str((opp_active or {}).get("species") or "")
        observed = list((opp_active or {}).get("moves_seen") or [])
        probs_map: Dict[str, float] = {}
        if builder.set_dex and species:
            probs_map = aggregate_move_probabilities(
                builder.set_dex, species, observed, builder.mismatch_penalty
            )
        branches: List[Tuple[str, str, float]] = []
        for mv in moves:
            w = probs_map.get(mv, 1.0 / max(len(moves), 1))
            branches.append(("move", mv, float(max(w, 1e-6))))
        for sp in enumerate_opponent_switches(state, top_k=self.opp_switch_top_k):
            branches.append(("switch", sp, 0.2))
        return branches

    def _eval_move_win_prob(self, state: Dict[str, object], token: str) -> float:
        # Engine path: simulate the real turn with @pkmn/sim.
        if self.use_engine and self._ensure_engine():
            battle = getattr(self, "_search_battle", None)
            if battle is not None and self._engine_eval is not None:
                try:
                    branches = self._engine_opp_branches(state)
                    res = self._engine_eval.evaluate_my_move(
                        battle, my_move=token, opp_branches=branches
                    )
                    if not res.setup_failed:
                        return float(res.win_prob)
                except Exception as exc:
                    print(f"[RbSearchPlayer] engine move eval failed: {exc}; fallback")
        # Fallback: model-based 1-ply on approximate state.
        assert self._winrate is not None
        if self.use_one_ply_turn:
            return self._winrate.predict_after_one_ply_turn(
                state, token, **self._winrate_kwargs()
            )
        return self._winrate.predict_state(state_after_move(state, token))

    def _eval_switch_win_prob(self, state: Dict[str, object], species: str) -> float:
        if self.use_engine and self._ensure_engine():
            battle = getattr(self, "_search_battle", None)
            if battle is not None and self._engine_eval is not None:
                try:
                    branches = self._engine_opp_branches(state)
                    res = self._engine_eval.evaluate_my_switch(
                        battle, my_switch=species, opp_branches=branches
                    )
                    if not res.setup_failed:
                        return float(res.win_prob)
                except Exception as exc:
                    print(f"[RbSearchPlayer] engine switch eval failed: {exc}; fallback")
        assert self._winrate is not None
        if self.use_one_ply_turn:
            return self._winrate.predict_after_my_switch_one_ply(
                state, species, **self._winrate_kwargs()
            )
        from random_battle.players.turn_simulation import state_after_my_switch

        return self._winrate.predict_state(state_after_my_switch(state, species))

    def _compute_switch_scores(
        self,
        switch_probs: np.ndarray,
        state: Dict[str, object],
        available_switches: List[Pokemon],
    ) -> List[tuple[float, Pokemon]]:
        slot_order = state.get("my_team_slot_order") or []
        scored: List[tuple[float, Pokemon]] = []
        for slot, species in enumerate(slot_order[:6]):
            if slot >= len(switch_probs):
                break
            score = float(switch_probs[slot]) * self.switch_model_penalty
            target_norm = normalize_species_key(str(species))
            for mon in available_switches:
                if normalize_species_key(species_name(mon)) == target_norm:
                    scored.append((score, mon))
                    break
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored and available_switches:
            for mon in available_switches:
                scored.append((0.01, mon))
        return scored

    def _best_move_with_winrate(
        self,
        move_probs: np.ndarray,
        available_moves: List[Move],
        *,
        battle: AbstractBattle,
    ) -> Tuple[Optional[Move], float, float]:
        """(move, win_prob, score_modèle). win_prob=-1 si non évalué."""
        scored = self._compute_move_scores(
            move_probs, available_moves, battle=battle
        )
        if not scored:
            return None, -1.0, 0.0
        shortlist = self._tiebreak_shortlist_scores(
            scored,
            top_k=self.tiebreak_top_k,
            min_k=self.tiebreak_move_min_k,
            score_ratio=self.tiebreak_move_score_ratio,
        )
        if not shortlist:
            return None, -1.0, 0.0
        if self._winrate is None:
            return shortlist[0], -1.0, scored[0][0]

        state = self._state_dict(battle)
        score_by_move = {move: s for s, move in scored}
        ranked: List[tuple[float, float, Move]] = []
        for move in shortlist:
            if not isinstance(move, Move):
                continue
            token = normalize_token(move.id)
            if not token:
                continue
            ranked.append(
                (
                    self._eval_move_win_prob(state, token),
                    score_by_move.get(move, 0.0),
                    move,
                )
            )
        if not ranked:
            m = shortlist[0] if isinstance(shortlist[0], Move) else None
            return m, -1.0, scored[0][0]

        ranked.sort(key=lambda row: (-row[0], -row[1]))
        if (
            len(ranked) >= 2
            and ranked[0][0] - ranked[1][0] < self.tiebreak_min_winrate_gap
        ):
            ranked.sort(key=lambda row: -row[1])
        best = ranked[0]
        # Log decision (no-op when decision_log_path is None).
        if self.decision_log_path is not None:
            chosen_id = normalize_token(getattr(best[2], "id", "")) or ""
            candidates_log = [
                {
                    "label": normalize_token(getattr(mv, "id", "")) or "",
                    "winprob": round(float(wp), 4),
                    "model_score": round(float(ms), 4),
                    "chosen": (normalize_token(getattr(mv, "id", "")) == chosen_id),
                }
                for (wp, ms, mv) in ranked
            ]
            self._log_decision(
                battle,
                kind="move",
                candidates=candidates_log,
                chosen_label=chosen_id,
            )
        return best[2], best[0], best[1]

    def _best_switch_with_winrate(
        self,
        switch_probs: np.ndarray,
        state: Dict[str, object],
        available_switches: List[Pokemon],
        *,
        battle: AbstractBattle,
    ) -> Tuple[Optional[Pokemon], float, float]:
        scored = self._compute_switch_scores(
            switch_probs, state, available_switches
        )
        if not scored:
            return None, -1.0, 0.0
        shortlist = self._tiebreak_shortlist_scores(
            scored, top_k=self.tiebreak_switch_top_k
        )
        if not shortlist:
            return None, -1.0, 0.0

        def _as_mon(entry: object) -> Optional[Pokemon]:
            if isinstance(entry, Pokemon):
                return entry
            if isinstance(entry, tuple) and len(entry) >= 2:
                return entry[1] if isinstance(entry[1], Pokemon) else None
            return None

        if len(shortlist) <= 1:
            mon = _as_mon(shortlist[0])
            if mon is None:
                return None, -1.0, 0.0
            if self._winrate is None:
                return mon, -1.0, scored[0][0]
            species = species_name(mon)
            wp = (
                self._eval_switch_win_prob(self._state_dict(battle), species)
                if species
                else -1.0
            )
            return mon, wp, scored[0][0]

        if self._winrate is None:
            mon = _as_mon(shortlist[0])
            return mon, -1.0, scored[0][0] if scored else 0.0

        st = self._state_dict(battle)
        score_by_mon = {mon: s for s, mon in scored}
        ranked: List[tuple[float, float, Pokemon]] = []
        for item in shortlist:
            mon = item[1] if isinstance(item, tuple) else item
            if not isinstance(mon, Pokemon):
                continue
            species = species_name(mon)
            if not species:
                continue
            ranked.append(
                (
                    self._eval_switch_win_prob(st, species),
                    score_by_mon.get(mon, 0.0),
                    mon,
                )
            )
        if not ranked:
            return available_switches[0], -1.0, scored[0][0]

        ranked.sort(key=lambda row: (-row[0], -row[1]))
        if (
            len(ranked) >= 2
            and ranked[0][0] - ranked[1][0] < self.tiebreak_min_winrate_gap
        ):
            ranked.sort(key=lambda row: -row[1])
        best = ranked[0]
        if self.decision_log_path is not None:
            chosen_species = species_name(best[2]) or ""
            candidates_log = [
                {
                    "label": species_name(mon) or "",
                    "winprob": round(float(wp), 4),
                    "model_score": round(float(ms), 4),
                    "chosen": (species_name(mon) == chosen_species),
                }
                for (wp, ms, mon) in ranked
            ]
            self._log_decision(
                battle,
                kind="switch",
                candidates=candidates_log,
                chosen_label=chosen_species,
            )
        return best[2], best[0], best[1]

    def _pick_move(
        self,
        move_probs: np.ndarray,
        available_moves: List[Move],
        *,
        battle: Optional[AbstractBattle] = None,
    ) -> Optional[Move]:
        if (
            self.use_winrate_search
            and self._winrate is not None
            and battle is not None
        ):
            move, _, _ = self._best_move_with_winrate(
                move_probs, available_moves, battle=battle
            )
            if move is not None:
                return move
        return super()._pick_move(move_probs, available_moves, battle=battle)

    def _pick_switch(
        self,
        switch_probs: np.ndarray,
        state: Dict[str, object],
        available_switches: List,
    ):
        battle = getattr(self, "_search_battle", None)
        if self.use_winrate_search and self._winrate is not None and battle is not None:
            mon, _, _ = self._best_switch_with_winrate(
                switch_probs,
                state,
                list(available_switches),
                battle=battle,
            )
            if mon is not None:
                return mon
        return super()._pick_switch(switch_probs, state, available_switches)

    async def choose_move(self, battle: AbstractBattle):
        if battle.in_team_preview:
            return self.choose_random_teampreview(battle)
        self._search_battle = battle
        if self.engine_verbose and self.use_engine:
            my = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            t0 = time.time()
            print(
                f"[engine] === turn {getattr(battle, 'turn', '?')} "
                f"{species_name(my) if my else '?'} vs "
                f"{species_name(opp) if opp else '?'} ===",
                file=sys.stderr,
                flush=True,
            )
        else:
            t0 = None
        try:
            override = await self._try_heuristic_override(battle)
            if override is not None:
                return override
            return await self._choose_with_model(battle)
        finally:
            self._search_battle = None
            if t0 is not None:
                print(
                    f"[engine] === turn {getattr(battle, 'turn', '?')} "
                    f"decided in {time.time() - t0:.1f}s ===",
                    file=sys.stderr,
                    flush=True,
                )

    async def _choose_with_model(self, battle: AbstractBattle):
        if not self.use_winrate_search or self._winrate is None:
            return await super()._choose_with_model(battle)

        available_moves = list(battle.available_moves or [])
        available_switches = list(battle.available_switches or [])
        can_tera = getattr(battle, "can_terastallize", False)

        if not available_moves and not available_switches:
            return self.choose_default_move()

        preds = self._predict(battle)
        if preds is None:
            return self._fallback(available_moves, available_switches, can_tera)

        move_probs = preds["move_id"][0]
        switch_probs = preds["switch_slot"][0]
        state = self._state_dict(battle)

        best_move: Optional[Move] = None
        move_wp = -1.0
        if available_moves:
            best_move, move_wp, _ = self._best_move_with_winrate(
                move_probs, available_moves, battle=battle
            )

        # Consider switches every turn (offensive revenge-kills included),
        # not only when the active is in defensive danger. The engine
        # win-rate decides; a larger margin gates non-defensive pivots so we
        # don't switch-spam.
        defensively_needed = self._should_consider_switch(battle)
        switch_candidates = self._switch_candidates(
            battle, available_switches, defensively_needed
        )

        best_switch: Optional[Pokemon] = None
        switch_wp = -1.0
        if switch_candidates:
            best_switch, switch_wp, _ = self._best_switch_with_winrate(
                switch_probs, state, switch_candidates, battle=battle
            )

        margin = (
            self.switch_vs_move_margin
            if defensively_needed
            else self.offensive_switch_margin
        )

        if (
            best_move is not None
            and best_switch is not None
            and move_wp >= 0
            and switch_wp >= 0
        ):
            if switch_wp > move_wp + margin:
                return self.create_order(best_switch)
            tera_dec = self._decide_terastallize(best_move, battle, can_tera=can_tera)
            order = self.create_order(best_move, terastallize=tera_dec)
            self._remember_move(battle, order)
            return order

        if best_move is not None and (best_switch is None or move_wp >= switch_wp):
            tera_dec = self._decide_terastallize(best_move, battle, can_tera=can_tera)
            order = self.create_order(best_move, terastallize=tera_dec)
            self._remember_move(battle, order)
            return order

        if best_switch is not None and not available_moves:
            return self.create_order(best_switch)

        return await super()._choose_with_model(battle)
