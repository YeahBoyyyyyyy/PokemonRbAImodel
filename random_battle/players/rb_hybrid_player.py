"""RbModelPlayer + règles heuristiques (types, switch, recovery)."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional, Set

from poke_env.battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.project_paths import setup_import_paths

setup_import_paths()

from materials import type_effectiveness

from common.combat_helpers import (
    best_damaging_move,
    estimate_enemy_worst_damage,
    estimate_my_damage_on_enemy,
    find_clean_guaranteed_ko_move,
    find_fastest_revenge_killer,
)
from common.defensive_switch import estimate_pokemon_speed
from common.defensive_switch import (
    has_resist_switch_option,
    needs_defensive_switch,
    pick_best_defensive_switch,
    switch_improvement_ok,
)
from random_battle.models.IA_multihead_predictor import normalize_token
from random_battle.players.rb_model_player import RbModelPlayer, _NON_SETUP_STATUS

_RECOVERY_MOVES: Set[str] = {
    "roost",
    "recover",
    "softboiled",
    "slackoff",
    "wish",
    "healorder",
    "moonlight",
    "synthesis",
    "rest",
    "strengthsap",
}


class RbHybridPlayer(RbModelPlayer):
    """
    Modèle multi-tête par défaut, avec overrides heuristiques pour les cas critiques.
    """

    def __init__(
        self,
        *,
        emergency_hp: float = 0.28,
        switch_critical_adv: float = 4.0,
        switch_enemy_adv: float = 2.0,
        switch_chance: float = 0.12,
        switch_bad_matchup_ratio: float = 4.0,
        switch_min_improvement: float = 0.5,
        switch_hp_max_hit: float = 0.5,
        use_heuristic_overrides: bool = True,
        use_heuristic_attack_fallback: bool = True,
        min_model_move_prob: float = 0.08,
        **kwargs,
    ) -> None:
        kwargs.setdefault("action_prob_threshold", 0.52)
        super().__init__(**kwargs)
        self.use_heuristic_overrides = use_heuristic_overrides
        self.emergency_hp = emergency_hp
        self.switch_critical_adv = switch_critical_adv
        self.switch_enemy_adv = switch_enemy_adv
        self.switch_chance = switch_chance
        self.switch_bad_matchup_ratio = switch_bad_matchup_ratio
        self.switch_min_improvement = switch_min_improvement
        self.switch_hp_max_hit = switch_hp_max_hit
        self.use_heuristic_attack_fallback = use_heuristic_attack_fallback
        self.min_model_move_prob = min_model_move_prob

    def _pokemon_types(self, mon: Optional[Pokemon]) -> List[object]:
        if mon is None:
            return []
        types = getattr(mon, "types", None) or []
        if types:
            return list(types)
        t1 = getattr(mon, "type_1", None)
        t2 = getattr(mon, "type_2", None)
        out = [t for t in (t1, t2) if t is not None]
        return out

    def _offensive_matchup(self, attacker: Pokemon, defender: Pokemon) -> float:
        eff = 1.0
        for atk_type in self._pokemon_types(attacker):
            eff *= type_effectiveness(atk_type, self._pokemon_types(defender))
        return eff

    def _defensive_matchup(self, defender: Pokemon, attacker: Pokemon) -> float:
        eff = 1.0
        for atk_type in self._pokemon_types(attacker):
            eff *= type_effectiveness(atk_type, self._pokemon_types(defender))
        return eff

    def _heuristic_best_switch(
        self, switches: List[Pokemon], enemy: Pokemon, *, active: Optional[Pokemon] = None
    ) -> Optional[Pokemon]:
        if active is not None:
            picked = pick_best_defensive_switch(switches, active, enemy)
            if picked is not None:
                return picked  # type: ignore[return-value]
        best: Optional[Pokemon] = None
        best_score = float("inf")
        for candidate in switches:
            if not self._pokemon_types(candidate):
                continue
            taken = self._defensive_matchup(candidate, enemy)
            if taken < best_score:
                best_score = taken
                best = candidate
        return best or (switches[0] if switches else None)

    def _heuristic_best_attack(
        self, moves: List[Move], my: Pokemon, enemy: Optional[Pokemon]
    ) -> Optional[Move]:
        damaging = [
            m for m in moves
            if not self._is_setup_move(m) and (getattr(m, "base_power", 0) or 0) > 0
            and normalize_token(m.id) not in _NON_SETUP_STATUS
        ]
        if enemy is not None:
            picked = best_damaging_move(damaging, my, enemy)
            if picked is not None:
                return picked  # type: ignore[return-value]

        best: Optional[Move] = None
        best_score = -1.0
        enemy_types = self._pokemon_types(enemy)
        my_types = self._pokemon_types(my)
        for move in damaging:
            type_eff = type_effectiveness(move.type, enemy_types)
            if type_eff <= 0.0:
                continue
            base = getattr(move, "base_power", 0) or 0
            score = type_eff * type_eff * base
            if move.type in my_types:
                score *= 1.5
            if type_eff < 1.0:
                score *= 0.35
            if enemy is not None and (enemy.current_hp_fraction or 1.0) < 0.25:
                score *= 1.25
            if score > best_score:
                best_score = score
                best = move
        return best

    def _i_can_ko(
        self, moves: List[Move], my: Pokemon, enemy: Optional[Pokemon]
    ) -> bool:
        if enemy is None or my is None:
            return False
        enemy_hp = float(getattr(enemy, "current_hp_fraction", None) or 1.0)
        for move in moves:
            if self._is_setup_move(move):
                continue
            if (getattr(move, "base_power", 0) or 0) <= 0:
                continue
            est = estimate_my_damage_on_enemy(move, my, enemy)
            if est is None:
                continue
            if est.avg_frac >= enemy_hp:
                return True
        return False

    def _enemy_can_ko_us(
        self, my: Pokemon, enemy: Optional[Pokemon]
    ) -> bool:
        if enemy is None or my is None:
            return False
        my_hp = float(getattr(my, "current_hp_fraction", None) or 1.0)
        worst = estimate_enemy_worst_damage(enemy, my)
        return worst >= my_hp


    def _find_recovery_move(self, moves: List[Move]) -> Optional[Move]:
        for move in moves:
            if normalize_token(move.id) in _RECOVERY_MOVES:
                return move
        return None

    def _max_legal_model_prob(
        self, move_probs, moves: List[Move]
    ) -> float:
        best = 0.0
        for move in moves:
            vid = self.vocab.get(normalize_token(move.id))
            if vid is not None and vid < len(move_probs):
                best = max(best, float(move_probs[vid]))
        return best

    def _damaging_moves_for_ko_check(self, moves: List[Move]) -> List[Move]:
        return [
            m
            for m in moves
            if not self._is_setup_move(m)
            and (getattr(m, "base_power", 0) or 0) > 0
            and normalize_token(m.id) not in _NON_SETUP_STATUS
        ]

    async def _try_priority_combat_override(self, battle: AbstractBattle):
        """Guaranteed clean KOs and faster revenge switches (always-on)."""
        available_moves = list(battle.available_moves or [])
        available_switches = list(battle.available_switches or [])
        can_tera = getattr(battle, "can_terastallize", False)
        my = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        if enemy is None:
            return None

        # 1) Fully-accurate, no-stat-drop KO — skip search entirely.
        if available_moves and my is not None:
            damaging = self._damaging_moves_for_ko_check(available_moves)
            ko_move = find_clean_guaranteed_ko_move(damaging, my, enemy)
            if ko_move is not None:
                tera = self._decide_terastallize(ko_move, battle, can_tera=can_tera)
                order = self.create_order(ko_move, terastallize=tera)
                self._remember_move(battle, order)
                return order

        if not available_switches:
            return None

        # 2) Pivot to a resist (e.g. Torkoal → Vaporeon vs Hydro Pump).
        if my is not None and has_resist_switch_option(
            my, enemy, available_switches
        ):
            pivot = pick_best_defensive_switch(available_switches, my, enemy)
            if pivot is not None:
                return self.create_order(pivot)

        forced_switch = not available_moves
        enemy_hp = float(getattr(enemy, "current_hp_fraction", None) or 1.0)
        my_spe = estimate_pokemon_speed(my) if my is not None else -1
        enemy_spe = estimate_pokemon_speed(enemy)

        # 3) Forced switch after a faint: send the fastest clean revenge killer.
        if forced_switch:
            revenge = find_fastest_revenge_killer(
                available_switches, enemy, active_mon=my
            )
            if revenge is not None:
                return self.create_order(revenge)

        # 4) Optional switch: foe is in KO range but we are slower — pivot faster.
        if (
            available_moves
            and my is not None
            and enemy_hp <= 0.55
            and my_spe < enemy_spe
        ):
            revenge = find_fastest_revenge_killer(
                available_switches,
                enemy,
                active_mon=my,
                require_faster_than_active=True,
            )
            if revenge is not None:
                return self.create_order(revenge)

        return None

    def _maybe_force_switch(self, battle: AbstractBattle):
        available_switches = list(battle.available_switches or [])
        if not available_switches:
            return None

        my = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        if my is None or enemy is None:
            return None
        if not self._pokemon_types(my) or not self._pokemon_types(enemy):
            return None

        my_hp = float(my.current_hp_fraction if my.current_hp_fraction is not None else 1.0)

        if not needs_defensive_switch(
            my,
            enemy,
            hp_fraction=my_hp,
            product_taken_threshold=self.switch_critical_adv,
            max_hit_threshold=self.switch_enemy_adv,
            hp_max_hit=self.switch_hp_max_hit,
        ):
            return None

        taken = self._defensive_matchup(my, enemy)
        if taken < self.switch_bad_matchup_ratio and random.random() > self.switch_chance:
            return None

        target = self._heuristic_best_switch(available_switches, enemy, active=my)
        if target is None:
            return None
        if not switch_improvement_ok(
            my,
            target,
            enemy,
            min_product_ratio=self.switch_min_improvement,
        ):
            return None
        return self.create_order(target)

    async def _try_heuristic_override(self, battle: AbstractBattle):
        if not self.use_heuristic_overrides:
            return None
        available_moves = list(battle.available_moves or [])
        available_switches = list(battle.available_switches or [])
        can_tera = getattr(battle, "can_terastallize", False)

        my = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        if not available_moves and not available_switches:
            return None

        # 1) Switch prioritaire si matchup catastrophique
        forced_switch = self._maybe_force_switch(battle)
        if forced_switch is not None:
            return forced_switch

        hp = (my.current_hp_fraction if my else 1.0) or 1.0

        # 2) Recovery conditionnelle : on heal seulement si :
        #    - HP bas ET on ne peut PAS KO l'adversaire ce tour
        #    - ET soit l'adversaire peut nous KO, soit on est en HP critique
        if hp <= self.emergency_hp and available_moves:
            recovery = self._find_recovery_move(available_moves)
            if recovery is not None:
                can_ko = self._i_can_ko(available_moves, my, enemy)
                enemy_dangerous = self._enemy_can_ko_us(my, enemy)
                critical = hp <= self.emergency_hp * 0.6
                if (not can_ko) and (enemy_dangerous or critical):
                    order = self.create_order(recovery)
                    self._remember_move(battle, order)
                    return order

        # 3) Modèle très incertain sur les coups → meilleure attaque (damage-aware)
        if self.use_heuristic_attack_fallback and available_moves and my is not None:
            preds = self._predict(battle)
            if preds is not None:
                move_probs = preds["move_id"][0]
                filtered = self._filter_available_moves(available_moves, battle)
                if not filtered:
                    filtered = available_moves
                if self._max_legal_model_prob(move_probs, filtered) < self.min_model_move_prob:
                    attack = self._heuristic_best_attack(filtered, my, enemy)
                    if attack is not None:
                        tera = self._decide_terastallize(
                            attack, battle, can_tera=can_tera
                        )
                        order = self.create_order(attack, terastallize=tera)
                        self._remember_move(battle, order)
                        return order

        return None

    async def _choose_with_model(self, battle: AbstractBattle):
        # Switch forcé déjà géré dans _try_heuristic_override (évite double switch / tour).
        return await super()._choose_with_model(battle)

    async def choose_move(self, battle: AbstractBattle):
        if battle.in_team_preview:
            return self.choose_random_teampreview(battle)

        override = await self._try_priority_combat_override(battle)
        if override is not None:
            return override

        override = await self._try_heuristic_override(battle)
        if override is not None:
            return override

        return await self._choose_with_model(battle)
