"""poke-env player driven by the Random Battle multi-head Keras model."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import tensorflow as tf
from poke_env.battle import AbstractBattle
from poke_env.player import Player
from poke_env.battle.move import Move

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.project_paths import setup_import_paths
from random_battle.config import (
    FORMAT_ID,
    MOVE_VOCAB_PATH,
    MULTIHEAD_MODEL_DIR,
    RB_SET_DEX_PATH,
)
from random_battle.players.poke_env_to_state import battle_to_state_dict, species_name
from random_battle.players.rb_move_filters import (
    MAJOR_STATUS,
    MOVE_APPLIES_STATUS,
    is_status_inflictor_move,
    normalize_status,
    status_inflictor_blocked_by_types,
    status_move_blocked,
)
from random_battle.data_extractors.rb_team_slots import normalize_species_key

setup_import_paths(shared_data=True, rb_data=True)

from materials import type_effectiveness

from common.combat_helpers import should_terastallize

from random_battle.models.IA_multihead_predictor import (  # noqa: E402
    Config,
    FeatureBuilder,
    load_set_dex,
    normalize_token,
)

# Stat-boost / setup moves (normalized ids) — pénalisés si déjà boosté ou répétés
_SETUP_MOVE_TOKENS: Set[str] = {
    "swordsdance",
    "nastyplot",
    "calmmind",
    "dragondance",
    "quiverdance",
    "bulkup",
    "irondefense",
    "agility",
    "rockpolish",
    "shellsmash",
    "bellydrum",
    "victorydance",
    "tailglow",
    "geomancy",
    "coil",
    "workup",
    "growth",
    "honeclaws",
    "autotomize",
    "tidyup",
    "clangoroussoul",
    "noretreat",
    "takeheart",
    "filletaway",
    "curse",
    "acidarmor",
    "amnesia",
    "barrier",
    "cottonguard",
    "cosmicpower",
    "defensecurl",
    "doubleteam",
    "minimize",
    "focusenergy",
    "tailwind",
    "trickroom",
    "raindance",
    "sunnyday",
    "snowscape",
    "chillyreception",
}

# Coups statut non-setup (base_power 0 mais pas des boosts)
_NON_SETUP_STATUS: Set[str] = {
    "protect",
    "substitute",
    "roost",
    "recover",
    "wish",
    "healorder",
    "rest",
    "sleeptalk",
    "willowisp",
    "thunderwave",
    "toxic",
    "spore",
    "encore",
    "taunt",
    "haze",
    "defog",
    "rapidspin",
    "stealthrock",
    "spikes",
    "toxicspikes",
    "stickyweb",
}


class RbModelPlayer(Player):
    """Play gen9randombattle using the trained multi-head model."""

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
        set_dex_path: Optional[Path] = None,
        action_prob_threshold: float = 0.5,
        temperature: float = 0.9,
        top_k_moves: int = 5,
        setup_repeat_penalty: float = 0.05,
        setup_move_score_multiplier: float = 0.08,
        max_setup_boosts: int = 0,
        max_setups_per_battle: int = 0,
        max_setups_per_active: int = 0,
        prefer_attacks_over_setup: bool = True,
        ban_setup_when_attack_available: bool = True,
        ban_setup_when_risky: bool = True,
        setup_risk_taken: float = 1.25,
        setup_risk_hp: float = 0.65,
        setup_safe_taken_max: float = 0.75,
        setup_safe_dealt_min: float = 2.0,
        battle_format: str = FORMAT_ID,
        mismatch_penalty: float = 0.15,
        **kwargs,
    ) -> None:
        kwargs.setdefault("battle_format", battle_format)
        super().__init__(**kwargs)

        model_path = Path(model_path or (MULTIHEAD_MODEL_DIR / "model.keras"))
        vocab_path = Path(vocab_path or MOVE_VOCAB_PATH)
        set_dex_path = Path(set_dex_path or RB_SET_DEX_PATH)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train with IA_multihead_predictor.py first."
            )
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Move vocab not found: {vocab_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.vocab: Dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.set_dex = load_set_dex(str(set_dex_path))
        self.builder = FeatureBuilder(
            Config(),
            self.vocab,
            self.set_dex,
            mismatch_penalty=mismatch_penalty,
            slot_mode="exact",
        )
        self.action_prob_threshold = action_prob_threshold
        self.temperature = max(0.05, float(temperature))
        self.top_k_moves = max(1, int(top_k_moves))
        self.setup_repeat_penalty = float(setup_repeat_penalty)
        self.setup_move_score_multiplier = float(setup_move_score_multiplier)
        # Bloque setup si boosts positifs > N (0 = dès +1). -1 = désactive ce critère.
        self.max_setup_boosts = int(max_setup_boosts)
        self.max_setups_per_battle = int(max_setups_per_battle)
        self.max_setups_per_active = int(max_setups_per_active)
        self.prefer_attacks_over_setup = bool(prefer_attacks_over_setup)
        self.ban_setup_when_attack_available = bool(ban_setup_when_attack_available)
        self.ban_setup_when_risky = bool(ban_setup_when_risky)
        self.setup_risk_taken = float(setup_risk_taken)
        self.setup_risk_hp = float(setup_risk_hp)
        self.setup_safe_taken_max = float(setup_safe_taken_max)
        self.setup_safe_dealt_min = float(setup_safe_dealt_min)

        self._my_preview_orders: Dict[str, List[str]] = {}
        self._opp_preview_orders: Dict[str, List[str]] = {}
        self._last_my_move: Dict[str, Optional[str]] = {}
        self._last_opp_move: Dict[str, Optional[str]] = {}
        self._setups_used: Dict[str, int] = {}
        self._setups_used_active: Dict[str, Dict[str, int]] = {}
        # Si poke-env n'a pas encore le statut (1 tour de retard), on se souvient du dernier statut visé
        self._pending_opp_status: Dict[str, str] = {}

    def _state_dict(self, battle: AbstractBattle) -> Dict[str, object]:
        return battle_to_state_dict(
            battle,
            self._my_preview_orders,
            self._opp_preview_orders,
            my_last_move=self._last_my_move.get(battle.battle_tag),
            opp_last_move=self._last_opp_move.get(battle.battle_tag),
        )

    def _predict(self, battle: AbstractBattle) -> Optional[Dict[str, np.ndarray]]:
        state = self._state_dict(battle)
        example = {"state": state, "action_type": "move", "action_target": ""}
        built = self.builder.build_example(example)
        if built is None:
            return None
        features, _, _ = built
        batch = {key: np.expand_dims(val, 0) for key, val in features.items()}
        return self.model.predict(batch, verbose=0)

    def _active_boosts(self, battle: AbstractBattle) -> Dict[str, int]:
        mon = battle.active_pokemon
        if mon is None:
            return {}
        return dict(getattr(mon, "boosts", None) or {})

    def _positive_boost_sum(self, boosts: Dict[str, int]) -> int:
        return sum(v for v in boosts.values() if v > 0)

    def _pokemon_types(self, mon) -> List[object]:
        if mon is None:
            return []
        types = getattr(mon, "types", None) or []
        if types:
            return list(types)
        t1 = getattr(mon, "type_1", None)
        t2 = getattr(mon, "type_2", None)
        return [t for t in (t1, t2) if t is not None]

    def _type_attack_multiplier(self, attacker, defender) -> float:
        mult = 1.0
        for atk_type in self._pokemon_types(attacker):
            mult *= type_effectiveness(atk_type, self._pokemon_types(defender))
        return mult

    def _is_damaging_move(self, move: Move) -> bool:
        if self._is_setup_move(move):
            return False
        return (getattr(move, "base_power", 0) or 0) > 0

    def _effective_move_type(self, move: Move, battle: AbstractBattle) -> object:
        """Real (possibly form/tera-dependent) type of a move.

        poke-env exposes the *static* dex type for a Move, which is wrong for
        form- or tera-dependent moves: Ivy Cudgel is listed as Grass but is
        Rock/Fire/Water for the Cornerstone/Hearthflame/Wellspring Ogerpon
        masks; Raging Bull / Aura Wheel / Tera Blast likewise change type.
        Using the static type made the type-effectiveness pre-filter discard
        these moves before the engine could even evaluate them.
        """
        token = normalize_token(getattr(move, "id", "") or "")
        my = battle.active_pokemon if battle is not None else None
        species = normalize_token(str(getattr(my, "species", "") or "")) if my else ""

        if token == "ivycudgel":
            if "cornerstone" in species:
                return "rock"
            if "hearthflame" in species:
                return "fire"
            if "wellspring" in species:
                return "water"
            return "grass"
        if token == "ragingbull":
            if "blaze" in species:
                return "fire"
            if "aqua" in species:
                return "water"
            if "tauros" in species:
                return "fighting"
        if token == "aurawheel":
            if "hangry" in species:
                return "dark"
            if "morpeko" in species:
                return "electric"
        if token == "terablast":
            if my is not None and getattr(my, "is_terastallized", False):
                tera = getattr(my, "tera_type", None)
                if tera is not None:
                    return tera
            return "normal"

        return getattr(move, "type", None)

    def _raw_move_type_effectiveness(self, move: Move, battle: AbstractBattle) -> float:
        """Multiplicateur type chart seul (sans STAB) — pour neutre vs résisté."""
        enemy = battle.opponent_active_pokemon
        if enemy is None:
            return 1.0
        move_type = self._effective_move_type(move, battle)
        if move_type is None:
            return 1.0
        return float(type_effectiveness(move_type, self._pokemon_types(enemy)))

    def _move_offensive_effectiveness(self, move: Move, battle: AbstractBattle) -> float:
        enemy = battle.opponent_active_pokemon
        my = battle.active_pokemon
        if enemy is None:
            return 1.0
        move_type = self._effective_move_type(move, battle)
        if move_type is None:
            return 1.0
        eff = self._raw_move_type_effectiveness(move, battle)
        if my is not None and self._has_type(my, move_type):
            eff *= 1.5
        return float(eff)

    @staticmethod
    def _has_type(mon: Pokemon, move_type: object) -> bool:
        """True if ``mon`` shares ``move_type`` (STAB), comparing by name."""
        target = str(getattr(move_type, "name", move_type) or "").lower()
        if not target:
            return False
        for t in getattr(mon, "types", None) or []:
            if str(getattr(t, "name", t) or "").lower() == target:
                return True
        return False

    def _move_is_type_useless(self, move: Move, battle: AbstractBattle) -> bool:
        enemy = battle.opponent_active_pokemon
        if enemy is None:
            return False
        enemy_types = self._pokemon_types(enemy)
        token = normalize_token(move.id)
        applies = MOVE_APPLIES_STATUS.get(token)
        if applies and status_inflictor_blocked_by_types(applies, enemy_types):
            return True
        move_type = self._effective_move_type(move, battle)
        if move_type is not None and enemy_types:
            if type_effectiveness(move_type, enemy_types) <= 0.0:
                return True
        return False

    def _filter_type_matchup(
        self,
        available_moves: List[Move],
        battle: Optional[AbstractBattle],
    ) -> List[Move]:
        """Retire immunités / coups très résistés quand de meilleures options existent."""
        if battle is None or len(available_moves) <= 1:
            return available_moves

        enemy = battle.opponent_active_pokemon
        if enemy is None or not self._pokemon_types(enemy):
            return available_moves

        useful = [m for m in available_moves if not self._move_is_type_useless(m, battle)]
        if not useful:
            return available_moves
        available_moves = useful

        damaging = [m for m in available_moves if self._is_damaging_move(m)]
        if len(damaging) <= 1:
            return available_moves

        raw_effs = [self._raw_move_type_effectiveness(m, battle) for m in damaging]
        best_raw = max(raw_effs)

        if best_raw >= 1.0:
            damaging = [
                m
                for m in damaging
                if self._raw_move_type_effectiveness(m, battle) >= 1.0
            ]
        else:
            damaging = [
                m
                for m in damaging
                if self._raw_move_type_effectiveness(m, battle) >= best_raw * 0.85
            ]

        if not damaging:
            return available_moves

        utility = [m for m in available_moves if not self._is_damaging_move(m)]
        utility.extend(damaging)
        return utility

    def _risky_to_stay(self, battle: AbstractBattle) -> bool:
        """True = dangereux de rester (ne pas setup). False = matchup safe, setup OK."""
        my = battle.active_pokemon
        enemy = battle.opponent_active_pokemon
        if my is None or enemy is None:
            return False
        if not self._pokemon_types(my) or not self._pokemon_types(enemy):
            return False

        taken = self._type_attack_multiplier(enemy, my)
        dealt = self._type_attack_multiplier(my, enemy)
        my_hp = float(my.current_hp_fraction if my.current_hp_fraction is not None else 1.0)
        enemy_hp = float(
            enemy.current_hp_fraction if enemy.current_hp_fraction is not None else 1.0
        )

        if dealt >= self.setup_safe_dealt_min and taken <= self.setup_safe_taken_max:
            return False
        if dealt >= 2.0 and enemy_hp < 0.45 and my_hp > 0.35:
            return False

        if taken >= 2.0:
            return True
        if taken >= self.setup_risk_taken and my_hp <= self.setup_risk_hp:
            return True
        if my_hp < 0.4 and dealt < taken:
            return True
        if taken > 1.0 and dealt < 1.0 and my_hp < 0.65:
            return True
        return False

    def _active_mon_setup_key(self, battle: AbstractBattle) -> str:
        mon = battle.active_pokemon
        if mon is None:
            return ""
        return normalize_species_key(species_name(mon)) or str(getattr(mon, "species", ""))

    def _is_setup_token(self, token: str) -> bool:
        if not token or token in _NON_SETUP_STATUS:
            return False
        return token in _SETUP_MOVE_TOKENS

    def _is_setup_move(self, move: Move) -> bool:
        token = normalize_token(move.id)
        if token in _NON_SETUP_STATUS:
            return False
        if token in _SETUP_MOVE_TOKENS:
            return True
        self_boost = getattr(move, "self_boost", None) or {}
        if isinstance(self_boost, dict) and any(int(v) > 0 for v in self_boost.values()):
            return True
        return False

    def _has_offensive_option(
        self,
        available_moves: List[Move],
        battle: AbstractBattle,
        *,
        min_effectiveness: float = 1.0,
    ) -> bool:
        for move in available_moves:
            if not self._is_damaging_move(move):
                continue
            if self._move_offensive_effectiveness(move, battle) >= min_effectiveness:
                return True
        return False

    def _should_block_setup(
        self,
        move: Move,
        *,
        battle: AbstractBattle,
        boosts: Dict[str, int],
        last_token: str,
        available_moves: Optional[List[Move]] = None,
    ) -> bool:
        if not self._is_setup_move(move):
            return False

        tag = battle.battle_tag
        if self.max_setups_per_battle <= 0:
            return True
        if self._setups_used.get(tag, 0) >= self.max_setups_per_battle:
            return True

        mon_key = self._active_mon_setup_key(battle)
        if self.max_setups_per_active >= 0 and mon_key:
            per_active = self._setups_used_active.get(tag, {}).get(mon_key, 0)
            if per_active >= self.max_setups_per_active:
                return True

        if self.ban_setup_when_risky and self._risky_to_stay(battle):
            return True

        if (
            self.ban_setup_when_attack_available
            and available_moves
            and self._has_offensive_option(available_moves, battle, min_effectiveness=0.5)
        ):
            return True

        token = normalize_token(move.id)
        if token and token == last_token:
            return True
        if last_token and self._is_setup_token(last_token):
            return True
        if self.max_setup_boosts >= 0:
            if self._positive_boost_sum(boosts) > self.max_setup_boosts:
                return True
        return False

    def _opponent_status(self, battle: AbstractBattle) -> Optional[str]:
        enemy = battle.opponent_active_pokemon
        if enemy is None:
            return None
        status = getattr(enemy, "status", None)
        if status is None:
            return None
        name = getattr(status, "name", status)
        return normalize_status(str(name).lower() if name else None)

    def _effective_opponent_status(self, battle: AbstractBattle) -> Optional[str]:
        tag = battle.battle_tag
        current = self._opponent_status(battle)
        if current and current in MAJOR_STATUS:
            self._pending_opp_status.pop(tag, None)
            return current
        pending = self._pending_opp_status.get(tag)
        if pending and pending in MAJOR_STATUS:
            return pending
        return None

    def _is_redundant_status_move(self, move: Move, battle: AbstractBattle) -> bool:
        token = normalize_token(move.id)
        applies = MOVE_APPLIES_STATUS.get(token)
        if not applies and not is_status_inflictor_move(token):
            return False
        if not applies:
            applies = "any"
        return status_move_blocked(
            applies,
            self._effective_opponent_status(battle),
            block_if_any_major_status=True,
        )

    def _filter_moves_for_setup(
        self,
        available_moves: List[Move],
        battle: Optional[AbstractBattle],
    ) -> List[Move]:
        if battle is None:
            return available_moves
        boosts = self._active_boosts(battle)
        last_token = self._last_my_move.get(battle.battle_tag) or ""
        allowed = [
            m
            for m in available_moves
            if not self._should_block_setup(
                m,
                battle=battle,
                boosts=boosts,
                last_token=last_token,
                available_moves=available_moves,
            )
        ]
        if allowed:
            moves = allowed
        else:
            moves = available_moves
        if self.prefer_attacks_over_setup:
            damaging = [m for m in moves if self._is_damaging_move(m)]
            if damaging:
                non_setup = [m for m in moves if not self._is_setup_move(m)]
                if non_setup:
                    return non_setup
        return moves if allowed else (
            [m for m in available_moves if not self._is_setup_move(m)] or available_moves
        )

    def _filter_available_moves(
        self,
        available_moves: List[Move],
        battle: Optional[AbstractBattle],
    ) -> List[Move]:
        moves = self._filter_moves_for_setup(available_moves, battle)
        if battle is None:
            return moves
        moves = self._filter_type_matchup(moves, battle)
        without_status = [
            m for m in moves if not self._is_redundant_status_move(m, battle)
        ]
        if without_status:
            return without_status
        without_setup = [m for m in moves if not self._is_setup_move(m)]
        if without_setup:
            return without_setup
        return moves

    def _compute_move_scores(
        self,
        move_probs: np.ndarray,
        available_moves: List[Move],
        *,
        battle: Optional[AbstractBattle] = None,
    ) -> List[tuple[float, Move]]:
        """Scores déterministes (modèle + types + setup), triés décroissants."""
        if not available_moves:
            return []

        available_moves = self._filter_available_moves(available_moves, battle)

        last_token = ""
        if battle is not None:
            last_token = self._last_my_move.get(battle.battle_tag) or ""

        raw_effs: List[float] = []
        for move in available_moves:
            if battle is not None and self._is_damaging_move(move):
                raw_effs.append(self._raw_move_type_effectiveness(move, battle))
            else:
                raw_effs.append(1.0)
        max_raw_eff = max(raw_effs) if raw_effs else 1.0

        scores: List[tuple[float, Move]] = []

        for move, raw_eff in zip(available_moves, raw_effs):
            token = normalize_token(move.id)
            vid = self.vocab.get(token)
            if vid is not None and vid < len(move_probs):
                score = float(move_probs[vid])
            else:
                score = 1e-4

            if self._is_setup_move(move) and self.setup_move_score_multiplier > 0:
                score *= self.setup_move_score_multiplier
            elif (
                self.setup_repeat_penalty > 0
                and self._is_setup_move(move)
                and token
                and token == last_token
            ):
                score *= self.setup_repeat_penalty

            if battle is not None and self._is_damaging_move(move):
                score *= max(raw_eff, 0.0) ** 2
                if max_raw_eff >= 1.0 and raw_eff < 1.0:
                    score *= 0.02

            scores.append((score, move))

        scores.sort(key=lambda pair: pair[0], reverse=True)
        return scores

    def _pick_move(
        self,
        move_probs: np.ndarray,
        available_moves: List[Move],
        *,
        battle: Optional[AbstractBattle] = None,
    ) -> Optional[Move]:
        if not available_moves:
            return None

        scores = self._compute_move_scores(
            move_probs, available_moves, battle=battle
        )

        if not scores:
            filtered = self._filter_available_moves(available_moves, battle)
            return (filtered or available_moves)[0]

        arr = np.array([s for s, _ in scores], dtype=np.float64)
        if self.temperature != 1.0:
            arr = np.log(np.maximum(arr, 1e-12)) / self.temperature
        arr = arr - np.max(arr)
        probs = np.exp(arr)
        probs /= probs.sum() or 1.0

        if self.top_k_moves < len(scores):
            keep = np.argpartition(-probs, self.top_k_moves)[: self.top_k_moves]
            mask = np.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask / (mask.sum() or 1.0)

        idx = int(np.random.choice(len(scores), p=probs))
        return scores[idx][1]

    def _pick_switch(
        self,
        switch_probs: np.ndarray,
        state: Dict[str, object],
        available_switches: List,
    ):
        slot_order = state.get("my_team_slot_order") or []
        best_mon = None
        best_score = -1.0

        for slot, species in enumerate(slot_order[:6]):
            if slot >= len(switch_probs):
                break
            score = float(switch_probs[slot])
            target_norm = normalize_species_key(str(species))
            for mon in available_switches:
                if normalize_species_key(species_name(mon)) == target_norm:
                    if score > best_score:
                        best_score = score
                        best_mon = mon
                    break

        if best_mon is not None:
            return best_mon
        return available_switches[0] if available_switches else None

    def _remember_move(self, battle: AbstractBattle, order) -> None:
        move = getattr(order, "order", None)
        if isinstance(move, Move):
            token = normalize_token(move.id)
            if token:
                tag = battle.battle_tag
                self._last_my_move[tag] = token
                if self._is_setup_move(move):
                    self._setups_used[tag] = self._setups_used.get(tag, 0) + 1
                    mon_key = self._active_mon_setup_key(battle)
                    if mon_key:
                        per = self._setups_used_active.setdefault(tag, {})
                        per[mon_key] = per.get(mon_key, 0) + 1
                applies = MOVE_APPLIES_STATUS.get(token)
                if applies:
                    self._pending_opp_status[tag] = applies
                return
        # Secours : dernier coup connu côté poke-env
        active = battle.active_pokemon
        if active is not None and getattr(active, "moved", False):
            if active.moves:
                used = next(iter(active.moves.values()), None)
                if used is not None:
                    token = normalize_token(getattr(used, "id", ""))
                    if token:
                        self._last_my_move[battle.battle_tag] = token

    async def choose_move(self, battle: AbstractBattle):
        if battle.in_team_preview:
            return self.choose_random_teampreview(battle)
        return await self._choose_with_model(battle)

    async def _choose_with_model(self, battle: AbstractBattle):
        available_moves = list(battle.available_moves or [])
        available_switches = list(battle.available_switches or [])
        can_tera = getattr(battle, "can_terastallize", False)

        if not available_moves and not available_switches:
            return self.choose_default_move()

        preds = self._predict(battle)
        if preds is None:
            return self._fallback(available_moves, available_switches, can_tera)

        action_prob = float(preds["action_prob"][0][0])
        move_probs = preds["move_id"][0]
        switch_probs = preds["switch_slot"][0]
        state = self._state_dict(battle)

        prefer_move = action_prob >= self.action_prob_threshold

        if prefer_move and available_moves:
            move = self._pick_move(move_probs, available_moves, battle=battle)
            if move is not None:
                tera = self._decide_terastallize(move, battle, can_tera=can_tera)
                order = self.create_order(move, terastallize=tera)
                self._remember_move(battle, order)
                return order

        if available_switches:
            target = self._pick_switch(switch_probs, state, available_switches)
            if target is not None:
                return self.create_order(target)

        if available_moves:
            move = self._pick_move(move_probs, available_moves, battle=battle)
            if move is not None:
                order = self.create_order(move)
                self._remember_move(battle, order)
                return order

        return self._fallback(available_moves, available_switches, can_tera)

    def _decide_terastallize(
        self,
        move: Optional[Move],
        battle: AbstractBattle,
        *,
        can_tera: bool,
    ) -> bool:
        if not can_tera or move is None:
            return False
        return should_terastallize(
            move, battle.active_pokemon, battle.opponent_active_pokemon
        )

    def _fallback(self, available_moves, available_switches, can_tera: bool):
        if available_moves:
            move = random.choice(available_moves)
            tera_now = can_tera and random.random() < 0.02
            return self.create_order(move, terastallize=tera_now)
        if available_switches:
            return self.create_order(random.choice(available_switches))
        return self.choose_default_move()
