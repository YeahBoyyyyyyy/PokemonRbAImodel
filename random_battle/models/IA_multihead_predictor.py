"""
Random Battle multi-head model (gen9randombattle).

Predicts action type, move, and switch slot from partially observed states
(hp_percent=-1 when unknown, revealed flag for opponent).

Uses rb_set_dex (pkmn/randbats) for move priors on active Pokémon.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.project_paths import setup_import_paths
from random_battle.config import (
    ACTION_CHUNKS_DIR,
    MOVE_VOCAB_PATH,
    MULTIHEAD_MODEL_DIR,
    RB_SET_DEX_PATH,
)

setup_import_paths()

from common.set_dex_prior import compute_move_prior, load_set_dex
from materials import TYPE_NAMES
from pokedex_9G_complete import pokemon_data_gen9
UNKNOWN_HP = -1.0  # must match rb_visibility.UNKNOWN_HP

MAX_TEAM_SIZE = 6
TOTAL_POKEMON = MAX_TEAM_SIZE * 2

STATUS_LIST = ["none", "brn", "par", "slp", "frz", "psn", "tox"]
WEATHER_LIST = ["none", "sunnyday", "raindance", "sandstorm", "hail", "snow"]
TERRAIN_LIST = ["none", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]

TYPE_NAME_MAP = {t.lower(): t for t in TYPE_NAMES}

SPECIES_ALIASES = {
    "sinistchamasterpiece": "sinistcha",
    "sinistchaunremarkable": "sinistcha",
    "polteageistantique": "polteageist",
    "polteageistphony": "polteageist",
    "gastrodoneast": "gastrodon",
    "gastrodonwest": "gastrodon",
    "pikachuoriginal": "pikachu",
    "dudunsparcethreesegment": "dudunsparce",
    "dudunsparcetwosegment": "dudunsparce",
    "alcremiematchacream": "alcremie",
    "mimikyubusted": "mimikyu",
    "zarudedada": "zarude",
    "toxtricitylowkey": "toxtricity",
    "mausholdfour": "maushold",
    "mausholdthree": "maushold",
    "mausholdfamilyoffour": "maushold",
    "mausholdfamilyofthree": "maushold",
    "miniorblue": "minior",
    "miniorgreen": "minior",
    "miniorindigo": "minior",
    "miniororange": "minior",
    "minioryellow": "minior",
    "miniorviolet": "minior",
    "miniorred": "minior",
    "miniorcore": "minior",
    "ogerpontealtera": "ogerpon",
    "ogerponwellspringtera": "ogerponwellspring",
    "ogerponhearthflametera": "ogerponhearthflame",
    "ogerponcornerstonetera": "ogerponcornerstone",
    "vivillongarden": "vivillon",
    "sneaselhisui": "sneasel",
}


def normalize_species_name(name: str) -> str:
    if not name:
        return ""
    lowered = name.lower().strip()
    return re.sub(r"[^a-z0-9]", "", lowered)


def normalize_token(token: Optional[str]) -> str:
    if not token:
        return ""
    lowered = token.lower().strip()
    return re.sub(r"[^a-z0-9]", "", lowered)


def canonicalize_species_norm(norm: str) -> str:
    if not norm:
        return norm
    base = norm[:-4] if norm.endswith("tera") else norm
    if base in SPECIES_ALIASES:
        return SPECIES_ALIASES[base]
    for prefix, target in (
        ("pikachu", "pikachu"),
        ("alcremie", "alcremie"),
        ("minior", "minior"),
        ("gastrodon", "gastrodon"),
        ("mimikyu", "mimikyu"),
        ("zarude", "zarude"),
        ("toxtricity", "toxtricity"),
        ("dudunsparce", "dudunsparce"),
        ("polteageist", "polteageist"),
        ("sinistcha", "sinistcha"),
        ("maushold", "maushold"),
        ("vivillon", "vivillon"),
    ):
        if base != target and base.startswith(prefix):
            return target
    return base


def stable_hash(token: str, dim: int) -> int:
    return zlib.crc32(token.encode("utf-8")) % dim


def hash_tokens(tokens: Iterable[str], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        if not token:
            continue
        idx = stable_hash(token, dim)
        vec[idx] += 1.0
    return vec


def build_species_index() -> Tuple[Dict[str, str], Dict[str, int]]:
    norm_to_key: Dict[str, str] = {}
    species_to_id: Dict[str, int] = {}
    for i, key in enumerate(sorted(pokemon_data_gen9.keys()), start=1):
        norm_to_key[normalize_species_name(key)] = key
        species_to_id[key] = i
    return norm_to_key, species_to_id


def load_set_dex(path: Optional[str]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_species_entry(
    set_dex: Dict[str, object], species: str
) -> Tuple[Dict[str, int], int]:
    species_dict = set_dex.get("species", {})
    if not species_dict:
        return {}, 0

    candidates: set[str] = set()
    if species in species_dict:
        candidates.add(species)
    lower = species.lower()
    for key in species_dict:
        if key.lower() == lower:
            candidates.add(key)
    norm = normalize_species_name(species)
    for key in species_dict:
        if normalize_species_name(key) == norm:
            candidates.add(key)
    base = species.split("-", 1)[0]
    norm_base = normalize_species_name(base)
    for key in species_dict:
        if normalize_species_name(key).startswith(norm_base):
            candidates.add(key)

    merged_sets: Dict[str, int] = {}
    total = 0
    for key in sorted(candidates):
        entry = species_dict.get(key, {})
        for set_key, count in entry.get("sets", {}).items():
            merged_sets[set_key] = merged_sets.get(set_key, 0) + int(count)
        total += int(entry.get("total", 0))
    return merged_sets, total


def compute_move_prior(
    set_dex: Optional[Dict[str, object]],
    species: str,
    observed_moves: List[str],
    mismatch_penalty: float,
    vocab: Dict[str, int],
) -> np.ndarray:
    if not set_dex:
        return np.zeros(len(vocab) + 1, dtype=np.float32)
    sets, total = resolve_species_entry(set_dex, species)
    if not sets or total <= 0:
        return np.zeros(len(vocab) + 1, dtype=np.float32)

    move_probs: Dict[str, float] = {}
    for set_key, count in sets.items():
        parts = set_key.split("|") if set_key else []
        moves = [p for p in parts if not p.startswith("item=")]
        missing = [m for m in observed_moves if m not in moves]
        if missing and mismatch_penalty <= 0:
            continue
        score = count / total
        if missing:
            score *= mismatch_penalty ** len(missing)
        for mv in moves:
            move_probs[mv] = move_probs.get(mv, 0.0) + score

    if not move_probs:
        return np.zeros(len(vocab) + 1, dtype=np.float32)

    total_score = sum(move_probs.values()) or 1.0
    vec = np.zeros(len(vocab) + 1, dtype=np.float32)
    for mv, score in move_probs.items():
        mv_id = vocab.get(mv)
        if mv_id:
            vec[mv_id] = score / total_score
    return vec


def build_move_vocab(
    input_files: List[Path],
    max_examples: int,
    val_every: int,
    filter_voluntary: bool,
    top_moves: int,
    min_move_freq: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    seen = 0
    for input_file in input_files:
        data = json.loads(input_file.read_text(encoding="utf-8"))
        for example in data:
            if filter_voluntary and not example.get("is_voluntary", True):
                continue
            seen += 1
            is_val = (seen % val_every) == 0 if val_every > 0 else False
            if is_val:
                continue
            move = normalize_token(example.get("action_target"))
            if not move:
                continue
            counts[move] = counts.get(move, 0) + 1
            if max_examples and seen >= max_examples:
                break
        if max_examples and seen >= max_examples:
            break

    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if min_move_freq > 1:
        items = [kv for kv in items if kv[1] >= min_move_freq]
    if top_moves > 0:
        items = items[:top_moves]
    vocab = {move: i + 1 for i, (move, _count) in enumerate(items)}
    return vocab


@dataclass
class Config:
    side_hash_dim: int = 32
    move_hash_dim: int = 64
    volatile_hash_dim: int = 8
    turn_cap: int = 100
    item_hash_dim: int = 8
    ability_hash_dim: int = 8


class FeatureBuilder:
    def __init__(
        self,
        config: Config,
        vocab: Dict[str, int],
        set_dex: Optional[Dict[str, object]],
        mismatch_penalty: float,
        slot_mode: str,
    ):
        self.config = config
        self.vocab = vocab
        self.set_dex = set_dex
        self.mismatch_penalty = mismatch_penalty
        self.slot_mode = slot_mode
        self.norm_to_key, self.species_to_id = build_species_index()
        self.type_to_id = {t.lower(): i + 1 for i, t in enumerate(TYPE_NAMES)}
        self.status_to_id = {s: i + 1 for i, s in enumerate(STATUS_LIST)}
        self.weather_to_id = {w: i for i, w in enumerate(WEATHER_LIST)}
        self.terrain_to_id = {t: i for i, t in enumerate(TERRAIN_LIST)}

        self.numeric_dim = (
            1  # hp
            + 1  # hp_known
            + 1  # fainted
            + 1  # is_active
            + 1  # revealed
            + 6  # base stats
            + 1  # weight
            + 5  # boosts
            + 1  # vol_count
            + self.config.volatile_hash_dim
            + self.config.item_hash_dim
            + self.config.ability_hash_dim
            + 2  # tera
        )
        self.global_dim = (self.config.side_hash_dim * 2) + (self.config.move_hash_dim * 2) + 1

    def output_signature(
        self,
    ) -> Tuple[
        Dict[str, tf.TensorSpec],
        Dict[str, tf.TensorSpec],
        Dict[str, tf.TensorSpec],
    ]:
        features = {
            "species_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
            "type1_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
            "type2_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
            "status_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
            "numeric_feats": tf.TensorSpec(shape=(TOTAL_POKEMON, self.numeric_dim), dtype=tf.float32),
            "global_num": tf.TensorSpec(shape=(self.global_dim,), dtype=tf.float32),
            "weather_id": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            "terrain_id": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            "moves_seen": tf.TensorSpec(shape=(len(self.vocab) + 1,), dtype=tf.float32),
            "move_prior": tf.TensorSpec(shape=(len(self.vocab) + 1,), dtype=tf.float32),
            "opp_move_prior": tf.TensorSpec(shape=(len(self.vocab) + 1,), dtype=tf.float32),
        }
        labels = {
            "action_prob": tf.TensorSpec(shape=(), dtype=tf.float32),
            "move_id": tf.TensorSpec(shape=(), dtype=tf.int32),
            "switch_slot": tf.TensorSpec(shape=(), dtype=tf.int32),
        }
        weights = {
            "action_prob": tf.TensorSpec(shape=(), dtype=tf.float32),
            "move_id": tf.TensorSpec(shape=(), dtype=tf.float32),
            "switch_slot": tf.TensorSpec(shape=(), dtype=tf.float32),
        }
        return features, labels, weights

    def _resolve_species_key(self, species: str) -> Optional[str]:
        norm = normalize_species_name(species)
        key = self.norm_to_key.get(norm)
        if key:
            return key
        alt = canonicalize_species_norm(norm)
        if alt != norm:
            key = self.norm_to_key.get(alt)
            if key:
                return key
        if "-" in species:
            base_raw = species.split("-", 1)[0]
            base_norm = normalize_species_name(base_raw)
            key = self.norm_to_key.get(base_norm)
            if key:
                return key
        return None

    def _pokedex_entry(self, species: str) -> Optional[dict]:
        key = self._resolve_species_key(species)
        if not key:
            return None
        return pokemon_data_gen9.get(key)

    def _species_id(self, species: str) -> int:
        norm = normalize_species_name(species)
        if not norm or norm == "unknown":
            return 0
        key = self._resolve_species_key(species)
        if not key:
            return 0
        return self.species_to_id.get(key, 0)

    def _type_ids(self, entry: Optional[dict]) -> Tuple[int, int]:
        if not entry:
            return 0, 0
        types = entry.get("types", [])
        if not types:
            return 0, 0
        t1 = self.type_to_id.get(types[0].lower(), 0)
        t2 = self.type_to_id.get(types[1].lower(), 0) if len(types) > 1 else 0
        return t1, t2

    def _status_id(self, status: Optional[str]) -> int:
        if not status:
            return 0
        raw = status.lower().strip()
        aliases = {
            "paralyzed": "par",
            "burned": "brn",
            "burn": "brn",
            "sleep": "slp",
            "asleep": "slp",
            "frozen": "frz",
            "freeze": "frz",
            "poison": "psn",
            "poisoned": "psn",
            "toxic": "tox",
            "badlypoisoned": "tox",
        }
        norm = aliases.get(raw, raw)
        return self.status_to_id.get(norm, 0)

    def _base_stats(self, entry: Optional[dict]) -> List[float]:
        if not entry:
            return [0.0] * 6
        stats = entry.get("stats", {})
        return [
            stats.get("hp", 0.0) / 255.0,
            stats.get("atk", 0.0) / 255.0,
            stats.get("def", 0.0) / 255.0,
            stats.get("spa", 0.0) / 255.0,
            stats.get("spd", 0.0) / 255.0,
            stats.get("spe", 0.0) / 255.0,
        ]

    def _clamp_boost(self, value: int) -> float:
        return max(-6, min(6, int(value))) / 6.0

    def _team_order(
        self,
        team_list: List[Dict[str, object]],
        slot_order: Optional[List[str]] = None,
    ) -> List[Dict[str, object]]:
        if slot_order:
            by_norm: Dict[str, Dict[str, object]] = {}
            for poke in team_list:
                norm = normalize_species_name((poke or {}).get("species", ""))
                if norm:
                    by_norm[norm] = poke
            ordered = []
            for species in slot_order[:MAX_TEAM_SIZE]:
                norm = normalize_species_name(species)
                ordered.append(by_norm.get(norm, {"species": species}))
        else:
            ordered = list(team_list)
        while len(ordered) < MAX_TEAM_SIZE:
            ordered.append({})
        return ordered[:MAX_TEAM_SIZE]

    def _parse_slot_index(self, action_target: object) -> Optional[int]:
        if isinstance(action_target, (int, float)):
            idx = int(action_target)
            if 1 <= idx <= MAX_TEAM_SIZE:
                return idx - 1
            if 0 <= idx < MAX_TEAM_SIZE:
                return idx
            return None
        if isinstance(action_target, str):
            raw = action_target.strip()
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= MAX_TEAM_SIZE:
                    return idx - 1
                if 0 <= idx < MAX_TEAM_SIZE:
                    return idx
        return None

    def _resolve_switch_species(self, action_target: object, my_team_list: List[Dict[str, object]]) -> str:
        if action_target is None:
            return ""
        # Slot number from inputlog (1-6)
        if isinstance(action_target, (int, float)):
            idx = int(action_target)
            if 0 <= idx < MAX_TEAM_SIZE:
                return (my_team_list[idx] or {}).get("species", "")
            if 1 <= idx <= MAX_TEAM_SIZE:
                return (my_team_list[idx - 1] or {}).get("species", "")
        if isinstance(action_target, str):
            raw = action_target.strip()
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= MAX_TEAM_SIZE:
                    return (my_team_list[idx - 1] or {}).get("species", "")
                if 0 <= idx < MAX_TEAM_SIZE:
                    return (my_team_list[idx] or {}).get("species", "")
            return raw
        return ""

    def build_example(
        self, example: Dict[str, object]
    ) -> Optional[
        Tuple[
            Dict[str, np.ndarray],
            Dict[str, np.ndarray],
            Dict[str, np.ndarray],
        ]
    ]:

        state = example.get("state", {})
        if not state:
            return None

        my_team_list = state.get("my_team", []) or []
        opp_team_list = state.get("opp_team", []) or []
        slot_order = state.get("my_team_slot_order") or [
            (p or {}).get("species", "") for p in my_team_list
        ]

        our_order = self._team_order(my_team_list, slot_order=slot_order)
        opp_order = self._team_order(opp_team_list)

        species_ids: List[int] = []
        type1_ids: List[int] = []
        type2_ids: List[int] = []
        status_ids: List[int] = []
        numeric_rows: List[List[float]] = []

        for team in (our_order, opp_order):
            for info in team:
                species = (info or {}).get("species", "")
                entry = self._pokedex_entry(species)
                species_ids.append(self._species_id(species))
                t1, t2 = self._type_ids(entry)
                type1_ids.append(t1)
                type2_ids.append(t2)
                status_ids.append(self._status_id((info or {}).get("status")))

                raw_hp = (info or {}).get("hp_percent", 1.0)
                hp_val = float(raw_hp if raw_hp is not None else UNKNOWN_HP)
                hp_known = 0.0 if hp_val < 0 else 1.0
                hp = max(hp_val, 0.0)
                fainted = 1.0 if (info or {}).get("fainted") else 0.0
                is_active = 1.0 if (info or {}).get("is_active") else 0.0
                revealed = 1.0 if (info or {}).get("revealed", True) else 0.0
                base_stats = self._base_stats(entry)
                weight = (entry.get("weight", 0.0) / 1000.0) if entry else 0.0
                boosts = (info or {}).get("stats_boosts", {}) or {}
                boosts_vec = [
                    self._clamp_boost(boosts.get("atk", 0)),
                    self._clamp_boost(boosts.get("def", 0)),
                    self._clamp_boost(boosts.get("spa", 0)),
                    self._clamp_boost(boosts.get("spd", 0)),
                    self._clamp_boost(boosts.get("spe", 0)),
                ]
                volatiles = (info or {}).get("volatile_conditions", []) or []
                vol_count = float(len(volatiles))
                vol_hash = hash_tokens([normalize_token(v) for v in volatiles], self.config.volatile_hash_dim)

                item = normalize_token((info or {}).get("item"))
                ability = normalize_token((info or {}).get("ability"))
                item_hash = hash_tokens([item], self.config.item_hash_dim) if item else np.zeros(
                    self.config.item_hash_dim, dtype=np.float32
                )
                ability_hash = (
                    hash_tokens([ability], self.config.ability_hash_dim)
                    if ability
                    else np.zeros(self.config.ability_hash_dim, dtype=np.float32)
                )
                tera_active = 1.0 if (info or {}).get("tera_active") else 0.0
                tera_type = normalize_token((info or {}).get("tera_type"))
                tera_type_id = self.type_to_id.get(tera_type, 0)
                tera_type_norm = (
                    tera_type_id / float(len(self.type_to_id)) if tera_type_id else 0.0
                )

                numeric_row: List[float] = []
                numeric_row.extend([hp, hp_known, fainted, is_active, revealed])
                numeric_row.extend(base_stats)
                numeric_row.append(weight)
                numeric_row.extend(boosts_vec)
                numeric_row.append(vol_count)
                numeric_row.extend(vol_hash.tolist())
                numeric_row.extend(item_hash.tolist())
                numeric_row.extend(ability_hash.tolist())
                numeric_row.extend([tera_active, tera_type_norm])
                numeric_rows.append(numeric_row)

        if len(species_ids) != TOTAL_POKEMON:
            return None

        my_side = state.get("my_side_conditions", {}) or {}
        opp_side = state.get("opp_side_conditions", {}) or {}
        my_hazards = state.get("my_hazards", {}) or {}
        opp_hazards = state.get("opp_hazards", {}) or {}

        my_side_tokens = [f"{normalize_token(k)}:{v}" for k, v in my_side.items()]
        my_side_tokens += [f"{normalize_token(k)}:{v}" for k, v in my_hazards.items()]
        opp_side_tokens = [f"{normalize_token(k)}:{v}" for k, v in opp_side.items()]
        opp_side_tokens += [f"{normalize_token(k)}:{v}" for k, v in opp_hazards.items()]

        side_vec = np.concatenate(
            [
                hash_tokens(my_side_tokens, self.config.side_hash_dim),
                hash_tokens(opp_side_tokens, self.config.side_hash_dim),
            ]
        )

        my_last_move = normalize_token(state.get("my_last_move"))
        opp_last_move = normalize_token(state.get("opp_last_move"))
        move_vec = np.concatenate(
            [
                hash_tokens([my_last_move], self.config.move_hash_dim),
                hash_tokens([opp_last_move], self.config.move_hash_dim),
            ]
        )

        weather = normalize_token(state.get("weather"))
        terrain = normalize_token(state.get("terrain"))
        weather_id = self.weather_to_id.get(weather, 0)
        terrain_id = self.terrain_to_id.get(terrain, 0)

        turn = int(state.get("turn", 0) or 0)
        turn_norm = min(turn, self.config.turn_cap) / float(self.config.turn_cap)
        global_num = np.concatenate([side_vec, move_vec, np.array([turn_norm], dtype=np.float32)])

        active_info = next((p for p in my_team_list if p.get("is_active")), {})
        active_species = (active_info or {}).get("species", "")
        moves_seen_raw = (active_info or {}).get("moves_seen", []) or []
        if not moves_seen_raw and my_last_move:
            moves_seen_raw = [my_last_move]
        moves_seen = [normalize_token(m) for m in moves_seen_raw if normalize_token(m)]

        moves_seen_vec = np.zeros(len(self.vocab) + 1, dtype=np.float32)
        for mv in moves_seen:
            mv_id = self.vocab.get(mv)
            if mv_id:
                moves_seen_vec[mv_id] = 1.0

        move_prior = compute_move_prior(
            set_dex=self.set_dex,
            species=active_species,
            observed_moves=moves_seen,
            mismatch_penalty=self.mismatch_penalty,
            vocab=self.vocab,
        )

        opp_active = next((p for p in opp_team_list if p.get("is_active")), {})
        opp_species = (opp_active or {}).get("species", "")
        opp_moves_raw = (opp_active or {}).get("moves_seen", []) or []
        opp_moves = [normalize_token(m) for m in opp_moves_raw if normalize_token(m)]
        opp_move_prior = compute_move_prior(
            set_dex=self.set_dex,
            species=opp_species,
            observed_moves=opp_moves,
            mismatch_penalty=self.mismatch_penalty,
            vocab=self.vocab,
        )

        features = {
            "species_ids": np.array(species_ids, dtype=np.int32),
            "type1_ids": np.array(type1_ids, dtype=np.int32),
            "type2_ids": np.array(type2_ids, dtype=np.int32),
            "status_ids": np.array(status_ids, dtype=np.int32),
            "numeric_feats": np.array(numeric_rows, dtype=np.float32),
            "global_num": global_num.astype(np.float32),
            "weather_id": np.array([weather_id], dtype=np.int32),
            "terrain_id": np.array([terrain_id], dtype=np.int32),
            "moves_seen": moves_seen_vec,
            "move_prior": move_prior,
            "opp_move_prior": opp_move_prior,
        }

        action_type = (example.get("action_type") or "").lower()
        action_label = 1.0 if action_type == "move" else 0.0

        move_label = 0
        move_weight = 0.0
        if action_type == "move":
            move_label = self.vocab.get(normalize_token(example.get("action_target")), 0)
            move_weight = 1.0

        switch_label = 0
        switch_weight = 0.0
        if action_type == "switch":
            slot_idx = self._parse_slot_index(example.get("action_target"))
            if slot_idx is not None and 0 <= slot_idx < MAX_TEAM_SIZE:
                switch_label = slot_idx
                switch_weight = 1.0

        labels = {
            "action_prob": np.array(action_label, dtype=np.float32),
            "move_id": np.array(move_label, dtype=np.int32),
            "switch_slot": np.array(switch_label, dtype=np.int32),
        }
        weights = {
            "action_prob": np.array(1.0, dtype=np.float32),
            "move_id": np.array(move_weight, dtype=np.float32),
            "switch_slot": np.array(switch_weight, dtype=np.float32),
        }
        return features, labels, weights


def iter_examples(
    input_files: List[Path],
    split: str,
    val_every: int,
    max_examples: int,
    filter_voluntary: bool,
) -> Iterable[Dict[str, object]]:
    yielded = 0
    seen = 0
    for input_file in input_files:
        data = json.loads(input_file.read_text(encoding="utf-8"))
        for example in data:
            if filter_voluntary and not example.get("is_voluntary", True):
                continue
            seen += 1
            is_val = (seen % val_every) == 0 if val_every > 0 else False
            if split == "train" and is_val:
                continue
            if split == "val" and not is_val:
                continue
            yield example
            yielded += 1
            if max_examples and yielded >= max_examples:
                return


def _examples_from_file(
    path: Path,
    builder: FeatureBuilder,
    split: str,
    val_every: int,
    max_examples: int,
    filter_voluntary: bool,
) -> Iterable:
    for example in iter_examples(
        input_files=[path],
        split=split,
        val_every=val_every,
        max_examples=max_examples,
        filter_voluntary=filter_voluntary,
    ):
        built = builder.build_example(example)
        if built is not None:
            yield built


def make_dataset(
    input_files: List[Path],
    builder: FeatureBuilder,
    split: str,
    val_every: int,
    max_examples: int,
    shuffle_buffer: int,
    filter_voluntary: bool,
    interleave_files: int = 4,
    cache: bool = False,
) -> tf.data.Dataset:
    if not input_files:
        raise FileNotFoundError("No input files for dataset")

    signature = builder.output_signature()
    paths = list(input_files)
    if split == "train":
        random.shuffle(paths)

    def multi_file_generator():
        yielded = 0
        for path in paths:
            for example in iter_examples(
                input_files=[path],
                split=split,
                val_every=val_every,
                max_examples=0,
                filter_voluntary=filter_voluntary,
            ):
                built = builder.build_example(example)
                if built is None:
                    continue
                yield built
                yielded += 1
                if max_examples and yielded >= max_examples:
                    return

    dataset = tf.data.Dataset.from_generator(
        multi_file_generator,
        output_signature=signature,
    )

    if split == "train":
        dataset = dataset.shuffle(shuffle_buffer)
    if cache:
        dataset = dataset.cache()
    return dataset


def build_model(
    builder: FeatureBuilder,
    dense_units: int,
    dense_units2: int,
    dropout: float,
    dropout2: float,
    l2_reg: float,
    vocab_size: int,
) -> tf.keras.Model:
    species_in = tf.keras.Input(shape=(TOTAL_POKEMON,), dtype="int32", name="species_ids")
    type1_in = tf.keras.Input(shape=(TOTAL_POKEMON,), dtype="int32", name="type1_ids")
    type2_in = tf.keras.Input(shape=(TOTAL_POKEMON,), dtype="int32", name="type2_ids")
    status_in = tf.keras.Input(shape=(TOTAL_POKEMON,), dtype="int32", name="status_ids")
    numeric_in = tf.keras.Input(
        shape=(TOTAL_POKEMON, builder.numeric_dim), dtype="float32", name="numeric_feats"
    )
    global_in = tf.keras.Input(shape=(builder.global_dim,), dtype="float32", name="global_num")
    weather_in = tf.keras.Input(shape=(1,), dtype="int32", name="weather_id")
    terrain_in = tf.keras.Input(shape=(1,), dtype="int32", name="terrain_id")
    moves_seen_in = tf.keras.Input(shape=(vocab_size + 1,), dtype="float32", name="moves_seen")
    move_prior_in = tf.keras.Input(shape=(vocab_size + 1,), dtype="float32", name="move_prior")
    opp_move_prior_in = tf.keras.Input(
        shape=(vocab_size + 1,), dtype="float32", name="opp_move_prior"
    )

    species_emb = tf.keras.layers.Embedding(input_dim=len(builder.species_to_id) + 1, output_dim=16)(
        species_in
    )
    type_emb = tf.keras.layers.Embedding(input_dim=len(builder.type_to_id) + 1, output_dim=4)
    type1_emb = type_emb(type1_in)
    type2_emb = type_emb(type2_in)
    status_emb = tf.keras.layers.Embedding(input_dim=len(builder.status_to_id) + 1, output_dim=3)(
        status_in
    )

    per_poke = tf.keras.layers.Concatenate(axis=-1)(
        [species_emb, type1_emb, type2_emb, status_emb, numeric_in]
    )
    per_poke = tf.keras.layers.Flatten()(per_poke)

    weather_emb = tf.keras.layers.Embedding(input_dim=len(WEATHER_LIST), output_dim=3)(weather_in)
    terrain_emb = tf.keras.layers.Embedding(input_dim=len(TERRAIN_LIST), output_dim=3)(terrain_in)
    global_cat = tf.keras.layers.Concatenate(axis=-1)(
        [
            global_in,
            tf.keras.layers.Flatten()(weather_emb),
            tf.keras.layers.Flatten()(terrain_emb),
            moves_seen_in,
            move_prior_in,
            opp_move_prior_in,
        ]
    )

    x = tf.keras.layers.Concatenate(axis=-1)([per_poke, global_cat])
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units2, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    action_out = tf.keras.layers.Dense(1, activation="sigmoid", name="action_prob")(x)
    move_out = tf.keras.layers.Dense(vocab_size + 1, activation="softmax", name="move_id")(x)
    switch_out = tf.keras.layers.Dense(MAX_TEAM_SIZE, activation="softmax", name="switch_slot")(x)

    model = tf.keras.Model(
        inputs=[
            species_in,
            type1_in,
            type2_in,
            status_in,
            numeric_in,
            global_in,
            weather_in,
            terrain_in,
            moves_seen_in,
            move_prior_in,
            opp_move_prior_in,
        ],
        outputs={
            "action_prob": action_out,
            "move_id": move_out,
            "switch_slot": switch_out,
        },
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "action_prob": tf.keras.losses.BinaryCrossentropy(),
            "move_id": tf.keras.losses.SparseCategoricalCrossentropy(),
            "switch_slot": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "action_prob": [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            "move_id": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
            ],
            "switch_slot": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        },
    )
    return model


def train(args: argparse.Namespace) -> None:
    input_paths: List[Path] = []
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Missing input_dir: {input_dir}")
        input_paths = sorted(input_dir.glob(f"{args.base_name}_*.json"))
        if not input_paths:
            raise FileNotFoundError(f"No chunk files found in {input_dir}")
    else:
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Missing training data: {input_path}")
        input_paths = [input_path]

    if args.vocab_file and Path(args.vocab_file).exists():
        vocab = json.loads(Path(args.vocab_file).read_text(encoding="utf-8"))
        vocab = {k: int(v) for k, v in vocab.items()}
    else:
        vocab = build_move_vocab(
            input_files=input_paths,
            max_examples=args.max_examples,
            val_every=args.val_every,
            filter_voluntary=args.filter_voluntary,
            top_moves=args.top_moves,
            min_move_freq=args.min_move_freq,
        )
        if args.vocab_file:
            Path(args.vocab_file).write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    set_dex = load_set_dex(args.set_dex) if args.set_dex else None
    if args.set_dex and not set_dex:
        print(f"Warning: set_dex not found at {args.set_dex} — run import_randbats_dex.py")

    builder = FeatureBuilder(
        Config(
            side_hash_dim=args.side_hash_dim,
            move_hash_dim=args.move_hash_dim,
            volatile_hash_dim=args.volatile_hash_dim,
            turn_cap=args.turn_cap,
            item_hash_dim=args.item_hash_dim,
            ability_hash_dim=args.ability_hash_dim,
        ),
        vocab=vocab,
        set_dex=set_dex,
        mismatch_penalty=args.mismatch_penalty,
        slot_mode=args.slot_mode,
    )

    train_ds = make_dataset(
        input_paths,
        builder,
        split="train",
        val_every=args.val_every,
        max_examples=args.max_examples,
        shuffle_buffer=args.shuffle_buffer,
        filter_voluntary=args.filter_voluntary,
        interleave_files=args.interleave_files,
        cache=args.cache_dataset,
    )
    val_ds = make_dataset(
        input_paths,
        builder,
        split="val",
        val_every=args.val_every,
        max_examples=args.max_val_examples,
        shuffle_buffer=args.shuffle_buffer,
        filter_voluntary=args.filter_voluntary,
        interleave_files=max(1, args.interleave_files // 2),
        cache=False,
    )

    if args.repeat:
        train_ds = train_ds.repeat()
        if args.validation_steps or args.max_val_examples:
            val_ds = val_ds.repeat()

    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)

    train_ds = train_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model(
        builder=builder,
        dense_units=args.dense_units,
        dense_units2=args.dense_units2,
        dropout=args.dropout,
        dropout2=args.dropout2,
        l2_reg=args.l2,
        vocab_size=len(vocab),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_move_id_top5", mode="max", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_move_id_top5", mode="max", patience=2),
    ]

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch == 0 and args.max_examples:
        steps_per_epoch = math.ceil(args.max_examples / args.batch_size)
    validation_steps = args.validation_steps
    if validation_steps == 0 and args.max_val_examples:
        validation_steps = math.ceil(args.max_val_examples / args.batch_size)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch if steps_per_epoch else None,
        validation_steps=validation_steps if validation_steps else None,
    )

    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / "model.keras")
    if args.vocab_file:
        print(f"Saved move vocab to {args.vocab_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RB multi-head action model.")
    parser.add_argument(
        "--input_file",
        default=str(ACTION_CHUNKS_DIR / "rb_action_data_00001.json"),
    )
    parser.add_argument("--input_dir", default=str(ACTION_CHUNKS_DIR))
    parser.add_argument("--base_name", default="rb_action_data")
    parser.add_argument("--model_dir", default=str(MULTIHEAD_MODEL_DIR))
    parser.add_argument("--vocab_file", default=str(MOVE_VOCAB_PATH))
    parser.add_argument("--set_dex", default=str(RB_SET_DEX_PATH))
    parser.add_argument("--mismatch_penalty", type=float, default=0.15)
    parser.add_argument("--top_moves", type=int, default=250)
    parser.add_argument("--min_move_freq", type=int, default=5)
    parser.add_argument(
        "--slot_mode",
        default="exact",
        choices=["exact", "sorted_active"],
        help="exact = team preview slot order (required for switch labels).",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=20)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_val_examples", type=int, default=0)
    parser.add_argument("--shuffle_buffer", type=int, default=4096)
    parser.add_argument("--filter_voluntary", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--validation_steps", type=int, default=0)
    parser.add_argument("--dense_units", type=int, default=256)
    parser.add_argument("--dense_units2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dropout2", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--side_hash_dim", type=int, default=32)
    parser.add_argument("--move_hash_dim", type=int, default=64)
    parser.add_argument("--volatile_hash_dim", type=int, default=8)
    parser.add_argument("--turn_cap", type=int, default=100)
    parser.add_argument("--item_hash_dim", type=int, default=8)
    parser.add_argument("--ability_hash_dim", type=int, default=8)
    parser.add_argument(
        "--interleave_files",
        type=int,
        default=4,
        help="(ignored) chunk files are read sequentially with shuffled order",
    )
    parser.add_argument(
        "--cache_dataset",
        action="store_true",
        help="Cache parsed features in RAM after 1st epoch (faster later epochs)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
