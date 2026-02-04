"""
Train a TensorFlow model to predict win probability from OU winrate chunks.

Expected data: ou_winrate_chunks/ou_winrate_data_*.json
Each example is a game state with a winner label (1 = our side wins).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from materials import TYPE_CHART
from pokedex_9G_complete import pokemon_data_gen9


TYPE_LIST = [
    "unknown",
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]

STATUS_LIST = ["none", "brn", "par", "slp", "frz", "psn", "tox"]
WEATHER_LIST = ["none", "sunnyday", "raindance", "sandstorm", "hail", "snow"]
TERRAIN_LIST = ["none", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]

ACTIVE_BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]

MAX_TEAM_SIZE = 6
TOTAL_POKEMON = MAX_TEAM_SIZE * 2

TYPE_NAME_MAP = {key.lower(): key for key in TYPE_CHART}
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


def normalize_type_name(type_name: str) -> Optional[str]:
    if not type_name:
        return None
    return TYPE_NAME_MAP.get(type_name.strip().lower())


def canonicalize_species_norm(norm: str) -> str:
    if not norm:
        return norm
    base = norm
    if base.endswith("tera"):
        base = base[:-4]
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


def type_multiplier(attacker_type: str, defender_types: List[str]) -> float:
    atk = normalize_type_name(attacker_type)
    if not atk or not defender_types:
        return 1.0
    chart = TYPE_CHART.get(atk, {})
    mult = 1.0
    for defender in defender_types:
        def_name = normalize_type_name(defender)
        if not def_name:
            continue
        mult *= chart.get(def_name, 1.0)
    return mult


def best_type_multiplier(attacker_types: List[str], defender_types: List[str]) -> float:
    if not attacker_types or not defender_types:
        return 1.0
    return max(type_multiplier(t, defender_types) for t in attacker_types)


def speed_multiplier(boost: int) -> float:
    if boost >= 0:
        return (2 + boost) / 2
    return 2 / (2 - boost)

def build_species_index() -> Tuple[Dict[str, str], Dict[str, int]]:
    norm_to_key: Dict[str, str] = {}
    for key, entry in pokemon_data_gen9.items():
        for candidate in [key, entry.get("name", "")]:
            norm = normalize_species_name(candidate)
            if norm and norm not in norm_to_key:
                norm_to_key[norm] = key
    species_keys = sorted(pokemon_data_gen9.keys())
    species_to_id = {key: idx + 1 for idx, key in enumerate(species_keys)}
    return norm_to_key, species_to_id


def normalize_weather(value: Optional[str]) -> str:
    token = normalize_token(value)
    if not token:
        return "none"
    if "sun" in token:
        return "sunnyday"
    if "rain" in token:
        return "raindance"
    if "sand" in token:
        return "sandstorm"
    if "hail" in token:
        return "hail"
    if "snow" in token:
        return "snow"
    return "none"


def normalize_terrain(value: Optional[str]) -> str:
    token = normalize_token(value)
    if not token:
        return "none"
    if "electric" in token:
        return "electricterrain"
    if "grassy" in token:
        return "grassyterrain"
    if "misty" in token:
        return "mistyterrain"
    if "psychic" in token:
        return "psychicterrain"
    return "none"


@dataclass
class FeatureConfig:
    side_hash_dim: int = 32
    move_hash_dim: int = 64
    volatile_hash_dim: int = 8
    turn_cap: int = 100


class FeatureBuilder:
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.norm_to_key, self.species_to_id = build_species_index()
        self.type_to_id = {name: idx for idx, name in enumerate(TYPE_LIST)}
        self.status_to_id = {name: idx for idx, name in enumerate(STATUS_LIST)}
        self.weather_to_id = {name: idx for idx, name in enumerate(WEATHER_LIST)}
        self.terrain_to_id = {name: idx for idx, name in enumerate(TERRAIN_LIST)}
        self.numeric_dim = self._numeric_dim()
        self.global_dim = self._global_dim()
        self._warned_species: set[str] = set()

    def _numeric_dim(self) -> int:
        base = 0
        base += 4  # hp, fainted, revealed, is_active
        base += 6  # base stats
        base += 1  # weight
        base += 1  # volatile count
        base += self.config.volatile_hash_dim
        base += 1  # has_substitute
        return base

    def _base_global_dim(self) -> int:
        return 7

    def _matchup_dim(self) -> int:
        base = 11
        base += len(ACTIVE_BOOST_KEYS) * 2
        return base

    def _global_dim(self) -> int:
        return (
            (self.config.side_hash_dim * 2)
            + (self.config.move_hash_dim * 2)
            + self._base_global_dim()
            + self._matchup_dim()
        )
    
    def _team_order(self, team: Dict[str, dict], active: Optional[str]) -> List[str]:
        names = list(team.keys())
        ordered: List[str] = []
        if active and active in team:
            ordered.append(active)
        ordered.extend(sorted([n for n in names if n != active]))
        while len(ordered) < MAX_TEAM_SIZE:
            ordered.append("")
        return ordered[:MAX_TEAM_SIZE]

    def _pokedex_entry(self, species: str) -> Optional[dict]:
        norm = normalize_species_name(species)
        key = self.norm_to_key.get(norm)
        if not key:
            alt = canonicalize_species_norm(norm)
            if alt != norm:
                key = self.norm_to_key.get(alt)
        if not key:
            if species not in self._warned_species:
                self._warned_species.add(species)
            #print(f"Warning: Unknown species '{species}'", file=sys.stderr)
            return None
        return pokemon_data_gen9.get(key)

    def _species_id(self, species: str) -> int:
        norm = normalize_species_name(species)
        key = self.norm_to_key.get(norm)
        if not key:
            alt = canonicalize_species_norm(norm)
            if alt != norm:
                key = self.norm_to_key.get(alt)
        if not key:
            if species not in self._warned_species:
                self._warned_species.add(species)
            #print(f"Warning: Unknown species '{species}'", file=sys.stderr)
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

    def _base_stats(self, entry: Optional[dict]) -> List[float]:
        if not entry:
            #print("Warning: Missing pokedex entry for base stats", file=sys.stderr)
            return [0.0] * 6
        stats = entry.get("stats", {})
        return [
            stats.get("HP", 0) / 255.0,
            stats.get("Attack", 0) / 255.0,
            stats.get("Defense", 0) / 255.0,
            stats.get("Sp. Atk", 0) / 255.0,
            stats.get("Sp. Def", 0) / 255.0,
            stats.get("Speed", 0) / 255.0,
        ]

    def _active_boosts(self, boosts: Dict[str, int]) -> List[float]:
        values: List[float] = []
        for key in ACTIVE_BOOST_KEYS:
            raw = int(boosts.get(key, 0)) if boosts else 0
            clipped = max(min(raw, 6), -6)
            values.append(clipped / 6.0)
        return values

    def _status_id(self, status: Optional[str]) -> int:
        token = (status or "none").lower()
        return self.status_to_id.get(token, 0)

    def _volatile_features(self, volatiles: List[str]) -> Tuple[float, np.ndarray, float]:
        tokens = [normalize_token(v) for v in (volatiles or []) if v]
        volatile_count = min(len(tokens), 10) / 10.0
        hashed = hash_tokens(tokens, self.config.volatile_hash_dim)
        has_substitute = 1.0 if any("substitute" in t for t in tokens) else 0.0
        return volatile_count, hashed, has_substitute

    def _find_team_member(self, team: Dict[str, dict], active_name: Optional[str]) -> Tuple[str, dict]:
        if not active_name or not team:
            return "", {}
        if active_name in team:
            return active_name, team[active_name]
        active_norm = normalize_species_name(active_name)
        for name, info in team.items():
            if normalize_species_name(name) == active_norm:
                return name, info
        return active_name, team.get(active_name, {})

    def _estimated_speed(self, entry: Optional[dict], boosts: Dict[str, int]) -> float:
        if not entry:
            return 0.0
        base_speed = entry.get("stats", {}).get("Speed", 0)
        boost = int(boosts.get("spe", 0)) if boosts else 0
        return base_speed * speed_multiplier(boost)

    def _team_matchup_ratios(self, team: Dict[str, dict], opp_types: List[str]) -> Tuple[float, float]:
        if not team or not opp_types:
            return 0.0, 0.0
        resist = 0
        weak = 0
        count = 0
        for name, info in team.items():
            species = info.get("species") or name
            entry = self._pokedex_entry(species)
            if not entry:
                continue
            member_types = entry.get("types", [])
            if not member_types:
                continue
            mult = best_type_multiplier(opp_types, member_types)
            if mult >= 2.0:
                weak += 1
            elif mult <= 0.5:
                resist += 1
            count += 1
        if count == 0:
            return 0.0, 0.0
        return resist / count, weak / count

    def build_example(self, example: dict) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        state = example.get("state")
        if not state:
            return None

        our_team = state.get("our_team", {})
        opp_team = state.get("opponent_team", {})
        our_active = state.get("our_active")
        opp_active = state.get("opponent_active")
        our_active_name, our_active_info = self._find_team_member(our_team, our_active)
        opp_active_name, opp_active_info = self._find_team_member(opp_team, opp_active)

        our_active_entry = self._pokedex_entry(our_active_info.get("species") or our_active_name)
        opp_active_entry = self._pokedex_entry(opp_active_info.get("species") or opp_active_name)
        our_active_types = our_active_entry.get("types", []) if our_active_entry else []
        opp_active_types = opp_active_entry.get("types", []) if opp_active_entry else []

        species_ids: List[int] = []
        type1_ids: List[int] = []
        type2_ids: List[int] = []
        status_ids: List[int] = []
        numeric_rows: List[List[float]] = []

        for team, active in [(our_team, our_active), (opp_team, opp_active)]:
            for name in self._team_order(team, active):
                info = team.get(name, {})
                species = info.get("species") or name
                entry = self._pokedex_entry(species)
                species_ids.append(self._species_id(species))
                t1, t2 = self._type_ids(entry)
                type1_ids.append(t1)
                type2_ids.append(t2)
                status_ids.append(self._status_id(info.get("status")))

                hp = float(info.get("hp_percent", 1.0))
                fainted = 1.0 if info.get("fainted") else 0.0
                revealed = 1.0 if info.get("revealed") else 0.0
                is_active = 1.0 if name and active and name == active else 0.0
                base_stats = self._base_stats(entry)
                weight = (entry.get("weight", 0.0) / 1000.0) if entry else 0.0
                volatile_count, volatile_h, has_sub = self._volatile_features(info.get("volatiles", []))

                numeric_row: List[float] = []
                numeric_row.extend([hp, fainted, revealed, is_active])
                numeric_row.extend(base_stats)
                numeric_row.append(weight)
                numeric_row.append(volatile_count)
                numeric_row.extend(volatile_h.tolist())
                numeric_row.append(has_sub)
                numeric_rows.append(numeric_row)

        if len(species_ids) != TOTAL_POKEMON:
            return None

        field = state.get("field_conditions", {})
        weather = normalize_weather(field.get("weather"))
        terrain = normalize_terrain(field.get("terrain"))
        weather_id = self.weather_to_id.get(weather, 0)
        terrain_id = self.terrain_to_id.get(terrain, 0)

        our_side = state.get("our_side_conditions", {}) or {}
        opp_side = state.get("opponent_side_conditions", {}) or {}
        our_side_tokens = [f"{normalize_token(k)}:{v}" for k, v in our_side.items()]
        opp_side_tokens = [f"{normalize_token(k)}:{v}" for k, v in opp_side.items()]
        side_vec = np.concatenate(
            [
                hash_tokens(our_side_tokens, self.config.side_hash_dim),
                hash_tokens(opp_side_tokens, self.config.side_hash_dim),
            ]
        )

        our_last_move = normalize_token(state.get("our_last_move"))
        opp_last_move = normalize_token(state.get("opponent_last_move"))
        move_vec = np.concatenate(
            [
                hash_tokens([our_last_move], self.config.move_hash_dim),
                hash_tokens([opp_last_move], self.config.move_hash_dim),
            ]
        )

        turn = int(state.get("turn", example.get("turn", 0)) or 0)
        turn_norm = min(turn, self.config.turn_cap) / float(self.config.turn_cap)

        def aggregate(team: Dict[str, dict]) -> Tuple[float, float, float]:
            hp_sum = 0.0
            fainted = 0.0
            revealed = 0.0
            for info in team.values():
                hp_sum += float(info.get("hp_percent", 1.0))
                fainted += 1.0 if info.get("fainted") else 0.0
                revealed += 1.0 if info.get("revealed") else 0.0
            if not team:
                return 0.0, 0.0, 0.0
            size = max(len(team), 1)
            return hp_sum / size, fainted / size, revealed / size

        our_hp, our_fainted, our_revealed = aggregate(our_team)
        opp_hp, opp_fainted, opp_revealed = aggregate(opp_team)

        our_active_hp = float(our_active_info.get("hp_percent", 0.0)) if our_active_info else 0.0
        opp_active_hp = float(opp_active_info.get("hp_percent", 0.0)) if opp_active_info else 0.0
        active_hp_diff = our_active_hp - opp_active_hp

        our_speed = self._estimated_speed(our_active_entry, our_active_info.get("boosts", {}))
        opp_speed = self._estimated_speed(opp_active_entry, opp_active_info.get("boosts", {}))
        speed_diff = (our_speed - opp_speed) / 400.0
        speed_diff = max(min(speed_diff, 1.0), -1.0)
        if our_speed == 0.0 or opp_speed == 0.0:
            speed_adv = 0.5
        else:
            speed_adv = 1.0 if our_speed >= opp_speed else 0.0

        our_active_boosts = self._active_boosts(our_active_info.get("boosts", {}))
        opp_active_boosts = self._active_boosts(opp_active_info.get("boosts", {}))

        our_offense_mult = best_type_multiplier(our_active_types, opp_active_types)
        opp_offense_mult = best_type_multiplier(opp_active_types, our_active_types)

        our_resist_ratio, our_weak_ratio = self._team_matchup_ratios(our_team, opp_active_types)
        opp_resist_ratio, opp_weak_ratio = self._team_matchup_ratios(opp_team, our_active_types)

        global_num = np.concatenate(
            [
                side_vec,
                move_vec,
                np.array(
                    [
                        turn_norm,
                        our_hp,
                        opp_hp,
                        our_fainted,
                        opp_fainted,
                        our_revealed,
                        opp_revealed,
                        our_active_hp,
                        opp_active_hp,
                        active_hp_diff,
                        speed_diff,
                        speed_adv,
                        *our_active_boosts,
                        *opp_active_boosts,
                        our_offense_mult,
                        opp_offense_mult,
                        our_resist_ratio,
                        our_weak_ratio,
                        opp_resist_ratio,
                        opp_weak_ratio,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        features = {
            "species_ids": np.array(species_ids, dtype=np.int32),
            "type1_ids": np.array(type1_ids, dtype=np.int32),
            "type2_ids": np.array(type2_ids, dtype=np.int32),
            "status_ids": np.array(status_ids, dtype=np.int32),
            "numeric_feats": np.array(numeric_rows, dtype=np.float32),
            "global_num": global_num,
            "weather_id": np.array([weather_id], dtype=np.int32),
            "terrain_id": np.array([terrain_id], dtype=np.int32),
        }

        label = np.array([float(example.get("winner", 0))], dtype=np.float32)
        return features, label

    def output_signature(self) -> Tuple[Dict[str, tf.TensorSpec], tf.TensorSpec]:
        return (
            {
                "species_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
                "type1_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
                "type2_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
                "status_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
                "numeric_feats": tf.TensorSpec(
                    shape=(TOTAL_POKEMON, self.numeric_dim), dtype=tf.float32
                ),
                "global_num": tf.TensorSpec(shape=(self.global_dim,), dtype=tf.float32),
                "weather_id": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "terrain_id": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )


def iter_examples(file_paths: List[Path]) -> Iterable[dict]:
    for path in file_paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for example in data:
            yield example


def iter_filtered_examples(
    file_paths: List[Path],
    split_mode: str,
    split: str,
    val_every: int,
    max_examples: int,
) -> Iterable[dict]:
    idx = 0
    yielded = 0
    for example in iter_examples(file_paths):
        if split_mode == "example":
            is_val = (idx % val_every) == 0
            idx += 1
            if split == "train" and is_val:
                continue
            if split == "val" and not is_val:
                continue
        elif split_mode != "file":
            raise ValueError(f"Unknown split mode: {split_mode}")
        yield example
        yielded += 1
        if max_examples and yielded >= max_examples:
            break


def split_files(
    file_paths: List[Path],
    split_mode: str,
    val_file_count: int,
    val_file_fraction: float,
) -> Tuple[List[Path], List[Path]]:
    if split_mode != "file":
        return file_paths, file_paths
    total = len(file_paths)
    if total < 2:
        raise ValueError("Need at least 2 files for file split mode.")
    if val_file_count > 0:
        val_count = min(val_file_count, total - 1)
    else:
        if not (0.0 < val_file_fraction < 1.0):
            raise ValueError("val_file_fraction must be between 0 and 1.")
        val_count = max(1, int(round(total * val_file_fraction)))
        if total - val_count < 1:
            val_count = total - 1
    return file_paths[:-val_count], file_paths[-val_count:]


def compute_baseline(
    file_paths: List[Path],
    split_mode: str,
    split: str,
    val_every: int,
    max_examples: int,
) -> Optional[Tuple[int, float, float]]:
    total = 0
    wins = 0
    for example in iter_filtered_examples(
        file_paths=file_paths,
        split_mode=split_mode,
        split=split,
        val_every=val_every,
        max_examples=max_examples,
    ):
        total += 1
        wins += int(example.get("winner", 0))
    if total == 0:
        return None
    win_rate = wins / total
    acc = max(win_rate, 1.0 - win_rate)
    return total, win_rate, acc


def make_dataset(
    file_paths: List[Path],
    builder: FeatureBuilder,
    split: str,
    split_mode: str,
    val_every: int,
    max_examples: int,
    shuffle_buffer: int,
) -> tf.data.Dataset:
    def gen():
        for example in iter_filtered_examples(
            file_paths=file_paths,
            split_mode=split_mode,
            split=split,
            val_every=val_every,
            max_examples=max_examples,
        ):
            built = builder.build_example(example)
            if built is None:
                continue
            yield built

    dataset = tf.data.Dataset.from_generator(gen, output_signature=builder.output_signature())
    if split == "train":
        dataset = dataset.shuffle(shuffle_buffer)
    return dataset


def build_model(
    builder: FeatureBuilder,
    dense_units: int,
    dense_units2: int,
    dropout: float,
    dropout2: float,
    l2_reg: float,
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

    species_emb = tf.keras.layers.Embedding(
        input_dim=len(builder.species_to_id) + 1, output_dim=16, name="species_emb"
    )(species_in)
    type_emb_layer = tf.keras.layers.Embedding(
        input_dim=len(TYPE_LIST), output_dim=4, name="type_emb"
    )
    type_emb = tf.keras.layers.Add()([type_emb_layer(type1_in), type_emb_layer(type2_in)])
    status_emb = tf.keras.layers.Embedding(
        input_dim=len(STATUS_LIST), output_dim=3, name="status_emb"
    )(status_in)

    per_poke = tf.keras.layers.Concatenate(axis=-1)([species_emb, type_emb, status_emb, numeric_in])
    per_poke = tf.keras.layers.Flatten()(per_poke)

    weather_emb = tf.keras.layers.Embedding(
        input_dim=len(WEATHER_LIST), output_dim=3, name="weather_emb"
    )(weather_in)
    terrain_emb = tf.keras.layers.Embedding(
        input_dim=len(TERRAIN_LIST), output_dim=3, name="terrain_emb"
    )(terrain_in)
    global_cat = tf.keras.layers.Concatenate(axis=-1)(
        [global_in, tf.keras.layers.Flatten()(weather_emb), tf.keras.layers.Flatten()(terrain_emb)]
    )

    x = tf.keras.layers.Concatenate(axis=-1)([per_poke, global_cat])
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units2, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="win_prob")(x)

    model = tf.keras.Model(
        inputs=[species_in, type1_in, type2_in, status_in, numeric_in, global_in, weather_in, terrain_in],
        outputs=output,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def save_config(path: Path, builder: FeatureBuilder, args: argparse.Namespace) -> None:
    payload = {
        "side_hash_dim": builder.config.side_hash_dim,
        "move_hash_dim": builder.config.move_hash_dim,
        "volatile_hash_dim": builder.config.volatile_hash_dim,
        "turn_cap": builder.config.turn_cap,
        "numeric_dim": builder.numeric_dim,
        "global_dim": builder.global_dim,
        "val_every": args.val_every,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_config(path: Path) -> FeatureConfig:
    if not path.exists():
        return FeatureConfig()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return FeatureConfig(
        side_hash_dim=int(data.get("side_hash_dim", 32)),
        move_hash_dim=int(data.get("move_hash_dim", 64)),
        volatile_hash_dim=int(data.get("volatile_hash_dim", 8)),
        turn_cap=int(data.get("turn_cap", 100)),
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.split_mode == "example" and args.val_every < 2:
        raise ValueError("val_every must be >= 2")

    data_dir = Path(args.data_dir)
    file_paths = sorted(data_dir.glob("ou_winrate_data_*.json"))
    if args.max_files:
        file_paths = file_paths[: args.max_files]
    if not file_paths:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    train_files, val_files = split_files(
        file_paths=file_paths,
        split_mode=args.split_mode,
        val_file_count=args.val_file_count,
        val_file_fraction=args.val_file_fraction,
    )
    print(
        f"Split mode: {args.split_mode} | train files: {len(train_files)} | val files: {len(val_files)}"
    )

    config = FeatureConfig(
        side_hash_dim=args.side_hash_dim,
        move_hash_dim=args.move_hash_dim,
        volatile_hash_dim=args.volatile_hash_dim,
        turn_cap=args.turn_cap,
    )
    builder = FeatureBuilder(config)

    if args.baseline or args.baseline_only:
        train_base = compute_baseline(
            file_paths=train_files,
            split_mode=args.split_mode,
            split="train",
            val_every=args.val_every,
            max_examples=args.max_examples,
        )
        val_base = compute_baseline(
            file_paths=val_files,
            split_mode=args.split_mode,
            split="val",
            val_every=args.val_every,
            max_examples=args.max_val_examples,
        )
        if train_base:
            total, win_rate, acc = train_base
            print(f"Baseline train: n={total} win_rate={win_rate:.4f} acc={acc:.4f}")
        if val_base:
            total, win_rate, acc = val_base
            print(f"Baseline val:   n={total} win_rate={win_rate:.4f} acc={acc:.4f}")
        if args.baseline_only:
            return

    train_ds = make_dataset(
        train_files,
        builder,
        split="train",
        split_mode=args.split_mode,
        val_every=args.val_every,
        max_examples=args.max_examples,
        shuffle_buffer=args.shuffle_buffer,
    )
    val_ds = make_dataset(
        val_files,
        builder,
        split="val",
        split_mode=args.split_mode,
        val_every=args.val_every,
        max_examples=args.max_val_examples,
        shuffle_buffer=args.shuffle_buffer,
    )

    train_ds = train_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model(
        builder=builder,
        dense_units=args.dense_units,
        dense_units2=args.dense_units2,
        dropout=args.dropout,
        dropout2=args.dropout2,
        l2_reg=args.l2,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=2),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
    )

    model_path = Path(args.model_dir)
    if model_path.suffix in {".keras", ".h5"}:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        save_config(model_path.parent / "config.json", builder, args)
    else:
        model_path.mkdir(parents=True, exist_ok=True)
        save_path = model_path / "model.keras"
        model.save(save_path)
        save_config(model_path / "config.json", builder, args)


def predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model_dir)
    if model_path.is_dir():
        candidate = model_path / "model.keras"
        load_path = candidate if candidate.exists() else model_path
    else:
        load_path = model_path
    model = tf.keras.models.load_model(load_path)

    config_dir = model_path if model_path.is_dir() else model_path.parent
    config = load_config(config_dir / "config.json")
    builder = FeatureBuilder(config)

    input_path = Path(args.predict_file)
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    examples = data if isinstance(data, list) else [data]
    outputs = []
    for example in examples[: args.max_predict]:
        built = builder.build_example(example)
        if not built:
            continue
        features, _ = built
        batch = {k: np.expand_dims(v, axis=0) for k, v in features.items()}
        prob = float(model.predict(batch, verbose=0)[0][0])
        outputs.append(prob)

    for idx, prob in enumerate(outputs, 1):
        print(f"{idx:04d} win_prob={prob:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="ou_winrate_chunks")
    parser.add_argument("--model_dir", default="PokemonOUaimodel/ou_winrate_model")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--split_mode", choices=["example", "file"], default="file")
    parser.add_argument("--val_file_count", type=int, default=0)
    parser.add_argument("--val_file_fraction", type=float, default=0.1)
    parser.add_argument("--val_every", type=int, default=20)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_val_examples", type=int, default=0)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--dense_units", type=int, default=192)
    parser.add_argument("--dense_units2", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dropout2", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--side_hash_dim", type=int, default=32)
    parser.add_argument("--move_hash_dim", type=int, default=64)
    parser.add_argument("--volatile_hash_dim", type=int, default=8)
    parser.add_argument("--turn_cap", type=int, default=100)
    parser.add_argument("--predict_file", default="")
    parser.add_argument("--max_predict", type=int, default=20)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.predict_file:
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
