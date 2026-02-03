"""
Train a model to predict whether the next action is a move or a switch.

Input dataset: JSON list produced by Replay_data/extract_training_data.py
Each example contains:
  - state (game state)
  - action_type: "move" or "switch"
  - is_voluntary: True/False
"""

from __future__ import annotations

import argparse
import math
import json
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import os
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pokedex_9G_complete import pokemon_data_gen9
from materials import TYPE_CHART

MAX_TEAM_SIZE = 6
TOTAL_POKEMON = MAX_TEAM_SIZE * 2

STATUS_LIST = ["none", "brn", "par", "slp", "frz", "psn", "tox"]
WEATHER_LIST = ["none", "sunnyday", "raindance", "sandstorm", "hail", "snow"]
TERRAIN_LIST = ["none", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]

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


def normalize_type_name(type_name: str) -> Optional[str]:
    if not type_name:
        return None
    return TYPE_NAME_MAP.get(type_name.strip().lower())


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
    for key in pokemon_data_gen9.keys():
        norm_to_key[normalize_species_name(key)] = key
    species_to_id = {key: idx + 1 for idx, key in enumerate(sorted(pokemon_data_gen9.keys()))}
    return norm_to_key, species_to_id


@dataclass
class Config:
    side_hash_dim: int = 32
    move_hash_dim: int = 64
    volatile_hash_dim: int = 8
    turn_cap: int = 100


class FeatureBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.norm_to_key, self.species_to_id = build_species_index()
        self.type_to_id = {t.lower(): i + 1 for i, t in enumerate(TYPE_CHART.keys())}
        self.status_to_id = {s: i + 1 for i, s in enumerate(STATUS_LIST)}
        self.weather_to_id = {w: i for i, w in enumerate(WEATHER_LIST)}
        self.terrain_to_id = {t: i for i, t in enumerate(TERRAIN_LIST)}
        self._warned_species: set[str] = set()

        self.numeric_dim = 1 + 1 + 1 + 6 + 1 + 5 + 1 + self.config.volatile_hash_dim
        # hp, fainted, is_active, 6 base stats, weight, 5 boosts, vol_count, vol_hash
        self.global_dim = (self.config.side_hash_dim * 2) + (self.config.move_hash_dim * 2) + 1
        # side hash (ours/opp), move hash (ours/opp), turn_norm; weather/terrain handled separately

    def output_signature(self) -> Tuple[Dict[str, tf.TensorSpec], tf.TensorSpec]:
        features = {
            "species_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
            "type1_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
            "type2_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
            "status_ids": tf.TensorSpec(shape=(TOTAL_POKEMON,), dtype=tf.int32),
            "numeric_feats": tf.TensorSpec(shape=(TOTAL_POKEMON, self.numeric_dim), dtype=tf.float32),
            "global_num": tf.TensorSpec(shape=(self.global_dim,), dtype=tf.float32),
            "weather_id": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            "terrain_id": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        }
        label = tf.TensorSpec(shape=(1,), dtype=tf.float32)
        return features, label

    def build_example(self, example: Dict[str, object]) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        state = example.get("state", {})
        if not state:
            return None

        my_team_list = state.get("my_team", []) or []
        opp_team_list = state.get("opp_team", []) or []

        our_order = self._team_order(my_team_list)
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

                hp = float((info or {}).get("hp_percent", 1.0))
                fainted = 1.0 if (info or {}).get("fainted") else 0.0
                is_active = 1.0 if (info or {}).get("is_active") else 0.0
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

                numeric_row: List[float] = []
                numeric_row.extend([hp, fainted, is_active])
                numeric_row.extend(base_stats)
                numeric_row.append(weight)
                numeric_row.extend(boosts_vec)
                numeric_row.append(vol_count)
                numeric_row.extend(vol_hash.tolist())
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

        features = {
            "species_ids": np.array(species_ids, dtype=np.int32),
            "type1_ids": np.array(type1_ids, dtype=np.int32),
            "type2_ids": np.array(type2_ids, dtype=np.int32),
            "status_ids": np.array(status_ids, dtype=np.int32),
            "numeric_feats": np.array(numeric_rows, dtype=np.float32),
            "global_num": global_num.astype(np.float32),
            "weather_id": np.array([weather_id], dtype=np.int32),
            "terrain_id": np.array([terrain_id], dtype=np.int32),
        }

        label = 1.0 if example.get("action_type") == "move" else 0.0
        return features, np.array([label], dtype=np.float32)

    def _team_order(self, team_list: List[Dict[str, object]]) -> List[Dict[str, object]]:
        active = [p for p in team_list if p.get("is_active")]
        inactive = [p for p in team_list if not p.get("is_active")]
        inactive_sorted = sorted(inactive, key=lambda p: normalize_species_name(p.get("species", "")))
        ordered = active + inactive_sorted
        while len(ordered) < MAX_TEAM_SIZE:
            ordered.append({})
        return ordered[:MAX_TEAM_SIZE]

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
            if "-" in species and species not in self._warned_species:
                print(f"Warning: Unknown species '{species}'")
                self._warned_species.add(species)
            return None
        return pokemon_data_gen9.get(key)

    def _species_id(self, species: str) -> int:
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
        return self.status_to_id.get(status.lower(), 0)

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


def make_dataset(
    input_files: List[Path],
    builder: FeatureBuilder,
    split: str,
    val_every: int,
    max_examples: int,
    shuffle_buffer: int,
    filter_voluntary: bool,
) -> tf.data.Dataset:
    def gen():
        for example in iter_examples(
            input_files=input_files,
            split=split,
            val_every=val_every,
            max_examples=max_examples,
            filter_voluntary=filter_voluntary,
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
        [global_in, tf.keras.layers.Flatten()(weather_emb), tf.keras.layers.Flatten()(terrain_emb)]
    )

    x = tf.keras.layers.Concatenate(axis=-1)([per_poke, global_cat])
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    x = tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units2, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="move_prob")(x)

    model = tf.keras.Model(
        inputs=[species_in, type1_in, type2_in, status_in, numeric_in, global_in, weather_in, terrain_in],
        outputs=output,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")],
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

    builder = FeatureBuilder(
        Config(
            side_hash_dim=args.side_hash_dim,
            move_hash_dim=args.move_hash_dim,
            volatile_hash_dim=args.volatile_hash_dim,
            turn_cap=args.turn_cap,
        )
    )

    train_ds = make_dataset(
        input_paths,
        builder,
        split="train",
        val_every=args.val_every,
        max_examples=args.max_examples,
        shuffle_buffer=args.shuffle_buffer,
        filter_voluntary=args.filter_voluntary,
    )
    val_ds = make_dataset(
        input_paths,
        builder,
        split="val",
        val_every=args.val_every,
        max_examples=args.max_val_examples,
        shuffle_buffer=args.shuffle_buffer,
        filter_voluntary=args.filter_voluntary,
    )

    if args.repeat:
        train_ds = train_ds.repeat()
        if args.validation_steps or args.max_val_examples:
            val_ds = val_ds.repeat()

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

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch == 0 and args.repeat and args.max_examples:
        steps_per_epoch = math.ceil(args.max_examples / args.batch_size)
    validation_steps = args.validation_steps
    if validation_steps == 0 and args.repeat and args.max_val_examples:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="Replay_data/training_data.json")
    parser.add_argument("--input_dir", default="")
    parser.add_argument("--base_name", default="action_data")
    parser.add_argument("--model_dir", default="TensorFlows/action_model")
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
    parser.add_argument("--dense_units", type=int, default=192)
    parser.add_argument("--dense_units2", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dropout2", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--side_hash_dim", type=int, default=32)
    parser.add_argument("--move_hash_dim", type=int, default=64)
    parser.add_argument("--volatile_hash_dim", type=int, default=8)
    parser.add_argument("--turn_cap", type=int, default=100)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
