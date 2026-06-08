"""
Random Battle win-rate model (gen9randombattle).

Prédit P(victoire | état partiel) pour le joueur `state["player"]`.
Entraînement sur les chunks d'actions : label `winner` (le joueur actif a gagné le combat).

Même encodeur d'état que IA_multihead_predictor (FeatureBuilder + move_vocab).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
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
    RB_SET_DEX_PATH,
    WINRATE_MODEL_DIR,
)

setup_import_paths()

from common.set_dex_prior import load_set_dex
from random_battle.models.IA_multihead_predictor import (  # noqa: E402
    TOTAL_POKEMON,
    WEATHER_LIST,
    TERRAIN_LIST,
    Config,
    FeatureBuilder,
    iter_examples,
)

MAX_TEAM_SIZE = 6


def winrate_output_signature(builder: FeatureBuilder) -> Tuple[
    Dict[str, tf.TensorSpec],
    Dict[str, tf.TensorSpec],
    Dict[str, tf.TensorSpec],
]:
    features, _, _ = builder.output_signature()
    labels = {"win_prob": tf.TensorSpec(shape=(), dtype=tf.float32)}
    weights = {"win_prob": tf.TensorSpec(shape=(), dtype=tf.float32)}
    return features, labels, weights


def example_to_win_labels(example: Dict[str, object]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    win = 1.0 if bool(example.get("winner")) else 0.0
    labels = {"win_prob": np.array(win, dtype=np.float32)}
    weights = {"win_prob": np.array(1.0, dtype=np.float32)}
    return labels, weights


def build_winrate_model(
    builder: FeatureBuilder,
    *,
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

    species_emb = tf.keras.layers.Embedding(
        input_dim=len(builder.species_to_id) + 1, output_dim=16
    )(species_in)
    type_emb = tf.keras.layers.Embedding(input_dim=len(builder.type_to_id) + 1, output_dim=4)
    type1_emb = type_emb(type1_in)
    type2_emb = type_emb(type2_in)
    status_emb = tf.keras.layers.Embedding(
        input_dim=len(builder.status_to_id) + 1, output_dim=3
    )(status_in)

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
    win_out = tf.keras.layers.Dense(1, activation="sigmoid", name="win_prob")(x)

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
        outputs={"win_prob": win_out},
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"win_prob": tf.keras.losses.BinaryCrossentropy()},
        metrics={
            "win_prob": [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="prec"),
                tf.keras.metrics.Recall(name="rec"),
            ],
        },
    )
    return model


def make_winrate_dataset(
    input_files: List[Path],
    builder: FeatureBuilder,
    *,
    split: str,
    val_every: int,
    max_examples: int,
    shuffle_buffer: int,
    filter_voluntary: bool,
    cache: bool = False,
) -> tf.data.Dataset:
    if not input_files:
        raise FileNotFoundError("No input files for dataset")

    signature = winrate_output_signature(builder)
    paths = list(input_files)
    if split == "train":
        random.shuffle(paths)

    def generator():
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
                features, _, _ = built
                labels, weights = example_to_win_labels(example)
                yield features, labels, weights
                yielded += 1
                if max_examples and yielded >= max_examples:
                    return

    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)
    if split == "train":
        dataset = dataset.shuffle(shuffle_buffer)
    if cache:
        dataset = dataset.cache()
    return dataset


def estimate_positive_rate(
    input_files: List[Path],
    *,
    max_examples: int,
    filter_voluntary: bool,
) -> float:
    pos = 0
    total = 0
    for path in input_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for example in data:
            if filter_voluntary and not example.get("is_voluntary", True):
                continue
            total += 1
            if example.get("winner"):
                pos += 1
            if max_examples and total >= max_examples:
                return pos / total if total else 0.5
    return pos / total if total else 0.5


def train(args: argparse.Namespace) -> None:
    input_paths: List[Path] = []
    if args.input_dir:
        input_dir = Path(args.input_dir)
        input_paths = sorted(input_dir.glob(f"{args.base_name}_*.json"))
        if not input_paths:
            raise FileNotFoundError(f"No chunk files in {input_dir}")
    else:
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Missing: {input_path}")
        input_paths = [input_path]

    vocab_path = Path(args.vocab_file)
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Move vocab required: {vocab_path}. Train multi-head first or pass --vocab_file."
        )
    vocab = {k: int(v) for k, v in json.loads(vocab_path.read_text(encoding="utf-8")).items()}

    set_dex = load_set_dex(args.set_dex) if args.set_dex else None
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

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch == 0 and args.max_examples:
        steps_per_epoch = math.ceil(args.max_examples / args.batch_size)
    elif steps_per_epoch == 0:
        steps_per_epoch = 1500

    validation_steps = args.validation_steps
    if validation_steps == 0 and args.max_val_examples:
        validation_steps = math.ceil(args.max_val_examples / args.batch_size)

    print(
        f"Train: jusqu'à {args.max_examples or 'tout'} ex./run, "
        f"{steps_per_epoch} steps/epoch x {args.epochs} epochs, "
        f"val {validation_steps} steps"
    )

    if args.balance_classes:
        pos_rate = estimate_positive_rate(
            input_paths,
            max_examples=min(args.max_examples or 200_000, 200_000),
            filter_voluntary=args.filter_voluntary,
        )
        print(f"Estimated positive rate (winner=1): {pos_rate:.3f}")

    train_ds = make_winrate_dataset(
        input_paths,
        builder,
        split="train",
        val_every=args.val_every,
        max_examples=args.max_examples,
        shuffle_buffer=args.shuffle_buffer,
        filter_voluntary=args.filter_voluntary,
        cache=args.cache_dataset,
    )
    val_ds = make_winrate_dataset(
        input_paths,
        builder,
        split="val",
        val_every=args.val_every,
        max_examples=args.max_val_examples,
        shuffle_buffer=args.shuffle_buffer,
        filter_voluntary=args.filter_voluntary,
        cache=False,
    )

    if args.repeat:
        train_ds = train_ds.repeat()
        if args.validation_steps or args.max_val_examples:
            val_ds = val_ds.repeat()

    train_ds = train_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_winrate_model(
        builder,
        dense_units=args.dense_units,
        dense_units2=args.dense_units2,
        dropout=args.dropout,
        dropout2=args.dropout2,
        l2_reg=args.l2,
        vocab_size=len(vocab),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_win_prob_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_win_prob_auc", mode="max", patience=2, factor=0.5
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch if steps_per_epoch else None,
        validation_steps=validation_steps if validation_steps else None,
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.keras")

    meta = {
        "label": "winner for state.player (battle outcome, not on-policy value)",
        "vocab_file": str(vocab_path),
        "set_dex": str(args.set_dex),
        "slot_mode": args.slot_mode,
        "filter_voluntary": args.filter_voluntary,
        "val_win_prob_auc": float(max(history.history.get("val_win_prob_auc", [0]))),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved win-rate model to {model_dir / 'model.keras'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RB win-rate model from action chunks.")
    parser.add_argument(
        "--input_file",
        default=str(ACTION_CHUNKS_DIR / "rb_action_data_00001.json"),
    )
    parser.add_argument("--input_dir", default=str(ACTION_CHUNKS_DIR))
    parser.add_argument("--base_name", default="rb_action_data")
    parser.add_argument("--model_dir", default=str(WINRATE_MODEL_DIR))
    parser.add_argument("--vocab_file", default=str(MOVE_VOCAB_PATH))
    parser.add_argument("--set_dex", default=str(RB_SET_DEX_PATH))
    parser.add_argument("--mismatch_penalty", type=float, default=0.15)
    parser.add_argument("--slot_mode", default="exact", choices=["exact", "sorted_active"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Nombre d'epochs (chaque epoch = steps_per_epoch batches, pas tout le disque).",
    )
    parser.add_argument("--val_every", type=int, default=20)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=400_000,
        help="Cap d'exemples train par epoch (0 = tout le corpus, très long).",
    )
    parser.add_argument("--max_val_examples", type=int, default=20_000)
    parser.add_argument("--shuffle_buffer", type=int, default=8192)
    parser.add_argument("--filter_voluntary", action="store_true")
    parser.add_argument("--balance_classes", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1500,
        help="Batches par epoch (1500 x 256 ≈ 384k exemples). 0 = déduit de max_examples.",
    )
    parser.add_argument("--validation_steps", type=int, default=80)
    parser.add_argument("--dense_units", type=int, default=256)
    parser.add_argument("--dense_units2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--dropout2", type=float, default=0.3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--side_hash_dim", type=int, default=32)
    parser.add_argument("--move_hash_dim", type=int, default=64)
    parser.add_argument("--volatile_hash_dim", type=int, default=8)
    parser.add_argument("--turn_cap", type=int, default=100)
    parser.add_argument("--item_hash_dim", type=int, default=8)
    parser.add_argument("--ability_hash_dim", type=int, default=8)
    parser.add_argument("--cache_dataset", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
