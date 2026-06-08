"""
Extract gen9randombattle (solo) from HolidayOugi/pokemon-showdown-replays on HF.

Fields: log, inputlog (optional), formatid, rating.
Uses RandomBattleTrainingExtractor + partial observability.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for _path in (_PROJECT_ROOT, _SCRIPT_DIR, _PROJECT_ROOT / "shared" / "data_extractors"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from chunk_io import ChunkBuffer  # noqa: E402
from extract_action_chunks import _escape_glob, choose_rating  # noqa: E402
from extract_pokechamp_training_data import TrainingExample  # noqa: E402
from rb_format_filter import accept_replay  # noqa: E402
from rb_training_extractor import RandomBattleTrainingExtractor  # noqa: E402
from random_battle.config import (  # noqa: E402
    ACTION_CHUNKS_HOLIDAYOUGI_DIR,
    FORMAT_ID,
    MIN_ELO_DEFAULT,
)

HOLIDAYOUGI_DATASET = "HolidayOugi/pokemon-showdown-replays"
DEFAULT_OUTPUT_DIR = ACTION_CHUNKS_HOLIDAYOUGI_DIR


def example_to_json(example: TrainingExample) -> Dict[str, object]:
    return asdict(example)


def iter_holidayougi_gen9_rb(
    *,
    dataset_name: str,
    data_files: List[str],
    split: str,
    min_rating: float,
    max_replays: int,
    require_log_check: bool,
    log_every_scanned: int = 5000,
):
    from datasets import load_dataset

    if data_files:
        escaped = [_escape_glob(name) for name in data_files]
        hf_paths = [f"hf://datasets/{dataset_name}/{name}" for name in escaped]
        dataset = load_dataset("parquet", data_files=hf_paths, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)

    accepted = 0
    scanned = 0
    for replay in dataset:
        scanned += 1
        if max_replays and accepted >= max_replays:
            break

        log_text = str(replay.get("log") or "")
        if not accept_replay(
            formatid=replay.get("formatid"),
            log_text=log_text,
            require_log_check=require_log_check,
        ):
            continue
        if choose_rating(replay) < min_rating:
            continue

        accepted += 1
        yield replay

    print(f"HolidayOugi {FORMAT_ID}: scanned={scanned}, accepted={accepted}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract gen9randombattle solo from HolidayOugi/pokemon-showdown-replays."
    )
    parser.add_argument("--dataset", default=HOLIDAYOUGI_DATASET)
    parser.add_argument(
        "--data_files",
        nargs="+",
        default=[],
        help='Optional parquet shards, e.g. "[Gen 9] Random Battle.parquet"',
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--min_rating", type=float, default=float(MIN_ELO_DEFAULT))
    parser.add_argument("--max_replays", type=int, default=0, help="0 = no limit")
    parser.add_argument(
        "--prefer_inputlog",
        action="store_true",
        help="Use inputlog for actions (better switch slot labels)",
    )
    parser.add_argument(
        "--skip_log_check",
        action="store_true",
        help="Only filter formatid (faster if data_files already target RB)",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_name", default="rb_action_data")
    parser.add_argument(
        "--max_chunk_bytes",
        type=int,
        default=50 * 1024 * 1024,
        help="Flush each JSON chunk at ~50 MiB",
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--log_every_scanned", type=int, default=1000)
    parser.add_argument("--log_every_accepted", type=int, default=100)
    parser.add_argument("--filter_voluntary", action="store_true")
    parser.add_argument(
        "--start_chunk_idx",
        type=int,
        default=0,
        help="0 = auto-continue after last chunk file in output_dir",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = RandomBattleTrainingExtractor()
    buffer = ChunkBuffer(
        output_dir=output_dir,
        base_name=args.base_name,
        max_bytes=args.max_chunk_bytes,
        pretty=args.pretty,
        start_chunk_idx=args.start_chunk_idx,
    )
    print(f"Writing chunks starting at index {buffer.chunk_idx:05d}", flush=True)
    total_examples = 0
    accepted = 0

    print(
        f"Streaming {args.dataset} ({FORMAT_ID}, rating>={args.min_rating}, "
        f"inputlog={'yes' if args.prefer_inputlog else 'no'})"
    )
    for replay in iter_holidayougi_gen9_rb(
        dataset_name=args.dataset,
        data_files=args.data_files,
        split=args.split,
        min_rating=args.min_rating,
        max_replays=args.max_replays,
        require_log_check=not args.skip_log_check,
        log_every_scanned=args.log_every_scanned,
    ):

        log_text = replay.get("log") or ""
        inputlog = replay.get("inputlog")
        examples = extractor.extract_from_log(
            str(log_text),
            inputlog=str(inputlog) if inputlog else None,
            prefer_inputlog=args.prefer_inputlog,
        )
        if not examples:
            continue

        accepted += 1
        if args.log_every_accepted and accepted % args.log_every_accepted == 0:
            print(
                f"Processed {accepted} replays, examples={total_examples}, "
                f"buffered={len(buffer.rows)}",
                flush=True,
            )

        for ex in examples:
            if args.filter_voluntary and not ex.is_voluntary:
                continue
            if buffer.append(example_to_json(ex)):
                buffer.flush()
            total_examples += 1

    buffer.flush()
    print(
        f"Done. format={FORMAT_ID} replays={accepted} "
        f"examples={total_examples} files={buffer.chunks_written} -> {output_dir}"
    )


if __name__ == "__main__":
    main()
