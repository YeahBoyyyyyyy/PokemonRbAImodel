"""
Extract winrate training data from the Hugging Face dataset:
HolidayOugi/pokemon-showdown-replays
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for _path in (_PROJECT_ROOT, _SCRIPT_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from extract_showdown_winrate_data import (
    ShowdownWinrateExtractor,
    example_to_json,
    write_chunk,
)

def _escape_glob(text: str) -> str:
    placeholder_l = "\0LBR\0"
    placeholder_r = "\0RBR\0"
    return (
        text.replace("[", placeholder_l)
        .replace("]", placeholder_r)
        .replace(placeholder_l, "[[]")
        .replace(placeholder_r, "[]]")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract winrate dataset from HF replays.")
    parser.add_argument(
        "--dataset",
        default="HolidayOugi/pokemon-showdown-replays",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--data_files",
        nargs="+",
        default=[],
        help="Optional HF data_files list (e.g. \"[Gen 9] OU_part1.parquet\")",
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output_dir", default="showdown_winrate_chunks", help="Output directory")
    parser.add_argument("--formatid", default="gen9ou", help="Format id filter")
    parser.add_argument("--format_name", default="", help="Exact match on 'format' field")
    parser.add_argument("--min_rating", type=float, default=0.0, help="Minimum rating filter")
    parser.add_argument("--max_replays", type=int, default=0, help="Stop after N replays")
    parser.add_argument("--examples_per_file", type=int, default=40000, help="Chunk size")
    parser.add_argument("--base_name", default="rb_winrate_data", help="Output filename prefix")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--log_every_scanned", type=int, default=1000, help="Log every N scanned")
    parser.add_argument("--log_every_accepted", type=int, default=100, help="Log every N accepted")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    extractor = ShowdownWinrateExtractor()
    current_chunk: List[Dict[str, object]] = []
    chunk_idx = 1
    chunks_written = 0

    def flush_chunk() -> None:
        nonlocal chunk_idx, chunks_written
        if not current_chunk:
            return
        out_path = write_chunk(
            chunk=current_chunk,
            output_dir=output_dir,
            base_name=args.base_name,
            chunk_idx=chunk_idx,
            pretty_json=args.pretty,
        )
        print(f"Wrote {len(current_chunk)} examples to {out_path}")
        current_chunk.clear()
        chunk_idx += 1
        chunks_written += 1

    data_files = args.data_files or None
    if data_files:
        escaped = [_escape_glob(name) for name in data_files]
        hf_paths = [f"hf://datasets/{args.dataset}/{name}" for name in escaped]
        dataset = load_dataset("parquet", data_files=hf_paths, split="train", streaming=True)
    else:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    scanned = 0
    accepted = 0

    for replay in dataset:
        scanned += 1
        if args.log_every_scanned and scanned % args.log_every_scanned == 0:
            print(
                f"Scanned {scanned} replays, accepted={accepted}, examples={extractor.stats['total_examples']}"
            )
        if args.max_replays and accepted >= args.max_replays:
            break

        format_id = replay.get("formatid") or ""
        format_name = replay.get("format") or ""
        if args.format_name:
            if format_name != args.format_name:
                continue
        elif args.formatid and format_id != args.formatid:
            continue

        rating = replay.get("rating") or 0.0
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            rating_value = 0.0
        if rating_value < args.min_rating:
            continue

        extractor.stats["total_replays"] += 1
        p1_examples, p2_examples = extractor.extract_from_replay(replay)

        if p1_examples or p2_examples:
            extractor.stats["successful_extractions"] += 1
        else:
            extractor.stats["failed_extractions"] += 1

        for ex in p1_examples + p2_examples:
            extractor.stats["total_examples"] += 1
            if ex.perspective == "p1" and ex.winner == 1:
                extractor.stats["p1_wins"] += 1
            elif ex.perspective == "p2" and ex.winner == 1:
                extractor.stats["p2_wins"] += 1
            current_chunk.append(example_to_json(ex))
            if len(current_chunk) >= args.examples_per_file:
                flush_chunk()

        accepted += 1
        if args.log_every_accepted and accepted % args.log_every_accepted == 0:
            print(f"Processed {accepted} replays, examples={extractor.stats['total_examples']}")

    flush_chunk()

    print("Extraction summary")
    print(f"- replays: {extractor.stats['total_replays']}")
    print(f"- success: {extractor.stats['successful_extractions']}")
    print(f"- failed: {extractor.stats['failed_extractions']}")
    print(f"- examples: {extractor.stats['total_examples']}")
    if extractor.stats["p1_examples"]:
        p1_rate = 100 * extractor.stats["p1_wins"] / extractor.stats["p1_examples"]
    else:
        p1_rate = 0.0
    if extractor.stats["p2_examples"]:
        p2_rate = 100 * extractor.stats["p2_wins"] / extractor.stats["p2_examples"]
    else:
        p2_rate = 0.0
    print(f"- p1 wins: {extractor.stats['p1_wins']} ({p1_rate:.1f}%)")
    print(f"- p2 wins: {extractor.stats['p2_wins']} ({p2_rate:.1f}%)")
    print(f"- output_dir: {output_dir} ({chunks_written} file(s))")


if __name__ == "__main__":
    main()
