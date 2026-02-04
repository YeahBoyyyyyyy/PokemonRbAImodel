"""
Extract action (move vs switch) training data into chunks.

Supports:
- local replays (with log + optional inputlog)
- Hugging Face HolidayOugi/pokemon-showdown-replays
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from extract_pokechamp_training_data import TrainingDataExtractor, TrainingExample


def _escape_glob(text: str) -> str:
    placeholder_l = "\0LBR\0"
    placeholder_r = "\0RBR\0"
    return (
        text.replace("[", placeholder_l)
        .replace("]", placeholder_r)
        .replace(placeholder_l, "[[]")
        .replace(placeholder_r, "[]]")
    )


def parse_inputlog_actions(inputlog: str) -> List[Tuple[int, str, str, str]]:
    actions: List[Tuple[int, str, str, str]] = []
    current_turn = 0
    for line in inputlog.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">start") or line.startswith(">player"):
            continue
        if not line.startswith(">"):
            continue
        parts = line.split()
        if not parts:
            continue
        player = parts[0][1:]
        if player not in ("p1", "p2"):
            continue
        if player == "p1":
            current_turn += 1
        elif current_turn == 0:
            current_turn = 1

        if len(parts) < 2:
            continue
        action_type = parts[1]
        if action_type == "move":
            move_name = parts[2] if len(parts) > 2 else "unknown"
            actions.append((current_turn, player, "move", move_name))
        elif action_type == "switch":
            target = parts[2] if len(parts) > 2 else "1"
            actions.append((current_turn, player, "switch", target))
    return actions


def example_to_json(example: TrainingExample) -> Dict[str, object]:
    return asdict(example)


def write_chunk(
    chunk: List[Dict[str, object]],
    output_dir: Path,
    base_name: str,
    chunk_idx: int,
    pretty_json: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{base_name}_{chunk_idx:05d}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(chunk, handle, indent=2 if pretty_json else None)
    return out_path


def extract_examples_from_log(
    extractor: TrainingDataExtractor,
    log_text: str,
    inputlog: Optional[str],
    prefer_inputlog: bool,
) -> List[TrainingExample]:
    extractor.reset_state()
    if not log_text:
        return []
    log_lines = log_text.strip().split("\n")
    extractor._extract_winner(log_lines)
    state_history = extractor._parse_log_to_states(log_lines)
    actions: List[Tuple[int, str, str, str]] = []
    if prefer_inputlog and inputlog:
        actions = parse_inputlog_actions(inputlog)
    if not actions:
        actions = extractor._infer_actions_from_log(log_lines)
    return extractor._create_training_examples(state_history, actions)


def iter_local_replays(input_dir: Path) -> Iterable[Dict[str, object]]:
    for path in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        yield data


def choose_rating(replay: Dict[str, object]) -> float:
    value = replay.get("rating") or replay.get("p1rating") or replay.get("p2rating") or 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract action chunks (move vs switch).")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--input_dir", default="replays_data")
    parser.add_argument("--dataset", default="HolidayOugi/pokemon-showdown-replays")
    parser.add_argument("--data_files", nargs="+", default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--formatid", default="")
    parser.add_argument("--format_name", default="")
    parser.add_argument("--min_rating", type=float, default=0.0)
    parser.add_argument("--max_replays", type=int, default=0)
    parser.add_argument("--prefer_inputlog", action="store_true")
    parser.add_argument("--output_dir", default="action_chunks")
    parser.add_argument("--base_name", default="action_data")
    parser.add_argument("--examples_per_file", type=int, default=40000)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--log_every_scanned", type=int, default=1000)
    parser.add_argument("--log_every_accepted", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    extractor = TrainingDataExtractor()
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

    scanned = 0
    accepted = 0
    total_examples = 0

    if args.source == "hf":
        from datasets import load_dataset

        if args.data_files:
            escaped = [_escape_glob(name) for name in args.data_files]
            hf_paths = [f"hf://datasets/{args.dataset}/{name}" for name in escaped]
            dataset = load_dataset("parquet", data_files=hf_paths, split="train", streaming=True)
        else:
            dataset = load_dataset(args.dataset, split=args.split, streaming=True)

        for replay in dataset:
            scanned += 1
            if args.log_every_scanned and scanned % args.log_every_scanned == 0:
                print(
                    f"Scanned {scanned} replays, accepted={accepted}, examples={len(current_chunk)}"
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

            if choose_rating(replay) < args.min_rating:
                continue

            log_text = replay.get("log") or ""
            examples = extract_examples_from_log(extractor, log_text, None, False)
            if not examples:
                continue

            accepted += 1
            if args.log_every_accepted and accepted % args.log_every_accepted == 0:
                print(f"Processed {accepted} replays, examples={total_examples}")

            for ex in examples:
                current_chunk.append(example_to_json(ex))
                total_examples += 1
                if len(current_chunk) >= args.examples_per_file:
                    flush_chunk()
    else:
        input_dir = Path(args.input_dir)
        for replay in iter_local_replays(input_dir):
            scanned += 1
            if args.log_every_scanned and scanned % args.log_every_scanned == 0:
                print(
                    f"Scanned {scanned} replays, accepted={accepted}, examples={len(current_chunk)}"
                )
            if args.max_replays and accepted >= args.max_replays:
                break

            format_id = replay.get("formatid") or ""
            if args.formatid and format_id != args.formatid:
                continue
            if choose_rating(replay) < args.min_rating:
                continue

            log_text = replay.get("log") or ""
            inputlog = replay.get("inputlog")
            examples = extract_examples_from_log(
                extractor, log_text, inputlog, args.prefer_inputlog
            )
            if not examples:
                continue

            accepted += 1
            if args.log_every_accepted and accepted % args.log_every_accepted == 0:
                print(f"Processed {accepted} replays, examples={total_examples}")

            for ex in examples:
                current_chunk.append(example_to_json(ex))
                total_examples += 1
                if len(current_chunk) >= args.examples_per_file:
                    flush_chunk()

    flush_chunk()
    print(f"Done. Wrote {chunks_written} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
