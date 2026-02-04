"""
Build a moveset dex from replay logs.

This scans replays (local JSON or HF HolidayOugi dataset) and collects
observed moves per species. Each species gets a distribution of move sets.

Output JSON schema:
{
  "meta": {...},
  "species": {
     "dragonite": {
        "total": 123,
        "sets": {
            "dragondance|extremespeed|...": 42,
            ...
        }
     },
     ...
  }
}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _escape_glob(text: str) -> str:
    placeholder_l = "\0LBR\0"
    placeholder_r = "\0RBR\0"
    return (
        text.replace("[", placeholder_l)
        .replace("]", placeholder_r)
        .replace(placeholder_l, "[[]")
        .replace(placeholder_r, "[]]")
    )


def normalize_move(name: str) -> str:
    if not name:
        return ""
    lowered = name.lower().strip()
    return re.sub(r"[^a-z0-9]", "", lowered)


def parse_moves_from_log(log_text: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    nickname_map: Dict[Tuple[str, str], str] = {}
    moves_by_species: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    item_by_species: Dict[Tuple[str, str], str] = {}

    for raw in log_text.splitlines():
        if not raw.startswith("|"):
            continue
        parts = raw.split("|")
        if len(parts) < 3:
            continue
        msg_type = parts[1]

        if msg_type in ("switch", "drag", "replace"):
            if len(parts) < 4:
                continue
            actor = parts[2]
            player_id = actor.split(":")[0][:2]
            nickname = actor.split(":", 1)[1].strip() if ":" in actor else actor.strip()
            species = parts[3].split(",")[0].strip()
            if player_id and nickname and species:
                nickname_map[(player_id, nickname)] = species
            continue

        if msg_type in ("-item", "item", "-enditem"):
            if len(parts) < 4:
                continue
            actor = parts[2]
            player_id = actor.split(":")[0][:2]
            nickname = actor.split(":", 1)[1].strip() if ":" in actor else actor.strip()
            species = nickname_map.get((player_id, nickname))
            if not species:
                continue
            item = normalize_move(parts[3])
            if item and item != "unknown":
                item_by_species[(player_id, species)] = item
            continue

        if msg_type == "move":
            if len(parts) < 4:
                continue
            actor = parts[2]
            player_id = actor.split(":")[0][:2]
            nickname = actor.split(":", 1)[1].strip() if ":" in actor else actor.strip()
            species = nickname_map.get((player_id, nickname))
            if not species:
                continue
            move = normalize_move(parts[3])
            if move:
                moves_by_species[(player_id, species)].add(move)

    summarized: Dict[str, List[str]] = {}
    items: Dict[str, str] = {}
    for (player_id, species), moves in moves_by_species.items():
        summarized[species] = sorted(moves)
        if (player_id, species) in item_by_species:
            items[species] = item_by_species[(player_id, species)]
    return summarized, items


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
    parser = argparse.ArgumentParser(description="Build a moveset dex from replays.")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--input_dir", default="replays_data")
    parser.add_argument("--dataset", default="HolidayOugi/pokemon-showdown-replays")
    parser.add_argument("--data_files", nargs="+", default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--formatid", default="")
    parser.add_argument("--format_name", default="")
    parser.add_argument("--min_rating", type=float, default=0.0)
    parser.add_argument("--max_replays", type=int, default=0)
    parser.add_argument("--min_moves", type=int, default=2)
    parser.add_argument("--max_moves", type=int, default=4)
    parser.add_argument("--output_file", default="PokemonOUaimodel/data_extracors/set_dex.json")
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()

    set_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    species_totals: Dict[str, int] = defaultdict(int)

    scanned = 0
    accepted = 0

    def handle_replay(replay: Dict[str, object]) -> None:
        nonlocal accepted
        log_text = replay.get("log") or ""
        if not log_text:
            return
        species_moves, species_items = parse_moves_from_log(log_text)
        if not species_moves:
            return
        accepted += 1
        for species, moves in species_moves.items():
            if not moves:
                continue
            if args.min_moves and len(moves) < args.min_moves:
                continue
            if args.max_moves and len(moves) > args.max_moves:
                continue
            item = species_items.get(species, "unknown")
            key = "|".join([f"item={item}"] + moves)
            set_counts[species][key] += 1
            species_totals[species] += 1

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
            if args.log_every and scanned % args.log_every == 0:
                print(f"Scanned {scanned} replays, accepted={accepted}")
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
            handle_replay(replay)
    else:
        input_dir = Path(args.input_dir)
        for replay in iter_local_replays(input_dir):
            scanned += 1
            if args.log_every and scanned % args.log_every == 0:
                print(f"Scanned {scanned} replays, accepted={accepted}")
            if args.max_replays and accepted >= args.max_replays:
                break
            format_id = replay.get("formatid") or ""
            if args.formatid and format_id != args.formatid:
                continue
            if choose_rating(replay) < args.min_rating:
                continue
            handle_replay(replay)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "source": args.source,
            "formatid": args.formatid,
            "format_name": args.format_name,
            "min_rating": args.min_rating,
            "min_moves": args.min_moves,
            "max_moves": args.max_moves,
            "scanned": scanned,
            "accepted": accepted,
        },
        "species": {},
    }
    for species, sets in set_counts.items():
        payload["species"][species] = {
            "total": species_totals.get(species, 0),
            "sets": dict(sets),
        }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote set dex to {output_path}")


if __name__ == "__main__":
    main()
