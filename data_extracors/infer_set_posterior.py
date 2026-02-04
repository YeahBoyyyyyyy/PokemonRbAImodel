"""
Infer probable sets for a species given observed moves using a Bayesian filter.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_move(name: str) -> str:
    if not name:
        return ""
    lowered = name.lower().strip()
    return re.sub(r"[^a-z0-9]", "", lowered)


def normalize_species(name: str) -> str:
    if not name:
        return ""
    lowered = name.lower().strip()
    return re.sub(r"[^a-z0-9]", "", lowered)


def parse_moves(text: str) -> List[str]:
    raw = [m.strip() for m in text.split(",") if m.strip()]
    return [normalize_move(m) for m in raw if normalize_move(m)]


def load_set_dex(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    norm = normalize_species(species)
    for key in species_dict:
        if normalize_species(key) == norm:
            candidates.add(key)
    base = species.split("-", 1)[0]
    norm_base = normalize_species(base)
    for key in species_dict:
        if normalize_species(key).startswith(norm_base):
            candidates.add(key)

    merged_sets: Dict[str, int] = {}
    total = 0
    for key in sorted(candidates):
        entry = species_dict.get(key, {})
        for set_key, count in entry.get("sets", {}).items():
            merged_sets[set_key] = merged_sets.get(set_key, 0) + int(count)
        total += int(entry.get("total", 0))
    return merged_sets, total


def infer(
    set_dex: Dict[str, object],
    species: str,
    observed_moves: List[str],
    observed_item: str,
    mismatch_penalty: float,
) -> List[Tuple[str, float, int]]:
    sets, total = resolve_species_entry(set_dex, species)
    if not sets or total <= 0:
        return []

    results: List[Tuple[str, float, int]] = []
    for set_key, count in sets.items():
        parts = set_key.split("|") if set_key else []
        item = ""
        moves = []
        for part in parts:
            if part.startswith("item="):
                item = part.split("=", 1)[1]
            else:
                moves.append(part)
        missing = [m for m in observed_moves if m not in moves]
        item_mismatch = bool(
            observed_item and item and item != "unknown" and observed_item != item
        )
        if missing:
            if mismatch_penalty <= 0:
                continue
            score = (count / total) * (mismatch_penalty ** len(missing))
        else:
            score = count / total
        if item_mismatch:
            if mismatch_penalty <= 0:
                continue
            score *= mismatch_penalty
        results.append((set_key, score, count))

    results.sort(key=lambda x: x[1], reverse=True)
    total_score = sum(r[1] for r in results) or 1.0
    return [(k, s / total_score, c) for (k, s, c) in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer sets for a species.")
    parser.add_argument("--set_dex", default="PokemonOUaimodel/data_extracors/set_dex.json")
    parser.add_argument("--species", required=True)
    parser.add_argument("--observed_moves", default="")
    parser.add_argument("--observed_item", default="")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--mismatch_penalty", type=float, default=0.0)
    args = parser.parse_args()

    set_dex = load_set_dex(Path(args.set_dex))
    species = args.species.strip()
    observed = parse_moves(args.observed_moves)
    observed_item = normalize_move(args.observed_item)

    results = infer(set_dex, species, observed, observed_item, args.mismatch_penalty)
    if not results:
        print("No matching sets found.")
        return

    for i, (set_key, prob, count) in enumerate(results[: args.top_k], start=1):
        moves = ", ".join(set_key.split("|"))
        print(f"{i:02d}. P={prob:.4f}  count={count}  moves=[{moves}]")


if __name__ == "__main__":
    main()
