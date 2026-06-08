"""
Import Random Battle set probabilities from pkmn/randbats into set_dex format.

Source: https://github.com/pkmn/randbats (100k simulated teams / format, updated hourly)
Data:  https://data.pkmn.cc/randbats/gen9randombattle.json

Each role in randbats becomes one weighted moveset in our dex (equal weight per role
unless --stats-dir provides per-role counts).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.set_dex_prior import normalize_species_name, normalize_token
from random_battle.config import RB_SET_DEX_PATH

DEFAULT_URL = (
    "https://raw.githubusercontent.com/pkmn/randbats/main/data/gen9randombattle.json"
)


def normalize_display_species(name: str) -> str:
    """Charizard-Galar -> charizardgalar (matches showdown ids roughly)."""
    return normalize_species_name(name.replace(" ", "").replace("'", ""))


def moves_to_set_key(moves: List[str]) -> str:
    normed = sorted({normalize_token(m) for m in moves if normalize_token(m)})
    return "|".join(normed)


def convert_randbats_entry(
    display_name: str,
    entry: Dict[str, object],
    role_weights: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    roles = entry.get("roles") or {}
    sets: Dict[str, int] = {}
    role_meta: Dict[str, object] = {}

    for role_name, role_data in roles.items():
        if not isinstance(role_data, dict):
            continue
        moves = role_data.get("moves") or []
        if not moves:
            continue
        set_key = moves_to_set_key(moves)
        weight = 1
        if role_weights and role_name in role_weights:
            weight = int(role_weights[role_name])
        sets[set_key] = sets.get(set_key, 0) + weight

        role_meta[role_name] = {
            "moves": [normalize_token(m) for m in moves],
            "abilities": role_data.get("abilities") or [],
            "items": role_data.get("items") or [],
            "teraTypes": role_data.get("teraTypes") or [],
        }

    total = sum(sets.values())
    return {
        "display_name": display_name,
        "level": entry.get("level"),
        "total": total,
        "sets": sets,
        "roles": role_meta,
    }


def convert_randbats(
    randbats: Dict[str, object],
    stats: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    species_out: Dict[str, object] = {}
    for display_name, entry in randbats.items():
        if not isinstance(entry, dict):
            continue
        norm = normalize_display_species(display_name)
        role_weights = None
        if stats and display_name in stats:
            stat_entry = stats[display_name]
            if isinstance(stat_entry, dict) and "roles" in stat_entry:
                role_weights = {
                    k: int(v.get("count", v) if isinstance(v, dict) else v)
                    for k, v in stat_entry["roles"].items()
                }
        species_out[norm] = convert_randbats_entry(display_name, entry, role_weights)
    return {
        "meta": {
            "source": "pkmn/randbats",
            "format": "gen9randombattle",
            "description": "Empirical RB sets from ~100k Showdown simulations per role.",
        },
        "species": species_out,
    }


def download_json(url: str, timeout: int = 120) -> Dict[str, object]:
    print(f"Downloading {url} ...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PokemonRbAImodel/1.0 (rb-set-dex-import)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def load_local_json(path: Path) -> Dict[str, object]:
    print(f"Loading {path} ...")
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_stats(stats_dir: Path) -> Optional[Dict[str, object]]:
    if not stats_dir.is_dir():
        return None
    merged: Dict[str, object] = {}
    for path in stats_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            merged.update(data)
    return merged if merged else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Import pkmn/randbats into rb_set_dex.json")
    parser.add_argument("--url", default=DEFAULT_URL, help="Randbats JSON URL")
    parser.add_argument(
        "--input_file",
        default="",
        help="Local randbats JSON (skip download)",
    )
    parser.add_argument(
        "--stats_dir",
        default="",
        help="Optional pkmn/randbats data/stats/gen9randombattle/ folder",
    )
    parser.add_argument("--output_file", default=str(RB_SET_DEX_PATH))
    args = parser.parse_args()

    if args.input_file:
        randbats = load_local_json(Path(args.input_file))
    else:
        try:
            randbats = download_json(args.url)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"Download failed: {exc}")
            print("Use --input_file with a local copy from:")
            print("  https://github.com/pkmn/randbats/blob/main/data/gen9randombattle.json")
            sys.exit(1)

    stats = try_load_stats(Path(args.stats_dir)) if args.stats_dir else None
    set_dex = convert_randbats(randbats, stats=stats)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(set_dex, indent=2), encoding="utf-8")

    n_species = len(set_dex.get("species", {}))
    n_sets = sum(
        len(s.get("sets", {}))
        for s in set_dex.get("species", {}).values()
        if isinstance(s, dict)
    )
    print(f"Wrote {out_path}")
    print(f"  species: {n_species}")
    print(f"  unique movesets (across species): {n_sets}")


if __name__ == "__main__":
    main()
