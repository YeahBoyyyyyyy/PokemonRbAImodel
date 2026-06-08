"""
Extract winrate training data from local Pokemon Showdown replay JSON files.

Output schema matches RB winrate chunks (saved to a configurable folder).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def load_pokedex_norm_map() -> Dict[str, str]:
    root_dir = Path(__file__).resolve().parents[1]
    pokedex_path = root_dir / "pokedex_9G_complete.py"
    if not pokedex_path.exists():
        return {}
    try:
        spec = importlib.util.spec_from_file_location("pokedex_9G_complete", pokedex_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        pokedex = getattr(module, "pokemon_data_gen9", {})
    except Exception:
        return {}
    return {normalize_species_name(key): key for key in pokedex.keys()}


POKEDEX_NORM_TO_KEY = load_pokedex_norm_map()


@dataclass
class PokemonState:
    species: str
    hp_percent: float
    status: Optional[str]
    fainted: bool
    boosts: Dict[str, int]
    volatiles: List[str]
    revealed: bool


@dataclass
class GameState:
    turn: int
    our_team: Dict[str, PokemonState]
    opponent_team: Dict[str, PokemonState]
    our_active: Optional[str]
    opponent_active: Optional[str]
    our_side_conditions: Dict[str, int]
    opponent_side_conditions: Dict[str, int]
    field_conditions: Dict[str, Optional[str]]
    our_last_move: Optional[str]
    opponent_last_move: Optional[str]


@dataclass
class WinPredictionExample:
    state: GameState
    winner: int
    turn: int
    perspective: str


def example_to_json(example: WinPredictionExample) -> Dict[str, object]:
    return {
        "turn": example.turn,
        "perspective": example.perspective,
        "winner": example.winner,
        "state": {
            "turn": example.state.turn,
            "our_team": {name: asdict(poke) for name, poke in example.state.our_team.items()},
            "opponent_team": {
                name: asdict(poke) for name, poke in example.state.opponent_team.items()
            },
            "our_active": example.state.our_active,
            "opponent_active": example.state.opponent_active,
            "our_side_conditions": example.state.our_side_conditions,
            "opponent_side_conditions": example.state.opponent_side_conditions,
            "field_conditions": example.state.field_conditions,
            "our_last_move": example.state.our_last_move,
            "opponent_last_move": example.state.opponent_last_move,
        },
    }


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


def _empty_member() -> Dict[str, object]:
    return {
        "hp_percent": 1.0,
        "status": None,
        "fainted": False,
        "boosts": {},
        "volatiles": [],
        "revealed": False,
    }


class ShowdownWinrateExtractor:
    def __init__(self) -> None:
        self.stats = {
            "total_replays": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_examples": 0,
            "p1_examples": 0,
            "p2_examples": 0,
            "p1_wins": 0,
            "p2_wins": 0,
        }
        self.nickname_map: Dict[str, Dict[str, str]] = {"p1": {}, "p2": {}}

    def extract_from_replay(
        self, replay_data: Dict[str, object]
    ) -> Tuple[List[WinPredictionExample], List[WinPredictionExample]]:
        log = replay_data.get("log")
        if not isinstance(log, str) or not log:
            return [], []

        self.nickname_map = {"p1": {}, "p2": {}}
        state_history, winner = self._parse_log_to_states_and_winner(log)
        if not state_history or winner is None:
            return [], []

        p1_examples = self._create_winrate_examples(state_history, winner, "p1")
        p2_examples = self._create_winrate_examples(state_history, winner, "p2")

        self.stats["p1_examples"] += len(p1_examples)
        self.stats["p2_examples"] += len(p2_examples)

        return p1_examples, p2_examples

    def _parse_log_to_states_and_winner(
        self, log: str
    ) -> Tuple[Dict[int, Dict[str, object]], Optional[str]]:
        lines = [line for line in log.split("\n") if line]
        player_usernames: Dict[str, str] = {}
        team_preview = {"p1": {}, "p2": {}}

        for line in lines:
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            command = parts[1]
            if command == "player" and len(parts) >= 4:
                player_id = parts[2]
                username = parts[3]
                player_usernames[username] = player_id
            elif command == "poke" and len(parts) >= 4:
                player_id = parts[2]
                pokemon_name = self._canonical_species(parts[3].split(",")[0].strip())
                if player_id in team_preview:
                    team_preview[player_id][pokemon_name] = _empty_member()

        state_history: Dict[int, Dict[str, object]] = {
            0: {
                "p1": {
                    "team": {name: data.copy() for name, data in team_preview["p1"].items()},
                    "active": None,
                    "side_conditions": {},
                    "last_move": None,
                },
                "p2": {
                    "team": {name: data.copy() for name, data in team_preview["p2"].items()},
                    "active": None,
                    "side_conditions": {},
                    "last_move": None,
                },
                "field": {"weather": None, "terrain": None},
            }
        }

        current_turn = 0
        winner: Optional[str] = None

        for line in lines:
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            command = parts[1]

            if command == "turn" and len(parts) >= 3:
                current_turn = int(parts[2])
                if current_turn - 1 in state_history:
                    state_history[current_turn] = self._deep_copy_state(
                        state_history[current_turn - 1]
                    )
                else:
                    state_history[current_turn] = self._deep_copy_state(state_history[0])
                continue

            if command == "win" and len(parts) >= 3:
                winner_username = parts[2]
                winner = player_usernames.get(winner_username)
                continue

            if current_turn not in state_history:
                state_history[current_turn] = self._deep_copy_state(state_history[0])

            state = state_history[current_turn]

            if command in ("switch", "drag") and len(parts) >= 4:
                player_id, nickname = self._split_player_slot(parts[2])
                pokemon_name = self._canonical_species(parts[3].split(",")[0].strip())
                hp_data = parts[4] if len(parts) > 4 else "100/100"
                hp_percent = self._parse_hp(hp_data)
                status = self._parse_status(hp_data)

                if nickname:
                    self.nickname_map[player_id][nickname] = pokemon_name

                self._ensure_member(state, player_id, pokemon_name)
                state[player_id]["active"] = pokemon_name
                member = state[player_id]["team"][pokemon_name]
                member["hp_percent"] = hp_percent
                member["status"] = status
                member["revealed"] = True

            elif command == "faint" and len(parts) >= 3:
                player_id, raw_name = self._split_player_slot(parts[2])
                pokemon_name = self._resolve_species(player_id, raw_name)
                self._ensure_member(state, player_id, pokemon_name)
                member = state[player_id]["team"][pokemon_name]
                member["fainted"] = True
                member["hp_percent"] = 0.0
                member["revealed"] = True

            elif command in ("-damage", "-heal") and len(parts) >= 4:
                player_id, raw_name = self._split_player_slot(parts[2])
                pokemon_name = self._resolve_species(player_id, raw_name)
                hp_data = parts[3]
                hp_percent = self._parse_hp(hp_data)
                status = self._parse_status(hp_data)
                self._ensure_member(state, player_id, pokemon_name)
                member = state[player_id]["team"][pokemon_name]
                member["hp_percent"] = hp_percent
                if status is not None:
                    member["status"] = status

            elif command == "-status" and len(parts) >= 4:
                player_id, raw_name = self._split_player_slot(parts[2])
                pokemon_name = self._resolve_species(player_id, raw_name)
                status = parts[3]
                self._ensure_member(state, player_id, pokemon_name)
                state[player_id]["team"][pokemon_name]["status"] = status

            elif command == "-curestatus" and len(parts) >= 3:
                player_id, raw_name = self._split_player_slot(parts[2])
                pokemon_name = self._resolve_species(player_id, raw_name)
                self._ensure_member(state, player_id, pokemon_name)
                state[player_id]["team"][pokemon_name]["status"] = None

            elif command in ("-boost", "-unboost") and len(parts) >= 5:
                player_id, raw_name = self._split_player_slot(parts[2])
                pokemon_name = self._resolve_species(player_id, raw_name)
                stat = parts[3]
                amount = int(parts[4])
                if command == "-unboost":
                    amount = -amount
                self._ensure_member(state, player_id, pokemon_name)
                boosts = state[player_id]["team"][pokemon_name]["boosts"]
                boosts[stat] = boosts.get(stat, 0) + amount

            elif command == "-start" and len(parts) >= 4:
                player_id, raw_name = self._split_player_slot(parts[2])
                pokemon_name = self._resolve_species(player_id, raw_name)
                volatile = parts[3]
                self._ensure_member(state, player_id, pokemon_name)
                volatiles = state[player_id]["team"][pokemon_name]["volatiles"]
                if volatile not in volatiles:
                    volatiles.append(volatile)

            elif command == "-end" and len(parts) >= 4:
                player_id, raw_name = self._split_player_slot(parts[2])
                pokemon_name = self._resolve_species(player_id, raw_name)
                volatile = parts[3]
                self._ensure_member(state, player_id, pokemon_name)
                volatiles = state[player_id]["team"][pokemon_name]["volatiles"]
                if volatile in volatiles:
                    volatiles.remove(volatile)

            elif command == "-sidestart" and len(parts) >= 4:
                player_id, _ = self._split_player_slot(parts[2])
                condition = parts[3]
                side = state[player_id]["side_conditions"]
                side[condition] = side.get(condition, 0) + 1

            elif command == "-sideend" and len(parts) >= 4:
                player_id, _ = self._split_player_slot(parts[2])
                condition = parts[3]
                side = state[player_id]["side_conditions"]
                if condition in side:
                    side[condition] = max(0, side[condition] - 1)
                    if side[condition] == 0:
                        del side[condition]

            elif command == "-weather" and len(parts) >= 3:
                weather = parts[2]
                state["field"]["weather"] = None if weather == "none" else weather

            elif command == "-fieldstart" and len(parts) >= 3:
                state["field"]["terrain"] = parts[2]

            elif command == "-fieldend":
                state["field"]["terrain"] = None

            elif command == "move" and len(parts) >= 4:
                player_id, _ = self._split_player_slot(parts[2])
                move = parts[3]
                state[player_id]["last_move"] = move

        return state_history, winner

    def _create_winrate_examples(
        self, state_history: Dict[int, Dict[str, object]], winner: str, perspective: str
    ) -> List[WinPredictionExample]:
        examples: List[WinPredictionExample] = []
        we_won = 1 if winner == perspective else 0

        for turn in sorted(state_history.keys()):
            if turn == 0:
                continue
            state = state_history[turn]
            our_data = state[perspective]
            opp_data = state["p2" if perspective == "p1" else "p1"]

            our_team = {
                name: PokemonState(
                    species=name,
                    hp_percent=info["hp_percent"],
                    status=info["status"],
                    fainted=info["fainted"],
                    boosts=info["boosts"].copy(),
                    volatiles=info["volatiles"].copy(),
                    revealed=info.get("revealed", False),
                )
                for name, info in our_data["team"].items()
            }

            opp_team = {
                name: PokemonState(
                    species=name,
                    hp_percent=info["hp_percent"],
                    status=info["status"],
                    fainted=info["fainted"],
                    boosts=info["boosts"].copy(),
                    volatiles=info["volatiles"].copy(),
                    revealed=info.get("revealed", False),
                )
                for name, info in opp_data["team"].items()
            }

            game_state = GameState(
                turn=turn,
                our_team=our_team,
                opponent_team=opp_team,
                our_active=our_data.get("active"),
                opponent_active=opp_data.get("active"),
                our_side_conditions=our_data["side_conditions"].copy(),
                opponent_side_conditions=opp_data["side_conditions"].copy(),
                field_conditions=state["field"].copy(),
                our_last_move=our_data.get("last_move"),
                opponent_last_move=opp_data.get("last_move"),
            )

            examples.append(
                WinPredictionExample(
                    state=game_state, winner=we_won, turn=turn, perspective=perspective
                )
            )

        return examples

    def _deep_copy_state(self, state: Dict[str, object]) -> Dict[str, object]:
        return {
            "p1": {
                "team": {
                    name: {
                        "hp_percent": data["hp_percent"],
                        "status": data["status"],
                        "fainted": data["fainted"],
                        "boosts": data["boosts"].copy(),
                        "volatiles": data["volatiles"].copy(),
                        "revealed": data.get("revealed", False),
                    }
                    for name, data in state["p1"]["team"].items()
                },
                "active": state["p1"]["active"],
                "side_conditions": state["p1"]["side_conditions"].copy(),
                "last_move": state["p1"].get("last_move"),
            },
            "p2": {
                "team": {
                    name: {
                        "hp_percent": data["hp_percent"],
                        "status": data["status"],
                        "fainted": data["fainted"],
                        "boosts": data["boosts"].copy(),
                        "volatiles": data["volatiles"].copy(),
                        "revealed": data.get("revealed", False),
                    }
                    for name, data in state["p2"]["team"].items()
                },
                "active": state["p2"]["active"],
                "side_conditions": state["p2"]["side_conditions"].copy(),
                "last_move": state["p2"].get("last_move"),
            },
            "field": state["field"].copy(),
        }

    def _parse_hp(self, hp_string: str) -> float:
        if "fnt" in hp_string.lower():
            return 0.0
        hp_part = hp_string.split()[0]
        if "/" in hp_part:
            current, maximum = hp_part.split("/")
            try:
                return float(current) / float(maximum)
            except (ValueError, ZeroDivisionError):
                return 1.0
        return 1.0

    def _parse_status(self, hp_string: str) -> Optional[str]:
        parts = hp_string.split()
        if len(parts) > 1 and parts[1] not in ("fnt",):
            return parts[1]
        return None

    def _ensure_member(self, state: Dict[str, object], player_id: str, name: str) -> None:
        team = state[player_id]["team"]
        if name not in team:
            team[name] = _empty_member()

    def _split_player_slot(self, player_slot: str) -> Tuple[str, str]:
        if ":" in player_slot:
            prefix, name = player_slot.split(":", 1)
            return prefix[:2], name.strip()
        return player_slot[:2], player_slot.strip()

    def _canonical_species(self, name: str) -> str:
        if not name:
            return ""
        norm = normalize_species_name(name)
        if not norm:
            return name
        norm = canonicalize_species_norm(norm)
        key = POKEDEX_NORM_TO_KEY.get(norm)
        if key:
            return key
        return name.strip()

    def _resolve_species(self, player_id: str, raw_name: str) -> str:
        if not raw_name:
            return ""
        mapping = self.nickname_map.get(player_id, {})
        if raw_name in mapping:
            return mapping[raw_name]
        canonical = self._canonical_species(raw_name)
        mapping[raw_name] = canonical
        return canonical


def extract_format_id(replay_data: Dict[str, object]) -> Optional[str]:
    value = replay_data.get("formatid")
    if isinstance(value, str):
        return value
    inputlog = replay_data.get("inputlog")
    if isinstance(inputlog, str):
        for line in inputlog.split("\n"):
            if line.startswith(">start"):
                try:
                    payload = json.loads(line.split(" ", 1)[1])
                    format_id = payload.get("formatid")
                    if isinstance(format_id, str):
                        return format_id
                except Exception:
                    return None
    return None


def extract_ratings(replay_data: Dict[str, object]) -> List[int]:
    ratings: List[int] = []
    for key in ("rating", "p1rating", "p2rating"):
        value = replay_data.get(key)
        if isinstance(value, int):
            ratings.append(value)
        elif isinstance(value, str):
            try:
                ratings.append(int(value))
            except ValueError:
                pass

    inputlog = replay_data.get("inputlog")
    if isinstance(inputlog, str):
        for line in inputlog.split("\n"):
            if line.startswith(">player"):
                try:
                    payload = json.loads(line.split(" ", 2)[2])
                    rating = payload.get("rating")
                    if isinstance(rating, int):
                        ratings.append(rating)
                except Exception:
                    continue
    return ratings


def choose_rating(ratings: List[int], mode: str) -> int:
    if not ratings:
        return 0
    if mode == "min":
        return min(ratings)
    if mode == "avg":
        return int(sum(ratings) / len(ratings))
    return max(ratings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract winrate dataset from Showdown replays.")
    parser.add_argument("--input_dir", default="replays_data", help="Replay JSON directory")
    parser.add_argument("--output_dir", default="showdown_winrate_chunks", help="Output directory")
    parser.add_argument("--formatid", default="gen9ou", help="Format id filter")
    parser.add_argument("--min_rating", type=int, default=0, help="Minimum rating filter")
    parser.add_argument(
        "--rating_mode", choices=["max", "min", "avg"], default="max", help="Rating aggregation"
    )
    parser.add_argument("--max_replays", type=int, default=0, help="Stop after N replays")
    parser.add_argument("--examples_per_file", type=int, default=20000, help="Chunk size")
    parser.add_argument("--base_name", default="rb_winrate_data", help="Output filename prefix")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    replay_files = sorted(input_dir.glob("*.json"))
    if not replay_files:
        print(f"No replay JSON found in {input_dir}")
        return

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

    processed = 0
    for replay_path in replay_files:
        if args.max_replays and processed >= args.max_replays:
            break
        try:
            replay_data = json.loads(replay_path.read_text(encoding="utf-8"))
        except Exception:
            extractor.stats["failed_extractions"] += 1
            continue

        format_id = extract_format_id(replay_data)
        if format_id != args.formatid:
            continue

        rating = choose_rating(extract_ratings(replay_data), args.rating_mode)
        if rating < args.min_rating:
            continue

        extractor.stats["total_replays"] += 1
        p1_examples, p2_examples = extractor.extract_from_replay(replay_data)

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

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} replays, examples={extractor.stats['total_examples']}")

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
