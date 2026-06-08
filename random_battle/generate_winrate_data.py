"""
Generate win-rate training data from the LIVE state pipeline (poke_env_to_state).

The historical win-rate chunks were built from human replays, where a player's
OWN team is revealed only progressively. At inference the bot knows its full
team from turn 1 (poke-env), so those states are out-of-distribution and the
model collapses toward 0. This script removes that train/inference gap: it plays
games on the local Showdown server and records ``battle_to_state_dict(battle)``
exactly as the bot sees it at decision time, labelling each state with whether
that side eventually won.

Both players record (from their own perspective), so every game yields balanced
win and loss examples. Output chunks are schema-compatible with
``IA_winrate_predictor.train`` (fields: state, winner, is_voluntary,
action_type, action_target).

Example:
  python random_battle/generate_winrate_data.py --n_battles 400 --mix
  python random_battle/generate_winrate_data.py --p1 low --p2 high --n_battles 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from poke_env import AccountConfiguration

from common.players.heuristics_pokemon_ai import (
    HighHeuristicAI,
    LowHeuristicAI,
    RandomAI,
)
from common.project_paths import setup_import_paths
from random_battle.config import FORMAT_ID, LOCAL_SERVER_CONFIGURATION
from random_battle.players.local_login_patch import apply_local_login_patch
from random_battle.players.poke_env_to_state import battle_to_state_dict

setup_import_paths()
apply_local_login_patch()

AI_CLASSES = {"random": RandomAI, "low": LowHeuristicAI, "high": HighHeuristicAI}

# Default matchup rotation for --mix: skill variety + balanced outcomes.
# (high-vs-high dropped: HighHeuristicAI is slow per turn; random kept light.)
MIX_PAIRS: List[Tuple[str, str]] = [
    ("low", "low"),
    ("low", "high"),
    ("random", "low"),
]


class _StateRecorder:
    """Mixin: record battle_to_state_dict at each of our decision points.

    Stateless w.r.t. __init__ (lazy buffers) so it composes with any poke-env
    Player subclass regardless of its constructor signature.
    """

    async def choose_move(self, battle):  # type: ignore[override]
        store: Dict[str, List[dict]] = self.__dict__.setdefault("_rec_store", {})
        myc: Dict[str, List[str]] = self.__dict__.setdefault("_rec_myc", {})
        oppc: Dict[str, List[str]] = self.__dict__.setdefault("_rec_oppc", {})
        try:
            state = battle_to_state_dict(battle, myc, oppc)
            store.setdefault(battle.battle_tag, []).append(state)
        except Exception:
            pass
        return await super().choose_move(battle)


class RecRandomAI(_StateRecorder, RandomAI):
    pass


class RecLowAI(_StateRecorder, LowHeuristicAI):
    pass


class RecHighAI(_StateRecorder, HighHeuristicAI):
    pass


REC_CLASSES = {"random": RecRandomAI, "low": RecLowAI, "high": RecHighAI}


def _make_player(kind: str, battle_format: str, max_concurrent: int):
    cls = REC_CLASSES[kind]
    account = AccountConfiguration.generate(f"wr_{kind}", rand=True)
    return cls(
        account_configuration=account,
        server_configuration=LOCAL_SERVER_CONFIGURATION,
        max_concurrent_battles=max_concurrent,
        battle_format=battle_format,
    )


def _harvest(player, rows: List[dict]) -> Tuple[int, int]:
    """Append (state, winner) rows from a finished player. Returns (wins, losses) states."""
    store: Dict[str, List[dict]] = getattr(player, "_rec_store", {})
    wins = losses = 0
    for tag, states in store.items():
        battle = player.battles.get(tag)
        if battle is None or battle.won is None:
            continue  # tie / unfinished -> drop
        won = bool(battle.won)
        for state in states:
            rows.append(
                {
                    "state": state,
                    "winner": won,
                    "is_voluntary": True,
                    "action_type": "move",
                    "action_target": "",
                }
            )
        if won:
            wins += len(states)
        else:
            losses += len(states)
    # Clear so a player reused across matchups doesn't double-count.
    store.clear()
    return wins, losses


def _write_chunks(rows: List[dict], out_dir: Path, base_name: str, per_chunk: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.stem.rsplit("_", 1)[-1])
        for p in out_dir.glob(f"{base_name}_*.json")
        if p.stem.rsplit("_", 1)[-1].isdigit()
    ]
    idx = (max(existing) + 1) if existing else 1
    written = 0
    for start in range(0, len(rows), per_chunk):
        chunk = rows[start : start + per_chunk]
        if not chunk:
            continue
        path = out_dir / f"{base_name}_{idx:05d}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  wrote {len(chunk)} states ({size_mb:.1f} MiB) -> {path.name}", flush=True)
        idx += 1
        written += 1
    return written


async def _run_pair(
    p1_kind: str,
    p2_kind: str,
    n_battles: int,
    battle_format: str,
    max_concurrent: int,
    rows: List[dict],
) -> None:
    p1 = _make_player(p1_kind, battle_format, max_concurrent)
    p2 = _make_player(p2_kind, battle_format, max_concurrent)
    print(f"[{p1_kind} vs {p2_kind}] lancement de {n_battles} combats...", flush=True)
    await p1.battle_against(p2, n_battles=n_battles)
    w1, l1 = _harvest(p1, rows)
    w2, l2 = _harvest(p2, rows)
    print(
        f"[{p1_kind} vs {p2_kind}] termin\u00e9: "
        f"{p1.n_won_battles}/{n_battles} c\u00f4t\u00e9 p1 | "
        f"states win={w1 + w2} loss={l1 + l2}",
        flush=True,
    )


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pairs = MIX_PAIRS if args.mix else [(args.p1, args.p2)]
    per_pair = max(1, args.n_battles // len(pairs))

    total_states = total_wins = total_chunks = 0
    for p1_kind, p2_kind in pairs:
        rows: List[dict] = []
        await _run_pair(
            p1_kind, p2_kind, per_pair, args.battle_format,
            args.max_concurrent_battles, rows,
        )
        if not rows:
            continue
        # Flush this matchup's states immediately so a long run never loses
        # progress if it stalls or is interrupted.
        total_chunks += _write_chunks(rows, out_dir, args.base_name, args.chunk_size)
        total_states += len(rows)
        total_wins += sum(1 for r in rows if r["winner"])

    if total_states == 0:
        print("Aucun \u00e9tat enregistr\u00e9 (parties nulles ou serveur indisponible ?).")
        return

    print(
        f"\nTotal: {total_states} \u00e9tats | winner=1: {total_wins} "
        f"({total_wins / total_states:.1%}) | winner=0: {total_states - total_wins}",
        flush=True,
    )
    print(f"\u00c9crit {total_chunks} chunk(s) dans {args.out_dir} (base={args.base_name}).")
    print(
        "Pour entra\u00eener:\n"
        f"  python random_battle/run_train_winrate_model.py "
        f"--input_dir {args.out_dir} --base_name {args.base_name}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G\u00e9n\u00e8re des donn\u00e9es win-rate via le pipeline live (poke_env_to_state)."
    )
    parser.add_argument("--p1", choices=tuple(AI_CLASSES), default="low")
    parser.add_argument("--p2", choices=tuple(AI_CLASSES), default="high")
    parser.add_argument(
        "--mix",
        action="store_true",
        help="Ignore --p1/--p2 et tourne une rotation de matchups (diversit\u00e9 + \u00e9quilibre).",
    )
    parser.add_argument("--n_battles", type=int, default=200)
    parser.add_argument("--battle_format", default=FORMAT_ID)
    parser.add_argument("--max_concurrent_battles", type=int, default=10)
    parser.add_argument(
        "--out_dir",
        default=str(_PROJECT_ROOT / "random_battle" / "chunks" / "winrate_live"),
    )
    parser.add_argument("--base_name", default="rb_winrate_live")
    parser.add_argument(
        "--chunk_size", type=int, default=5000, help="\u00c9tats par fichier chunk."
    )
    return parser


def main() -> None:
    asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    main()
