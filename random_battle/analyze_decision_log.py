"""Quick post-mortem of a decision JSONL log produced by RbSearchPlayer.

Usage::

    python random_battle/analyze_decision_log.py path/to/decisions.jsonl
        [--show_wins] [--max_battles 5] [--min_alive_diff 1]

Groups decisions by ``battle_tag``, infers win/loss from the final
``my_team_alive`` vs ``opp_team_alive`` differential, and prints the
turn-by-turn choices with their candidate scores for a handful of
matches (defaults to 5 losses).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_log(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_battle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            tag = entry.get("battle_tag") or "?"
            by_battle[tag].append(entry)
    for tag, entries in by_battle.items():
        entries.sort(key=lambda e: int(e.get("turn") or 0))
    return by_battle


def battle_outcome(entries: List[Dict[str, Any]]) -> str:
    """Heuristic outcome from the last logged decision.

    - "win"  : at end my_team_alive > opp_team_alive
    - "loss" : reverse
    - "draw" : equal (rare or unfinished)
    """
    if not entries:
        return "draw"
    last = entries[-1]
    mine = int(last.get("my_team_alive") or 0)
    opp = int(last.get("opp_team_alive") or 0)
    if mine > opp:
        return "win"
    if opp > mine:
        return "loss"
    return "draw"


def format_decision(entry: Dict[str, Any]) -> str:
    turn = entry.get("turn", 0)
    kind = entry.get("kind", "?")
    my = entry.get("my_active") or "?"
    opp = entry.get("opp_active") or "?"
    chosen = entry.get("chosen") or "?"
    cands = entry.get("candidates") or []
    cand_strs = []
    for c in cands[:5]:
        mark = "*" if c.get("chosen") else " "
        cand_strs.append(
            f"{mark} {c.get('label', '?'):<14} wp={c.get('winprob', 0):.3f} ms={c.get('model_score', 0):.3f}"
        )
    cand_block = "\n      ".join(cand_strs)
    return (
        f"  T{turn:>2} {kind:<6} {my:<14} vs {opp:<14} "
        f"alive {entry.get('my_team_alive', '?')}/{entry.get('opp_team_alive', '?')} "
        f"hp {entry.get('my_hp_total', 0):.1f}/{entry.get('opp_hp_total', 0):.1f}\n"
        f"      => {chosen}\n      {cand_block}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--show_wins", action="store_true")
    parser.add_argument("--max_battles", type=int, default=5)
    args = parser.parse_args()

    by_battle = load_log(args.log_path)
    if not by_battle:
        print("No decisions found in", args.log_path)
        return 1

    outcomes: Dict[str, str] = {tag: battle_outcome(e) for tag, e in by_battle.items()}
    summary = defaultdict(int)
    for o in outcomes.values():
        summary[o] += 1

    print(f"Total battles logged : {len(by_battle)}")
    print(f"  wins  : {summary['win']}")
    print(f"  losses: {summary['loss']}")
    print(f"  draws : {summary['draw']}")
    print()

    target_outcomes = {"loss"}
    if args.show_wins:
        target_outcomes.add("win")

    shown = 0
    for tag, entries in sorted(by_battle.items()):
        if outcomes[tag] not in target_outcomes:
            continue
        if shown >= args.max_battles:
            break
        shown += 1
        print(f"=== Battle {tag}  [{outcomes[tag].upper()}]  {len(entries)} decisions ===")
        for entry in entries:
            print(format_decision(entry))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
