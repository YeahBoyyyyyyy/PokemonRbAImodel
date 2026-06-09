"""Run local benchmarks vs RandomAI / LowHeuristicAI and compare results.

Examples::

    # 100 games vs random + low (3-ply engine, 2-ply switches)
    python random_battle/benchmark.py run \\
        --n_battles 100 --opponents random low \\
        --search --no-hybrid --use_engine \\
        --engine_depth 3 --engine_switch_min_depth 2 \\
        --engine_n_worlds 1 --prune_delta 0.13 \\
        --engine_workers 4 --engine_prune_switches

    # Compare the two most recent benchmark runs
    python random_battle/benchmark.py compare --latest 2

    # Compare specific JSON files
    python random_battle/benchmark.py compare \\
        random_battle/artifacts/benchmarks/bench_a.json \\
        random_battle/artifacts/benchmarks/bench_b.json

    # List saved runs
    python random_battle/benchmark.py list
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from random_battle.config import ARTIFACTS_DIR, FORMAT_ID  # noqa: E402

BENCHMARK_DIR = ARTIFACTS_DIR / "benchmarks"
INDEX_PATH = ARTIFACTS_DIR / "benchmark_comparison.json"
RUNNER = _PROJECT_ROOT / "random_battle" / "run_rb_model_player.py"
PYTHON = sys.executable

OPPONENT_LABELS = {
    "random": "RandomAI",
    "low": "LowHeuristicAI",
    "high": "HighHeuristicAI",
}


@dataclass
class SeriesResult:
    opponent: str
    n_battles: int
    wins: int
    losses: int
    win_rate: float
    avg_turns: float
    elapsed_s: float
    stats_jsonl: Optional[str] = None
    summary_json: Optional[str] = None
    bot_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opponent": self.opponent,
            "opponent_label": OPPONENT_LABELS.get(self.opponent, self.opponent),
            "n_battles": self.n_battles,
            "wins": self.wins,
            "losses": self.losses,
            "finished": self.wins + self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_turns": round(self.avg_turns, 2),
            "elapsed_s": self.elapsed_s,
            "stats_jsonl": self.stats_jsonl,
            "summary_json": self.summary_json,
            "bot_name": self.bot_name,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stats_from_jsonl(path: Path) -> Tuple[int, int, float]:
    """Return (wins, losses, avg_turns)."""
    wins = 0
    losses = 0
    turns: List[int] = []
    if not path.is_file():
        return 0, 0, 0.0
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if bool(row.get("won")):
                wins += 1
            else:
                losses += 1
            t = int(row.get("turns") or 0)
            if t > 0:
                turns.append(t)
    avg_turns = sum(turns) / len(turns) if turns else 0.0
    return wins, losses, avg_turns


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes:02d}m"


def _format_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _row_win_rate(row: Dict[str, Any]) -> float:
    if "win_rate" in row and row["win_rate"] is not None:
        return float(row["win_rate"])
    wins = int(row.get("wins", 0))
    finished = int(row.get("finished", row.get("n_battles", 0)))
    if finished <= 0:
        finished = wins + int(row.get("losses", 0))
    return (wins / finished) if finished else 0.0


def _pct_delta(a: float, b: float) -> str:
    delta = (a - b) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}pp"


def _build_runner_cmd(
    *,
    opponent: str,
    n_battles: int,
    stats_path: Path,
    summary_path: Path,
    runner_args: Sequence[str],
) -> List[str]:
    cmd = [
        PYTHON,
        str(RUNNER),
        "--mode",
        "battle",
        "--vs",
        opponent,
        "--n_battles",
        str(n_battles),
        "--session_stats",
        str(stats_path),
        "--session_summary",
        str(summary_path),
        "--max_concurrent_battles",
        "1",
    ]
    cmd.extend(runner_args)
    return cmd


def run_series(
    opponent: str,
    *,
    n_battles: int,
    run_id: str,
    runner_args: Sequence[str],
) -> SeriesResult:
    stats_path = BENCHMARK_DIR / f"{run_id}_{opponent}.jsonl"
    summary_path = BENCHMARK_DIR / f"{run_id}_{opponent}_summary.json"
    cmd = _build_runner_cmd(
        opponent=opponent,
        n_battles=n_battles,
        stats_path=stats_path,
        summary_path=summary_path,
        runner_args=runner_args,
    )
    label = OPPONENT_LABELS.get(opponent, opponent)
    print(f"\n=== Bench vs {label} ({n_battles} combats) ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark vs {opponent} failed (exit {proc.returncode}). "
            "Vérifiez que le serveur local tourne (localhost:8000)."
        )

    wins, losses, avg_turns = _stats_from_jsonl(stats_path)
    finished = wins + losses
    if summary_path.is_file():
        summary = _load_json(summary_path)
        wins = int(summary.get("wins", wins))
        finished = int(summary.get("battles", finished))
        losses = max(0, finished - wins)
        bot_name = summary.get("bot")
    else:
        bot_name = None
    win_rate = (wins / finished) if finished else 0.0
    return SeriesResult(
        opponent=opponent,
        n_battles=n_battles,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_turns=avg_turns,
        elapsed_s=elapsed,
        stats_jsonl=str(stats_path),
        summary_json=str(summary_path),
        bot_name=bot_name,
    )


def _config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "label": args.label,
        "n_battles": args.n_battles,
        "opponents": list(args.opponents),
        "battle_format": FORMAT_ID,
        "search": bool(args.search),
        "hybrid": bool(args.hybrid),
        "use_engine": bool(args.use_engine),
        "engine_depth": args.engine_depth,
        "engine_switch_min_depth": args.engine_switch_min_depth,
        "engine_n_worlds": args.engine_n_worlds,
        "engine_prune_delta": args.engine_prune_delta,
        "engine_workers": args.engine_workers,
        "engine_prune_switches": bool(args.engine_prune_switches),
        "runner_extra": list(args.runner_extra or []),
    }


def _runner_args_from_namespace(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    if args.search:
        out.append("--search")
    if not args.hybrid:
        out.append("--no-hybrid")
    if args.use_engine:
        out.append("--use_engine")
    out.extend(
        [
            "--engine_depth",
            str(args.engine_depth),
            "--engine_switch_min_depth",
            str(args.engine_switch_min_depth),
            "--engine_n_worlds",
            str(args.engine_n_worlds),
            "--prune_delta",
            str(args.engine_prune_delta),
            "--engine_workers",
            str(args.engine_workers),
        ]
    )
    if args.engine_prune_switches:
        out.append("--engine_prune_switches")
    if args.save_replays:
        out.append("--save_replays")
    out.extend(args.runner_extra or [])
    return out


def _make_run_record(
    *,
    run_id: str,
    config: Dict[str, Any],
    results: Sequence[SeriesResult],
) -> Dict[str, Any]:
    total_elapsed = sum(r.elapsed_s for r in results)
    return {
        "run_id": run_id,
        "ts": _utc_now(),
        "config": config,
        "total_elapsed_s": total_elapsed,
        "results": [r.to_dict() for r in results],
    }


def _load_index() -> Dict[str, Any]:
    if not INDEX_PATH.is_file():
        return {
            "battle_format": FORMAT_ID,
            "runs": [],
            "results": [],
        }
    try:
        data = _load_json(INDEX_PATH)
    except (json.JSONDecodeError, OSError):
        return {"battle_format": FORMAT_ID, "runs": [], "results": []}
    data.setdefault("runs", [])
    return data


def _append_run_to_index(run: Dict[str, Any]) -> None:
    index = _load_index()
    runs: List[Dict[str, Any]] = list(index.get("runs") or [])
    runs.append(run)
    index["runs"] = runs[-30:]
    index["battle_format"] = run.get("config", {}).get("battle_format", FORMAT_ID)
    index["n_battles_per_series"] = run.get("config", {}).get("n_battles")
    index["last_run_id"] = run.get("run_id")
    index["last_updated"] = _utc_now()
    # Legacy flat list: latest run only (same shape as old benchmark_comparison.json).
    index["results"] = list(run.get("results") or [])
    _save_json(INDEX_PATH, index)


def _load_run_file(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    if "results" in data and "run_id" in data:
        return data
    # Legacy flat file.
    return {
        "run_id": path.stem,
        "ts": None,
        "config": {
            "n_battles": data.get("n_battles_per_series"),
            "label": path.stem,
        },
        "results": list(data.get("results") or []),
    }


def _legacy_runs_from_index() -> List[Dict[str, Any]]:
    """Rebuild pseudo-runs from a flat benchmark_comparison.json."""
    if not INDEX_PATH.is_file():
        return []
    try:
        data = _load_json(INDEX_PATH)
    except (json.JSONDecodeError, OSError):
        return []
    if data.get("runs"):
        return []
    flat = list(data.get("results") or [])
    if not flat:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in flat:
        if row.get("use_search"):
            key = "search"
        else:
            key = "hybrid"
        grouped.setdefault(key, []).append(row)

    runs: List[Dict[str, Any]] = []
    n_battles = data.get("n_battles_per_series")
    for key, rows in grouped.items():
        wins = sum(int(r.get("wins", 0)) for r in rows)
        finished = sum(int(r.get("finished", r.get("n_battles", 0))) for r in rows)
        runs.append(
            {
                "run_id": f"legacy_{key}",
                "ts": data.get("last_updated"),
                "config": {
                    "label": f"legacy ({key})",
                    "n_battles": n_battles,
                    "search": key == "search",
                    "battle_format": data.get("battle_format", FORMAT_ID),
                },
                "total_elapsed_s": sum(float(r.get("elapsed_s", 0.0)) for r in rows),
                "results": rows,
            }
        )
    return runs


def _resolve_run_paths(paths: Sequence[str], latest: int) -> List[Path]:
    if paths:
        return [Path(p) for p in paths]
    all_runs = sorted(BENCHMARK_DIR.glob("bench_*.json"))
    if all_runs:
        n = max(1, latest)
        return all_runs[-n:]
    if _legacy_runs_from_index():
        return []
    raise FileNotFoundError(
        f"Aucun benchmark dans {BENCHMARK_DIR} ni dans {INDEX_PATH}. "
        "Lancez d'abord: python random_battle/benchmark.py run ..."
    )


def print_run_table(run: Dict[str, Any]) -> None:
    run_id = run.get("run_id", "?")
    cfg = run.get("config") or {}
    label = cfg.get("label") or run_id
    depth = cfg.get("engine_depth")
    sw_depth = cfg.get("engine_switch_min_depth")
    search = cfg.get("search")
    engine = cfg.get("use_engine")
    n = cfg.get("n_battles")
    print(f"\n--- {label} ({run_id}) ---")
    flags = []
    if search:
        flags.append("search")
    if engine:
        flags.append(f"engine depth={depth}")
    if sw_depth is not None:
        flags.append(f"switch_depth={sw_depth}")
    if flags:
        print("Config : " + ", ".join(flags))
    print(
        f"{'Adversaire':<16} {'Score':<12} {'Win%':<8} "
        f"{'Tours moy.':<10} {'Durée':<10}"
    )
    print("-" * 58)
    for row in run.get("results") or []:
        opp = row.get("opponent_label") or OPPONENT_LABELS.get(
            row.get("opponent", ""), row.get("opponent", "?")
        )
        wins = int(row.get("wins", 0))
        finished = int(row.get("finished", row.get("n_battles", 0)))
        if finished <= 0:
            finished = wins + int(row.get("losses", 0))
        losses = max(0, finished - wins)
        rate = _row_win_rate(row)
        avg_turns = float(row.get("avg_turns", 0.0))
        elapsed = float(row.get("elapsed_s", 0.0))
        print(
            f"{opp:<16} {wins}/{finished:<10} {_format_pct(rate):<8} "
            f"{avg_turns:<10.1f} {_format_duration(elapsed):<10}"
        )
    total_elapsed = float(run.get("total_elapsed_s", 0.0))
    if not total_elapsed:
        total_elapsed = sum(
            float(r.get("elapsed_s", 0.0)) for r in (run.get("results") or [])
        )
    print(f"Total session : {_format_duration(total_elapsed)}")


def print_compare_runs(runs: Sequence[Dict[str, Any]]) -> None:
    if len(runs) == 1:
        print_run_table(runs[0])
        return

    print("\n=== Comparaison de benchmarks ===")
    for run in runs:
        print_run_table(run)

    opponents = sorted(
        {
            r.get("opponent")
            for run in runs
            for r in (run.get("results") or [])
            if r.get("opponent")
        }
    )
    if not opponents:
        return

    print("\n--- Écarts de win rate (vs première série) ---")
    base = runs[0]
    base_by_opp = {
        r.get("opponent"): r for r in (base.get("results") or []) if r.get("opponent")
    }
    headers = [runs[0].get("config", {}).get("label") or runs[0].get("run_id", "A")]
    headers.extend(
        (r.get("config", {}).get("label") or r.get("run_id", f"run{i}"))
        for i, r in enumerate(runs[1:], start=1)
    )
    print(f"{'Adversaire':<16} " + " | ".join(f"{h:<14}" for h in headers))
    print("-" * (18 + 17 * len(headers)))
    for opp in opponents:
        cells: List[str] = []
        base_rate = _row_win_rate(base_by_opp.get(opp) or {})
        for i, run in enumerate(runs):
            row = next(
                (r for r in (run.get("results") or []) if r.get("opponent") == opp),
                None,
            )
            if row is None:
                cells.append("—")
                continue
            rate = _row_win_rate(row)
            if i == 0:
                cells.append(_format_pct(rate))
            else:
                cells.append(f"{_format_pct(rate)} ({_pct_delta(rate, base_rate)})")
        opp_label = OPPONENT_LABELS.get(opp, opp)
        print(f"{opp_label:<16} " + " | ".join(f"{c:<14}" for c in cells))


def cmd_run(args: argparse.Namespace) -> int:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"bench_{stamp}"
    runner_args = _runner_args_from_namespace(args)
    config = _config_from_args(args)
    results: List[SeriesResult] = []
    for opponent in args.opponents:
        results.append(
            run_series(
                opponent,
                n_battles=args.n_battles,
                run_id=run_id,
                runner_args=runner_args,
            )
        )
    run_record = _make_run_record(run_id=run_id, config=config, results=results)
    out_path = BENCHMARK_DIR / f"{run_id}.json"
    _save_json(out_path, run_record)
    _append_run_to_index(run_record)
    print(f"\nRésultats enregistrés : {out_path}")
    print(f"Index global          : {INDEX_PATH}")
    print_run_table(run_record)
    if len(_load_index().get("runs") or []) >= 2:
        recent = [_load_run_file(p) for p in _resolve_run_paths([], latest=2)]
        if len(recent) == 2:
            print_compare_runs(recent)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    paths = _resolve_run_paths(args.runs, args.latest)
    if paths:
        runs = [_load_run_file(p) for p in paths]
    else:
        legacy_runs = _legacy_runs_from_index()
        if legacy_runs:
            n = max(1, args.latest)
            runs = legacy_runs[-n:]
        else:
            runs = []
    if not runs:
        print("Aucun run à comparer.")
        return 1
    print_compare_runs(runs)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    index = _load_index()
    runs = list(index.get("runs") or [])
    files = sorted(BENCHMARK_DIR.glob("bench_*.json"))
    print(f"Index : {INDEX_PATH} ({len(runs)} runs)")
    print(f"Fichiers dans {BENCHMARK_DIR} : {len(files)}")
    for run in runs[-max(1, args.last) :]:
        cfg = run.get("config") or {}
        label = cfg.get("label") or run.get("run_id")
        n = cfg.get("n_battles")
        opponents = ", ".join(cfg.get("opponents") or [])
        ts = (run.get("ts") or "")[:19]
        summary_parts = []
        for row in run.get("results") or []:
            opp = OPPONENT_LABELS.get(row.get("opponent", ""), row.get("opponent"))
            rate = _row_win_rate(row)
            summary_parts.append(f"{opp} {_format_pct(rate)}")
        print(
            f"  {run.get('run_id')}  {ts}  n={n}  [{opponents}]  "
            + " | ".join(summary_parts)
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark local vs RandomAI / LowHeuristicAI + comparaison."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Lancer une série de benchmarks.")
    run_p.add_argument("--n_battles", type=int, default=100)
    run_p.add_argument(
        "--opponents",
        nargs="+",
        choices=tuple(OPPONENT_LABELS.keys()),
        default=["random", "low"],
    )
    run_p.add_argument("--label", default="", help="Nom lisible pour cette config.")
    run_p.add_argument("--search", action="store_true")
    run_p.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_p.add_argument("--use_engine", action="store_true")
    run_p.add_argument("--engine_depth", type=int, default=3)
    run_p.add_argument("--engine_switch_min_depth", type=int, default=2)
    run_p.add_argument("--engine_n_worlds", type=int, default=1)
    run_p.add_argument("--prune_delta", dest="engine_prune_delta", type=float, default=0.13)
    run_p.add_argument("--engine_workers", type=int, default=4)
    run_p.add_argument("--engine_prune_switches", action="store_true")
    run_p.add_argument("--save_replays", action="store_true")
    run_p.add_argument(
        "--runner_extra",
        nargs=argparse.REMAINDER,
        help="Arguments supplémentaires passés à run_rb_model_player.py.",
    )

    cmp_p = sub.add_parser("compare", help="Comparer des runs enregistrés.")
    cmp_p.add_argument(
        "runs",
        nargs="*",
        help="Fichiers bench_*.json (défaut: --latest fichiers).",
    )
    cmp_p.add_argument(
        "--latest",
        type=int,
        default=2,
        help="Nombre de derniers runs à comparer si aucun fichier fourni.",
    )

    list_p = sub.add_parser("list", help="Lister les runs enregistrés.")
    list_p.add_argument("--last", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        if not args.label:
            parts = []
            if args.use_engine:
                parts.append(f"engine{args.engine_depth}")
            if args.search:
                parts.append("search")
            if not args.hybrid:
                parts.append("no-hybrid")
            args.label = "_".join(parts) if parts else "baseline"
        return cmd_run(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "list":
        return cmd_list(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
