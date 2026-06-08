"""Session logging: per-battle JSONL + cumulative win stats."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from poke_env.battle import AbstractBattle


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_battle_record(
    stats_path: Path,
    *,
    battle: AbstractBattle,
    bot_username: str,
    rating_before: Optional[int] = None,
    rating_after: Optional[int] = None,
    rating_delta: Optional[int] = None,
    opponent_rating: Optional[int] = None,
) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    opponent = getattr(battle, "opponent_username", None) or "?"
    entry = {
        "ts": _utc_now(),
        "battle_tag": battle.battle_tag,
        "won": bool(getattr(battle, "won", False)),
        "opponent": opponent,
        "turns": int(getattr(battle, "turn", 0) or 0),
        "format": getattr(battle, "format", None),
        "bot": bot_username,
    }
    if rating_before is not None:
        entry["rating_before"] = rating_before
    if rating_after is not None:
        entry["rating_after"] = rating_after
    if rating_delta is not None:
        entry["rating_delta"] = rating_delta
    if opponent_rating is not None:
        entry["opponent_rating"] = opponent_rating
    with stats_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_cumulative_stats(cumulative_path: Path) -> Dict[str, Any]:
    if not cumulative_path.is_file():
        return {"total_battles": 0, "total_wins": 0, "total_losses": 0, "sessions": []}
    try:
        return json.loads(cumulative_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"total_battles": 0, "total_wins": 0, "total_losses": 0, "sessions": []}


def write_session_summary(
    summary_path: Path,
    *,
    bot_username: str,
    n_finished: int,
    n_won: int,
    mode: str,
    server: str,
    battle_format: str,
    stats_path: Optional[Path] = None,
    replays_dir: Optional[Path] = None,
    cumulative_path: Optional[Path] = None,
    elo_start: Optional[int] = None,
    elo_end: Optional[int] = None,
) -> Dict[str, Any]:
    n_lost = max(0, n_finished - n_won)
    win_rate = (n_won / n_finished) if n_finished else 0.0
    summary = {
        "ts": _utc_now(),
        "bot": bot_username,
        "mode": mode,
        "server": server,
        "format": battle_format,
        "battles": n_finished,
        "wins": n_won,
        "losses": n_lost,
        "win_rate": round(win_rate, 4),
        "stats_jsonl": str(stats_path) if stats_path else None,
        "replays_dir": str(replays_dir) if replays_dir else None,
    }
    if elo_start is not None:
        summary["elo_start"] = elo_start
    if elo_end is not None:
        summary["elo_end"] = elo_end
    if elo_start is not None and elo_end is not None:
        summary["elo_delta"] = elo_end - elo_start
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if cumulative_path is not None:
        cum = load_cumulative_stats(cumulative_path)
        cum["total_battles"] = int(cum.get("total_battles", 0)) + n_finished
        cum["total_wins"] = int(cum.get("total_wins", 0)) + n_won
        cum["total_losses"] = int(cum.get("total_losses", 0)) + n_lost
        cum["last_updated"] = _utc_now()
        sessions = list(cum.get("sessions") or [])
        sessions.append(summary)
        cum["sessions"] = sessions[-50:]
        cumulative_path.parent.mkdir(parents=True, exist_ok=True)
        cumulative_path.write_text(
            json.dumps(cum, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def attach_session_logging(
    player: Any,
    *,
    stats_path: Optional[Path],
    bot_username: str,
) -> None:
    """Hook poke-env battle-finished callback to append JSONL records."""
    if stats_path is None:
        return
    original = player._battle_finished_callback

    def _wrapped(battle: AbstractBattle) -> None:
        original(battle)
        if getattr(player, "_elo_tracker", None) is not None:
            return
        try:
            append_battle_record(stats_path, battle=battle, bot_username=bot_username)
        except Exception as exc:
            print(f"[session] failed to log battle: {exc}")

    player._battle_finished_callback = _wrapped
