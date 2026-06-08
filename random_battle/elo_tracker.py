"""Track Showdown ladder Elo across battles (live + JSONL + cumulative)."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from poke_env.battle import AbstractBattle

SHOWDOWN_USER_API = "https://pokemonshowdown.com/users/{username}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int_rating(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _player_start_rating(battle: AbstractBattle, username: str) -> Optional[int]:
    for player in getattr(battle, "_players", None) or []:
        if player.get("username") == username and "rating" in player:
            return _to_int_rating(player.get("rating"))
    return None


def fetch_showdown_rating(username: str, battle_format: str) -> Optional[Dict[str, Any]]:
    """Synchronous fetch of current ladder stats from Showdown's public API."""
    url = SHOWDOWN_USER_API.format(username=username.lower())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PokemonRbAImodel/1.0 (elo-tracker)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    ratings = payload.get("ratings") or {}
    fmt_key = re.sub(r"[^a-z0-9]+", "", battle_format.lower())
    entry = ratings.get(fmt_key)
    if not isinstance(entry, dict):
        return None
    elo = _to_int_rating(entry.get("elo"))
    if elo is None:
        return None
    return {
        "elo": elo,
        "gxe": entry.get("gxe"),
        "w": entry.get("w"),
        "l": entry.get("l"),
    }


async def fetch_showdown_rating_async(
    username: str, battle_format: str
) -> Optional[Dict[str, Any]]:
    return await asyncio.to_thread(fetch_showdown_rating, username, battle_format)


class EloTracker:
    """Per-session Elo state; persists each rated battle."""

    def __init__(
        self,
        *,
        username: str,
        battle_format: str,
        history_path: Optional[Path] = None,
        cumulative_path: Optional[Path] = None,
    ) -> None:
        self.username = username
        self.battle_format = battle_format
        self.history_path = history_path
        self.cumulative_path = cumulative_path
        self.session_start_elo: Optional[int] = None
        self.last_elo: Optional[int] = None
        self.session_elo_delta: int = 0

    async def refresh_from_api(self) -> Optional[int]:
        stats = await fetch_showdown_rating_async(self.username, self.battle_format)
        if stats is None:
            return None
        elo = int(stats["elo"])
        if self.session_start_elo is None:
            self.session_start_elo = elo
        self.last_elo = elo
        return elo

    def record_battle(self, battle: AbstractBattle) -> Dict[str, Any]:
        won = bool(getattr(battle, "won", False))
        rating_before = _player_start_rating(battle, self.username)
        if rating_before is None:
            rating_before = self.last_elo

        rating_after = _to_int_rating(getattr(battle, "rating", None))
        opponent_rating = _to_int_rating(getattr(battle, "opponent_rating", None))

        delta: Optional[int] = None
        if rating_before is not None and rating_after is not None:
            delta = rating_after - rating_before
            self.session_elo_delta += delta

        if rating_after is not None:
            self.last_elo = rating_after
        elif rating_before is not None:
            self.last_elo = rating_before

        return {
            "ts": _utc_now(),
            "username": self.username,
            "format": self.battle_format,
            "battle_tag": getattr(battle, "battle_tag", None),
            "won": won,
            "opponent": getattr(battle, "opponent_username", None),
            "rating_before": rating_before,
            "rating_after": rating_after,
            "rating_delta": delta,
            "opponent_rating": opponent_rating,
            "session_elo": self.last_elo,
        }

    def persist(self, entry: Dict[str, Any]) -> None:
        if self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.cumulative_path is not None:
            self._update_cumulative(entry)

    def _update_cumulative(self, entry: Dict[str, Any]) -> None:
        from random_battle.session_log import load_cumulative_stats

        cum = load_cumulative_stats(self.cumulative_path)
        elo_block = dict(cum.get("elo") or {})
        key = f"{self.username}:{self.battle_format}"
        prev = dict(elo_block.get(key) or {})
        history = list(prev.get("history") or [])
        history.append(
            {
                "ts": entry.get("ts"),
                "battle_tag": entry.get("battle_tag"),
                "won": entry.get("won"),
                "rating_before": entry.get("rating_before"),
                "rating_after": entry.get("rating_after"),
                "rating_delta": entry.get("rating_delta"),
            }
        )
        elo_block[key] = {
            "username": self.username,
            "format": self.battle_format,
            "current_elo": entry.get("rating_after") or entry.get("session_elo"),
            "session_start_elo": self.session_start_elo,
            "last_delta": entry.get("rating_delta"),
            "last_updated": _utc_now(),
            "history": history[-200:],
        }
        cum["elo"] = elo_block
        cum["last_updated"] = _utc_now()
        self.cumulative_path.parent.mkdir(parents=True, exist_ok=True)
        self.cumulative_path.write_text(
            json.dumps(cum, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def format_live_battle_line(self, entry: Dict[str, Any], battle: AbstractBattle) -> str:
        tag = "WIN" if entry.get("won") else "LOSS"
        opp = entry.get("opponent") or "?"
        turns = getattr(battle, "turn", "?")
        before = entry.get("rating_before")
        after = entry.get("rating_after")
        delta = entry.get("rating_delta")
        if after is not None and delta is not None:
            sign = "+" if delta >= 0 else ""
            elo_part = f"Elo {before} → {after} ({sign}{delta})"
        elif after is not None:
            elo_part = f"Elo {after}"
        else:
            elo_part = "Elo ?"
        return f"=== {tag} vs {opp} ({turns} tours) | {elo_part} ==="

    def format_session_elo_summary(self) -> Optional[str]:
        if self.session_start_elo is None and self.last_elo is None:
            return None
        start = self.session_start_elo if self.session_start_elo is not None else "?"
        end = self.last_elo if self.last_elo is not None else "?"
        if self.session_start_elo is not None and self.last_elo is not None:
            delta = self.last_elo - self.session_start_elo
            sign = "+" if delta >= 0 else ""
            return f"Elo session : {start} → {end} ({sign}{delta})"
        return f"Elo actuel : {end}"


def attach_elo_tracking(
    player: Any,
    tracker: EloTracker,
    *,
    stats_path: Optional[Path] = None,
    live_print_fn: Optional[Any] = None,
) -> None:
    """Hook battle-finished to record Elo after each rated game."""
    from random_battle.session_log import append_battle_record

    player._elo_tracker = tracker
    original = player._battle_finished_callback

    def _wrapped(battle: AbstractBattle) -> None:
        original(battle)
        try:
            entry = tracker.record_battle(battle)
            tracker.persist(entry)
            if stats_path is not None:
                append_battle_record(
                    stats_path,
                    battle=battle,
                    bot_username=tracker.username,
                    rating_before=entry.get("rating_before"),
                    rating_after=entry.get("rating_after"),
                    rating_delta=entry.get("rating_delta"),
                    opponent_rating=entry.get("opponent_rating"),
                )
            if live_print_fn is not None:
                w = player.n_won_battles
                n = player.n_finished_battles
                line = tracker.format_live_battle_line(entry, battle)
                live_print_fn(f"{line} | session {w}/{n} wins")
        except Exception as exc:
            print(f"[elo] failed to record rating: {exc}")

    player._battle_finished_callback = _wrapped
