"""Strict gen9randombattle solo filters for replay datasets."""

from __future__ import annotations

from random_battle.config import FORMAT_ID

REJECTED_FORMATID_SUBSTRINGS = (
    "double",
    "factory",
    "bss",
    "baby",
    "blitz",
    "freeforall",
    "multi",
    "coop",
)


def is_gen9_solo_random_formatid(formatid: object) -> bool:
    fid = str(formatid or "").strip().lower()
    if fid != FORMAT_ID:
        return False
    return not any(token in fid for token in REJECTED_FORMATID_SUBSTRINGS)


def log_is_gen9_solo_random_battle(log_text: str) -> bool:
    """Confirm singles random battle from Showdown log header."""
    gametype_singles = False
    tier_ok = False
    for line in (log_text or "").split("\n")[:40]:
        if not line.startswith("|"):
            continue
        parts = [p.strip().lower() for p in line.split("|")]
        if len(parts) < 3:
            continue
        kind = parts[1]
        value = parts[2]
        if kind == "gametype":
            if value == "doubles":
                return False
            if value == "singles":
                gametype_singles = True
        elif kind == "tier":
            if any(x in value for x in ("double", "baby", "factory", "bss", "blitz")):
                return False
            if "random" in value and "battle" in value:
                tier_ok = True
    return gametype_singles and tier_ok


def accept_replay(
    *,
    formatid: object,
    log_text: str,
    require_log_check: bool = True,
) -> bool:
    if not is_gen9_solo_random_formatid(formatid):
        return False
    if require_log_check and not log_is_gen9_solo_random_battle(log_text):
        return False
    return True
