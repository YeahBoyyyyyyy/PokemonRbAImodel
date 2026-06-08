"""Team preview order and switch slot resolution for Random Battle."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

MAX_TEAM_SIZE = 6

SPECIES_ALIASES = {
    "sinistchamasterpiece": "sinistcha",
    "sinistchaunremarkable": "sinistcha",
    "polteageistantique": "polteageist",
    "polteageistphony": "polteageist",
    "gastrodoneast": "gastrodon",
    "gastrodonwest": "gastrodon",
    "ho-oh": "hooh",
    "pikachuoriginal": "pikachu",
}


def normalize_species_key(name: str) -> str:
    if not name:
        return ""
    lowered = name.lower().strip()
    lowered = re.sub(r"[^a-z0-9]", "", lowered)
    if lowered in SPECIES_ALIASES:
        return SPECIES_ALIASES[lowered]
    return lowered


def _append_species_to_order(
    order: List[str],
    seen: set[str],
    species: str,
) -> None:
    norm = normalize_species_key(species)
    if not norm or norm in seen or len(order) >= MAX_TEAM_SIZE:
        return
    seen.add(norm)
    order.append(species)


def infer_preview_order_from_switches(log_lines: List[str]) -> Dict[str, List[str]]:
    """First-seen species on |switch| (Pokéchamp logs often omit |poke|)."""
    order: Dict[str, List[str]] = {"p1": [], "p2": []}
    seen: Dict[str, set[str]] = {"p1": set(), "p2": set()}

    for line in log_lines:
        if not line.startswith("|switch|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        player_id = parts[2].split(":")[0][:2]
        if player_id not in order:
            continue
        species = parts[3].split(",")[0].strip()
        _append_species_to_order(order[player_id], seen[player_id], species)
    return order


def merge_preview_orders(
    primary: Dict[str, List[str]],
    secondary: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {"p1": [], "p2": []}
    for player in ("p1", "p2"):
        seen: set[str] = set()
        for source in (primary, secondary):
            for species in source.get(player, []):
                _append_species_to_order(merged[player], seen, species)
    return merged


def parse_team_preview_order(log_lines: List[str]) -> Dict[str, List[str]]:
    """Team preview slot order: |poke| when present, else first |switch| reveal order."""
    from_poke: Dict[str, List[str]] = {"p1": [], "p2": []}
    seen: Dict[str, set[str]] = {"p1": set(), "p2": set()}

    for line in log_lines:
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 4 or parts[1] != "poke":
            continue
        player_id = parts[2]
        if player_id not in from_poke:
            continue
        species = parts[3].split(",")[0].strip()
        _append_species_to_order(from_poke[player_id], seen[player_id], species)

    from_switch = infer_preview_order_from_switches(log_lines)
    return merge_preview_orders(from_poke, from_switch)


def parse_slot_index(action_target: object) -> Optional[int]:
    if isinstance(action_target, (int, float)):
        idx = int(action_target)
        if 1 <= idx <= MAX_TEAM_SIZE:
            return idx - 1
        if 0 <= idx < MAX_TEAM_SIZE:
            return idx
        return None
    if isinstance(action_target, str):
        raw = action_target.strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= MAX_TEAM_SIZE:
                return idx - 1
            if 0 <= idx < MAX_TEAM_SIZE:
                return idx
    return None


def resolve_switch_slot(
    action_target: object,
    preview_order: List[str],
    *,
    active_species: Optional[str] = None,
) -> Optional[int]:
    """
    Map a switch action to slot index 0-5 in team preview order.
    Skips the currently active slot when matching by species name.
    """
    slot = parse_slot_index(action_target)
    if slot is not None:
        return slot

    if not isinstance(action_target, str) or not preview_order:
        return None

    target_norm = normalize_species_key(action_target)
    if not target_norm:
        return None

    active_norm = normalize_species_key(active_species or "")

    for idx, species in enumerate(preview_order[:MAX_TEAM_SIZE]):
        if normalize_species_key(species) == target_norm:
            if active_norm and normalize_species_key(species) == active_norm:
                continue
            return idx
    return None


def normalize_switch_target(
    action_target: object,
    preview_order: List[str],
    *,
    active_species: Optional[str] = None,
) -> Optional[str]:
    """Return slot index as string 0-5 for storage in training JSON."""
    slot = resolve_switch_slot(action_target, preview_order, active_species=active_species)
    if slot is None:
        return None
    return str(slot)
