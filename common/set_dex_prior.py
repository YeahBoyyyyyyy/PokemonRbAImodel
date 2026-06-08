"""Shared set-dex loading and Bayesian move priors for Random Battle."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def normalize_token(token: Optional[str]) -> str:
    if not token:
        return ""
    return re.sub(r"[^a-z0-9]", "", token.lower().strip())


def normalize_species_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())


def load_set_dex(path: Optional[str]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_species_entry(
    set_dex: Dict[str, object], species: str
) -> Tuple[Dict[str, int], int]:
    species_dict = set_dex.get("species", {})
    if not species_dict:
        return {}, 0

    candidates: set[str] = set()
    norm = normalize_species_name(species)
    if norm in species_dict:
        candidates.add(norm)
    if species in species_dict:
        candidates.add(species)
    lower = species.lower()
    for key in species_dict:
        if key.lower() == lower:
            candidates.add(key)
    for key in species_dict:
        if normalize_species_name(key) == norm:
            candidates.add(key)
    base = species.split("-", 1)[0]
    norm_base = normalize_species_name(base)
    for key in species_dict:
        if normalize_species_name(key).startswith(norm_base):
            candidates.add(key)

    merged_sets: Dict[str, int] = {}
    total = 0
    for key in sorted(candidates):
        entry = species_dict.get(key, {})
        for set_key, count in entry.get("sets", {}).items():
            merged_sets[set_key] = merged_sets.get(set_key, 0) + int(count)
        total += int(entry.get("total", 0))
    if not total and merged_sets:
        total = sum(merged_sets.values())
    return merged_sets, total


def aggregate_move_probabilities(
    set_dex: Optional[Dict[str, object]],
    species: str,
    observed_moves: List[str],
    mismatch_penalty: float,
) -> Dict[str, float]:
    """Probabilités relatives par token de coup (set-dex + coups déjà vus)."""
    observed = [normalize_token(m) for m in observed_moves if normalize_token(m)]
    move_probs: Dict[str, float] = {}

    if set_dex:
        sets, total = resolve_species_entry(set_dex, species)
        if sets and total > 0:
            for set_key, count in sets.items():
                parts = set_key.split("|") if set_key else []
                moves = [p for p in parts if not p.startswith("item=")]
                missing = [m for m in observed if m not in moves]
                if missing and mismatch_penalty <= 0:
                    continue
                score = count / total
                if missing:
                    score *= mismatch_penalty ** len(missing)
                for mv in moves:
                    move_probs[mv] = move_probs.get(mv, 0.0) + score

    for mv in observed:
        move_probs[mv] = max(move_probs.get(mv, 0.0), 1.0)

    total_score = sum(move_probs.values()) or 1.0
    return {mv: score / total_score for mv, score in move_probs.items()}


def enumerate_likely_moves(
    set_dex: Optional[Dict[str, object]],
    species: str,
    observed_moves: List[str],
    vocab: Dict[str, int],
    *,
    top_k: int = 6,
    min_prob: float = 0.03,
    mismatch_penalty: float = 0.15,
) -> List[str]:
    """Coups plausibles : moves_seen + top du prior set-dex (même logique que le modèle)."""
    observed = [normalize_token(m) for m in observed_moves if normalize_token(m)]
    probs = aggregate_move_probabilities(
        set_dex, species, observed_moves, mismatch_penalty
    )
    if not probs:
        return observed[:top_k]

    ranked = sorted(probs.items(), key=lambda pair: pair[1], reverse=True)
    chosen: List[str] = []
    for mv, p in ranked:
        if mv not in vocab and mv not in observed:
            continue
        if p < min_prob and mv not in observed:
            continue
        if mv not in chosen:
            chosen.append(mv)
        if len(chosen) >= top_k:
            break

    for mv in observed:
        if mv not in chosen:
            chosen.insert(0, mv)
    return chosen[:top_k]


def compute_move_prior(
    set_dex: Optional[Dict[str, object]],
    species: str,
    observed_moves: List[str],
    mismatch_penalty: float,
    vocab: Dict[str, int],
) -> np.ndarray:
    probs = aggregate_move_probabilities(
        set_dex, species, observed_moves, mismatch_penalty
    )
    if not probs:
        return np.zeros(len(vocab) + 1, dtype=np.float32)

    vec = np.zeros(len(vocab) + 1, dtype=np.float32)
    for mv, score in probs.items():
        mv_id = vocab.get(mv)
        if mv_id:
            vec[mv_id] = score
    return vec


def compute_opponent_move_prior(
    set_dex: Optional[Dict[str, object]],
    species: str,
    observed_moves: List[str],
    mismatch_penalty: float,
    vocab: Dict[str, int],
) -> np.ndarray:
    """Same logic as compute_move_prior — for the active opponent species."""
    return compute_move_prior(set_dex, species, observed_moves, mismatch_penalty, vocab)
