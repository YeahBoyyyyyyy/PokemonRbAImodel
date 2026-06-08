"""Build Showdown-compatible team sets from a poke-env battle state.

This module is the bridge between observed battle state (poke-env) and the
Showdown simulator (@pkmn/sim) called from Node.js. It produces "set" dicts
that ``@pkmn/sim`` understands directly (no need for packed-team parsing).

Two strategies coexist:

* For our own team: extract everything we know directly. Missing fields are
  filled in with Random-Battle conventions (level from set-dex, nature
  Hardy, 85 EVs / 31 IVs).
* For the opponent: pick the most likely role from ``rb_set_dex`` (filtered
  by ``moves_seen``) and use its ability/item/moves/teraTypes top-1. Unknown
  team members are filled with neutral placeholders.

The output is a list of 6 sets per side, ready to be JSON-serialised and
sent to the Node bridge.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Random-Battle conventions
# ---------------------------------------------------------------------------

# All RB Pokemon use neutral natures; Hardy is the conventional one.
RB_NEUTRAL_NATURE = "Hardy"

# Default Gen-9 RB EV/IV spread: 85 in every stat, 31 IVs.
RB_DEFAULT_EVS: Dict[str, int] = {"hp": 85, "atk": 85, "def": 85, "spa": 85, "spd": 85, "spe": 85}
RB_DEFAULT_IVS: Dict[str, int] = {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}

# A move whose ID belongs to this set zeroes the corresponding offensive stat
# (mirrors Showdown's getRandomTeam logic).
_TRICKROOM_MOVES = {"trickroom"}
_GYROBALL_MOVES = {"gyroball"}

# Fallback placeholder used when we don't even know the species (slot never
# revealed and no set-dex available). We deliberately avoid Ditto + Imposter
# here: when Imposter triggers it transforms into our active Pokemon, which
# produces wildly unrealistic mirror-match simulations. Smeargle with passive
# moves is functionally inert: low stats, no STAB, status moves only.
PLACEHOLDER_SET: Dict[str, Any] = {
    "name": "Placeholder",
    "species": "Smeargle",
    "level": 80,
    "ability": "Own Tempo",
    "item": "Focus Sash",
    "moves": ["Splash", "Defense Curl", "Snore", "Endure"],
    "nature": RB_NEUTRAL_NATURE,
    "evs": dict(RB_DEFAULT_EVS),
    "ivs": dict(RB_DEFAULT_IVS),
    "teraType": "Normal",
    "gender": "",
}

_SLUG_RE = re.compile(r"[^a-z0-9]")
_ID_RE = re.compile(r"[^a-z0-9]")


def _slug(token: Optional[object]) -> str:
    """Normalize a token to a Showdown ID (lowercase alphanumeric only)."""
    if token is None:
        return ""
    raw = getattr(token, "name", token)
    return _ID_RE.sub("", str(raw).lower().strip())


def _species_id(token: Optional[object]) -> str:
    """Like _slug but preserves species hyphens (gen9 forms)."""
    if token is None:
        return ""
    raw = getattr(token, "name", token)
    text = str(raw).strip().lower().replace("_", "-")
    # keep alphanumerics + hyphens
    return re.sub(r"[^a-z0-9-]", "", text)


def _resolve_species_entry(set_dex: Optional[Dict[str, Any]], species: str) -> Tuple[Dict[str, Any], int]:
    """Return (species_entry, total_count) from rb_set_dex, robust to form name variants."""
    if not set_dex:
        return {}, 0
    species_dict = set_dex.get("species", {})
    if not species_dict:
        return {}, 0
    norm = _slug(species)
    # exact id match
    for key, entry in species_dict.items():
        if _slug(key) == norm:
            return entry, int(entry.get("total", 0))
    # base-name match (e.g. "tauros-paldea-aqua" -> "tauros")
    base = species.split("-", 1)[0]
    base_norm = _slug(base)
    for key, entry in species_dict.items():
        if _slug(key).startswith(base_norm):
            return entry, int(entry.get("total", 0))
    return {}, 0


def _score_role(role: Dict[str, Any], observed: set) -> float:
    """Compatibility score between a role and a set of observed move ids."""
    moves = {_slug(m) for m in (role.get("moves") or [])}
    matched = len(observed & moves)
    score = 100.0 * matched + len(moves)
    score -= 50.0 * len(observed - moves)
    return score


def _pick_best_role(
    roles: Dict[str, Any], observed_moves_ids: Sequence[str]
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Pick the role whose moveset best matches observed_moves_ids.

    Tie-broken by largest move-count (preferring fuller sets).
    """
    if not roles:
        return None, None
    obs = set(observed_moves_ids)
    best_name, best_role, best_score = None, None, -1.0
    for name, role in roles.items():
        score = _score_role(role, obs)
        if score > best_score:
            best_score = score
            best_name = name
            best_role = role
    return best_name, best_role


def _sample_role(
    roles: Dict[str, Any],
    observed_moves_ids: Sequence[str],
    rng: random.Random,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Sample a role with probability proportional to compatibility score.

    Roles that contradict observed moves are heavily down-weighted, but not
    eliminated entirely (so a wrongly-classified move doesn't lock us out
    of the right role). Falls back to ``_pick_best_role`` when ``roles`` is
    empty or unique.
    """
    if not roles:
        return None, None
    items = list(roles.items())
    if len(items) == 1:
        name, role = items[0]
        return name, role
    obs = set(observed_moves_ids)
    scored: List[Tuple[str, Dict[str, Any], float]] = []
    for name, role in items:
        scored.append((name, role, _score_role(role, obs)))
    base = min(s for _, _, s in scored)
    weights = [max(1e-3, s - base + 1.0) for _, _, s in scored]
    idx = _weighted_choice(weights, rng)
    name, role, _ = scored[idx]
    return name, role


def _weighted_choice(weights: Sequence[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return 0
    threshold = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= threshold:
            return i
    return len(weights) - 1


def _sample_from_list(items: Sequence[Any], rng: Optional[random.Random]) -> Optional[Any]:
    """Return a single element from ``items`` (random when rng is provided)."""
    if not items:
        return None
    if rng is None:
        return items[0]
    return rng.choice(list(items))


def _ev_iv_adjustments(moves_ids: Iterable[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Apply Showdown's stat-zeroing rules for problematic stats.

    - Trick Room / Gyro Ball: Speed 0 EV, 0 IV
    - (We could add: physical-only attackers should be SpA 0, but it's
      cosmetic for damage calc only; leaving 85/31 is safe for hidden info.)
    """
    evs = dict(RB_DEFAULT_EVS)
    ivs = dict(RB_DEFAULT_IVS)
    move_set = {m for m in moves_ids if m}
    if move_set & (_TRICKROOM_MOVES | _GYROBALL_MOVES):
        evs["spe"] = 0
        ivs["spe"] = 0
    return evs, ivs


# ---------------------------------------------------------------------------
# Set builders (own + opponent)
# ---------------------------------------------------------------------------


def _safe_attr(obj: Any, *names: str) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _moves_from_poke_env(mon: Any, *, is_own: bool) -> List[str]:
    moves = getattr(mon, "moves", None) or {}
    seen: List[str] = []
    for move in moves.values():
        token = _slug(getattr(move, "id", None) or getattr(move, "name", move))
        if token and token not in seen:
            seen.append(token)
    return seen


def _level_for(mon: Any, fallback_role_level: Optional[int]) -> int:
    level = getattr(mon, "level", None)
    if isinstance(level, int) and level > 0:
        return level
    if fallback_role_level:
        return int(fallback_role_level)
    return 80


def _ability_from_role(
    role: Optional[Dict[str, Any]], rng: Optional[random.Random] = None
) -> Optional[str]:
    if not role:
        return None
    abilities = role.get("abilities") or []
    picked = _sample_from_list(abilities, rng)
    return str(picked) if picked else None


def _item_from_role(
    role: Optional[Dict[str, Any]], rng: Optional[random.Random] = None
) -> Optional[str]:
    if not role:
        return None
    items = role.get("items") or []
    picked = _sample_from_list(items, rng)
    return str(picked) if picked else None


def _tera_from_role(
    role: Optional[Dict[str, Any]], rng: Optional[random.Random] = None
) -> Optional[str]:
    if not role:
        return None
    types = role.get("teraTypes") or []
    picked = _sample_from_list(types, rng)
    return str(picked) if picked else None


def _complete_moves(
    observed_ids: Sequence[str],
    role: Optional[Dict[str, Any]],
    *,
    desired: int = 4,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Combine observed moves with the role's moves to reach ``desired`` size.

    When ``rng`` is provided, missing slots are sampled from the role pool
    instead of taking the top of the list. The first elements (observed)
    are always kept in their original order so the move slots stay stable
    when poke-env reveals new attacks across turns.
    """
    result: List[str] = []
    seen_set: set = set()
    for token in observed_ids:
        if token and token not in seen_set:
            result.append(token)
            seen_set.add(token)
            if len(result) >= desired:
                return result
    if role:
        pool = [_slug(m) for m in (role.get("moves") or []) if _slug(m)]
        pool = [m for m in pool if m not in seen_set]
        if rng is not None:
            rng.shuffle(pool)
        for token in pool:
            result.append(token)
            seen_set.add(token)
            if len(result) >= desired:
                break
    if not result:
        result = ["tackle"]
    return result[:desired]


@dataclass(frozen=True)
class SetBuildConfig:
    """Knobs controlling set construction for the opponent side."""

    # When set, override the role selection (debug / testing).
    forced_role: Optional[str] = None
    # If True, force teraType from set_dex even if poke-env exposes one.
    prefer_role_tera: bool = False
    # When provided, the builder samples among compatible roles / items /
    # abilities / tera types / moves instead of always taking the top
    # candidate. Use a different seed per "world" to generate varied
    # opponent hypotheses for aggregation.
    sampling_seed: Optional[int] = None


def build_set_for_pokemon(
    mon: Any,
    *,
    is_own: bool,
    set_dex: Optional[Dict[str, Any]] = None,
    config: Optional[SetBuildConfig] = None,
) -> Dict[str, Any]:
    """Build a Showdown set dict for a poke-env Pokemon object.

    `is_own=True` extracts everything from poke-env (we know it all).
    `is_own=False` uses `set_dex` to fill unknown fields.
    """
    cfg = config or SetBuildConfig()
    species_raw = _safe_attr(mon, "species", "_species", "name") or "unknown"
    species = _species_id(species_raw)

    # Per-mon rng so each Pokemon picks its own variant deterministically.
    rng: Optional[random.Random] = None
    if cfg.sampling_seed is not None and not is_own:
        # Combine the variant seed with the species so different mons in the
        # same variant don't all pick the same role index. Use a stable hash
        # (Python's hash() is salted across processes, so we mix manually).
        species_hash = sum(ord(c) * (31 ** i) for i, c in enumerate(species)) & 0xFFFFFFFF
        rng = random.Random(int(cfg.sampling_seed) ^ species_hash)

    species_entry, _total = _resolve_species_entry(set_dex, species)
    roles = species_entry.get("roles") if species_entry else {}
    fallback_level = species_entry.get("level") if species_entry else None

    observed_moves = _moves_from_poke_env(mon, is_own=is_own)

    if cfg.forced_role and roles and cfg.forced_role in roles:
        role_name, role = cfg.forced_role, roles[cfg.forced_role]
    elif rng is not None:
        role_name, role = _sample_role(roles or {}, observed_moves, rng)
    else:
        role_name, role = _pick_best_role(roles or {}, observed_moves)

    # Ability
    poke_env_ability = _slug(_safe_attr(mon, "ability"))
    if poke_env_ability:
        ability = poke_env_ability
    else:
        ability = _slug(_ability_from_role(role, rng)) or "noability"

    # Item
    poke_env_item = _slug(_safe_attr(mon, "item"))
    if poke_env_item == "unknown_item":
        poke_env_item = ""
    if poke_env_item:
        item = poke_env_item
    else:
        item = _slug(_item_from_role(role, rng))

    # Moves: own pokemon — keep everything from poke-env (up to 4); opponent
    # — use observed + top (or sampled) of role to reach 4 entries.
    if is_own:
        moves = observed_moves[:4] if observed_moves else _complete_moves([], role)
    else:
        moves = _complete_moves(observed_moves, role, rng=rng)

    # Tera type
    poke_env_tera = _slug(_safe_attr(mon, "tera_type"))
    if cfg.prefer_role_tera or not poke_env_tera:
        tera = _slug(_tera_from_role(role, rng)) or poke_env_tera or "normal"
    else:
        tera = poke_env_tera

    evs, ivs = _ev_iv_adjustments(moves)

    level = _level_for(mon, fallback_level)

    return {
        "name": species,
        "species": species,
        "item": item or None,
        "ability": ability,
        "moves": moves,
        "nature": RB_NEUTRAL_NATURE,
        "evs": evs,
        "ivs": ivs,
        "level": int(level),
        "teraType": (tera or "normal").capitalize(),
        "gender": "",
    }


# ---------------------------------------------------------------------------
# Whole-team builders
# ---------------------------------------------------------------------------


def _iter_team(battle: Any, *, opponent: bool) -> List[Any]:
    """Return poke-env Pokemon objects for one side, **active first**.

    Showdown puts whoever is at position 1 on the field after team preview.
    Failing to put the actually-active mon at slot 1 will make the engine
    simulate the wrong matchup. The remaining mons keep their original
    discovery order.
    """
    if opponent:
        team = getattr(battle, "opponent_team", None) or {}
        active_ref = getattr(battle, "opponent_active_pokemon", None)
    else:
        team = getattr(battle, "team", None) or {}
        active_ref = getattr(battle, "active_pokemon", None)

    if isinstance(team, dict):
        members = list(team.values())
    else:
        members = list(team or [])

    if not members:
        return members

    def is_active(mon: Any) -> bool:
        if mon is active_ref:
            return True
        if active_ref is not None and getattr(mon, "species", None) == getattr(active_ref, "species", None):
            return True
        return bool(getattr(mon, "active", False))

    active = [m for m in members if is_active(m)]
    rest = [m for m in members if not is_active(m)]
    return active + rest


def _sample_placeholder_from_set_dex(
    set_dex: Optional[Dict[str, Any]],
    rng: random.Random,
    *,
    avoid_species: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Pick a plausible random RB Pokemon from ``set_dex`` and build a full set.

    Sampling is weighted by ``species[*].total`` (usage in the empirical
    dex), so commonly-played mons appear more often. Returns ``None`` if
    no set_dex is available or all candidates were rejected.
    """
    if not set_dex:
        return None
    species_dict = set_dex.get("species") or {}
    if not species_dict:
        return None
    avoid_species = avoid_species or set()
    pool = [
        (species_id, entry)
        for species_id, entry in species_dict.items()
        if entry.get("roles")
        and _slug(species_id) not in avoid_species
        and int(entry.get("total", 0)) > 0
    ]
    if not pool:
        return None
    weights = [int(entry.get("total", 0)) for _, entry in pool]
    idx = _weighted_choice(weights, rng)
    species_id, entry = pool[idx]

    # Pick a random role weighted by appearance count (entry.sets has
    # frequency info; for simplicity we sample uniformly among roles).
    roles = entry.get("roles") or {}
    if not roles:
        return None
    role_names = list(roles.keys())
    role_name = rng.choice(role_names)
    role = roles[role_name]
    ability = _sample_from_list(role.get("abilities") or [], rng) or "noability"
    item = _sample_from_list(role.get("items") or [], rng)
    tera = _sample_from_list(role.get("teraTypes") or [], rng) or "Normal"
    moves = _complete_moves([], role, rng=rng)
    level = int(entry.get("level") or 80)
    evs, ivs = _ev_iv_adjustments(moves)
    return {
        "name": species_id,
        "species": species_id,
        "ability": _slug(ability) or "noability",
        "item": _slug(item) or None,
        "moves": moves,
        "nature": RB_NEUTRAL_NATURE,
        "evs": evs,
        "ivs": ivs,
        "level": level,
        "teraType": str(tera).capitalize(),
        "gender": "",
    }


def build_team_sets(
    battle: Any,
    *,
    opponent: bool,
    set_dex: Optional[Dict[str, Any]] = None,
    desired_size: int = 6,
    config: Optional[SetBuildConfig] = None,
) -> List[Dict[str, Any]]:
    """Build a list of Showdown sets for one side, padding with placeholders.

    Padding strategy (opponent only):
    * If ``set_dex`` + sampling_seed are available, sample plausible random
      mons (weighted by usage) for the unrevealed slots. This avoids the
      pathological "all Smeargle" hidden bench that lets opponents pivot
      into completely inert placeholders during simulations.
    * Otherwise, fall back to the static :data:`PLACEHOLDER_SET`.
    """
    cfg = config or SetBuildConfig()
    sets: List[Dict[str, Any]] = []
    revealed_species: set = set()
    for mon in _iter_team(battle, opponent=opponent):
        try:
            built = build_set_for_pokemon(
                mon, is_own=not opponent, set_dex=set_dex, config=cfg
            )
            sets.append(built)
            revealed_species.add(_slug(built.get("species") or ""))
        except Exception as exc:
            sets.append({**PLACEHOLDER_SET, "name": "BuildErr", "comment": str(exc)})

    # Pad unrevealed slots. Use random sampling only on the opponent side
    # and only when we have a set_dex + a deterministic seed.
    can_sample = (
        opponent
        and set_dex is not None
        and cfg.sampling_seed is not None
    )
    pad_rng: Optional[random.Random] = None
    if can_sample:
        pad_rng = random.Random(int(cfg.sampling_seed) ^ 0xC0FFEE)

    while len(sets) < desired_size:
        slot_idx = len(sets)
        placeholder: Optional[Dict[str, Any]] = None
        if pad_rng is not None:
            placeholder = _sample_placeholder_from_set_dex(
                set_dex, pad_rng, avoid_species=revealed_species
            )
            if placeholder is not None:
                revealed_species.add(_slug(placeholder.get("species") or ""))
        if placeholder is None:
            placeholder = {**PLACEHOLDER_SET, "name": f"Placeholder{slot_idx + 1}"}
        sets.append(placeholder)
    return sets[:desired_size]


def build_battle_teams(
    battle: Any,
    *,
    set_dex: Optional[Dict[str, Any]] = None,
    config: Optional[SetBuildConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (our_team, opponent_team) as lists of Showdown sets.

    The two teams are intended to be passed to the sim_bridge as p1/p2.
    Our team is always p1.
    """
    p1 = build_team_sets(battle, opponent=False, set_dex=set_dex, config=config)
    p2 = build_team_sets(battle, opponent=True, set_dex=set_dex, config=config)
    return p1, p2


def build_battle_teams_variants(
    battle: Any,
    *,
    n_variants: int,
    set_dex: Optional[Dict[str, Any]] = None,
    base_config: Optional[SetBuildConfig] = None,
    base_seed: int = 0,
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Build ``n_variants`` plausible opponent hypotheses.

    The player team (p1) is identical across variants -- we know it. Only the
    opponent's unknown attributes (role, item, ability, tera type, missing
    moves) vary. The first variant always uses the top-match role (same as
    a single ``build_battle_teams`` call), and subsequent variants sample
    alternative roles/items/etc. so the simulation can hedge against
    uncertainty.
    """
    n = max(1, n_variants)
    base = base_config or SetBuildConfig()
    p1_known = build_team_sets(battle, opponent=False, set_dex=set_dex, config=base)
    variants: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    # When sampling more than one world we want every variant (including
    # variant 0) to use the set_dex for padding, otherwise variant 0 would
    # be the only one polluted with Smeargle placeholders and would bias
    # the averaged score downwards.
    force_sampling_on_first = n > 1
    for i in range(n):
        if i == 0 and base.sampling_seed is None and not force_sampling_on_first:
            cfg = base  # deterministic top-match (single-world legacy)
        else:
            cfg = SetBuildConfig(
                forced_role=base.forced_role,
                prefer_role_tera=base.prefer_role_tera,
                sampling_seed=(base.sampling_seed or 0) ^ (base_seed + i + 1),
            )
        p2 = build_team_sets(battle, opponent=True, set_dex=set_dex, config=cfg)
        variants.append((list(p1_known), p2))
    return variants
