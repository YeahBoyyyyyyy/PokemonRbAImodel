"""Convert an ``EngineSnapshot`` back into the state-dict format used by
:class:`WinRateEvaluator` / the multi-head model.

We don't rebuild the state from scratch (that would lose the preview order,
the move vocabulary mapping, the revealed/preview distinctions, ...). Instead
we take the **base state-dict** that the caller already computed for the
current turn and patch it with the post-turn HP / status / boosts / hazards
extracted from the engine. This keeps the model's input format perfectly
consistent with what it saw during training.

Typical usage::

    snap_after = sim.apply_choices(...)
    state_after = patch_state_from_engine(state_before, snap_after)
    win_prob = winrate.predict_state(state_after)
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["patch_state_from_engine"]


_SPECIES_RE = re.compile(r"\[Species:([^\]]+)\]", re.IGNORECASE)


def _species_from_engine(mon: Dict[str, Any]) -> str:
    """Extract a lowercase species token from a Showdown serialized mon."""
    spec = mon.get("species")
    if isinstance(spec, str):
        m = _SPECIES_RE.match(spec)
        if m:
            return m.group(1).strip().lower()
        return spec.strip().lower()
    details = mon.get("details") or ""
    if isinstance(details, str):
        return details.split(",", 1)[0].strip().lower()
    return ""


def _norm(token: Any) -> str:
    if token is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(token).lower())


def _parse_condition(condition: Any) -> Tuple[float, Optional[str], bool]:
    """Parse Showdown 'cur/max status' string. Returns (hp_pct, status, fainted)."""
    if not isinstance(condition, str) or not condition.strip():
        return 1.0, None, False
    cleaned = condition.strip()
    if cleaned.endswith(" fnt"):
        return 0.0, None, True
    status: Optional[str] = None
    parts = cleaned.split(" ")
    main = parts[0]
    if len(parts) > 1:
        status = parts[1].strip() or None
    if "/" in main:
        try:
            cur_s, max_s = main.split("/", 1)
            cur = int(cur_s)
            mx = int(max_s)
            if mx <= 0:
                return 1.0, status, False
            return max(0.0, min(1.0, cur / mx)), status, cur == 0
        except ValueError:
            return 1.0, status, False
    return 1.0, status, False


# Showdown serialized state -> internal state-dict keys
_HAZARD_TAGS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
_SIDE_TAGS = {
    "reflect", "lightscreen", "auroraveil", "tailwind", "safeguard", "mist"
}


def _side_dicts_from_state_side(
    state_side: Dict[str, Any],
) -> Tuple[Dict[str, bool], Dict[str, int]]:
    side: Dict[str, bool] = {}
    hazards: Dict[str, int] = {}
    side_conds = state_side.get("sideConditions") or {}
    if isinstance(side_conds, dict):
        for raw_id, raw_entry in side_conds.items():
            tag = _norm(raw_id)
            if not tag:
                continue
            if tag in _HAZARD_TAGS:
                layers = 1
                if isinstance(raw_entry, dict):
                    layers = int(raw_entry.get("layers") or 1)
                hazards[tag] = layers
            elif tag in _SIDE_TAGS:
                side[tag] = True
    return side, hazards


def _weather_from_state(state: Dict[str, Any]) -> Optional[str]:
    field = state.get("field") or {}
    w = field.get("weather")
    if not w:
        return None
    return _norm(w) or None


def _terrain_from_state(state: Dict[str, Any]) -> Optional[str]:
    field = state.get("field") or {}
    t = field.get("terrain")
    if not t:
        return None
    tnorm = _norm(t)
    # Strip the trailing "terrain" so it matches the model vocab
    # (electricterrain -> electric).
    for suffix in ("terrain",):
        if tnorm.endswith(suffix):
            tnorm = tnorm[: -len(suffix)]
    return tnorm or None


def _patch_pokemon_entry(target: Dict[str, Any], engine_mon: Dict[str, Any]) -> None:
    hp_pct, status, fainted = _parse_condition(engine_mon.get("condition"))
    if engine_mon.get("hp") is not None and engine_mon.get("maxhp"):
        try:
            hp_pct = max(0.0, min(1.0, float(engine_mon["hp"]) / float(engine_mon["maxhp"])))
        except Exception:
            pass
    if engine_mon.get("fainted"):
        fainted = True
    if engine_mon.get("status"):
        status_id = str(engine_mon["status"]).lower()
        if status_id:
            status = status_id
    if status == "":
        status = None
    target["hp_percent"] = hp_pct
    target["fainted"] = bool(fainted)
    target["status"] = status
    # Update active flag if the engine moved someone else to the front.
    target["is_active"] = bool(engine_mon.get("isActive") or engine_mon.get("active"))
    boosts = engine_mon.get("boosts")
    if isinstance(boosts, dict):
        target["stats_boosts"] = {
            k: int(v)
            for k, v in boosts.items()
            if k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion") and v
        }
    # Volatiles
    volatiles_obj = engine_mon.get("volatiles") or {}
    if isinstance(volatiles_obj, dict) and volatiles_obj:
        keys = []
        for k in volatiles_obj.keys():
            n = _norm(k)
            if n and n not in ("aquaring", "magnetrise"):  # keep most volatiles
                keys.append(n)
        if keys:
            target["volatile_conditions"] = keys
    # Terastallization
    if engine_mon.get("terastallized"):
        target["tera_active"] = True


def _patch_side(
    team: List[Dict[str, Any]],
    state_side: Dict[str, Any],
) -> None:
    """Match each engine mon with the closest existing team entry by species."""
    engine_mons = state_side.get("pokemon") or []
    # Index engine mons by normalized species token
    eng_by_species: Dict[str, Dict[str, Any]] = {}
    for mon in engine_mons:
        sp = _norm(_species_from_engine(mon))
        if sp:
            eng_by_species.setdefault(sp, mon)
    for entry in team:
        sp = _norm(entry.get("species") or "")
        if not sp:
            continue
        engine_mon = eng_by_species.get(sp)
        if engine_mon is None:
            # Try base-species match (e.g. tauros-paldea-aqua -> tauros)
            for k, v in eng_by_species.items():
                if k.startswith(sp) or sp.startswith(k):
                    engine_mon = v
                    break
        if engine_mon is not None:
            _patch_pokemon_entry(entry, engine_mon)


def patch_state_from_engine(
    base_state: Dict[str, Any],
    snap: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of ``base_state`` whose HP / status / boosts / hazards /
    weather / terrain reflect the engine snapshot.

    ``base_state`` should be the state-dict the caller computed for the
    *current* turn (preview order, revealed flags, moves_seen, ...). We
    keep all of that and overlay only the dynamic fields that the engine
    is authoritative about.
    """
    state = copy.deepcopy(base_state)
    engine_state = snap.get("state") or {}
    sides = engine_state.get("sides") or []
    if len(sides) >= 1:
        _patch_side(state.get("my_team") or [], sides[0])
        side, hazards = _side_dicts_from_state_side(sides[0])
        state["my_side_conditions"] = side
        state["my_hazards"] = hazards
    if len(sides) >= 2:
        _patch_side(state.get("opp_team") or [], sides[1])
        side, hazards = _side_dicts_from_state_side(sides[1])
        state["opp_side_conditions"] = side
        state["opp_hazards"] = hazards
    weather = _weather_from_state(engine_state)
    if weather is not None or "weather" in state:
        state["weather"] = weather
    terrain = _terrain_from_state(engine_state)
    if terrain is not None or "terrain" in state:
        state["terrain"] = terrain
    state["turn"] = int(engine_state.get("turn") or state.get("turn", 0))
    # The engine doesn't track which move each side picked beyond its log.
    # Keep base_state's last_move fields (the caller can update if needed).
    return state
