"""Patch a serialized Showdown state to match the live poke-env battle.

The Node bridge always starts a battle from a *fresh* state (full HP, no
status, no hazards, no boosts). To make the engine simulate from the actual
in-game situation we extract the live HP / status / boosts / hazards / field
from the poke-env ``Battle`` and overlay them onto the Showdown state dict
returned by :py:meth:`EngineSimulator.setup_from_battle`.

This module exposes two helpers:

* :func:`build_state_overlay` -- inspects the poke-env battle and produces a
  small dictionary describing per-side HP, status, boosts, side conditions
  and field conditions.
* :func:`apply_state_overlay` -- mutates a Showdown state dict (in-place) to
  reflect the overlay. The state dict is the same shape returned by the
  ``@pkmn/sim`` JSON serializer.

The implementation deliberately stays narrow: we only patch what we can
read with confidence from poke-env. Anything we cannot determine (volatile
substitute HP, exact remaining sleep turns for the opponent, ...) is left
at the engine's default value rather than guessed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["build_state_overlay", "apply_state_overlay"]


# ---------------------------------------------------------------------------
# Status / weather / terrain normalization
# ---------------------------------------------------------------------------


_STATUS_MAP = {
    "BRN": "brn",
    "BURN": "brn",
    "PAR": "par",
    "PARALYSIS": "par",
    "SLP": "slp",
    "SLEEP": "slp",
    "FRZ": "frz",
    "FREEZE": "frz",
    "PSN": "psn",
    "POISON": "psn",
    "TOX": "tox",
    "TOXIC": "tox",
    "FNT": "fnt",
    "FAINTED": "fnt",
}

_WEATHER_MAP = {
    "SUN": "sunnyday",
    "SUNNYDAY": "sunnyday",
    "HEAVYRAIN": "raindance",
    "RAIN": "raindance",
    "RAINDANCE": "raindance",
    "SAND": "sandstorm",
    "SANDSTORM": "sandstorm",
    "HAIL": "snow",
    "SNOW": "snow",
    "DESOLATELAND": "desolateland",
    "PRIMORDIALSEA": "primordialsea",
    "DELTASTREAM": "deltastream",
}

_TERRAIN_MAP = {
    "ELECTRIC": "electricterrain",
    "ELECTRICTERRAIN": "electricterrain",
    "PSYCHIC": "psychicterrain",
    "PSYCHICTERRAIN": "psychicterrain",
    "GRASSY": "grassyterrain",
    "GRASSYTERRAIN": "grassyterrain",
    "MISTY": "mistyterrain",
    "MISTYTERRAIN": "mistyterrain",
}


_SIDE_COND_MAP = {
    # poke-env SideCondition enum names -> Showdown id
    "STEALTH_ROCK": "stealthrock",
    "STEALTHROCK": "stealthrock",
    "SPIKES": "spikes",
    "TOXIC_SPIKES": "toxicspikes",
    "TOXICSPIKES": "toxicspikes",
    "STICKY_WEB": "stickyweb",
    "STICKYWEB": "stickyweb",
    "REFLECT": "reflect",
    "LIGHT_SCREEN": "lightscreen",
    "LIGHTSCREEN": "lightscreen",
    "AURORA_VEIL": "auroraveil",
    "AURORAVEIL": "auroraveil",
    "TAILWIND": "tailwind",
    "SAFEGUARD": "safeguard",
    "MIST": "mist",
}


def _enum_key(value: Any) -> str:
    """Coerce a value (Enum or string) to a normalized upper-case key.

    Strips underscores/spaces/dashes so callers don't need to mirror every
    spelling variant of poke-env's enums.
    """
    if hasattr(value, "name"):
        raw = str(value.name)
    else:
        raw = str(value)
    return raw.upper().replace("_", "").replace(" ", "").replace("-", "")


def _normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _STATUS_MAP.get(_enum_key(value))


def _normalize_weather(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _WEATHER_MAP.get(_enum_key(value))


def _normalize_terrain(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _TERRAIN_MAP.get(_enum_key(value))


def _normalize_side_condition(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _SIDE_COND_MAP.get(_enum_key(value))


def _boost_dict(mon: Any) -> Dict[str, int]:
    boosts = getattr(mon, "boosts", None) or {}
    result: Dict[str, int] = {}
    for key in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
        try:
            v = int(boosts.get(key, 0)) if isinstance(boosts, dict) else 0
        except Exception:
            v = 0
        if v:
            result[key] = max(-6, min(6, v))
    return result


def _species_token(mon: Any) -> str:
    species = getattr(mon, "species", None) or getattr(mon, "name", None) or ""
    return str(species).replace(" ", "").replace("-", "").replace("'", "").replace(".", "").lower()


# ---------------------------------------------------------------------------
# Overlay extraction
# ---------------------------------------------------------------------------


def _iter_side(battle: Any, *, opponent: bool) -> List[Any]:
    if opponent:
        team = getattr(battle, "opponent_team", None) or {}
    else:
        team = getattr(battle, "team", None) or {}
    if isinstance(team, dict):
        return list(team.values())
    return list(team or [])


def _side_overlay(battle: Any, *, opponent: bool) -> Dict[str, Any]:
    mons = _iter_side(battle, opponent=opponent)
    if opponent:
        active = getattr(battle, "opponent_active_pokemon", None)
        side_conds = getattr(battle, "opponent_side_conditions", None) or {}
    else:
        active = getattr(battle, "active_pokemon", None)
        side_conds = getattr(battle, "side_conditions", None) or {}

    # Per-mon overlay keyed by species (engine sets are also keyed by species).
    # We accept both the active pokemon and benched pokemon. For the active we
    # also capture stat boosts.
    per_mon: List[Dict[str, Any]] = []
    for mon in mons:
        species = _species_token(mon)
        if not species:
            continue
        entry: Dict[str, Any] = {"species": species}

        # HP fraction
        try:
            hp_frac = float(getattr(mon, "current_hp_fraction", 1.0) or 0.0)
        except Exception:
            hp_frac = 1.0
        if hp_frac < 0:
            hp_frac = 0.0
        if hp_frac > 1:
            hp_frac = 1.0
        entry["hp_fraction"] = hp_frac

        # Fainted
        fainted = bool(getattr(mon, "fainted", False)) or hp_frac <= 0.0
        if fainted:
            entry["fainted"] = True
            entry["hp_fraction"] = 0.0

        # Status
        st = _normalize_status(getattr(mon, "status", None))
        if st and st != "fnt":
            entry["status"] = st
        if fainted:
            entry["status"] = "fnt"

        # Boosts (only meaningful for the active mon)
        if mon is active or getattr(mon, "active", False):
            boosts = _boost_dict(mon)
            if boosts:
                entry["boosts"] = boosts
            entry["is_active"] = True

            # Volatile effects we can transcribe cheaply.
            effects = getattr(mon, "effects", None) or {}
            if effects:
                volatiles: List[str] = []
                for eff in effects:
                    name = getattr(eff, "name", None)
                    if not name:
                        continue
                    nm = str(name).upper()
                    if nm in ("LEECH_SEED", "LEECHSEED"):
                        volatiles.append("leechseed")
                    elif nm in ("SUBSTITUTE",):
                        volatiles.append("substitute")
                    elif nm in ("CONFUSION",):
                        volatiles.append("confusion")
                    elif nm in ("TAUNT",):
                        volatiles.append("taunt")
                    elif nm in ("ENCORE",):
                        volatiles.append("encore")
                    elif nm in ("YAWN",):
                        volatiles.append("yawn")
                if volatiles:
                    entry["volatiles"] = volatiles

            # Terastallized?
            if getattr(mon, "is_terastallized", False):
                tera_type = getattr(mon, "tera_type", None)
                if tera_type is not None:
                    name = getattr(tera_type, "name", str(tera_type))
                    entry["tera_active"] = True
                    entry["tera_type"] = str(name).capitalize()

        per_mon.append(entry)

    # Side conditions (hazards, screens, tailwind, ...)
    cond_overlay: Dict[str, int] = {}
    if isinstance(side_conds, dict):
        for cond, val in side_conds.items():
            tag = _normalize_side_condition(cond)
            if tag is None:
                continue
            try:
                layers = int(val) if val is not None else 1
            except Exception:
                layers = 1
            cond_overlay[tag] = max(1, layers)

    return {"pokemon": per_mon, "side_conditions": cond_overlay}


def build_state_overlay(battle: Any) -> Dict[str, Any]:
    """Extract a JSON-friendly overlay describing the live battle state."""
    overlay: Dict[str, Any] = {
        "p1": _side_overlay(battle, opponent=False),
        "p2": _side_overlay(battle, opponent=True),
        "field": {},
    }

    weather = getattr(battle, "weather", None)
    if isinstance(weather, dict):
        # poke-env exposes weather as {Weather.X: turn_started}; pick the first.
        for k in weather.keys():
            w = _normalize_weather(k)
            if w:
                overlay["field"]["weather"] = w
                break
    else:
        w = _normalize_weather(weather)
        if w:
            overlay["field"]["weather"] = w

    fields = getattr(battle, "fields", None)
    if isinstance(fields, dict):
        for k in fields.keys():
            t = _normalize_terrain(k)
            if t:
                overlay["field"]["terrain"] = t
                break

    return overlay


# ---------------------------------------------------------------------------
# Overlay application
# ---------------------------------------------------------------------------


def _find_mon_index(side: Dict[str, Any], species: str) -> Optional[int]:
    species_l = species.lower()
    pokemon = side.get("pokemon") or []
    for idx, mon in enumerate(pokemon):
        spec_ref = mon.get("species") or ""
        if isinstance(spec_ref, str):
            # Showdown stores "[Species:pikachu]" after serialization.
            ref_lower = spec_ref.lower()
            if ref_lower.endswith(f":{species_l}]") or ref_lower == species_l:
                return idx
        # Also try details which contain "Pikachu, L84, M"
        details = mon.get("details") or ""
        if isinstance(details, str) and details.split(",")[0].replace(" ", "").lower() == species_l:
            return idx
    return None


def _set_mon_hp(mon: Dict[str, Any], fraction: float) -> None:
    maxhp = int(mon.get("maxhp") or mon.get("baseMaxhp") or 100)
    new_hp = max(0, min(maxhp, int(round(fraction * maxhp))))
    mon["hp"] = new_hp


def _set_mon_status(mon: Dict[str, Any], status_id: Optional[str]) -> None:
    if status_id is None:
        return
    if status_id == "fnt":
        mon["status"] = ""
        mon["hp"] = 0
        mon["fainted"] = True
        mon["isActive"] = False
        st = mon.get("statusState") or {}
        st["id"] = ""
        mon["statusState"] = st
        return
    mon["status"] = status_id
    mon["fainted"] = False
    st = mon.get("statusState") or {}
    st["id"] = status_id
    mon["statusState"] = st


def _set_mon_boosts(mon: Dict[str, Any], boosts: Dict[str, int]) -> None:
    current = dict(mon.get("boosts") or {})
    for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
        if stat in boosts:
            current[stat] = int(boosts[stat])
        else:
            current.setdefault(stat, 0)
    mon["boosts"] = current


def _set_side_conditions(side: Dict[str, Any], conds: Dict[str, int], *, turn: int) -> None:
    if not conds:
        return
    side_conds = dict(side.get("sideConditions") or {})
    for tag, layers in conds.items():
        entry = side_conds.get(tag) or {"id": tag, "target": f"[Side:{side.get('id', 'p1')}]"}
        entry["id"] = tag
        if tag == "spikes" or tag == "toxicspikes":
            entry["layers"] = int(layers)
        if tag in ("reflect", "lightscreen", "auroraveil", "tailwind", "safeguard", "mist"):
            # 5-turn screens / 4-turn tailwind: we don't know the remaining
            # duration with certainty, default to 'max' (full duration).
            entry["duration"] = 5 if tag != "tailwind" else 4
        side_conds[tag] = entry
    side["sideConditions"] = side_conds


def _set_field(state: Dict[str, Any], field_overlay: Dict[str, Any]) -> None:
    field = state.get("field") or {}
    weather = field_overlay.get("weather")
    if weather:
        field["weather"] = weather
        ws = field.get("weatherState") or {}
        ws["id"] = weather
        field["weatherState"] = ws
    terrain = field_overlay.get("terrain")
    if terrain:
        field["terrain"] = terrain
        ts = field.get("terrainState") or {}
        ts["id"] = terrain
        field["terrainState"] = ts
    state["field"] = field


def apply_state_overlay(state: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate ``state`` to reflect ``overlay``. Returns the mutated state."""
    sides = state.get("sides") or []
    if not sides:
        return state

    side_keys = ("p1", "p2")
    for idx, key in enumerate(side_keys):
        if idx >= len(sides):
            break
        side_overlay = overlay.get(key) or {}
        side = sides[idx]
        pokemon_overlays: Sequence[Dict[str, Any]] = side_overlay.get("pokemon") or []
        for mon_overlay in pokemon_overlays:
            species = mon_overlay.get("species")
            if not species:
                continue
            mon_idx = _find_mon_index(side, species)
            if mon_idx is None:
                continue
            mon = side["pokemon"][mon_idx]

            if "hp_fraction" in mon_overlay:
                _set_mon_hp(mon, float(mon_overlay["hp_fraction"]))

            _set_mon_status(mon, mon_overlay.get("status"))

            if mon_overlay.get("boosts"):
                _set_mon_boosts(mon, mon_overlay["boosts"])

            if mon_overlay.get("tera_active"):
                mon["terastallized"] = mon_overlay.get("tera_type") or mon.get("teraType") or "Electric"

        _set_side_conditions(
            side,
            side_overlay.get("side_conditions") or {},
            turn=int(state.get("turn", 1) or 1),
        )

        # Update pokemonLeft / totalFainted to stay consistent with the
        # overlay so the engine doesn't think a fainted mon is still around.
        alive = sum(1 for m in side["pokemon"] if not m.get("fainted") and (m.get("hp") or 0) > 0)
        fainted = len(side["pokemon"]) - alive
        side["pokemonLeft"] = alive
        side["totalFainted"] = fainted

    _set_field(state, overlay.get("field") or {})
    return state
