"""
Shared type chart + helpers.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from pokedex_9G_complete import pokemon_data_gen9

# 18x18 matrix (Gen 6+ type chart) in this fixed order:
TYPE_NAMES = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]

TYPE_INDEX = {t: i for i, t in enumerate(TYPE_NAMES)}

_TYPE_CHART_DICT = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "dragon": 0.5,
        "steel": 2.0,
    },
    "water": {
        "fire": 2.0,
        "water": 0.5,
        "grass": 0.5,
        "ground": 2.0,
        "rock": 2.0,
        "dragon": 0.5,
    },
    "electric": {
        "water": 2.0,
        "electric": 0.5,
        "grass": 0.5,
        "ground": 0.0,
        "flying": 2.0,
        "dragon": 0.5,
    },
    "grass": {
        "fire": 0.5,
        "water": 2.0,
        "grass": 0.5,
        "poison": 0.5,
        "ground": 2.0,
        "flying": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "dragon": 0.5,
        "steel": 0.5,
    },
    "ice": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ground": 2.0,
        "flying": 2.0,
        "dragon": 2.0,
        "ice": 0.5,
        "steel": 0.5,
    },
    "fighting": {
        "normal": 2.0,
        "ice": 2.0,
        "rock": 2.0,
        "dark": 2.0,
        "steel": 2.0,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 0.5,
        "bug": 0.5,
        "ghost": 0.0,
        "fairy": 0.5,
    },
    "poison": {
        "grass": 2.0,
        "poison": 0.5,
        "ground": 0.5,
        "rock": 0.5,
        "ghost": 0.5,
        "steel": 0.0,
        "fairy": 2.0,
    },
    "ground": {
        "fire": 2.0,
        "electric": 2.0,
        "grass": 0.5,
        "poison": 2.0,
        "flying": 0.0,
        "bug": 0.5,
        "rock": 2.0,
        "steel": 2.0,
    },
    "flying": {
        "electric": 0.5,
        "grass": 2.0,
        "fighting": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "steel": 0.5,
    },
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "steel": 0.5, "dark": 0.0},
    "bug": {
        "fire": 0.5,
        "grass": 2.0,
        "fighting": 0.5,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 2.0,
        "ghost": 0.5,
        "dark": 2.0,
        "steel": 0.5,
        "fairy": 0.5,
    },
    "rock": {
        "fire": 2.0,
        "ice": 2.0,
        "fighting": 0.5,
        "ground": 0.5,
        "flying": 2.0,
        "bug": 2.0,
        "steel": 0.5,
    },
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {
        "fire": 0.5,
        "water": 0.5,
        "electric": 0.5,
        "ice": 2.0,
        "rock": 2.0,
        "fairy": 2.0,
        "steel": 0.5,
    },
    "fairy": {
        "fire": 0.5,
        "fighting": 2.0,
        "poison": 0.5,
        "dragon": 2.0,
        "dark": 2.0,
        "steel": 0.5,
    },
}

# Matrix (attacking type row, defending type column) in TYPE_NAMES order.
TYPE_CHART = [
    [_TYPE_CHART_DICT.get(atk, {}).get(def_, 1.0) for def_ in TYPE_NAMES] for atk in TYPE_NAMES
]


def _norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower().strip())


_SPECIES_ALIASES = {
    "sinistchamasterpiece": "sinistcha",
    "sinistchaunremarkable": "sinistcha",
    "polteageistantique": "polteageist",
    "polteageistphony": "polteageist",
    "gastrodoneast": "gastrodon",
    "gastrodonwest": "gastrodon",
    "pikachuoriginal": "pikachu",
    "dudunsparcethreesegment": "dudunsparce",
    "dudunsparcetwosegment": "dudunsparce",
    "alcremiematchacream": "alcremie",
    "mimikyubusted": "mimikyu",
    "zarudedada": "zarude",
    "toxtricitylowkey": "toxtricity",
    "mausholdfour": "maushold",
    "mausholdthree": "maushold",
    "mausholdfamilyoffour": "maushold",
    "mausholdfamilyofthree": "maushold",
    "miniorblue": "minior",
    "miniorgreen": "minior",
    "miniorindigo": "minior",
    "miniororange": "minior",
    "minioryellow": "minior",
    "miniorviolet": "minior",
    "miniorred": "minior",
    "miniorcore": "minior",
    "ogerpontealtera": "ogerpon",
    "ogerponwellspringtera": "ogerponwellspring",
    "ogerponhearthflametera": "ogerponhearthflame",
    "ogerponcornerstonetera": "ogerponcornerstone",
    "vivillongarden": "vivillon",
    "sneaselhisui": "sneasel",
}

_NORM_TO_KEY = {_norm_token(k): k for k in pokemon_data_gen9.keys()}


def _normalize_type(value: object) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "name"):
        value = value.name
    elif hasattr(value, "type"):
        value = value.type
    return str(value).lower().strip()


def type_effectiveness(attacking_type: object, defending_types: Iterable[object]) -> float:
    atk = _normalize_type(attacking_type)
    if not atk or atk not in TYPE_INDEX:
        return 1.0
    if not defending_types:
        return 1.0
    atk_idx = TYPE_INDEX[atk]
    mult = 1.0
    for t in defending_types:
        dt = _normalize_type(t)
        if not dt or dt not in TYPE_INDEX:
            continue
        def_idx = TYPE_INDEX[dt]
        mult *= TYPE_CHART[atk_idx][def_idx]
    return mult


def get_name(pokemon: object) -> str:
    if pokemon is None:
        return ""
    raw = getattr(pokemon, "species", None) or getattr(pokemon, "name", None) or getattr(
        pokemon, "_name", None
    )
    if raw is None:
        raw = str(pokemon)
    raw = str(raw).split(",", 1)[0].strip()
    norm = _norm_token(raw)
    if norm.endswith("tera"):
        norm = norm[:-4]
    norm = _SPECIES_ALIASES.get(norm, norm)
    return _NORM_TO_KEY.get(norm, raw)
