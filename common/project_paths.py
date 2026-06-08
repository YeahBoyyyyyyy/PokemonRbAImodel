"""Central import paths for the Pokemon Random Battle AI project."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMON_DIR = PROJECT_ROOT / "common"
SHARED_DATA_EXTRACTORS = PROJECT_ROOT / "shared" / "data_extractors"
SHARED_WINRATE_EXTRACTORS = PROJECT_ROOT / "shared" / "winrate_extractors"
RB_DIR = PROJECT_ROOT / "random_battle"


def setup_import_paths(
    *,
    shared_data: bool = False,
    shared_winrate: bool = False,
    rb_data: bool = False,
) -> None:
    paths = [PROJECT_ROOT, COMMON_DIR]
    if shared_data:
        paths.append(SHARED_DATA_EXTRACTORS)
    if shared_winrate:
        paths.append(SHARED_WINRATE_EXTRACTORS)
    if rb_data:
        paths.append(RB_DIR / "data_extractors")
    for path in paths:
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
