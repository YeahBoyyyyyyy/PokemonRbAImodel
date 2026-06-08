"""Launch RB action extraction (partial observability, gen9randombattle)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if __name__ == "__main__":
    from random_battle.data_extractors.extract_rb_action_chunks import main

    main()
