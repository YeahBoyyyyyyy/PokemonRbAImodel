"""Random Battle winrate extraction from Pokéchamp (gen9randombattle)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ou.winrate_extractors.extract_winrate_from_pokechamp import run_pokechamp_winrate_extraction
from random_battle.config import FORMAT_ID, MIN_ELO_DEFAULT, WINRATE_CHUNKS_DIR


def main() -> None:
    run_pokechamp_winrate_extraction(
        format_id=FORMAT_ID,
        min_elo=MIN_ELO_DEFAULT,
        output_dir=WINRATE_CHUNKS_DIR,
        output_base_name="rb_winrate_data",
        title="RANDOM BATTLE WIN PREDICTION DATA EXTRACTION",
    )


if __name__ == "__main__":
    main()
