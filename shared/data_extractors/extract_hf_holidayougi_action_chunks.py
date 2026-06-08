"""
Convenience wrapper to extract action chunks from the HolidayOugi HF dataset.

This forwards all arguments to extract_action_chunks.py while forcing --source hf.
"""

from __future__ import annotations

import sys

from extract_action_chunks import main as extract_main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--source", "hf"] + sys.argv[1:]
    extract_main()
