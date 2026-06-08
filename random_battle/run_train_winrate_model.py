"""
Entraîne le modèle win-rate (P(victoire | état)) sur les chunks rb_action_data_*.json.

Prérequis : move_vocab.json (entraînement multi-head ou build_move_vocab).

Exemple :
  python random_battle/run_train_winrate_model.py
  python random_battle/run_train_winrate_model.py
  python random_battle/run_train_winrate_model.py --max_examples 0 --steps_per_epoch 0 --epochs 12  # tout le corpus (long)
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from random_battle.models.IA_winrate_predictor import main

if __name__ == "__main__":
    main()
