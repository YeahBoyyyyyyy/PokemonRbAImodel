# Branche Random Battle (gen9randombattle)

- **Format** : `gen9randombattle` (`random_battle/config.py`)
- **set_dex RB** (`data/rb_set_dex.json`) : probas empiriques depuis [pkmn/randbats](https://github.com/pkmn/randbats) (~100k simulations)
- **Extraction dédiée** : `data_extractors/extract_rb_action_chunks.py`
  - Visibilité partielle : à chaque tour, seules les infos réellement révélées (espèce au team preview, HP/moves/item après reveal)
  - État capturé **avant** chaque coup (pas en fin de tour)
- **Sources HF** :
  - `milkkarten/pokechamp` (`--source pokechamp`)
  - `HolidayOugi/pokemon-showdown-replays` (`--source holidayougi`, défaut)
- **Replays locaux** : `--source local --input_dir replays_data --prefer_inputlog`
- **Modèles** : à entraîner dans `models/`
- **Artifacts** : `artifacts/`

## Importer / mettre à jour le set_dex RB

```bash
python random_battle/data_extractors/import_randbats_dex.py
# ou fichier local :
python random_battle/data_extractors/import_randbats_dex.py --input_file chemin/gen9randombattle.json
```

Produit `random_battle/data/rb_set_dex.json` (utilisé par `IA_multihead_predictor.py` via `--set_dex`).

## Extraction rapide

```bash
# HolidayOugi (replays Showdown, log + inputlog)
python random_battle/data_extractors/extract_rb_action_chunks_holidayougi.py --prefer_inputlog

# Pokéchamp (champ text, filtre gamemode)
python random_battle/run_extract_action_chunks.py --max_replays 5000
# ou directement :
python random_battle/data_extractors/extract_rb_action_chunks.py --max_replays 5000
```

Sortie par défaut : `random_battle/chunks/action/rb_action_data_00001.json`

## Entraînement multi-tête RB

```bash
python random_battle/models/IA_multihead_predictor.py \
  --input_dir random_battle/chunks/action \
  --base_name rb_action_data \
  --filter_voluntary \
  --epochs 10
```

Différences vs OU (`ou/models/IA_multihead_predictor.py`) :
- `set_dex` = randbats (`rb_set_dex.json`) + `move_prior` + `opp_move_prior`
- HP inconnu = `-1` + feature `hp_known`
- Feature `revealed` pour l'adversaire
- Derniers coups (`my_last_move` / `opp_last_move`) pris depuis l'état

Modèle sauvegardé : `random_battle/artifacts/multihead_model/model.keras`

## Modèle win-rate (P(victoire | état))

Label : `winner` dans les chunks d'actions (le joueur `state["player"]` a gagné le combat).
Même encodeur que le multi-tête ; sortie `win_prob` (sigmoïde).

```bash
# Prérequis : artifacts/move_vocab.json
python random_battle/run_train_winrate_model.py \
  --input_dir random_battle/chunks/action \
  --filter_voluntary
# Defaut: ~400k ex., 1500 steps/epoch, 6 epochs (~25 min selon machine)
# Corpus complet: --max_examples 0 --steps_per_epoch 0 --epochs 12

```

API Python :
- `win_rate_evaluator.py` — `WinRateEvaluator`, `predict_after_one_ply_turn`
- `turn_simulation.py` — `opponent_moves_for_state`, `state_after_opponent_move`
- `common/pkmn_engine_simulator.py` — `EngineSimulator` (haut niveau, poke-env -> Showdown)
- `common/engine_turn_eval.py` — `EngineTurnEvaluator` (1-ply real-engine)
- `common/showdown_set_builder.py` — `build_battle_teams` (poke-env -> sets Showdown)
- `common/pkmn_bridge.py` — `PkmnBridge` (sous-processus Node + JSON-line protocol)

Avec `--search`, le bot départage les coups via **min** des P(victoire) après un tour simulé :
- **Branches adverses** : coups plausibles (set-dex + `moves_seen`) + switchs banc révélé (`opp_switch_top_k`).
- **Switchs** : tiebreak win-rate seulement si matchup défensif critique (×2/×4 subis) et remplaçant résistant + bulk Def/SpD face aux coups vus ; pénalité modèle + marge +6 % vs attaque.
- **Simulation de tour** : par défaut, transitions structurelles uniquement (last_move + moves_seen, sans dégâts). Pour une simulation **exacte** avec talents/objets/météo/statuts/pivots, ajouter `--use_engine` qui branche le simulateur Showdown réel (`@pkmn/sim` via `sim_bridge/`, voir `sim_bridge/README.md`).

```powershell
# Tiebreak win-rate avec le vrai moteur Showdown
python random_battle/run_rb_model_player.py --mode battle --vs low --search --use_engine
```

Les heuristiques rapides du hybrid (best_damaging_move, Tera, switch défensif) utilisent un scoring **type-based** (`base_power × type_effectiveness × STAB`) — pas de formule de dégâts détaillée, qui était trop approximative.

## Jouer en local (serveur + bot)

**Terminal 1 — serveur Showdown** (depuis la racine `IA Pokemon`) :

```powershell
cd PokemonRbAImodel/random_battle
.\start_showdown_for_bots.ps1
# Alternative manuelle :
# cd "C:\Users\natha\MonBureau\IA Pokemon\pokemon-showdown"
# node pokemon-showdown start --no-security
```

**Terminal 2 — bot** (depuis `PokemonRbAImodel`) :

```powershell
cd PokemonRbAImodel

# Hybride classique (multi-tête + heuristiques)
python random_battle/run_rb_model_player.py --mode accept --username RbModelBot --opponent Natanyelle

# Hybride + départage win-rate entre les meilleurs coups (modèle win-rate requis)
python random_battle/run_rb_model_player.py --mode accept --username RbModelBot --opponent Natanyelle --search
```

Sur le client Showdown local (`http://localhost:8000`), défie **RbModelBot** en **gen9randombattle**.

## Benchmark / analyse post-mortem

```powershell
# Bench win-rate (ex. 200 combats vs LowHeuristicAI)
python random_battle/run_rb_model_player.py --mode battle --vs low --n_battles 200 --search --use_engine --engine_n_worlds 5

# Analyser un log de décisions JSONL
python random_battle/analyze_decision_log.py artifacts/debug/decisions_*.jsonl
```

Heuristique avancée orientée RB : `common/players/heuristics_pokemon_ai.py` (`HighHeuristicAI`).
