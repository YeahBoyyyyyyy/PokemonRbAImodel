# PokemonRbAImodel

AI for Pokémon **Random Battle** (gen9randombattle): multi-head policy, win-rate model, and optional Showdown engine search (`@pkmn/sim`).

```
PokemonRbAImodel/
├── common/                 # Shared engine bridge, heuristics, set builder
├── shared/                 # Replay extraction pipelines (HF / local)
│   ├── data_extractors/
│   └── winrate_extractors/
├── random_battle/          # RB training, players, artifacts
│   ├── config.py
│   ├── models/
│   ├── data_extractors/
│   ├── players/
│   ├── data/               # rb_set_dex.json
│   ├── chunks/             # training data (gitignored, generate locally)
│   └── artifacts/          # trained models (gitignored by default)
└── sim_bridge/             # Node.js @pkmn/sim bridge
```

## Quick start

**Prerequisites:** Python 3.11+, Node.js (for `sim_bridge`), local Showdown server for battles.

```powershell
cd PokemonRbAImodel
pip install -r random_battle/requirements.txt   # if present
cd sim_bridge; npm install; cd ..

# Terminal 1 — Showdown server (from repo root or pokemon-showdown clone)
cd random_battle
.\start_showdown_for_bots.ps1

# Terminal 2 — play
python random_battle/run_rb_model_player.py --mode battle --vs low --search --use_engine
```

See [random_battle/README.md](random_battle/README.md) for extraction, training, and engine options.

## Data extraction

```bash
python random_battle/data_extractors/extract_rb_action_chunks_holidayougi.py --prefer_inputlog
python random_battle/run_extract_action_chunks.py --source pokechamp --max_replays 5000
python shared/winrate_extractors/extract_hf_showdown_winrate_data.py --formatid gen9randombattle --output_dir random_battle/chunks/winrate
```

## Training

```bash
python random_battle/models/IA_multihead_predictor.py \
  --input_dir random_battle/chunks/action \
  --base_name rb_action_data --filter_voluntary --epochs 10

python random_battle/run_train_winrate_model.py \
  --input_dir random_battle/chunks/action --filter_voluntary
```

## License

Private repository — do not redistribute trained weights or scraped replay data without checking dataset licenses.
