# Sim Bridge

Pont Node.js entre le code Python (poke-env / IA) et le simulateur officiel
Pokemon Showdown (`@pkmn/sim`). Permet d'exécuter de vrais tours Gen 9 depuis
Python : on construit deux équipes, on envoie des choix `(p1, p2)`, et on
récupère l'état suivant — exactement comme un combat réel sur Showdown.

## Architecture

```
Python (RbSearchPlayer)
  └── common/pkmn_engine_simulator.py        (API haut niveau)
      ├── common/showdown_set_builder.py     (poke-env Battle -> Showdown sets)
      └── common/pkmn_bridge.py              (client subprocess JSON-line)
              │
              ▼
        Node.js (this directory)
          └── bridge.js                      (long-running process)
              └── @pkmn/sim                  (le simulateur)
```

Un seul processus Node est lancé par `EngineSimulator`. Chaque appel à
`bridge.step()` désérialise l'état sérialisé, applique les deux choix,
puis re-sérialise. Stateless côté Node — ce qui permet à Python de forker
un état autant de fois que voulu (search tree, multiple croyances).

## Protocole

Le bridge lit des lignes JSON sur stdin et écrit des lignes JSON sur stdout.

### Commandes

```json
{"cmd": "ping"}
{"cmd": "init", "format": "gen9customgame",
 "p1": {"name": "Bot", "team": [<sets>]},
 "p2": {"name": "Opp", "team": [<sets>]},
 "seed": [1, 2, 3, 4]}
{"cmd": "step", "state": <serialized>, "p1_choice": "move 1",
 "p2_choice": "move 1", "seed": [..], "include_log": false}
{"cmd": "requests", "state": <serialized>}
{"cmd": "quit"}
```

### Réponses

```json
{
  "ok": true,
  "state": { /* Showdown's serialized Battle */ },
  "requests": {
    "p1": { /* compact: active | forceSwitch | teamPreview | wait */ },
    "p2": { /* idem */ }
  },
  "ended": false,
  "winner": null,
  "turn": 2,
  "request_state": "move",
  "log_tail": ["|move|p1a:...", ...]   // when include_log=true
}
```

## Performances

Sur Windows + Node 22 + venv311 Python 3.11 :
- **~285 turn-steps/seconde** (Pikachu vs Blissey 1v1, full state round-trip)
- **~3.5 ms** par step (sérialisation JSON + désérialisation + simulation)
- **~0.23 ms** RTT pour un ping (overhead protocole)

Suffisant pour du 1-2 ply en temps réel et du sampling de 10-30 mondes par
décision.

## Installation

```bash
cd PokemonRbAImodel/sim_bridge
npm install
```

Aucun build natif requis. `@pkmn/sim` est du JavaScript pur (~10 Mo une fois
installé).

## Test rapide

```bash
# Smoke test côté Node uniquement
node smoke_test.js
```

## Limitations connues

- **Pivot moves (Volt Switch / U-turn / Flip Turn / Parting Shot)**
  déclenchent un `forceSwitch` post-tour qui n'est pas géré automatiquement
  par `EngineTurnEvaluator.evaluate_my_move`. Le score d'un pivot move
  est donc surévalué. À corriger en envoyant un second `step` avec une
  réponse de switch quand la requête suivante est `forceSwitch`.
- **Format** : `gen9customgame` (pas `gen9randombattle`). Les règles de
  niveau / EVs / IVs sont encodées dans les sets transmis, pas tirées au
  hasard par le moteur.
- **Sets adverses** : la qualité dépend du `rb_set_dex.json`. Un set mal
  inféré ne sera pas corrigé par l'engine — la simulation sera réaliste
  pour ce set précis, mais ce set peut être faux. Voir le TODO "sampling
  de croyances".
