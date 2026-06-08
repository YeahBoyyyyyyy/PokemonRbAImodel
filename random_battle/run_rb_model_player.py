"""
Run the Random Battle multi-head model on a local Showdown server via poke-env.

Examples:
  python random_battle/run_rb_model_player.py --mode accept --opponent Natanyelle
  python random_battle/run_rb_model_player.py --mode battle --n_battles 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from poke_env import AccountConfiguration

from common.players.heuristics_pokemon_ai import HighHeuristicAI, LowHeuristicAI, RandomAI
from common.project_paths import setup_import_paths
from random_battle.config import (
    FORMAT_ID,
    LOCAL_SERVER_CONFIGURATION,
    MOVE_VOCAB_PATH,
    MULTIHEAD_MODEL_DIR,
    RB_SET_DEX_PATH,
    WINRATE_MODEL_DIR,
)
from random_battle.players.local_login_patch import apply_local_login_patch
from random_battle.players.rb_hybrid_player import RbHybridPlayer
from random_battle.players.rb_model_player import RbModelPlayer
from random_battle.players.rb_search_player import RbSearchPlayer

setup_import_paths()
apply_local_login_patch()

OPPONENTS = {
    "random": RandomAI,
    "low": LowHeuristicAI,
    "high": HighHeuristicAI,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play gen9randombattle with RbModelPlayer.")
    parser.add_argument("--mode", choices=("accept", "battle"), default="battle")
    parser.add_argument(
        "--username",
        default="",
        help="Nom Showdown (vide = nom unique auto, recommandé en local).",
    )
    parser.add_argument("--password", default=None)
    parser.add_argument("--opponent", default="Natanyelle", help="Human name for accept mode.")
    parser.add_argument("--vs", choices=tuple(OPPONENTS.keys()), default="low")
    parser.add_argument("--n_battles", type=int, default=5)
    parser.add_argument(
        "--n_challenges",
        type=int,
        default=999,
        help="Nombre de defis acceptes avant arret (defaut: quasi illimite).",
    )
    parser.add_argument("--model", default=str(MULTIHEAD_MODEL_DIR / "model.keras"))
    parser.add_argument("--vocab", default=str(MOVE_VOCAB_PATH))
    parser.add_argument("--set_dex", default=str(RB_SET_DEX_PATH))
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help=">1 = plus aléatoire, <1 = plus greedy (défaut 0.9).",
    )
    parser.add_argument("--top_k_moves", type=int, default=5)
    parser.add_argument(
        "--max_setup_boosts",
        type=int,
        default=0,
        help="Bloque setup si boosts positifs > N (0 = dès +1). -1 = ignore les boosts.",
    )
    parser.add_argument(
        "--max_setups_per_battle",
        type=int,
        default=0,
        help="Nombre max de setups par combat (0 = aucun setup).",
    )
    parser.add_argument(
        "--max_setups_per_active",
        type=int,
        default=0,
        help="Nombre max de setups par Pokémon actif (0 = aucun sur ce slot).",
    )
    parser.add_argument(
        "--prefer_attacks_over_setup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retire les setups des coups légaux s'il existe une attaque offensives.",
    )
    parser.add_argument(
        "--ban_setup_when_attack_available",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interdit setup s'il existe une attaque au moins neutre (efficacité >= 0.5).",
    )
    parser.add_argument(
        "--ban_setup_when_risky",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interdit setup si rester est risqué (types/HP).",
    )
    parser.add_argument(
        "--setup_risk_taken",
        type=float,
        default=1.25,
        help="Seuil multiplicateur type subi pour considérer la position risquée.",
    )
    parser.add_argument("--setup_repeat_penalty", type=float, default=0.05)
    parser.add_argument(
        "--setup_move_score_multiplier",
        type=float,
        default=0.08,
        help="Multiplicateur des probas modèle pour les coups setup (si encore légaux).",
    )
    parser.add_argument("--max_concurrent_battles", type=int, default=1)
    parser.add_argument("--format", default=FORMAT_ID, dest="battle_format")
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "RbHybridPlayer (défaut) ou --no-hybrid pour le modèle seul. "
            "Avec --search : désactive les garde-fous (switch forcé, recovery, "
            "fallback attaque heuristique)."
        ),
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="RbSearchPlayer : win-rate pour départager les meilleurs coups du modèle.",
    )
    parser.add_argument(
        "--winrate_model",
        default=str(WINRATE_MODEL_DIR / "model.keras"),
        help="Chemin model.keras win-rate (avec --search).",
    )
    parser.add_argument(
        "--use_engine",
        action="store_true",
        help=(
            "Avec --search : remplace l'éval 1-ply approximative par le moteur "
            "Showdown (@pkmn/sim via sim_bridge/). Plus précis mais nécessite "
            "Node.js + npm install dans sim_bridge/."
        ),
    )
    parser.add_argument(
        "--engine_aggregation",
        choices=("min", "mean", "weighted_mean", "max"),
        default="weighted_mean",
        help="Agrégation des branches adverses avec --use_engine.",
    )
    parser.add_argument(
        "--engine_n_worlds",
        type=int,
        default=1,
        help=(
            "Nombre d'hypothèses adverses simulées (set/role/item/tera variants). "
            "Coût linéaire: n_worlds * branches sims par décision. 5 est un bon "
            "compromis."
        ),
    )
    parser.add_argument(
        "--engine_world_aggregation",
        choices=("min", "mean", "max"),
        default="mean",
        help=(
            "Agrégation des scores entre worlds pour chaque branche adverse "
            "(mean = robuste, min = pessimiste)."
        ),
    )
    parser.add_argument(
        "--decision_log",
        type=str,
        default=None,
        help=(
            "Chemin vers un fichier JSONL où chaque décision (move/switch) "
            "sera loguée avec les scores des candidats (post-mortem)."
        ),
    )
    parser.add_argument(
        "--engine_use_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Avec --use_engine : score post-tour via le modèle win-rate. "
            "Défaut : heuristique position (HP, hazards, setup)."
        ),
    )
    parser.add_argument(
        "--offensive_switch_margin",
        type=float,
        default=0.10,
        help=(
            "Avec --search : marge de win-prob qu'un switch offensif (actif "
            "pas en danger) doit dépasser pour être joué. Plus bas = switch "
            "plus volontiers (risque de switch-spam)."
        ),
    )
    parser.add_argument(
        "--engine_depth",
        type=int,
        default=1,
        help=(
            "Profondeur de recherche engine (avec --use_engine). 1 = 1-ply. "
            "2/3 = expectimax sur les tours suivants (mon coup -> réponse adv "
            "-> mon meilleur coup -> ...). Valorise les séquences A->B au lieu "
            "de spammer un coup. ATTENTION: 3-ply est lourd — utiliser "
            "--engine_n_worlds 1 ou 2 et un petit --n_battles."
        ),
    )
    parser.add_argument(
        "--engine_depth2_my_top_k",
        type=int,
        default=3,
        help="Profond: nb de mes coups de suivi explorés (après pré-tri rapide).",
    )
    parser.add_argument(
        "--engine_deep_opp_move_cap",
        type=int,
        default=4,
        help="Profond: cap sur les branches d'attaque adverses (un mon a <=4 coups).",
    )
    parser.add_argument(
        "--engine_opp_switch_base",
        type=int,
        default=1,
        help="Profond: nb de switchs adverses testés au minimum.",
    )
    parser.add_argument(
        "--engine_opp_switch_max",
        type=int,
        default=3,
        help=(
            "Profond: nb max de switchs adverses testés quand l'adversaire est "
            "désavantagé (il pivote plus). S'il a l'avantage: 0 switch testé."
        ),
    )
    parser.add_argument(
        "--engine_prune_delta",
        type=float,
        default=0.15,
        help=(
            "Élagage: ignore un coup de suivi dont le probe est > delta sous le "
            "meilleur. Plus petit = plus agressif (plus rapide)."
        ),
    )
    parser.add_argument(
        "--engine_win_cutoff",
        type=float,
        default=0.92,
        help=(
            "Élagage: positions au-dessus de ce seuil (ou sous 1-seuil) ne sont "
            "pas approfondies; arrête la recherche d'un nœud dès qu'un coup "
            "l'atteint."
        ),
    )
    parser.add_argument(
        "--engine_verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Logs de progression de la recherche engine (heartbeat: tour, "
            "candidat évalué, nb de sims, temps écoulé). Utile pour diagnostiquer "
            "une recherche lente/bloquée en 2-3 ply."
        ),
    )
    parser.add_argument(
        "--tiebreak_top_k",
        type=int,
        default=5,
        help="Max coups évalués par win-rate par tour (avec --search).",
    )
    parser.add_argument(
        "--tiebreak_move_min_k",
        type=int,
        default=2,
        help=(
            "Minimum de coups évalués même si le modèle favorise fortement "
            "un seul (utile pour alterner deux attaques en 1-ply)."
        ),
    )
    parser.add_argument(
        "--tiebreak_move_score_ratio",
        type=float,
        default=0.12,
        help=(
            "Seuil relatif au meilleur model_score pour inclure un coup "
            "dans la shortlist (0.12 = 12%% du meilleur)."
        ),
    )
    parser.add_argument("--emergency_hp", type=float, default=0.28)
    parser.add_argument(
        "--switch_critical_adv",
        type=float,
        default=4.0,
        help="Switch si produit des mult. subis >= seuil (4 = double faiblesse type).",
    )
    parser.add_argument(
        "--switch_enemy_adv",
        type=float,
        default=2.0,
        help="Mult. max sur un type attaquant (2 = au moins un x2 subi).",
    )
    parser.add_argument(
        "--switch_chance",
        type=float,
        default=0.12,
        help="Proba de switch sur cas limite (produit < seuil critique).",
    )
    parser.add_argument(
        "--switch_bad_matchup_ratio",
        type=float,
        default=4.0,
        help="Seuil produit pour autoriser le tirage switch_chance.",
    )
    parser.add_argument(
        "--switch_min_improvement",
        type=float,
        default=0.5,
        help="Le remplaçant doit avoir un produit def <= 50%% du actif.",
    )
    parser.add_argument(
        "--switch_hp_max_hit",
        type=float,
        default=0.5,
        help="HP max pour switch sur simple x2 subi (avec switch_enemy_adv).",
    )
    parser.add_argument(
        "--action_threshold",
        type=float,
        default=0.52,
        help="Au-dessus: le modèle attaque plutôt que switch (hybride défaut 0.52).",
    )
    return parser


async def main_async(args: argparse.Namespace) -> None:
    if args.username.strip():
        account = AccountConfiguration(args.username.strip(), args.password)
        bot_name = account.username
    else:
        account = AccountConfiguration.generate("RbBot", rand=True)
        bot_name = account.username

    server_configuration = LOCAL_SERVER_CONFIGURATION
    player_kwargs = dict(
        account_configuration=account,
        server_configuration=server_configuration,
        max_concurrent_battles=args.max_concurrent_battles,
        model_path=Path(args.model),
        vocab_path=Path(args.vocab),
        set_dex_path=Path(args.set_dex),
        action_prob_threshold=args.action_threshold,
        temperature=args.temperature,
        top_k_moves=args.top_k_moves,
        setup_repeat_penalty=args.setup_repeat_penalty,
        setup_move_score_multiplier=args.setup_move_score_multiplier,
        max_setup_boosts=args.max_setup_boosts,
        max_setups_per_battle=args.max_setups_per_battle,
        max_setups_per_active=args.max_setups_per_active,
        prefer_attacks_over_setup=args.prefer_attacks_over_setup,
        ban_setup_when_attack_available=args.ban_setup_when_attack_available,
        ban_setup_when_risky=args.ban_setup_when_risky,
        setup_risk_taken=args.setup_risk_taken,
        battle_format=args.battle_format,
    )
    hybrid_kwargs = dict(
        emergency_hp=args.emergency_hp,
        switch_critical_adv=args.switch_critical_adv,
        switch_enemy_adv=args.switch_enemy_adv,
        switch_chance=args.switch_chance,
        switch_bad_matchup_ratio=args.switch_bad_matchup_ratio,
        switch_min_improvement=args.switch_min_improvement,
        switch_hp_max_hit=args.switch_hp_max_hit,
        use_heuristic_overrides=args.hybrid,
    )
    if args.search:
        player = RbSearchPlayer(
            winrate_model_path=Path(args.winrate_model),
            use_engine=args.use_engine,
            engine_aggregation=args.engine_aggregation,
            engine_n_worlds=args.engine_n_worlds,
            engine_world_aggregation=args.engine_world_aggregation,
            engine_use_model=args.engine_use_model,
            engine_depth=args.engine_depth,
            engine_depth2_my_top_k=args.engine_depth2_my_top_k,
            engine_deep_opp_move_cap=args.engine_deep_opp_move_cap,
            engine_opp_switch_base=args.engine_opp_switch_base,
            engine_opp_switch_max=args.engine_opp_switch_max,
            engine_prune_delta=args.engine_prune_delta,
            engine_win_cutoff=args.engine_win_cutoff,
            engine_verbose=args.engine_verbose,
            offensive_switch_margin=args.offensive_switch_margin,
            tiebreak_top_k=args.tiebreak_top_k,
            tiebreak_move_min_k=args.tiebreak_move_min_k,
            tiebreak_move_score_ratio=args.tiebreak_move_score_ratio,
            decision_log_path=Path(args.decision_log) if args.decision_log else None,
            **hybrid_kwargs,
            **player_kwargs,
        )
        mode_label = (
            "hybride + tiebreak win-rate"
            if args.hybrid
            else "modèle + tiebreak win-rate"
        )
        if args.use_engine:
            mode_label += " + engine (pkmn/sim)"
            if args.engine_n_worlds > 1:
                mode_label += f" x{args.engine_n_worlds} worlds"
            if args.engine_depth > 1:
                mode_label += f" {args.engine_depth}-ply (élagué)"
            if not args.engine_use_model:
                mode_label += " [heuristic scorer]"
    elif args.hybrid:
        player = RbHybridPlayer(**hybrid_kwargs, **player_kwargs)
        mode_label = "hybride"
    else:
        player = RbModelPlayer(**player_kwargs)
        mode_label = "modèle seul"

    if args.mode == "accept":
        print(
            f"Connexion au serveur local ({LOCAL_SERVER_CONFIGURATION.websocket_url})...\n"
            f"Bot : {bot_name} [{mode_label}] | format : {args.battle_format}\n"
            f"Defis acceptes de : {args.opponent!r} (max {args.n_challenges})\n"
            "---\n"
            f"1) Ouvre http://localhost:8000 dans le navigateur\n"
            f"2) Connecte-toi avec le pseudo « {args.opponent} » (exact)\n"
            f"3) Defie : /challenge {bot_name}, gen9randombattle\n"
            f"   (ou menu Combat -> Defier -> {bot_name} -> Random Battle)\n"
            "---"
        )
        await player.accept_challenges(args.opponent, n_challenges=args.n_challenges)
        print(f"Combats terminés : {player.n_finished_battles}, victoires : {player.n_won_battles}")
        return

    opponent_cls = OPPONENTS[args.vs]
    opp_account = AccountConfiguration.generate(f"{args.vs}", rand=True)
    opponent = opponent_cls(
        account_configuration=opp_account,
        server_configuration=server_configuration,
        max_concurrent_battles=args.max_concurrent_battles,
        battle_format=args.battle_format,
    )
    print(
        f"Lancement de {args.n_battles} combats {bot_name} [{mode_label}] "
        f"vs {opp_account.username} ({args.battle_format})..."
    )
    await player.battle_against(opponent, n_battles=args.n_battles)
    total = max(player.n_finished_battles, 1)
    print(f"{bot_name}: {player.n_won_battles}/{args.n_battles} victoires ({player.n_won_battles / total:.1%})")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
