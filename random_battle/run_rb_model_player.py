"""
Run the Random Battle multi-head model on a local Showdown server via poke-env.

Examples:
  python random_battle/run_rb_model_player.py --mode accept --opponent Natanyelle
  python random_battle/run_rb_model_player.py --mode battle --n_battles 10
  python random_battle/run_rb_model_player.py --mode ladder --server official \\
      --username MonBot --password '***' --search --use_engine --duration_minutes 120
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from poke_env import AccountConfiguration, ShowdownServerConfiguration

from common.players.heuristics_pokemon_ai import HighHeuristicAI, LowHeuristicAI, RandomAI
from common.project_paths import setup_import_paths
from random_battle.config import (
    FORMAT_ID,
    LOCAL_SERVER_CONFIGURATION,
    MOVE_VOCAB_PATH,
    MULTIHEAD_MODEL_DIR,
    RB_SET_DEX_PATH,
    REPLAYS_DIR,
    SESSION_DIR,
    WINRATE_MODEL_DIR,
)
from random_battle.elo_tracker import EloTracker, attach_elo_tracking
from random_battle.live_log import attach_live_logging, live_print
from random_battle.session_log import (
    attach_session_logging,
    load_cumulative_stats,
    write_session_summary,
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
    parser.add_argument(
        "--mode",
        choices=("accept", "battle", "ladder"),
        default="battle",
        help="battle=vs bot local ; accept=défis ; ladder=queue officielle.",
    )
    parser.add_argument(
        "--server",
        choices=("local", "official"),
        default="local",
        help="local=localhost:8000 ; official=play.pokemonshowdown.com (ladder/accept).",
    )
    parser.add_argument(
        "--username",
        default="",
        help="Nom Showdown (obligatoire sur official ; vide = guest auto en local).",
    )
    parser.add_argument("--password", default=None)
    parser.add_argument("--opponent", default="Natanyelle", help="Human name for accept mode.")
    parser.add_argument(
        "--accept_all",
        action="store_true",
        help="Mode accept : accepter les défis de n'importe qui.",
    )
    parser.add_argument(
        "--duration_minutes",
        type=int,
        default=0,
        help="Mode ladder : durée max en minutes (0 = utiliser --n_battles).",
    )
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
    parser.add_argument(
        "--ping_timeout",
        type=float,
        default=None,
        help=(
            "Timeout websocket keepalive (s). Défaut auto : 180 sur official/engine, "
            "sinon 20. Augmenter si déconnexions pendant les longs calculs."
        ),
    )
    parser.add_argument(
        "--ping_interval",
        type=float,
        default=None,
        help="Intervalle ping websocket (s). Défaut auto : 45 sur official/engine.",
    )
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
        "--engine_switch_min_depth",
        type=int,
        default=2,
        help=(
            "Tours minimum simulés après un switch racine (pivot + réponse adv "
            "+ follow-ups). Défaut 2 pour voir les revenge-kills. Plus lent."
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
        "--prune_delta",
        dest="engine_prune_delta",
        type=float,
        default=0.15,
        metavar="DELTA",
        help=(
            "Élagage de l'arbre expectimax (--use_engine, --engine_depth >= 2) : "
            "un coup de suivi est ignoré si son probe est plus de DELTA sous le "
            "meilleur (win-prob 0–1). Plus petit = arbre plus petit, plus rapide "
            "(ex. 0.08 agressif, 0.25 conservateur). Défaut: 0.15."
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
        "--engine_workers",
        type=int,
        default=1,
        help=(
            "Nombre de bridges Node en parallèle pour évaluer les candidats "
            "moves/switches (un subprocess par worker). 4 est un bon départ."
        ),
    )
    parser.add_argument(
        "--engine_prune_switches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Avec --search : n'évalue en engine que les switchs défensifs "
            "ou plus rapides avec OHKO propre garanti (plus rapide en ladder)."
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
    parser.add_argument(
        "--save_replays",
        nargs="?",
        const="default",
        default=None,
        metavar="DIR",
        help=(
            "Sauvegarde les replays HTML (poke-env). Sans chemin : "
            "random_battle/artifacts/replays/"
        ),
    )
    parser.add_argument(
        "--session_stats",
        type=str,
        default=None,
        help=(
            "Fichier JSONL : une ligne par combat (won, opponent, turns). "
            "Défaut auto si --save_replays ou mode ladder/accept official."
        ),
    )
    parser.add_argument(
        "--session_summary",
        type=str,
        default=None,
        help="JSON récap en fin de session (wins/losses/win_rate).",
    )
    parser.add_argument(
        "--cumulative_stats",
        type=str,
        default=None,
        help=(
            "JSON cumulatif des victoires sur plusieurs sessions "
            "(défaut: artifacts/sessions/cumulative.json)."
        ),
    )
    parser.add_argument(
        "--live_log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Logs stderr lisibles : connexion, file ladder, tours, coups joués. "
            "Défaut : activé en ladder/accept official."
        ),
    )
    parser.add_argument(
        "--track_elo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Suivi Elo Showdown (live + JSONL + cumulative). "
            "Défaut : activé en ladder/accept official."
        ),
    )
    parser.add_argument(
        "--elo_history",
        type=str,
        default=None,
        help="JSONL historique Elo par combat (défaut auto si --track_elo).",
    )
    parser.add_argument(
        "--log_level",
        choices=("warning", "info", "debug"),
        default="info",
        help="Niveau de log poke-env (connexion websocket, etc.).",
    )
    return parser


def _resolve_session_paths(args: argparse.Namespace, bot_name: str) -> dict:
    """Pick replay/stats paths from CLI flags and mode."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_bot = re.sub(r"[^a-zA-Z0-9_-]+", "_", bot_name) or "bot"
    live = args.mode in ("ladder", "accept") and args.server == "official"
    auto_track = live or args.save_replays is not None

    replays_dir: Optional[Path] = None
    if args.save_replays is not None:
        replays_dir = (
            REPLAYS_DIR / f"{safe_bot}_{stamp}"
            if args.save_replays == "default"
            else Path(args.save_replays)
        )

    stats_path: Optional[Path] = None
    if args.session_stats:
        stats_path = Path(args.session_stats)
    elif auto_track:
        stats_path = SESSION_DIR / f"{safe_bot}_{stamp}.jsonl"

    summary_path: Optional[Path] = None
    if args.session_summary:
        summary_path = Path(args.session_summary)
    elif auto_track:
        summary_path = SESSION_DIR / f"{safe_bot}_{stamp}_summary.json"

    cumulative_path: Optional[Path] = None
    if args.cumulative_stats:
        cumulative_path = Path(args.cumulative_stats)
    elif auto_track:
        cumulative_path = SESSION_DIR / "cumulative.json"

    elo_history_path: Optional[Path] = None
    track_elo = args.track_elo
    if track_elo is None:
        track_elo = live
    if track_elo:
        if args.elo_history:
            elo_history_path = Path(args.elo_history)
        elif auto_track:
            elo_history_path = SESSION_DIR / f"{safe_bot}_elo.jsonl"

    return {
        "replays_dir": replays_dir,
        "stats_path": stats_path,
        "summary_path": summary_path,
        "cumulative_path": cumulative_path,
        "elo_history_path": elo_history_path,
        "track_elo": track_elo,
    }


def _print_session_footer(
    player,
    *,
    bot_name: str,
    args: argparse.Namespace,
    session_paths: dict,
) -> None:
    n_finished = player.n_finished_battles
    n_won = player.n_won_battles
    n_lost = max(0, n_finished - n_won)
    rate = (n_won / n_finished) if n_finished else 0.0
    print(
        f"\n=== Session {bot_name} ===\n"
        f"Combats : {n_finished} | Victoires : {n_won} | Défaites : {n_lost} "
        f"| Win rate : {rate:.1%}"
    )
    summary_path = session_paths.get("summary_path")
    if summary_path is not None:
        elo_tracker = getattr(player, "_elo_tracker", None)
        summary = write_session_summary(
            summary_path,
            bot_username=bot_name,
            n_finished=n_finished,
            n_won=n_won,
            mode=args.mode,
            server=args.server,
            battle_format=args.battle_format,
            stats_path=session_paths.get("stats_path"),
            replays_dir=session_paths.get("replays_dir"),
            cumulative_path=session_paths.get("cumulative_path"),
            elo_start=(
                elo_tracker.session_start_elo if elo_tracker is not None else None
            ),
            elo_end=elo_tracker.last_elo if elo_tracker is not None else None,
        )
        print(f"Résumé : {summary_path}")
        if elo_tracker is not None:
            elo_summary = elo_tracker.format_session_elo_summary()
            if elo_summary:
                print(elo_summary)
            if session_paths.get("elo_history_path"):
                print(f"Historique Elo : {session_paths['elo_history_path']}")
        cum_path = session_paths.get("cumulative_path")
        if cum_path is not None:
            cum = load_cumulative_stats(cum_path)
            total_b = int(cum.get("total_battles", 0))
            total_w = int(cum.get("total_wins", 0))
            rate_all = (total_w / total_b) if total_b else 0.0
            print(
                f"Cumul toutes sessions : {total_w}/{total_b} ({rate_all:.1%}) "
                f"→ {cum_path}"
            )
    if session_paths.get("stats_path"):
        print(f"Stats JSONL : {session_paths['stats_path']}")
    if session_paths.get("replays_dir"):
        print(f"Replays : {session_paths['replays_dir']}")


async def run_ladder_session(player, *, n_battles: int, duration_minutes: int) -> None:
    if duration_minutes > 0:
        deadline = time.monotonic() + duration_minutes * 60
        while time.monotonic() < deadline:
            await player.ladder(1)
        return
    await player.ladder(n_battles)


async def main_async(args: argparse.Namespace) -> None:
    if args.mode in ("ladder", "accept") and args.server == "official":
        if not args.username.strip():
            raise SystemExit(
                "Sur le serveur officiel, --username (compte enregistré) est requis."
            )
        if not args.password:
            print(
                "Attention : pas de --password — connexion invité (peut être instable)."
            )

    if args.username.strip():
        account = AccountConfiguration(args.username.strip(), args.password)
        bot_name = account.username
    else:
        account = AccountConfiguration.generate("RbBot", rand=True)
        bot_name = account.username

    server_configuration = (
        ShowdownServerConfiguration
        if args.server == "official"
        else LOCAL_SERVER_CONFIGURATION
    )
    if args.mode == "ladder" and args.server != "official":
        print("Mode ladder : bascule automatique sur --server official.")
        server_configuration = ShowdownServerConfiguration
    ping_timeout = args.ping_timeout
    ping_interval = args.ping_interval
    if args.server == "official" or args.use_engine:
        if ping_timeout is None:
            ping_timeout = 180.0
        if ping_interval is None:
            ping_interval = 45.0
    else:
        if ping_timeout is None:
            ping_timeout = 20.0
        if ping_interval is None:
            ping_interval = 20.0

    session_paths = _resolve_session_paths(args, bot_name)
    if session_paths["replays_dir"] is not None:
        session_paths["replays_dir"].mkdir(parents=True, exist_ok=True)

    log_levels = {
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    live_log = args.live_log
    if live_log is None:
        live_log = args.mode in ("ladder", "accept") and args.server == "official"

    player_kwargs = dict(
        account_configuration=account,
        server_configuration=server_configuration,
        max_concurrent_battles=args.max_concurrent_battles,
        ping_timeout=ping_timeout,
        ping_interval=ping_interval,
        save_replays=(
            str(session_paths["replays_dir"])
            if session_paths["replays_dir"] is not None
            else False
        ),
        log_level=log_levels.get(args.log_level, logging.INFO),
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
        # Defaults block all setups (0); search mode allows a few when safe.
        if args.max_setups_per_battle == 0:
            player_kwargs["max_setups_per_battle"] = 2
        if args.max_setups_per_active == 0:
            player_kwargs["max_setups_per_active"] = 1
        player = RbSearchPlayer(
            winrate_model_path=Path(args.winrate_model),
            use_engine=args.use_engine,
            engine_aggregation=args.engine_aggregation,
            engine_n_worlds=args.engine_n_worlds,
            engine_world_aggregation=args.engine_world_aggregation,
            engine_use_model=args.engine_use_model,
            engine_depth=args.engine_depth,
            engine_switch_min_depth=args.engine_switch_min_depth,
            engine_depth2_my_top_k=args.engine_depth2_my_top_k,
            engine_deep_opp_move_cap=args.engine_deep_opp_move_cap,
            engine_opp_switch_base=args.engine_opp_switch_base,
            engine_opp_switch_max=args.engine_opp_switch_max,
            engine_prune_delta=args.engine_prune_delta,
            engine_win_cutoff=args.engine_win_cutoff,
            engine_verbose=args.engine_verbose,
            engine_workers=args.engine_workers,
            engine_prune_switches=args.engine_prune_switches,
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
                mode_label += (
                    f" {args.engine_depth}-ply (élagué, prune_δ={args.engine_prune_delta})"
                )
            if args.engine_workers > 1:
                mode_label += f" x{args.engine_workers} workers"
            if not args.engine_use_model:
                mode_label += " [heuristic scorer]"
    elif args.hybrid:
        player = RbHybridPlayer(**hybrid_kwargs, **player_kwargs)
        mode_label = "hybride"
    else:
        player = RbModelPlayer(**player_kwargs)
        mode_label = "modèle seul"

    elo_tracker: Optional[EloTracker] = None
    if session_paths.get("track_elo") and args.server == "official":
        elo_tracker = EloTracker(
            username=bot_name,
            battle_format=args.battle_format,
            history_path=session_paths.get("elo_history_path"),
            cumulative_path=session_paths.get("cumulative_path"),
        )

    attach_session_logging(
        player,
        stats_path=session_paths.get("stats_path"),
        bot_username=bot_name,
    )
    attach_live_logging(player, bot_username=bot_name, enabled=live_log)
    if elo_tracker is not None:
        attach_elo_tracking(
            player,
            elo_tracker,
            stats_path=session_paths.get("stats_path"),
            live_print_fn=live_print if live_log else None,
        )
    if live_log:
        live_print("Logs live activés (--no-live_log pour désactiver).")
    live_print(f"Connexion à {server_configuration.websocket_url}...")
    if session_paths.get("replays_dir"):
        print(f"Replays → {session_paths['replays_dir']}")
    if session_paths.get("stats_path"):
        print(f"Stats JSONL → {session_paths['stats_path']}")

    if args.mode == "accept":
        accept_from = None if args.accept_all else args.opponent
        ws = server_configuration.websocket_url
        print(
            f"Connexion ({ws})...\n"
            f"Bot : {bot_name} [{mode_label}] | format : {args.battle_format}\n"
            f"Défis acceptés de : "
            f"{'tout le monde' if accept_from is None else accept_from!r} "
            f"(max {args.n_challenges})\n"
        )
        if args.server == "local":
            print(
                "---\n"
                f"1) Ouvre http://localhost:8000\n"
                f"2) Défie : /challenge {bot_name}, {args.battle_format}\n"
                "---"
            )
        else:
            only = (
                "n'importe qui"
                if accept_from is None
                else f"le joueur « {accept_from} » uniquement"
            )
            print(
                "---\n"
                "1) Ouvre https://play.pokemonshowdown.com et connecte-toi\n"
                f"2) Dans le chat : /challenge {bot_name}, {args.battle_format}\n"
                f"   (défis acceptés de : {only})\n"
                f"3) Le bot doit rester lancé dans ce terminal\n"
                f"   (websocket ping_timeout={ping_timeout}s)\n"
                "---"
            )
        if args.use_engine and args.max_concurrent_battles > 1:
            print("Accept + engine : max_concurrent_battles forcé à 1.")
            player.max_concurrent_battles = 1
        await player.accept_challenges(accept_from, n_challenges=args.n_challenges)
        _print_session_footer(player, bot_name=bot_name, args=args, session_paths=session_paths)
        return

    if args.mode == "ladder":
        if args.use_engine and args.max_concurrent_battles > 1:
            print("Ladder + engine : max_concurrent_battles forcé à 1.")
            player.max_concurrent_battles = 1
        print(
            f"Ladder {args.battle_format} — {bot_name} [{mode_label}]\n"
            f"Serveur : {server_configuration.websocket_url}"
        )
        if args.duration_minutes > 0:
            print(f"Durée : {args.duration_minutes} min")
        else:
            print(f"Combats : {args.n_battles}")
        await run_ladder_session(
            player,
            n_battles=args.n_battles,
            duration_minutes=args.duration_minutes,
        )
        _print_session_footer(player, bot_name=bot_name, args=args, session_paths=session_paths)
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
    _print_session_footer(player, bot_name=bot_name, args=args, session_paths=session_paths)


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
