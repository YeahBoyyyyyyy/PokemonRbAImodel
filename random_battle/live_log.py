"""Human-readable stderr logs for ladder / accept sessions."""

from __future__ import annotations

import sys
import time
from typing import Any, Optional

from poke_env.battle import AbstractBattle


def live_print(msg: str) -> None:
    print(f"[live] {msg}", file=sys.stderr, flush=True)


def _species(mon: Any) -> str:
    if mon is None:
        return "?"
    try:
        from random_battle.players.poke_env_to_state import species_name

        return species_name(mon) or "?"
    except Exception:
        raw = getattr(mon, "species", None) or getattr(mon, "name", "?")
        return str(getattr(raw, "name", raw))


def attach_live_logging(
    player: Any,
    *,
    bot_username: str,
    enabled: bool = True,
) -> None:
    if not enabled:
        return

    # --- Battle finished ---
    orig_finish = player._battle_finished_callback

    def _finish(battle: AbstractBattle) -> None:
        orig_finish(battle)
        if getattr(player, "_elo_tracker", None) is not None:
            return
        tag = "WIN" if getattr(battle, "won", False) else "LOSS"
        opp = getattr(battle, "opponent_username", None) or "?"
        n = player.n_finished_battles
        w = player.n_won_battles
        live_print(
            f"=== {tag} vs {opp} ({getattr(battle, 'turn', '?')} tours) "
            f"| session {w}/{n} wins ==="
        )

    player._battle_finished_callback = _finish

    # --- New battle ---
    orig_create = player._create_battle

    async def _create_battle(split_message):
        battle = await orig_create(split_message)
        opp = getattr(battle, "opponent_username", None) or "?"
        live_print(f"Combat #{player.n_finished_battles + 1} vs {opp}")
        return battle

    player._create_battle = _create_battle

    # --- Each turn decision ---
    orig_choose = player.choose_move

    async def _choose_move(battle: AbstractBattle):
        t0 = time.time()
        turn = getattr(battle, "turn", "?")
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        live_print(
            f"T{turn} {_species(my)} vs {_species(opp)} — calcul en cours..."
        )
        order = await orig_choose(battle)
        msg = getattr(order, "message", str(order)) or "?"
        dt = time.time() - t0
        short = msg if len(msg) <= 100 else msg[:97] + "..."
        live_print(f"T{turn} → {short} ({dt:.1f}s)")
        return order

    player.choose_move = _choose_move

    # --- Ladder queue ---
    orig_ladder = player._ladder

    async def _ladder(n_games: int) -> None:
        await player.ps_client.logged_in.wait()
        tracker = getattr(player, "_elo_tracker", None)
        if tracker is not None:
            elo = await tracker.refresh_from_api()
            if elo is not None:
                live_print(
                    f"Connecté sur Showdown ({bot_username}). "
                    f"Elo {tracker.battle_format} : {elo} — {n_games} match(s) ladder."
                )
            else:
                live_print(
                    f"Connecté sur Showdown ({bot_username}). "
                    f"Elo indisponible — {n_games} match(s) ladder."
                )
        else:
            live_print(
                f"Connecté sur Showdown ({bot_username}). {n_games} match(s) ladder."
            )
        for i in range(n_games):
            live_print(f"Recherche adversaire ({i + 1}/{n_games})...")
            async with player._battle_start_condition:
                await player.ps_client.search_ladder_game(
                    player._format, player.next_team
                )
                live_print("   En file d'attente sur le ladder (attente adversaire)...")
                await player._battle_start_condition.wait()
                live_print("   Adversaire trouvé — combat lancé.")
                while player._battle_count_queue.full():
                    async with player._battle_end_condition:
                        await player._battle_end_condition.wait()
                await player._battle_semaphore.acquire()
        await player._battle_count_queue.join()
        live_print("Session ladder terminée.")

    player._ladder = _ladder

    # --- Accept challenges ---
    orig_accept = player._accept_challenges

    async def _accept_challenges(opponent, n_challenges, packed_team):
        from poke_env.data import to_id_str

        opp_filter = opponent
        if opp_filter:
            if isinstance(opp_filter, list):
                opp_filter = [to_id_str(o) for o in opp_filter]
            else:
                opp_filter = to_id_str(opp_filter)
        await player.ps_client.logged_in.wait()
        who = "tout le monde" if opp_filter is None else opponent
        tracker = getattr(player, "_elo_tracker", None)
        if tracker is not None:
            elo = await tracker.refresh_from_api()
            if elo is not None:
                live_print(
                    f"Connecté ({bot_username}). Elo {tracker.battle_format} : {elo} — "
                    f"en attente de défis de {who} (max {n_challenges})."
                )
            else:
                live_print(
                    f"Connecté ({bot_username}). En attente de défis de {who} "
                    f"(max {n_challenges})."
                )
        else:
            live_print(
                f"Connecté ({bot_username}). En attente de défis de {who} "
                f"(max {n_challenges})."
            )
        for i in range(n_challenges):
            live_print(f"En attente défi ({i + 1}/{n_challenges})...")
            team = packed_team or player.next_team
            while True:
                username = to_id_str(await player._challenge_queue.get())
                if (
                    opp_filter is None
                    or opp_filter == username
                    or (isinstance(opp_filter, list) and username in opp_filter)
                ):
                    live_print(f"Défi accepté de {username}")
                    await player.ps_client.accept_challenge(username, team)
                    await player._battle_semaphore.acquire()
                    break
        await player._battle_count_queue.join()
        live_print("Session accept terminée.")

    player._accept_challenges = _accept_challenges
