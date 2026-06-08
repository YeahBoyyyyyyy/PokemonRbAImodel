import asyncio
import random
import statistics
from poke_env.player import Player
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.project_paths import setup_import_paths

setup_import_paths()

from materials import *

#======================================================== Random AI ==============================================================#

class RandomAI(Player):
    """IA Pokémon simple : choix totalement aléatoire avec pondération."""

    async def choose_move(self, battle):
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        can_tera = getattr(battle, "can_terastallize", False)

        possible_actions = []

        if available_moves:
            possible_actions.append("attack")

        if available_switches:
            possible_actions.append("switch")

        if can_tera and available_moves:
            possible_actions.append("terastallize")

        # Si aucune action n'est possible, fallback
        if not possible_actions:
            return self.choose_default_move()

        # Choix aléatoire pondéré
        if len(possible_actions) > 1:
            if "switch" in possible_actions and "terastallize" in possible_actions:
                weights = [0.8, 0.15, 0.05]
            elif "switch" in possible_actions:
                weights = [0.9, 0.1]
            elif "terastallize" in possible_actions:
                weights = [0.9, 0.1]
            else:
                weights = [1.0]
            action_type = random.choices(possible_actions, weights=weights)[0]
        else:
            action_type = possible_actions[0]

        # Exécution
        if action_type == "attack":
            move = random.choice(available_moves)
            return self.create_order(move)

        elif action_type == "switch":
            switch_target = random.choice(available_switches)
            return self.create_order(switch_target)

        elif action_type == "terastallize":
            move = random.choice(available_moves)
            return self.create_order(move, terastallize=True)

        return self.choose_default_move()

#======================================================== Low Heuristic AI ==============================================================#

class LowHeuristicAI(Player):
    """IA simple : choisit ses actions selon l'efficacité de type."""

    async def choose_move(self, battle):
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        can_tera = getattr(battle, "can_terastallize", False)

        my_poke = battle.active_pokemon
        enemy = battle.opponent_active_pokemon

        if not available_moves and not available_switches:
            return self.choose_default_move()

        # --- Calcul des avantages de type ---
        if available_switches and enemy and my_poke and my_poke.types and enemy.types:
            my_eff = 1
            for t in my_poke.types:
                my_eff *= type_effectiveness(t, enemy.types)
            enemy_eff = 1
            for t in enemy.types:
                enemy_eff *= type_effectiveness(t, my_poke.types)

            # switch si gros désavantage
            if enemy_eff > 1.5 and random.random() < 0.6:
                best_switch = self._choose_best_switch(battle, available_switches, enemy)
                if best_switch:
                    return self.create_order(best_switch)

        # --- Choisir la meilleure attaque ---
        if available_moves:
            best_move, best_score = None, -1
            for move in available_moves:
                if move.base_power == 0:
                    continue
                score = type_effectiveness(move.type, enemy.types if enemy else []) * (move.base_power or 1)
                if move.type in my_poke.types:
                    score *= 1.5  # STAB
                if enemy and enemy.current_hp_fraction < 0.2:
                    score *= 1.2
                if score > best_score:
                    best_score = score
                    best_move = move
            if best_move:
                return self.create_order(best_move)

        # --- Option Tera ---
        if can_tera and available_moves:
            move = random.choice(available_moves)
            return self.create_order(move, terastallize=True)

        return self.choose_default_move()

    def _choose_best_switch(self, battle, switches, enemy):
        """Choisit le switch avec le meilleur match-up de type."""
        best = None
        best_score = float("inf")
        for switch in switches:
            if not enemy or not switch.types:
                continue
            eff = type_effectiveness(enemy.types[0], switch.types)
            if eff < best_score:
                best_score = eff
                best = switch
        return best

#======================================================== High Heuristic AI ==============================================================#

class HighHeuristicAI(Player):
    """
    IA avancée prenant en compte :
    - les niveaux réels des Pokémon
    - la stat de vitesse max EV (priorité aux plus rapides)
    - une estimation réaliste des dégâts en Random Battle
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.switch_streak = 0

    async def choose_move(self, battle):
        # Gérer les switches forcés
        if battle.force_switch:
            return await self.choose_switch(battle)

        me = battle.active_pokemon
        foe = battle.opponent_active_pokemon
        moves = battle.available_moves
        switches = battle.available_switches
        can_tera = getattr(battle, "can_terastallize", False)

        if not moves and not switches:
            return self.choose_default_move()

        # --- SCORE DES OPTIONS ---
        attack_scores = self._score_attacks(battle, me, foe)
        stay_score = self._score_current_pokemon(me, foe, attack_scores)
        switch_scores = self._score_switches(battle, switches, foe)

        best_switch, best_switch_score = None, float("-inf")
        if switch_scores:
            best_switch = max(switch_scores, key=switch_scores.get)
            best_switch_score = switch_scores[best_switch]

        # Meilleure attaque
        best_attack = max(attack_scores, key=attack_scores.get) if attack_scores else None

        # --- LOGIQUE DE DÉCISION ---
        if best_switch and best_switch_score > stay_score and self.switch_streak < 2:
            self.switch_streak += 1
            return self.create_order(best_switch)
        elif best_attack:
            self.switch_streak = 0
            if can_tera and me.tera_type == best_attack.type:
                return self.create_order(best_attack, terastallize=True)
            return self.create_order(best_attack)
        elif switches:
            self.switch_streak += 1
            return self.create_order(random.choice(switches))
        return self.choose_default_move()

    # === SCORE DES ATTAQUES ===
    def _score_attacks(self, battle, me, foe):
        if not me or not foe or not me.moves:
            return {}

        scores = {}
        moves = battle.available_moves  # poke-env fournit déjà ceux utilisables
        if not moves:
            return {}

        for move in moves:

            dmg, ko_prob, dmg_min, dmg_max = self.damage_estimate(me, move, foe)
            eff = type_effectiveness(move.type, foe.types)

            # On corrige ici : score de base = dégâts * efficacité
            score = 2 * dmg * eff

            # Pénalité massive pour inefficacité
            if eff == 0:
                score -= 999
            elif eff < 1:
                score /= 2  # double réduction

            # Bonus si KO probable
            if ko_prob > 0.95:
                score += 400

            # Bonus vitesse
            me_speed = self.estimate_speed(me)
            foe_speed = self.estimate_speed(foe)
            if me_speed > foe_speed:
                score *= 1.1

            # Bonus priorité si ennemi low HP
            try:
                priority = move.priority if move.priority is not None else 0
            except:
                priority = 0
            if priority > 0 and foe.current_hp_fraction < 0.2:
                score *= 1.2

            # Bonus STAB (réduit, car déjà compté dans damage_estimate)
            if move.type in (me.types or []):
                score *= 1.1

            # Moves de soin
            if move.category.name == "STATUS" and "heal" in move.id and me.current_hp_fraction < 0.5:
                score += 300
            if move._id in ["swordsdance", "nastyplot", "dragondance", "calmmind"]:
                if me_speed > foe_speed:
                    score += 15  # plus sûr de se booster avant d’être frappé
                else:
                    score -= 20  # dangereux de setup quand on est lent
            if move._id in ["recover", "roost", "softboiled", "wish"]:
                if me.current_hp_fraction < 0.5:
                    score += 30  # utile si low HP

            scores[move] = score

        return scores


    # === SCORE DU SWITCH ===
    def _score_switches(self, battle, switches, foe):
        scores = {}
        if not foe or not switches:
            return scores

        for p in switches:
            eff = 1.0
            for t in foe.types or []:
                eff *= type_effectiveness(t, p.types)
            score = 100 / (eff + 0.01) * 1.25
            score *= (p.current_hp_fraction or 1.0)
            scores[p] = score/10
        return scores

    # === SCORE DU POKÉMON ACTUEL ===
    def _score_current_pokemon(self, me, foe, attack_scores):
        """Évalue la pertinence de garder le Pokémon actuel en fonction du matchup."""
        if not me or not foe:
            return 0

        score = 0

        # --- Estimation réaliste de la vitesse ---
        me_speed = self.estimate_speed(me)
        foe_speed = self.estimate_speed(foe)

        # --- Légère préférence pour les Pokémon plus rapides ---
        if me_speed >= foe_speed:
            score += 30
        else:
            score -= 30
            
        # --- Statuts négatifs ---
        if me._status == "BRN":  # burn
            score -= 15
        elif me._status == "PAR":  # paralysie
            score -= 10
        elif me._status in ["PSN", "TOX"]:  # poison/toxic
            score -= 20

        # --- Meilleure attaque disponible ---
        if attack_scores:
            score += max(attack_scores.values())

        return score


    # === SWITCH FORCÉ ===
    async def choose_switch(self, battle):
        switches = battle.available_switches
        if not switches:
            return self.choose_default_move()

        foe = battle.opponent_active_pokemon
        best_switch, best_score = None, float("-inf")
        for p in switches:
            eff = 1.0
            for t in foe.types or []:
                eff *= type_effectiveness(t, p.types)
            score = 100 / max(eff, 0.5)
            score *= (p.current_hp_fraction or 1.0)
            if score > best_score:
                best_score, best_switch = score, p

        if not best_switch:
            best_switch = random.choice(switches)
        
        return self.create_order(best_switch)

    def estimate_speed(self, pokemon):

        """Estime la vitesse d’un Pokémon à partir de son espèce, niveau et boosts visibles."""
        from pokedex_9G_complete import pokemon_data_gen9

        name = get_name(pokemon)

        base_speed = pokemon_data_gen9.get(name, {}).get("stats", {}).get(
            "Speed")

        level = getattr(pokemon, "level")

        iv = 31
        ev = 252
        nature = 1.0  # Hardy neutre

        # Calcul de la vitesse brute
        raw_speed = int(((2 * base_speed + iv + ev // 4) * level / 100) + 5) * nature

        # Application des boosts visibles (+1 = *1.5, -1 = *0.67, etc.)
        boost_stage = getattr(pokemon, "boosts", {}).get("spe", 0)
        if boost_stage > 0:
            mult = (2 + boost_stage) / 2
        elif boost_stage < 0:
            mult = 2 / (2 - boost_stage)
        else:
            mult = 1.0

        return int(raw_speed * mult)

    def damage_estimate(self, attacker, move, defender):
        """Estime les dégâts en se basant uniquement sur le niveau, la puissance du move et les types.
        Ignore les talents et objets de l'adversaire (inconnus au début)."""
        
        if move.category == "STATUS":
            return [0, 0, 0, 0]


        from pokedex_9G_complete import pokemon_data_gen9

        name_attacker = get_name(attacker)
        name_defender = get_name(defender)

        base_attack = pokemon_data_gen9.get(name_attacker, {}).get("stats", {}).get(
            "Attack" if move.category.name == "PHYSICAL" else "Sp. Atk")
        base_defense = pokemon_data_gen9.get(name_defender, {}).get("stats", {}).get(
            "Defense" if move.category.name == "PHYSICAL" else "Sp. Def")

        base_power = move.base_power or 40
        
        damage_max = (((2 * 50 / 5 + 2) * base_power * (base_attack / max(1, base_defense))) / 50) + 2

        # Multiplicateurs de base
        stab = 1.5 if move.type in (attacker.types or []) else 1.0
        eff = type_effectiveness(move.type, defender.types)

        # --- Talent de l'attaquant (si connu et générique) ---
        talent_mult = 1.0
        if attacker.ability:
            ab = attacker.ability.lower()
            # seulement quelques talents offensifs évidents
            if "adaptability" in ab and stab > 1:
                stab = 2.0
            elif "sheerforce" in ab and getattr(move, "secondary", None):
                talent_mult *= 1.3
            elif "toughclaws" in ab and getattr(move, "makes_contact", False):
                talent_mult *= 1.3

        # --- Objet de l'attaquant (si connu et courant) ---
        item_mult = 1.0
        if attacker.item:
            it = attacker.item.lower()
            if "choiceband" in it and move.category.name == "PHYSICAL":
                item_mult *= 1.5
            elif "choicespecs" in it and move.category.name == "SPECIAL":
                item_mult *= 1.5
            elif "lifeorb" in it:
                item_mult *= 1.3

        rdm = [0.85 + 0.01 * i for i in range(16)]

        # --- Formule des dégâts ---

        if attacker.status == "brn" and move.category.name == "PHYSICAL":
            burn = 0.5
        else:
            burn = 1.0

        raw_damage = (((2 * 50 / 5 + 2) * base_power * base_attack / max(1, base_defense)) / 50) + 2
        damage_max = raw_damage * stab * eff * item_mult * talent_mult * burn
        possible_damages = [damage_max * r for r in rdm]

        cases_of_ko = [d for d in possible_damages if d >= defender.current_hp_fraction * defender.max_hp]

        prob_ko = len(cases_of_ko) / len(possible_damages) if possible_damages else 0
        damage_avg = int((damage_max + damage_max * 0.85) / 2)
        damage_min = int(min(possible_damages))

        # --- Probabilité de KO estimée ---
        hp = defender.current_hp_fraction * defender.max_hp if defender.current_hp_fraction else defender.max_hp
        prob_ko = min(1.0, damage_max / hp) if hp and hp > 0 else 1.0

        return [damage_avg, prob_ko, damage_min, damage_max]


##################################################################################################################################################
########################################################## Boucle d'événements ###################################################################
##################################################################################################################################################


# --- Fonction principale : plusieurs seeds ---
async def benchmark(n_runs=10, n_battles=300):
    results = []

    for seed in range(n_runs):
        random.seed(seed)

        ai1 = HighHeuristicAI(
            account_configuration=AccountConfiguration(f"HighAI_{seed}", None),
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=5,  # mettre 5 combats à la fois
        )
        ai2 = LowHeuristicAI(
            account_configuration=AccountConfiguration(f"RandAI_{seed}", None),
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=5,
        )

        print(f" Run {seed+1}/{n_runs} — Seed = {seed}")
        await ai1.battle_against(ai2, n_battles=n_battles)

        # Attendre que le serveur termine bien le match précédent
        #await asyncio.sleep(0.5)

        win_rate = ai1.n_won_battles / n_battles
        results.append(win_rate)
        print(f"Résultat : {win_rate*100:.1f}% de victoires")

    mean = statistics.mean(results)
    stdev = statistics.pstdev(results)
    print("\n Résumé global :")
    print(f"Moyenne : {mean*100:.2f}%  |  Écart type : {stdev*100:.2f}%")

    return results

if __name__ == "__main__":  
    asyncio.run(benchmark())
    

"""
"""
async def main_bot_vs_bot():
    ai1 = LowHeuristicAI(
        account_configuration=AccountConfiguration("AI1", None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10
    )

    ai2 = LowHeuristicAI(
        account_configuration=AccountConfiguration("AI2", None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10
    )

    N_BATTLES = 1000
    print(f"Lancement de {N_BATTLES} combats entre AI1 et AI2...")
    await ai1.battle_against(ai2, n_battles=N_BATTLES)

    print("Résultats :")
    print(f"AI1 - {ai1.n_won_battles}/{N_BATTLES} ({ai1.n_won_battles / N_BATTLES:.1%})")
    print(f"AI2 - {ai2.n_won_battles}/{N_BATTLES} ({ai2.n_won_battles / N_BATTLES:.1%})")

    print(f"Nombre de combats terminés : {ai1.n_finished_battles}")
    print(f"Nombre de victoires AI1 : {ai1.n_won_battles}")

    avg_turns = sum(b.turn for b in ai1.battles.values()) / len(ai1.battles)
    print(f"Nombre moyen de tours par combat : {avg_turns:.2f}")
"""

async def main_human_vs_bot():
    bot = HighHeuristicAI(
        account_configuration=AccountConfiguration("HeuristicBot", None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )

    print("HeuristicBot prêt ! Connecte-toi sur ton serveur local et défie-le : 'HeuristicBot'")
    print("Le bot attend un défi de 'Natanyelle'...")

    await bot.accept_challenges("Natanyelle", n_challenges=1)   

    print(" Défi terminé.")

if __name__ == "__main__":
    asyncio.run(main_human_vs_bot())

"""
