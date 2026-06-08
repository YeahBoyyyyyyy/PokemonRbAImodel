"""
Extract training data from Pokéchamp Hugging Face dataset.

This script adapts the local replay extractor to work with the massive
Pokéchamp dataset (2.1M battles). It processes battles in streaming mode
and infers actions from the battle log.

Key differences from local extractor:
- Loads from HF dataset instead of local JSON files
- Reads 'text' field instead of 'log'/'inputlog'
- Infers actions from battle log (no inputlog available)
- All other parsing logic identical (95% code compatibility)
"""

import json
import copy
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from datasets import load_dataset


@dataclass
class PokemonState:
    """State of a single Pokemon."""
    species: str
    is_active: bool
    hp_percent: float
    status: Optional[str]
    fainted: bool
    stats_boosts: Dict[str, int]
    volatile_conditions: List[str]
    moves_seen: List[str]
    item: Optional[str]
    ability: Optional[str]
    tera_type: Optional[str]
    tera_active: bool
    revealed: bool  # True if Pokemon has switched in (HP/status known)


@dataclass
class GameState:
    """Complete game state at a turn."""
    turn: int
    my_team: List[PokemonState]
    opp_team: List[PokemonState]
    my_side_conditions: Dict[str, bool]
    opp_side_conditions: Dict[str, bool]
    my_hazards: Dict[str, int]
    opp_hazards: Dict[str, int]
    weather: Optional[str]
    terrain: Optional[str]
    my_last_move: Optional[str]
    opp_last_move: Optional[str]
    player: str
    my_team_slot_order: Optional[List[str]] = None  # team preview order (RB)


@dataclass
class TrainingExample:
    """Single training example (state, action) pair."""
    state: GameState
    action_type: str  # 'move' or 'switch'
    action_target: str  # move name or switch target
    is_voluntary: bool  # False if forced switch after KO
    winner: bool  # True if player won


class TrainingDataExtractor:
    """Extract training data from Pokéchamp battles."""
    
    def __init__(self):
        self.reset_state()
    
    def reset_state(self):
        """Reset internal state for new battle."""
        self.winner = None
        self.faints_this_turn = {'p1': False, 'p2': False}
        self.player_usernames = {}  # Map username -> player_id
        self.nickname_to_species = {}  # Map (player_id, nickname) -> species

    def _resolve_name(self, player_id: str, raw_name: str) -> str:
        """Resolve nickname to species when possible."""
        if not raw_name:
            return raw_name
        return self.nickname_to_species.get((player_id, raw_name), raw_name)
    
    def extract_from_pokechamp_battle(self, battle: Dict) -> List[TrainingExample]:
        """
        Extract training examples from Pokéchamp battle.
        
        Args:
            battle: Battle dict from Pokéchamp dataset with 'text' field
            
        Returns:
            List of training examples
        """
        self.reset_state()
        
        # Get battle log from 'text' field
        log_text = battle.get('text', '')
        if not log_text:
            return []
        
        # Split into lines
        log_lines = log_text.strip().split('\n')
        
        # Extract winner
        self._extract_winner(log_lines)
        
        # Parse log to build state history
        state_history = self._parse_log_to_states(log_lines)
        
        # Infer actions from battle log (no inputlog available)
        actions = self._infer_actions_from_log(log_lines)
        
        # Create training examples
        examples = self._create_training_examples(state_history, actions)
        
        return examples
    
    def _extract_winner(self, log_lines: List[str]) -> None:
        """Extract winner from battle log."""
        # First, parse player usernames
        for line in log_lines:
            if line.startswith('|player|'):
                parts = line.split('|')
                if len(parts) >= 4:
                    player_id = parts[2]  # 'p1' or 'p2'
                    username = parts[3]
                    self.player_usernames[username] = player_id
        
        # Then, parse winner (username format)
        for line in log_lines:
            if line.startswith('|win|'):
                parts = line.split('|')
                if len(parts) >= 3:
                    winner_username = parts[2].strip()
                    self.winner = self.player_usernames.get(winner_username)
                    return
    
    def _empty_player_state(self) -> Dict:
        """Create empty player state."""
        return {
            'active': None,
            'team': {},
            'side_conditions': {},
            'hazards': {},
            'last_move': None
        }
    
    def _parse_log_to_states(
        self,
        log_lines: List[str],
        pre_action_snapshots: Optional[List[Tuple[int, str, str, str, Dict]]] = None,
    ) -> Dict[int, Dict]:
        """
        Parse battle log to build turn-by-turn state history.

        If pre_action_snapshots is provided, append
        (turn, player, action_type, action_target, state) tuples captured
        immediately BEFORE each |move| or |switch| line is applied.

        Returns:
            Dict mapping turn number to game state
        """
        state_history = {}
        
        def new_state():
            return {
                'p1': self._empty_player_state(),
                'p2': self._empty_player_state(),
                'weather': None,
                'terrain': None,
                'turn': 0
            }
        
        state_history[0] = new_state()
        current_turn = 0
        
        # First pass: Parse team preview to get full teams
        for line in log_lines:
            if not line.strip() or not line.startswith('|'):
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue
            
            # Team preview: |poke|p1|Pokemon, F|item
            if parts[1] == 'poke':
                player_id = parts[2]
                pokemon_name = parts[3].split(',')[0].strip()

                # Initialize all team members at turn 0 with unknown status
                if pokemon_name not in state_history[0][player_id]['team']:
                    state_history[0][player_id]['team'][pokemon_name] = {
                        'hp_percent': 1.0,  # Unknown until revealed
                        'status': None,
                        'fainted': False,
                        'boosts': {},
                        'volatiles': [],
                        'moves_seen': [],
                        'item': None,
                        'ability': None,
                        'tera_type': None,
                        'tera_active': False,
                        'revealed': False,  # Track if we've seen it yet
                        'preview_species': True,  # Species visible in team preview (RB/OU)
                    }
        
        # Second pass: Parse battle events
        for line in log_lines:
            if not line.strip() or not line.startswith('|'):
                continue
            
            parts = line.split('|')
            
            if len(parts) < 2:
                continue
            
            msg_type = parts[1]
            
            # Turn marker
            if msg_type == 'turn':
                current_turn = int(parts[2])
                # Reset faint tracking for new turn
                self.faints_this_turn = {'p1': False, 'p2': False}
                # Deep copy previous state (including team preview from turn 0)
                if current_turn > 1 and (current_turn - 1) in state_history:
                    state_history[current_turn] = copy.deepcopy(state_history[current_turn - 1])
                    state_history[current_turn]['turn'] = current_turn
                elif current_turn == 1 and 0 in state_history:
                    # Turn 1: copy team preview from turn 0
                    state_history[current_turn] = copy.deepcopy(state_history[0])
                    state_history[current_turn]['turn'] = current_turn
                else:
                    state_history[current_turn] = new_state()
                    state_history[current_turn]['turn'] = current_turn
            
            if current_turn == 0:
                current_turn = 1
                if 1 not in state_history:
                    # Copy team preview from turn 0
                    state_history[1] = copy.deepcopy(state_history[0])
                    state_history[1]['turn'] = 1
            
            if current_turn not in state_history:
                if current_turn > 1 and (current_turn - 1) in state_history:
                    state_history[current_turn] = copy.deepcopy(state_history[current_turn - 1])
                elif 0 in state_history:
                    # Copy team preview from turn 0
                    state_history[current_turn] = copy.deepcopy(state_history[0])
                else:
                    state_history[current_turn] = new_state()
                state_history[current_turn]['turn'] = current_turn
            
            state = state_history[current_turn]

            def _record_pre_action_snapshot(
                player_id: str, action_type: str, action_target: str
            ) -> None:
                if pre_action_snapshots is None:
                    return
                pre_action_snapshots.append(
                    (
                        current_turn,
                        player_id,
                        action_type,
                        action_target,
                        copy.deepcopy(state),
                    )
                )
            
            # Switch
            if msg_type == 'switch' and len(parts) >= 5:
                player_id = parts[2].split(':')[0][:2]  # p1 or p2
                nickname = parts[2].split(':', 1)[1].strip() if ':' in parts[2] else ""
                pokemon_name = parts[3].split(',')[0].strip()
                hp_info = parts[4]
                _record_pre_action_snapshot(player_id, "switch", pokemon_name)

                if nickname:
                    self.nickname_to_species[(player_id, nickname)] = pokemon_name
                
                state[player_id]['active'] = pokemon_name
                
                # Parse HP
                hp_parts = hp_info.split()
                if hp_parts and '/' in hp_parts[0]:
                    current_hp, max_hp = hp_parts[0].split('/')
                    hp_percent = float(current_hp) / float(max_hp) if float(max_hp) > 0 else 0
                else:
                    hp_percent = 1.0
                
                # Initialize team entry if not exists (for battles without team preview)
                if pokemon_name not in state[player_id]['team']:
                    state[player_id]['team'][pokemon_name] = {
                        'hp_percent': 1.0,
                        'status': None,
                        'fainted': False,
                        'boosts': {},
                        'volatiles': [],
                        'moves_seen': [],
                        'item': None,
                        'ability': None,
                        'tera_type': None,
                        'tera_active': False,
                        'revealed': False
                    }
                
                # Update with actual HP and mark as revealed
                state[player_id]['team'][pokemon_name]['hp_percent'] = hp_percent
                state[player_id]['team'][pokemon_name]['status'] = hp_parts[1] if len(hp_parts) > 1 and hp_parts[1] not in ['fnt'] else None
                state[player_id]['team'][pokemon_name]['fainted'] = False
                state[player_id]['team'][pokemon_name]['revealed'] = True
                
                # Reset boosts and volatiles on switch
                state[player_id]['team'][pokemon_name]['boosts'] = {}
                state[player_id]['team'][pokemon_name]['volatiles'] = []
            
            # Damage
            elif msg_type == '-damage' and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                hp_info = parts[3]
                
                hp_parts = hp_info.split()
                if hp_parts:
                    if hp_parts[0] == '0' or hp_parts[0] == '0 fnt':
                        # Will be marked as fainted separately
                        pass
                    elif '/' in hp_parts[0]:
                        current_hp, max_hp = hp_parts[0].split('/')
                        hp_percent = float(current_hp) / float(max_hp) if float(max_hp) > 0 else 0
                        if pokemon_name not in state[player_id]['team']:
                            state[player_id]['team'][pokemon_name] = {
                                'hp_percent': 1.0,
                                'status': None,
                                'fainted': False,
                                'boosts': {},
                                'volatiles': [],
                                'moves_seen': [],
                                'item': None,
                                'ability': None,
                                'tera_type': None,
                                'tera_active': False
                            }
                        state[player_id]['team'][pokemon_name]['hp_percent'] = hp_percent
            
            # Faint
            elif msg_type == 'faint' and len(parts) >= 3:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                
                if pokemon_name not in state[player_id]['team']:
                    state[player_id]['team'][pokemon_name] = {
                        'hp_percent': 0,
                        'status': None,
                        'fainted': True,
                        'boosts': {},
                        'volatiles': [],
                        'moves_seen': [],
                        'item': None,
                        'ability': None,
                        'tera_type': None,
                        'tera_active': False
                    }
                
                state[player_id]['team'][pokemon_name]['fainted'] = True
                state[player_id]['team'][pokemon_name]['hp_percent'] = 0
                
                # Mark that this player had a faint this turn
                self.faints_this_turn[player_id] = True
            
            # Status
            elif msg_type == '-status' and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                status = parts[3].strip()
                
                if pokemon_name not in state[player_id]['team']:
                    state[player_id]['team'][pokemon_name] = {
                        'hp_percent': 1.0,
                        'status': status,
                        'fainted': False,
                        'boosts': {},
                        'volatiles': [],
                        'moves_seen': [],
                        'item': None,
                        'ability': None,
                        'tera_type': None,
                        'tera_active': False
                    }
                state[player_id]['team'][pokemon_name]['status'] = status

            # Item revealed/consumed
            elif msg_type in ('-item', 'item', '-enditem') and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                item_name = parts[3].strip().lower()
                if pokemon_name:
                    if pokemon_name not in state[player_id]['team']:
                        state[player_id]['team'][pokemon_name] = {
                            'hp_percent': 1.0,
                            'status': None,
                            'fainted': False,
                            'boosts': {},
                            'volatiles': [],
                            'moves_seen': [],
                            'item': None,
                            'ability': None,
                            'tera_type': None,
                            'tera_active': False,
                            'revealed': True
                        }
                    if item_name and item_name != 'unknown':
                        state[player_id]['team'][pokemon_name]['item'] = item_name

            # Ability revealed
            elif msg_type in ('-ability', 'ability') and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                ability_name = parts[3].strip().lower()
                if pokemon_name:
                    if pokemon_name not in state[player_id]['team']:
                        state[player_id]['team'][pokemon_name] = {
                            'hp_percent': 1.0,
                            'status': None,
                            'fainted': False,
                            'boosts': {},
                            'volatiles': [],
                            'moves_seen': [],
                            'item': None,
                            'ability': None,
                            'tera_type': None,
                            'tera_active': False,
                            'revealed': True
                        }
                    if ability_name and ability_name != 'unknown':
                        state[player_id]['team'][pokemon_name]['ability'] = ability_name

            elif msg_type == '-activate' and len(parts) >= 4 and 'ability:' in parts[3].lower():
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                ability_name = parts[3].split(':', 1)[1].strip().lower()
                if pokemon_name:
                    if pokemon_name not in state[player_id]['team']:
                        state[player_id]['team'][pokemon_name] = {
                            'hp_percent': 1.0,
                            'status': None,
                            'fainted': False,
                            'boosts': {},
                            'volatiles': [],
                            'moves_seen': [],
                            'item': None,
                            'ability': None,
                            'tera_type': None,
                            'tera_active': False,
                            'revealed': True
                        }
                    if ability_name and ability_name != 'unknown':
                        state[player_id]['team'][pokemon_name]['ability'] = ability_name

            # Terastallize
            elif msg_type in ('-terastallize', 'terastallize') and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                tera_type = parts[3].strip().lower()
                if pokemon_name:
                    if pokemon_name not in state[player_id]['team']:
                        state[player_id]['team'][pokemon_name] = {
                            'hp_percent': 1.0,
                            'status': None,
                            'fainted': False,
                            'boosts': {},
                            'volatiles': [],
                            'moves_seen': [],
                            'item': None,
                            'ability': None,
                            'tera_type': None,
                            'tera_active': False,
                            'revealed': True
                        }
                    state[player_id]['team'][pokemon_name]['tera_type'] = tera_type
                    state[player_id]['team'][pokemon_name]['tera_active'] = True
            
            # Boosts
            elif msg_type == '-boost' and len(parts) >= 5:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else state[player_id]['active'],
                )
                stat = parts[3].strip()
                amount = int(parts[4])
                
                if pokemon_name and pokemon_name in state[player_id]['team']:
                    if stat not in state[player_id]['team'][pokemon_name]['boosts']:
                        state[player_id]['team'][pokemon_name]['boosts'][stat] = 0
                    state[player_id]['team'][pokemon_name]['boosts'][stat] += amount
            
            elif msg_type == '-unboost' and len(parts) >= 5:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else state[player_id]['active'],
                )
                stat = parts[3].strip()
                amount = int(parts[4])
                
                if pokemon_name and pokemon_name in state[player_id]['team']:
                    if stat not in state[player_id]['team'][pokemon_name]['boosts']:
                        state[player_id]['team'][pokemon_name]['boosts'][stat] = 0
                    state[player_id]['team'][pokemon_name]['boosts'][stat] -= amount
            
            # Volatile conditions (Substitute, Leech Seed, etc.)
            elif msg_type == '-start' and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                condition = parts[3].strip().lower()
                
                # Filter side conditions vs volatile conditions
                side_conditions = ['reflect', 'lightscreen', 'auroraveil', 'tailwind', 'mist', 'safeguard']
                
                if any(sc in condition for sc in side_conditions):
                    # This is a side condition
                    state[player_id]['side_conditions'][condition] = True
                elif pokemon_name in state[player_id]['team']:
                    # Volatile condition on specific Pokemon
                    if condition not in state[player_id]['team'][pokemon_name]['volatiles']:
                        state[player_id]['team'][pokemon_name]['volatiles'].append(condition)
            
            # End volatile conditions
            elif msg_type == '-end' and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                condition = parts[3].strip().lower()
                
                if pokemon_name in state[player_id]['team']:
                    if condition in state[player_id]['team'][pokemon_name]['volatiles']:
                        state[player_id]['team'][pokemon_name]['volatiles'].remove(condition)
            
            # Side conditions end
            elif msg_type == '-sideend' and len(parts) >= 4:
                player_id = parts[2].split(':')[0]
                condition = parts[3].split(':')[-1].strip().lower()
                if condition in state[player_id]['side_conditions']:
                    del state[player_id]['side_conditions'][condition]
            
            # Entry Hazards
            elif msg_type == '-sidestart' and len(parts) >= 4:
                player_id = parts[2].split(':')[0]
                hazard = parts[3].split(':')[-1].strip().lower()
                
                # Distinguish hazards from side conditions
                hazards_list = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']
                
                if any(h in hazard for h in hazards_list):
                    # Entry hazard - can stack (spikes up to 3)
                    if hazard not in state[player_id]['hazards']:
                        state[player_id]['hazards'][hazard] = 0
                    state[player_id]['hazards'][hazard] += 1
                    # Cap spikes at 3
                    if 'spikes' in hazard:
                        state[player_id]['hazards'][hazard] = min(state[player_id]['hazards'][hazard], 3)
                    if 'toxicspikes' in hazard:
                        state[player_id]['hazards'][hazard] = min(state[player_id]['hazards'][hazard], 2)
            
            # Move used (track last move)
            elif msg_type == 'move' and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]
                pokemon_name = self._resolve_name(
                    player_id,
                    parts[2].split(':', 1)[1].strip() if ':' in parts[2] else parts[2].strip(),
                )
                move_name = parts[3].strip().lower()
                _record_pre_action_snapshot(player_id, "move", move_name)
                state[player_id]['last_move'] = move_name
                if pokemon_name:
                    if pokemon_name not in state[player_id]['team']:
                        state[player_id]['team'][pokemon_name] = {
                            'hp_percent': 1.0,
                            'status': None,
                            'fainted': False,
                            'boosts': {},
                            'volatiles': [],
                            'moves_seen': [],
                            'revealed': True,
                        }
                    if move_name not in state[player_id]['team'][pokemon_name]['moves_seen']:
                        state[player_id]['team'][pokemon_name]['moves_seen'].append(move_name)
            
            # Weather
            elif msg_type == '-weather' and len(parts) >= 3:
                weather = parts[2].strip()
                if weather != 'none':
                    state['weather'] = weather
                else:
                    state['weather'] = None
            
            # Terrain
            elif msg_type == '-terrain' and len(parts) >= 3:
                terrain = parts[2].strip()
                if terrain != 'none':
                    state['terrain'] = terrain
                else:
                    state['terrain'] = None
        
        return state_history
    
    def _infer_actions_from_log(self, log_lines: List[str]) -> List[Tuple[int, str, str, str]]:
        """
        Infer actions from battle log (no inputlog available).
        
        Actions are visible in the log as:
        - |move|p1a: Pokemon|Move Name|...
        - |switch|p1a: Pokemon|Species, L##|...
        
        Returns:
            List of (turn, player, action_type, action_target)
        """
        actions = []
        current_turn = 0
        
        for line in log_lines:
            if not line.strip() or not line.startswith('|'):
                continue
            
            parts = line.split('|')
            
            if len(parts) < 2:
                continue
            
            msg_type = parts[1]
            
            # Track turn
            if msg_type == 'turn':
                current_turn = int(parts[2])
                continue
            
            # Move action
            if msg_type == 'move' and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]  # p1 or p2
                move_name = parts[3].strip().lower()
                
                actions.append((current_turn, player_id, 'move', move_name))
            
            # Switch action
            elif msg_type == 'switch' and len(parts) >= 4:
                player_id = parts[2].split(':')[0][:2]  # p1 or p2
                pokemon_name = parts[3].split(',')[0].strip()
                
                # Use pokemon name as switch target (different from inputlog's slot number)
                actions.append((current_turn, player_id, 'switch', pokemon_name))
        
        return actions
    
    def _create_training_examples(self, 
                                 state_history: Dict[int, Dict],
                                 actions: List[Tuple[int, str, str, str]]) -> List[TrainingExample]:
        """
        Create training examples by matching states with actions.
        """
        examples = []
        
        # Track forced switches
        forced_switches: Set[Tuple[int, str]] = set()
        
        # Detect forced switches: any switch after a faint
        for turn, state_info in state_history.items():
            for player in ['p1', 'p2']:
                # Check if this player had a faint this turn
                fainted_this_turn = any(
                    poke_info.get('fainted', False) 
                    for poke_info in state_info[player]['team'].values()
                )
                
                if fainted_this_turn:
                    # Mark next switch for this player as forced
                    forced_switches.add((turn, player))
        
        # Create examples
        for turn, player, action_type, action_target in actions:
            if turn not in state_history:
                continue
            
            state_info = state_history[turn]
            player_state = state_info[player]
            opponent = 'p2' if player == 'p1' else 'p1'
            opp_state = state_info[opponent]
            
            # Check if this is a forced switch
            is_voluntary = (turn, player) not in forced_switches
            
            # Skip if no active pokemon
            if not player_state['active'] or not opp_state['active']:
                continue
            
            # Build complete team states
            my_team_states = []
            for poke_name, poke_data in player_state['team'].items():
                poke_state = PokemonState(
                    species=poke_name,
                    is_active=(poke_name == player_state['active']),
                    hp_percent=poke_data.get('hp_percent', 1.0),
                    status=poke_data.get('status', None),
                    fainted=poke_data.get('fainted', False),
                    stats_boosts=poke_data.get('boosts', {}).copy(),
                    volatile_conditions=poke_data.get('volatiles', []).copy(),
                    moves_seen=poke_data.get('moves_seen', []).copy(),
                    item=poke_data.get('item', None),
                    ability=poke_data.get('ability', None),
                    tera_type=poke_data.get('tera_type', None),
                    tera_active=bool(poke_data.get('tera_active', False)),
                    revealed=poke_data.get('revealed', True)  # Default True for backwards compat
                )
                my_team_states.append(poke_state)

            if action_type == 'move':
                for poke_state in my_team_states:
                    if poke_state.is_active:
                        move_name = (action_target or "").lower()
                        if move_name and move_name in poke_state.moves_seen:
                            poke_state.moves_seen = [
                                m for m in poke_state.moves_seen if m != move_name
                            ]
                        break
            
            opp_team_states = []
            for poke_name, poke_data in opp_state['team'].items():
                poke_state = PokemonState(
                    species=poke_name,
                    is_active=(poke_name == opp_state['active']),
                    hp_percent=poke_data.get('hp_percent', 1.0),
                    status=poke_data.get('status', None),
                    fainted=poke_data.get('fainted', False),
                    stats_boosts=poke_data.get('boosts', {}).copy(),
                    volatile_conditions=poke_data.get('volatiles', []).copy(),
                    moves_seen=poke_data.get('moves_seen', []).copy(),
                    item=poke_data.get('item', None),
                    ability=poke_data.get('ability', None),
                    tera_type=poke_data.get('tera_type', None),
                    tera_active=bool(poke_data.get('tera_active', False)),
                    revealed=poke_data.get('revealed', True)
                )
                opp_team_states.append(poke_state)
            
            # Create game state with complete information
            game_state = GameState(
                turn=turn,
                my_team=my_team_states,
                opp_team=opp_team_states,
                my_side_conditions=player_state['side_conditions'].copy(),
                opp_side_conditions=opp_state['side_conditions'].copy(),
                my_hazards=player_state['hazards'].copy(),
                opp_hazards=opp_state['hazards'].copy(),
                weather=state_info['weather'],
                terrain=state_info['terrain'],
                my_last_move=player_state['last_move'],
                opp_last_move=opp_state['last_move'],
                player=player
            )
            
            # Create training example
            example = TrainingExample(
                state=game_state,
                action_type=action_type,
                action_target=action_target,
                is_voluntary=is_voluntary,
                winner=(player == 'p1' and self.winner == 'p1')  # Simplified
            )
            
            examples.append(example)
        
        return examples


def extract_from_pokechamp_dataset(
    gamemode: str = 'gen9ou',
    min_elo: int = 1400,
    max_elo: int = 3000,
    max_battles: int = 10000,
    output_dir: str = "action_chunks",
    output_prefix: str = "action_data",
    chunk_size: int = 5000,
    filter_voluntary_only: bool = True
) -> Dict:
    """
    Extract training data from Pokéchamp Hugging Face dataset.
    
    Args:
        gamemode: Format to filter (e.g., 'gen9ou', 'gen9vgc2024')
        min_elo: Minimum Elo rating for battles
        max_elo: Maximum Elo rating for battles
        max_battles: Maximum number of battles to process
        output_dir: Output directory for chunked training data
        output_prefix: Base name for chunk files
        chunk_size: Number of examples per chunk
        filter_voluntary_only: Only include voluntary decisions
        
    Returns:
        Statistics dictionary
    """
    print("="*70)
    print("🎓 POKÉCHAMP TRAINING DATA EXTRACTION")
    print("="*70)
    print(f"Format: {gamemode}")
    print(f"Elo range: {min_elo}-{max_elo}")
    print(f"Max battles: {max_battles}")
    print(f"Filter voluntary only: {filter_voluntary_only}")
    print("="*70 + "\n")
    
    # Load dataset in streaming mode
    print("📦 Loading Pokéchamp dataset (streaming mode)...")
    dataset = load_dataset(
        "milkkarten/pokechamp",
        split="train",
        streaming=True
    )
    
    # Filter by gamemode and Elo
    # Note: Elo in dataset is a range string like "1400-1599" or "1800+"
    def parse_elo(elo_str):
        if '+' in elo_str:
            return int(elo_str.replace('+', ''))
        return int(elo_str.split('-')[0])
    
    filtered_dataset = dataset.filter(
        lambda x: (
            x['gamemode'].lower() == gamemode.lower() and
            min_elo <= parse_elo(x['elo']) <= max_elo
        )
    )
    
    extractor = TrainingDataExtractor()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    current_chunk = []
    chunk_index = 1
    
    stats = {
        'battles_processed': 0,
        'battles_failed': 0,
        'total_examples': 0,
        'voluntary_examples': 0,
        'forced_examples': 0,
        'moves': 0,
        'switches': 0,
        'chunks_written': 0
    }
    
    print(f"🎮 Processing battles...")
    
    # Process battles
    for i, battle in enumerate(filtered_dataset):
        if i >= max_battles:
            break
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{max_battles} - {stats['total_examples']} examples extracted")
        
        try:
            examples = extractor.extract_from_pokechamp_battle(battle)
            
            for ex in examples:
                if filter_voluntary_only and not ex.is_voluntary:
                    stats['forced_examples'] += 1
                    continue
                
                current_chunk.append(asdict(ex))
                stats['total_examples'] += 1
                
                if ex.is_voluntary:
                    stats['voluntary_examples'] += 1
                
                if ex.action_type == 'move':
                    stats['moves'] += 1
                else:
                    stats['switches'] += 1

                if chunk_size > 0 and len(current_chunk) >= chunk_size:
                    chunk_name = f"{output_prefix}_{chunk_index:05d}.json"
                    chunk_file = output_path / chunk_name
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        json.dump(current_chunk, f, indent=2)
                    stats['chunks_written'] += 1
                    current_chunk = []
                    chunk_index += 1
            
            stats['battles_processed'] += 1
            
        except Exception as e:
            if i < 10:  # Print first few errors for debugging
                print(f"❌ Error processing battle {i+1}: {e}")
            stats['battles_failed'] += 1
    
    if current_chunk:
        chunk_name = f"{output_prefix}_{chunk_index:05d}.json"
        chunk_file = output_path / chunk_name
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(current_chunk, f, indent=2)
        stats['chunks_written'] += 1

    print("\n" + "="*70)
    print("✅ EXTRACTION COMPLETE")
    print("="*70)
    print(f"Battles processed: {stats['battles_processed']}")
    print(f"Battles failed: {stats['battles_failed']}")
    print(f"Total examples: {stats['total_examples']}")
    print(f"  - Voluntary decisions: {stats['voluntary_examples']}")
    print(f"  - Forced switches (filtered): {stats['forced_examples']}")
    print(f"  - Moves: {stats['moves']}")
    print(f"  - Voluntary switches: {stats['switches']}")
    print(f"\nTraining data saved to: {output_dir}")
    print(f"Chunks written: {stats['chunks_written']}")
    print("="*70)
    
    return stats



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract action training data from Pokechamp dataset.")
    parser.add_argument("--gamemode", default="gen9ou")
    parser.add_argument("--min_elo", type=int, default=1500)
    parser.add_argument("--max_elo", type=int, default=3000)
    parser.add_argument("--max_battles", type=int, default=10000)
    parser.add_argument("--output_dir", default="action_chunks")
    parser.add_argument("--output_prefix", default="action_data")
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--voluntary_only", action="store_true", default=True)

    args = parser.parse_args()

    extract_from_pokechamp_dataset(
        gamemode=args.gamemode,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        max_battles=args.max_battles,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        chunk_size=args.chunk_size,
        filter_voluntary_only=args.voluntary_only,
    )
