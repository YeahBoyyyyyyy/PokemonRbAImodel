"""
Extract OU Win Prediction Training Data from Pokéchamp Dataset

This script extracts game states paired with the actual winner for training
a neural network to predict which player will win based on the current state.

Differences from Random Battle version:
- Format: gen9ou (team preview with 6v6 known from start)
- Team visibility: Both teams fully revealed from turn 1 (team preview)
- Higher Elo threshold: 1500+ (more competitive)

Output format: [(state, winner), ...]
where winner = 1 if we win, 0 if opponent wins
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class PokemonState:
    """State of a single Pokemon"""
    species: str
    hp_percent: float
    status: Optional[str]
    fainted: bool
    boosts: Dict[str, int]
    volatiles: List[str]
    revealed: bool  # Whether this Pokemon has been revealed (switched in)


@dataclass
class GameState:
    """Complete state of the game at a given turn"""
    turn: int
    our_team: Dict[str, PokemonState]
    opponent_team: Dict[str, PokemonState]
    our_active: Optional[str]
    opponent_active: Optional[str]
    our_side_conditions: Dict[str, int]
    opponent_side_conditions: Dict[str, int]
    field_conditions: Dict[str, str]
    our_last_move: Optional[str]
    opponent_last_move: Optional[str]


@dataclass
class WinPredictionExample:
    """Training example for win prediction"""
    state: GameState
    winner: int  # 1 if we win, 0 if opponent wins
    turn: int
    perspective: str  # 'p1' or 'p2'


def example_to_json(example: WinPredictionExample) -> Dict[str, object]:
    """
    Convert a WinPredictionExample to a JSON-serializable dict.
    """
    return {
        'turn': example.turn,
        'perspective': example.perspective,
        'winner': example.winner,
        'state': {
            'turn': example.state.turn,
            'our_team': {
                name: asdict(poke) for name, poke in example.state.our_team.items()
            },
            'opponent_team': {
                name: asdict(poke) for name, poke in example.state.opponent_team.items()
            },
            'our_active': example.state.our_active,
            'opponent_active': example.state.opponent_active,
            'our_side_conditions': example.state.our_side_conditions,
            'opponent_side_conditions': example.state.opponent_side_conditions,
            'field_conditions': example.state.field_conditions,
            'our_last_move': example.state.our_last_move,
            'opponent_last_move': example.state.opponent_last_move
        }
    }


def write_chunk(chunk: List[Dict[str, object]],
                output_dir: Path,
                base_name: str,
                chunk_idx: int,
                pretty_json: bool) -> Path:
    """
    Write a chunk of examples to disk as a JSON array.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{base_name}_{chunk_idx:05d}.json"
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(chunk, f, indent=2 if pretty_json else None)
    return out_path


class OUWinRateDataExtractor:
    """Extract win prediction training data from OU replays"""
    
    def __init__(self):
        self.stats = {
            'total_battles': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_examples': 0,
            'p1_wins': 0,
            'p2_wins': 0,
            'p1_examples': 0,
            'p2_examples': 0
        }
    
    def extract_from_pokechamp_battle(self, battle_data: dict) -> Tuple[List[WinPredictionExample], List[WinPredictionExample]]:
        """
        Extract win prediction examples from both perspectives.
        
        Returns:
            Tuple of (p1_examples, p2_examples)
        """
        try:
            log = battle_data.get('text', '')
            if not log:
                return [], []
            
            # Parse battle log to get state history and winner
            state_history, winner = self._parse_log_to_states_and_winner(log)
            
            if not state_history or winner is None:
                return [], []
            
            # Create examples from both perspectives
            p1_examples = self._create_winrate_examples(state_history, winner, perspective='p1')
            p2_examples = self._create_winrate_examples(state_history, winner, perspective='p2')
            
            self.stats['p1_examples'] += len(p1_examples)
            self.stats['p2_examples'] += len(p2_examples)
            
            return p1_examples, p2_examples
            
        except Exception as e:
            print(f"Error extracting from battle: {e}")
            return [], []
    
    def _parse_log_to_states_and_winner(self, log: str) -> Tuple[Dict[int, Dict], Optional[str]]:
        """
        Parse battle log to create state history and determine winner.
        
        Returns:
            Tuple of (state_history, winner) where winner is 'p1' or 'p2'
        """
        lines = log.strip().split('\n')
        
        # FIRST PASS: Parse team preview and player usernames
        team_preview_p1 = {}
        team_preview_p2 = {}
        player_usernames = {}  # Map username -> player_id
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 3:
                continue
            
            command = parts[1]
            
            # Parse player names
            if command == 'player':
                player_id = parts[2]  # 'p1' or 'p2'
                username = parts[3]
                player_usernames[username] = player_id
            
            # Parse team preview
            elif command == 'poke':
                player_id = parts[2]
                pokemon_name = parts[3].split(',')[0].strip()
                
                if player_id == 'p1':
                    team_preview_p1[pokemon_name] = {
                        'hp_percent': 1.0,
                        'status': None,
                        'fainted': False,
                        'boosts': {},
                        'volatiles': [],
                        'revealed': False
                    }
                elif player_id == 'p2':
                    team_preview_p2[pokemon_name] = {
                        'hp_percent': 1.0,
                        'status': None,
                        'fainted': False,
                        'boosts': {},
                        'volatiles': [],
                        'revealed': False
                    }
        
        # Initialize state history with turn 0 containing team preview
        state_history = {
            0: {
                'p1': {
                    'team': team_preview_p1.copy(),
                    'active': None,
                    'side_conditions': {},
                    'last_move': None
                },
                'p2': {
                    'team': team_preview_p2.copy(),
                    'active': None,
                    'side_conditions': {},
                    'last_move': None
                },
                'field': {'weather': None, 'terrain': None}
            }
        }
        
        current_turn = 0
        winner = None
        
        # SECOND PASS: Parse battle events
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 2:
                continue
            
            command = parts[1]
            
            # Track turn number
            if command == 'turn':
                current_turn = int(parts[2])
                # Copy previous state including team preview
                if current_turn - 1 in state_history:
                    state_history[current_turn] = self._deep_copy_state(state_history[current_turn - 1])
                else:
                    state_history[current_turn] = self._deep_copy_state(state_history[0])
            
            # Parse winner
            elif command == 'win':
                winner_username = parts[2]
                winner = player_usernames.get(winner_username)
            
            # Parse switch
            elif command == 'switch' or command == 'drag':
                player_slot = parts[2]
                player_id = player_slot.split(':')[0][:2]
                pokemon_data = parts[3]
                pokemon_name = pokemon_data.split(',')[0].strip()
                hp_data = parts[4] if len(parts) > 4 else '100/100'
                
                hp_percent = self._parse_hp(hp_data)
                status = self._parse_status(hp_data)
                
                if current_turn not in state_history:
                    state_history[current_turn] = self._deep_copy_state(state_history.get(current_turn - 1, state_history[0]))
                
                # Update Pokemon data (should already exist from team preview)
                if pokemon_name in state_history[current_turn][player_id]['team']:
                    state_history[current_turn][player_id]['team'][pokemon_name]['hp_percent'] = hp_percent
                    state_history[current_turn][player_id]['team'][pokemon_name]['status'] = status
                    state_history[current_turn][player_id]['team'][pokemon_name]['revealed'] = True
                
                state_history[current_turn][player_id]['active'] = pokemon_name
            
            # Parse faint
            elif command == 'faint':
                player_slot = parts[2]
                player_id = player_slot.split(':')[0][:2]
                pokemon_name = player_slot.split(': ')[1].split(',')[0].strip()
                
                if current_turn in state_history and pokemon_name in state_history[current_turn][player_id]['team']:
                    state_history[current_turn][player_id]['team'][pokemon_name]['fainted'] = True
                    state_history[current_turn][player_id]['team'][pokemon_name]['hp_percent'] = 0.0
            
            # Parse HP changes
            elif command == '-damage' or command == '-heal':
                player_slot = parts[2]
                player_id = player_slot.split(':')[0][:2]
                pokemon_name = player_slot.split(': ')[1].split(',')[0].strip()
                hp_data = parts[3]
                
                hp_percent = self._parse_hp(hp_data)
                status = self._parse_status(hp_data)
                
                if current_turn in state_history and pokemon_name in state_history[current_turn][player_id]['team']:
                    state_history[current_turn][player_id]['team'][pokemon_name]['hp_percent'] = hp_percent
                    if status:
                        state_history[current_turn][player_id]['team'][pokemon_name]['status'] = status
            
            # Parse status
            elif command == '-status':
                player_slot = parts[2]
                player_id = player_slot.split(':')[0][:2]
                pokemon_name = player_slot.split(': ')[1].split(',')[0].strip()
                status = parts[3]
                
                if current_turn in state_history and pokemon_name in state_history[current_turn][player_id]['team']:
                    state_history[current_turn][player_id]['team'][pokemon_name]['status'] = status
            
            # Parse boosts
            elif command == '-boost' or command == '-unboost':
                player_slot = parts[2]
                player_id = player_slot.split(':')[0][:2]
                pokemon_name = player_slot.split(': ')[1].split(',')[0].strip()
                stat = parts[3]
                amount = int(parts[4])
                
                if command == '-unboost':
                    amount = -amount
                
                if current_turn in state_history and pokemon_name in state_history[current_turn][player_id]['team']:
                    boosts = state_history[current_turn][player_id]['team'][pokemon_name]['boosts']
                    boosts[stat] = boosts.get(stat, 0) + amount
            
            # Parse volatiles
            elif command == '-start':
                player_slot = parts[2]
                player_id = player_slot.split(':')[0][:2]
                pokemon_name = player_slot.split(': ')[1].split(',')[0].strip()
                volatile = parts[3]
                
                if current_turn in state_history and pokemon_name in state_history[current_turn][player_id]['team']:
                    volatiles = state_history[current_turn][player_id]['team'][pokemon_name]['volatiles']
                    if volatile not in volatiles:
                        volatiles.append(volatile)
            
            # Parse side conditions
            elif command == '-sidestart':
                player_slot = parts[2]
                player_id = player_slot.split(':')[0][:2]
                condition = parts[3]
                
                if current_turn in state_history:
                    state_history[current_turn][player_id]['side_conditions'][condition] = \
                        state_history[current_turn][player_id]['side_conditions'].get(condition, 0) + 1
            
            # Parse weather
            elif command == '-weather':
                weather = parts[2] if parts[2] != 'none' else None
                if current_turn in state_history:
                    state_history[current_turn]['field']['weather'] = weather
            
            # Parse terrain
            elif command == '-fieldstart':
                terrain = parts[2]
                if current_turn in state_history:
                    state_history[current_turn]['field']['terrain'] = terrain
            
            # Parse moves (for last_move tracking)
            elif command == 'move':
                player_slot = parts[2]
                player_id = player_slot.split(':')[0][:2]
                move = parts[3]
                
                if current_turn in state_history:
                    state_history[current_turn][player_id]['last_move'] = move
        
        return state_history, winner
    
    def _create_winrate_examples(self, state_history: Dict[int, Dict], 
                                  winner: str, perspective: str) -> List[WinPredictionExample]:
        """
        Create win prediction examples from a perspective.
        
        Args:
            state_history: Turn-by-turn game states
            winner: 'p1' or 'p2'
            perspective: 'p1' or 'p2' - which player's perspective
        """
        examples = []
        
        # Determine if this perspective won
        we_won = 1 if winner == perspective else 0
        
        for turn in sorted(state_history.keys()):
            if turn == 0:
                continue  # Skip turn 0
            
            state = state_history[turn]
            
            # Determine "us" and "opponent" based on perspective
            if perspective == 'p1':
                our_data = state['p1']
                opponent_data = state['p2']
            else:
                our_data = state['p2']
                opponent_data = state['p1']
            
            # Create OUR team (all Pokemon visible from team preview)
            our_team = {}
            for poke_name, poke_data in our_data['team'].items():
                our_team[poke_name] = PokemonState(
                    species=poke_name,
                    hp_percent=poke_data['hp_percent'],
                    status=poke_data['status'],
                    fainted=poke_data['fainted'],
                    boosts=poke_data['boosts'].copy(),
                    volatiles=poke_data['volatiles'].copy(),
                    revealed=poke_data['revealed']
                )
            
            # Create OPPONENT team (all Pokemon visible from team preview)
            opponent_team = {}
            for poke_name, poke_data in opponent_data['team'].items():
                opponent_team[poke_name] = PokemonState(
                    species=poke_name,
                    hp_percent=poke_data['hp_percent'],
                    status=poke_data['status'],
                    fainted=poke_data['fainted'],
                    boosts=poke_data['boosts'].copy(),
                    volatiles=poke_data['volatiles'].copy(),
                    revealed=poke_data['revealed']
                )
            
            game_state = GameState(
                turn=turn,
                our_team=our_team,
                opponent_team=opponent_team,
                our_active=our_data['active'],
                opponent_active=opponent_data['active'],
                our_side_conditions=our_data['side_conditions'].copy(),
                opponent_side_conditions=opponent_data['side_conditions'].copy(),
                field_conditions=state['field'].copy(),
                our_last_move=our_data.get('last_move'),
                opponent_last_move=opponent_data.get('last_move')
            )
            
            example = WinPredictionExample(
                state=game_state,
                winner=we_won,
                turn=turn,
                perspective=perspective
            )
            
            examples.append(example)
        
        return examples
    
    def _deep_copy_state(self, state: Dict) -> Dict:
        """Deep copy a state dictionary"""
        return {
            'p1': {
                'team': {
                    name: {
                        'hp_percent': data['hp_percent'],
                        'status': data['status'],
                        'fainted': data['fainted'],
                        'boosts': data['boosts'].copy(),
                        'volatiles': data['volatiles'].copy(),
                        'revealed': data.get('revealed', False)
                    }
                    for name, data in state['p1']['team'].items()
                },
                'active': state['p1']['active'],
                'side_conditions': state['p1']['side_conditions'].copy(),
                'last_move': state['p1'].get('last_move')
            },
            'p2': {
                'team': {
                    name: {
                        'hp_percent': data['hp_percent'],
                        'status': data['status'],
                        'fainted': data['fainted'],
                        'boosts': data['boosts'].copy(),
                        'volatiles': data['volatiles'].copy(),
                        'revealed': data.get('revealed', False)
                    }
                    for name, data in state['p2']['team'].items()
                },
                'active': state['p2']['active'],
                'side_conditions': state['p2']['side_conditions'].copy(),
                'last_move': state['p2'].get('last_move')
            },
            'field': state['field'].copy()
        }
    
    def _parse_hp(self, hp_string: str) -> float:
        """Parse HP percentage from string like '85/100' or '0 fnt'"""
        if 'fnt' in hp_string.lower():
            return 0.0
        
        # Remove status if present
        hp_part = hp_string.split()[0]
        
        if '/' in hp_part:
            current, maximum = hp_part.split('/')
            try:
                return float(current) / float(maximum)
            except (ValueError, ZeroDivisionError):
                return 1.0
        
        return 1.0
    
    def _parse_status(self, hp_string: str) -> Optional[str]:
        """Parse status condition from HP string"""
        parts = hp_string.split()
        if len(parts) > 1 and parts[1] not in ['fnt']:
            return parts[1]
        return None
    
    def parse_elo(self, elo_str: str) -> int:
        """Parse Elo rating from string format"""
        if not elo_str:
            return 0
        
        # Handle "1800+" format
        if '+' in elo_str:
            return int(elo_str.replace('+', ''))
        
        # Handle "1400-1599" format - take lower bound
        if '-' in elo_str:
            return int(elo_str.split('-')[0])
        
        # Handle direct integer
        try:
            return int(elo_str)
        except ValueError:
            return 0


def main():
    """Main extraction function"""
    # Configuration
    FORMAT = 'gen9ou'
    MIN_ELO = 1600
    MAX_BATTLES = 20000
    OUTPUT_DIR = Path('ou_winrate_chunks_replay_data')
    OUTPUT_BASE_NAME = 'ou_winrate_data'
    EXAMPLES_PER_FILE = 40000  # Write a new file every N examples
    PRETTY_JSON = False  # Set True if you want indented output
   
    print(" OU WIN PREDICTION DATA EXTRACTION")
    print(f"Format: {FORMAT}")
    print(f"Elo range: {MIN_ELO}-3000")
    print(f"Max battles: {MAX_BATTLES}")
    print(f"Examples per file: {EXAMPLES_PER_FILE}")
    print(f"Output directory: {OUTPUT_DIR} (base: {OUTPUT_BASE_NAME})")
    print("Extracting from BOTH perspectives (2x data)")
    print("Team preview: Full 6v6 visible from turn 1")
    print()
    
    # Load dataset
    print(" Loading Pokéchamp dataset (streaming mode)...")
    dataset = load_dataset("milkkarten/pokechamp", split="train", streaming=True)
    
    # Initialize extractor
    extractor = OUWinRateDataExtractor()
    current_chunk: List[Dict[str, object]] = []
    chunk_idx = 1
    chunks_written = 0
    output_dir = OUTPUT_DIR

    def flush_chunk():
        """Write the current chunk to disk if it has data."""
        nonlocal chunk_idx, chunks_written
        if not current_chunk:
            return
        out_path = write_chunk(
            chunk=current_chunk,
            output_dir=output_dir,
            base_name=OUTPUT_BASE_NAME,
            chunk_idx=chunk_idx,
            pretty_json=PRETTY_JSON
        )
        print(f"Wrote {len(current_chunk)} examples to {out_path}")
        current_chunk.clear()
        chunk_idx += 1
        chunks_written += 1
    
    # Process battles
    print(" Processing battles...")
    battles_processed = 0
    
    for battle in tqdm(dataset, total=MAX_BATTLES, desc="Extracting"):
        # Filter by format and Elo
        if battle.get('gamemode') != FORMAT:
            continue
        
        elo = extractor.parse_elo(battle.get('elo', '0'))
        if elo < MIN_ELO:
            continue
        
        extractor.stats['total_battles'] += 1
        
        # Extract examples from both perspectives
        p1_examples, p2_examples = extractor.extract_from_pokechamp_battle(battle)
        
        if p1_examples or p2_examples:
            extractor.stats['successful_extractions'] += 1
        else:
            extractor.stats['failed_extractions'] += 1

        for ex in p1_examples + p2_examples:
            extractor.stats['total_examples'] += 1
            if ex.perspective == 'p1' and ex.winner == 1:
                extractor.stats['p1_wins'] += 1
            elif ex.perspective == 'p2' and ex.winner == 1:
                extractor.stats['p2_wins'] += 1

            json_example = example_to_json(ex)
            current_chunk.append(json_example)

            if len(current_chunk) >= EXAMPLES_PER_FILE:
                flush_chunk()
        
        battles_processed += 1
        if battles_processed >= MAX_BATTLES:
            break
    
    # Flush any remaining examples
    flush_chunk()
    
    # Print statistics
    p1_wins = extractor.stats['p1_wins']
    p2_wins = extractor.stats['p2_wins']
    
    print("\n" + "="*60)
    print(" EXTRACTION STATISTICS")
    print("="*60)
    print(f"Total battles processed: {extractor.stats['total_battles']}")
    print(f"Successful extractions: {extractor.stats['successful_extractions']}")
    print(f"Failed extractions: {extractor.stats['failed_extractions']}")
    print(f"\nTotal examples: {extractor.stats['total_examples']}")
    print(f"  - P1 perspective: {extractor.stats['p1_examples']}")
    print(f"  - P2 perspective: {extractor.stats['p2_examples']}")
    print(f"\nWin distribution:")
    if extractor.stats['p1_examples'] > 0:
        print(f"  - P1 wins: {extractor.stats['p1_wins']} ({100*p1_wins/extractor.stats['p1_examples']:.1f}%)")
    else:
        print(f"  - P1 wins: 0 (N/A)")
    if extractor.stats['p2_examples'] > 0:
        print(f"  - P2 wins: {extractor.stats['p2_wins']} ({100*p2_wins/extractor.stats['p2_examples']:.1f}%)")
    else:
        print(f"  - P2 wins: 0 (N/A)")
    if extractor.stats['total_examples'] == 0:
        print("\nNo examples were produced; check filters and dataset access.")
    else:
        print(f"\nOutput saved to directory: {OUTPUT_DIR} ({chunks_written} file(s) written)")
    print("="*60)


if __name__ == "__main__":
    main()
