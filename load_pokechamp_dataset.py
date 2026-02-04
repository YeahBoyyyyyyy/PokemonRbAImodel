"""
Load and filter the Pokéchamp dataset from Hugging Face.
Dataset: https://huggingface.co/datasets/milkkarten/pokechamp
Contains 2.1M battles (1.9M train, 213K test) across 37+ formats.
"""

import os
from datasets import load_dataset
from typing import Optional, List, Dict, Any
import json
from pathlib import Path


class PokechampDatasetLoader:
    """Load and filter the Pokéchamp dataset from Hugging Face."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Optional directory to cache the dataset (default: ~/.cache/huggingface)
        """
        self.cache_dir = cache_dir
        self.dataset = None
        
    def load_dataset(self, split: str = "train", streaming: bool = False):
        """
        Load the full dataset from Hugging Face.
        
        Args:
            split: Dataset split to load ("train" or "test")
            streaming: If True, stream the dataset without downloading everything
        
        Returns:
            The loaded dataset
        """
        print(f"Loading Pokéchamp dataset (split: {split})...")
        if streaming:
            print("Using streaming mode - no download required!")
        else:
            print("Downloading full dataset (~5.57 GB)...")
            print("This may take a while on first run, but will be cached locally.")
        
        self.dataset = load_dataset(
            "milkkarten/pokechamp",
            split=split,
            cache_dir=self.cache_dir,
            streaming=streaming
        )
        
        if not streaming:
            print(f"Dataset loaded! Total battles: {len(self.dataset):,}")
        else:
            print("Dataset streaming initialized!")
        
        return self.dataset
    
    def filter_dataset(
        self,
        gamemode: Optional[str] = None,
        elo_min: Optional[int] = None,
        elo_max: Optional[int] = None,
        min_month: Optional[str] = None,
        max_month: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """
        Filter the dataset by various criteria.
        
        Args:
            gamemode: Filter by format (e.g., "gen9ou", "gen9randombattle")
            elo_min: Minimum Elo rating
            elo_max: Maximum Elo rating
            min_month: Minimum month (e.g., "January2025")
            max_month: Maximum month (e.g., "March2025")
            limit: Maximum number of battles to return
        
        Returns:
            Filtered dataset
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded! Call load_dataset() first.")
        
        print("Applying filters...")
        filters = []
        
        if gamemode:
            filters.append(f"gamemode={gamemode}")
        if elo_min:
            filters.append(f"elo>={elo_min}")
        if elo_max:
            filters.append(f"elo<={elo_max}")
        if min_month:
            filters.append(f"month>={min_month}")
        if max_month:
            filters.append(f"month<={max_month}")
        
        print(f"Filters: {', '.join(filters) if filters else 'None'}")
        
        filtered = self.dataset
        
        # Apply filters - combine into single filter for streaming datasets
        def combined_filter(example):
            # Gamemode filter
            if gamemode:
                mode = example.get('gamemode', '').lower()
                if mode != gamemode.lower():
                    return False
            
            # Elo filter
            if elo_min is not None or elo_max is not None:
                elo = example.get('elo', 0)
                if isinstance(elo, str):
                    # Handle Elo ranges like "1600-1799"
                    try:
                        elo = int(elo.split('-')[0])
                    except:
                        return False
                if elo_min is not None and elo < elo_min:
                    return False
                if elo_max is not None and elo > elo_max:
                    return False
            
            return True
        
        if gamemode or elo_min is not None or elo_max is not None:
            filtered = filtered.filter(combined_filter)
        
        # Apply limit
        if limit:
            if hasattr(filtered, 'take'):  # Streaming dataset
                filtered = filtered.take(limit)
            else:
                filtered = filtered.select(range(min(limit, len(filtered))))
        
        print(f"Filtered dataset ready!")
        return filtered
    
    def get_available_formats(self, sample_size: int = 10000):
        """
        Get list of all available game formats in the dataset.
        
        Args:
            sample_size: Number of battles to sample (None for all)
        
        Returns:
            Dictionary with format counts
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded! Call load_dataset() first.")
        
        print(f"Analyzing formats (sampling {sample_size} battles)...")
        formats = {}
        
        sample = self.dataset.take(sample_size) if hasattr(self.dataset, 'take') else self.dataset[:sample_size]
        
        for battle in sample:
            fmt = battle.get('gamemode', 'unknown')
            formats[fmt] = formats.get(fmt, 0) + 1
        
        # Sort by count
        formats = dict(sorted(formats.items(), key=lambda x: x[1], reverse=True))
        return formats
    
    def get_battle_log(self, battle: Dict[str, Any]) -> str:
        """
        Extract the battle log from a battle example.
        
        Args:
            battle: A battle example from the dataset
        
        Returns:
            The battle log as a string
        """
        # The battle log is typically stored in a field like 'log' or 'battle_log'
        return battle.get('log', battle.get('battle_log', ''))
    
    def save_filtered_battles(
        self,
        output_path: str,
        battles: Any,
        format: str = "json"
    ):
        """
        Save filtered battles to disk.
        
        Args:
            output_path: Path to save the battles
            battles: Filtered dataset or iterable of battles
            format: Output format ("json" or "jsonl")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving battles to {output_path}...")
        
        if format == "json":
            # Save as single JSON array
            battles_list = list(battles) if not isinstance(battles, list) else battles
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(battles_list, f, indent=2)
            print(f"Saved {len(battles_list):,} battles to {output_path}")
        
        elif format == "jsonl":
            # Save as JSON lines (one battle per line)
            count = 0
            with open(output_path, 'w', encoding='utf-8') as f:
                for battle in battles:
                    f.write(json.dumps(battle) + '\n')
                    count += 1
            print(f"Saved {count:,} battles to {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    """Example usage of the Pokéchamp dataset loader."""
    
    # Initialize loader
    loader = PokechampDatasetLoader()
    
    # Option 1: Load full dataset (downloads ~5.57 GB)
    # dataset = loader.load_dataset(split="train", streaming=False)
    
    # Option 2: Use streaming mode (no download, processes on-the-fly)
    print("=" * 80)
    print("Loading Pokéchamp Dataset - Streaming Mode")
    print("=" * 80)
    dataset = loader.load_dataset(split="train", streaming=True)
    
    # Get available formats
    print("\n" + "=" * 80)
    print("Analyzing available formats...")
    print("=" * 80)
    formats = loader.get_available_formats(sample_size=5000)
    print("\nTop 20 formats:")
    for i, (fmt, count) in enumerate(list(formats.items())[:20], 1):
        print(f"{i:2d}. {fmt:30s} - {count:,} battles (in sample)")
    
    # Filter for Gen 9 OU battles with high Elo
    print("\n" + "=" * 80)
    print("Filtering for Gen 9 OU battles (Elo 1400+)")
    print("=" * 80)
    filtered = loader.filter_dataset(
        gamemode="gen9ou",
        elo_min=1400,
        limit=3000  # Get first 3000 battles matching criteria
    )
    
    # Save filtered battles
    output_dir = Path("Pokechamp_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save sample battles
    sample_battles = []
    for i, battle in enumerate(filtered):
        if i >= 100:  # Save first 100 for inspection
            break
        sample_battles.append(battle)
    
    if sample_battles:
        loader.save_filtered_battles(
            output_path=output_dir / "gen9ou_high_elo_sample.json",
            battles=sample_battles,
            format="json"
        )
        
        # Print example battle
        print("\n" + "=" * 80)
        print("Example battle structure:")
        print("=" * 80)
        example = sample_battles[0]
        print(f"Keys: {list(example.keys())}")
        print(f"\nGamemode: {example.get('gamemode')}")
        print(f"Elo: {example.get('elo')}")
        
        # Print first 500 characters of battle log
        log = loader.get_battle_log(example)
        if log:
            print(f"\nBattle log preview (first 500 chars):")
            print(log[:500])
    
    print("\n" + "=" * 80)
    print("Dataset loading complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run this script to download/analyze the dataset")
    print("2. Use extract_pokechamp_training_data.py to convert battles to training data")
    print("3. Train your neural network on millions of examples!")


if __name__ == "__main__":
    main()
