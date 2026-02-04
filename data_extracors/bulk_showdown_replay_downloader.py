"""
Bulk Replay Downloader for Pokémon Showdown
============================================

Downloads as many replays as possible with concurrent requests,
progress tracking, and automatic retry logic.
"""

import requests
import json
import time
from typing import List, Dict, Optional, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
from collections import defaultdict

class BulkReplayDownloader:
    """
    High-performance bulk replay downloader with concurrent downloads.
    """
    
    BASE_URL = "https://replay.pokemonshowdown.com"
    SEARCH_URL = f"{BASE_URL}/search.json"
    
    # Gen 9 Singles formats only
            
    POPULAR_FORMATS = [
        ##"gen9randombattle", # Random Battle
        "gen9ou",           # OverUsed
        "gen9uu",           # UnderUsed
        "gen9monotype",     # Monotype
    ]
    
    def __init__(self, 
                 output_dir: str = "replays_data",
                 max_workers: int = 10,
                 retry_attempts: int = 3,
                 delay_between_requests: float = 0.1):
        """
        Initialize the bulk downloader.
        
        Args:
            output_dir: Directory to save replays
            max_workers: Number of concurrent download threads
            retry_attempts: Number of retries for failed downloads
            delay_between_requests: Delay between requests (seconds)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.delay = delay_between_requests
        
        # Statistics
        self.stats = {
            'total_downloaded': 0,
            'total_failed': 0,
            'total_skipped': 0,
            'by_format': defaultdict(int),
            'start_time': None,
        }
        
        # Thread-safe lock for statistics
        self.stats_lock = threading.Lock()
        
        # Track seen replay IDs to avoid duplicates
        self.seen_ids: Set[str] = set()
        self._load_existing_replays()
    
    def _load_existing_replays(self):
        """Load IDs of already downloaded replays."""
        existing_files = list(self.output_dir.glob("*.json"))
        self.seen_ids = {f.stem for f in existing_files}
        print(f"Found {len(self.seen_ids)} existing replays")
    
    def _update_stats(self, stat_name: str, increment: int = 1, format_name: str = None):
        """Thread-safe statistics update."""
        with self.stats_lock:
            if stat_name in self.stats:
                self.stats[stat_name] += increment
            if format_name:
                self.stats['by_format'][format_name] += increment
    
    def search_replays(self, format: str, page: int) -> List[Dict]:
        """
        Search for replays with retry logic.
        
        Args:
            format: Battle format
            page: Page number
            
        Returns:
            List of replay metadata
        """
        params = {'format': format, 'page': page}
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(
                    self.SEARCH_URL, 
                    params=params, 
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                return data if data else []
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    print(f"Search failed for {format} page {page}: {e}")
                    return []
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return []
    
    def download_replay(self, replay_id: str, format_name: str) -> bool:
        """
        Download a single replay with retry logic.
        
        Args:
            replay_id: Replay ID
            format_name: Format name for statistics
            
        Returns:
            True if successful, False otherwise
        """
        # Check if already downloaded
        if replay_id in self.seen_ids:
            self._update_stats('total_skipped')
            return False
        
        filepath = self.output_dir / f"{replay_id}.json"
        
        # Double-check file existence
        if filepath.exists():
            self.seen_ids.add(replay_id)
            self._update_stats('total_skipped')
            return False
        
        url = f"{self.BASE_URL}/{replay_id}.json"
        
        for attempt in range(self.retry_attempts):
            try:
                time.sleep(self.delay)  # Rate limiting
                
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Validate replay data
                if not data.get('log'):
                    print(f"Warning: Invalid replay data for {replay_id}")
                    self._update_stats('total_failed', format_name=format_name)
                    return False
                
                # Save to disk
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                self.seen_ids.add(replay_id)
                self._update_stats('total_downloaded', format_name=format_name)
                return True
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # Replay doesn't exist, don't retry
                    self._update_stats('total_failed', format_name=format_name)
                    return False
                elif attempt == self.retry_attempts - 1:
                    self._update_stats('total_failed', format_name=format_name)
                    return False
                time.sleep(1 * (attempt + 1))
                
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    print(f"Failed to download {replay_id}: {e}")
                    self._update_stats('total_failed', format_name=format_name)
                    return False
                time.sleep(1 * (attempt + 1))
        
        return False
    
    def download_format_replays(self, 
                               format: str, 
                               max_pages: int = 1000,
                               rating_min: int = 0) -> int:
        """
        Download all available replays for a specific format.
        
        Args:
            format: Battle format
            max_pages: Maximum number of pages to search
            rating_min: Minimum rating filter
            
        Returns:
            Number of replays downloaded
        """
        print(f"\n{'='*70}")
        print(f"Downloading replays for: {format}")
        print(f"{'='*70}")
        
        downloaded_count = 0
        replay_tasks = []
        
        # Collect all replay IDs from pages
        for page in range(1, max_pages + 1):
            search_results = self.search_replays(format, page)
            
            if not search_results:
                print(f" No more replays found at page {page}")
                break
            
            for replay_meta in search_results:
                replay_id = replay_meta.get('id')
                if not replay_id or replay_id in self.seen_ids:
                    continue
                
                # Filter by rating if specified
                rating = replay_meta.get('rating') or replay_meta.get('p1rating') or 0
                if rating_min > 0 and rating < rating_min:
                    continue
                
                replay_tasks.append((replay_id, format))
            
            if page % 5 == 0:
                print(f" Searched {page} pages, queued {len(replay_tasks)} new replays")
        
        print(f"   Found {len(replay_tasks)} replays to download")
        
        # Download concurrently
        if replay_tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.download_replay, rid, fmt): rid 
                    for rid, fmt in replay_tasks
                }
                
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        success = future.result()
                        if success:
                            downloaded_count += 1
                        
                        # Progress update every 100 replays
                        if i % 100 == 0:
                            print(f"   Progress: {i}/{len(replay_tasks)} processed, "
                                  f"{downloaded_count} downloaded")
                    except Exception as e:
                        print(f"   Task error: {e}")
        
        print(f"   Downloaded {downloaded_count} replays for {format}")
        return downloaded_count
    
    def download_all_formats(self, 
                            formats: List[str] = None,
                            max_pages_per_format: int = 1000,
                            rating_min: int = 0) -> Dict:
        """
        Download replays from multiple formats.
        
        Args:
            formats: List of formats (uses POPULAR_FORMATS if None)
            max_pages_per_format: Max pages to search per format
            rating_min: Minimum rating filter
            
        Returns:
            Download statistics
        """
        if formats is None:
            formats = self.POPULAR_FORMATS
        
        self.stats['start_time'] = datetime.now()
        
        print("\n" + "="*70)
        print("BULK REPLAY DOWNLOADER")
        print("="*70)
        print(f"Formats to download: {len(formats)}")
        print(f"Max pages per format: {max_pages_per_format}")
        print(f"Min rating: {rating_min}")
        print(f"Concurrent workers: {self.max_workers}")
        print(f"Output directory: {self.output_dir}")
        print("="*70)
        
        for i, format_name in enumerate(formats, 1):
            print(f"\n[{i}/{len(formats)}] Processing format: {format_name}")
            try:
                self.download_format_replays(
                    format_name, 
                    max_pages_per_format,
                    rating_min
                )
            except Exception as e:
                print(f"Error processing {format_name}: {e}")
                continue
            
            # Print current statistics
            self._print_progress_stats()
        
        return self._get_final_stats()
    
    def download_continuous(self, 
                           format: str = "gen9randombattle",
                           duration_hours: float = 1.0,
                           rating_min: int = 0):
        """
        Continuously download replays for a specified duration.
        
        Args:
            format: Battle format
            duration_hours: How long to run (hours)
            rating_min: Minimum rating filter
        """
        self.stats['start_time'] = datetime.now()
        end_time = time.time() + (duration_hours * 3600)
        
        print("\n" + "="*70)
        print("CONTINUOUS DOWNLOAD MODE")
        print("="*70)
        print(f"Format: {format}")
        print(f"Duration: {duration_hours} hours")
        print(f"Will stop at: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
        print("="*70)
        
        page = 1
        cycle = 1
        
        while time.time() < end_time:
            print(f"\nCycle {cycle}, starting from page {page}")
            
            downloaded = self.download_format_replays(format, max_pages=20, rating_min=rating_min)
            
            if downloaded == 0:
                print("   No new replays found, waiting 5 minutes...")
                time.sleep(300)  # Wait 5 minutes
                page = 1  # Reset to beginning
            else:
                page += 20
            
            cycle += 1
            self._print_progress_stats()
        
        return self._get_final_stats()
    
    def _print_progress_stats(self):
        """Print current progress statistics."""
        with self.stats_lock:
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            rate = self.stats['total_downloaded'] / elapsed if elapsed > 0 else 0
            
            print(f"\n{'─'*70}")
            print(f"PROGRESS STATISTICS")
            print(f"{'─'*70}")
            print(f"Downloaded: {self.stats['total_downloaded']}")
            print(f"Skipped: {self.stats['total_skipped']}")
            print(f"Failed: {self.stats['total_failed']}")
            print(f"Rate: {rate:.2f} replays/second")
            print(f"Elapsed: {elapsed/60:.1f} minutes")
            print(f"{'─'*70}")
    
    def _get_final_stats(self) -> Dict:
        """Get final download statistics."""
        with self.stats_lock:
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            
            print("\n" + "="*70)
            print("FINAL STATISTICS")
            print("="*70)
            print(f"Total Downloaded: {self.stats['total_downloaded']}")
            print(f"Total Skipped: {self.stats['total_skipped']}")
            print(f"Total Failed: {self.stats['total_failed']}")
            print(f"Total Replays: {len(self.seen_ids)}")
            print(f"Total Time: {elapsed/60:.1f} minutes")
            print(f"Average Rate: {self.stats['total_downloaded']/elapsed:.2f} replays/second")
            
            print(f"\nBy Format:")
            for format_name, count in sorted(self.stats['by_format'].items(), 
                                            key=lambda x: x[1], 
                                            reverse=True):
                print(f"   {format_name}: {count}")
            
            print("="*70)
            
            return dict(self.stats)


def main():
    """Main function with different download modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bulk download Pokémon Showdown replays')
    parser.add_argument('--mode', 
                       choices=['all', 'format', 'continuous'],
                       default='all',
                       help='Download mode')
    parser.add_argument('--format', 
                       default='gen9randombattle',
                       help='Format to download (for format/continuous modes)')
    parser.add_argument('--formats',
                       nargs='+',
                       help='List of formats to download (for all mode)')
    parser.add_argument('--pages',
                       type=int,
                       default=50,
                       help='Max pages per format')
    parser.add_argument('--rating',
                       type=int,
                       default=0,
                       help='Minimum rating filter')
    parser.add_argument('--workers',
                       type=int,
                       default=10,
                       help='Number of concurrent workers')
    parser.add_argument('--duration',
                       type=float,
                       default=1.0,
                       help='Duration in hours (for continuous mode)')
    parser.add_argument('--output',
                       default='replays_data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = BulkReplayDownloader(
        output_dir=args.output,
        max_workers=args.workers,
        retry_attempts=3,
        delay_between_requests=0.1
    )
    
    # Execute based on mode
    if args.mode == 'all':
        formats = args.formats if args.formats else None
        downloader.download_all_formats(
            formats=formats,
            max_pages_per_format=args.pages,
            rating_min=args.rating
        )
    
    elif args.mode == 'format':
        downloader.download_format_replays(
            format=args.format,
            max_pages=args.pages,
            rating_min=args.rating
        )
    
    elif args.mode == 'continuous':
        downloader.download_continuous(
            format=args.format,
            duration_hours=args.duration,
            rating_min=args.rating
        )


if __name__ == "__main__":
    # Quick start examples (comment out main() and uncomment one of these)
    
    # Example 1: Download from all popular formats (recommended)
    downloader = BulkReplayDownloader(
        output_dir="replays_data",
        max_workers=15,  # Increase for faster downloads
        delay_between_requests=0.05
    )
    downloader.download_all_formats(max_pages_per_format=1000, rating_min=1500)
    
    # Example 2: Focus on one format with high quality
    # downloader = BulkReplayDownloader(output_dir="replays_data", max_workers=10)
    # downloader.download_format_replays("gen9ou", max_pages=100, rating_min=1600)
    
    # Example 3: Continuous download for 2 hours
    # downloader = BulkReplayDownloader(output_dir="replays_data", max_workers=10)
    # downloader.download_continuous("gen9randombattle", duration_hours=2.0, rating_min=1400)
    
    # Example 4: Use command line arguments
    # main()
