"""Fast chunked JSON writer (incremental size tracking)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from extract_action_chunks import write_chunk


def _row_json_bytes(row: Dict[str, object], *, pretty: bool) -> int:
    payload = json.dumps(row, indent=2 if pretty else None, separators=(",", ":"))
    return len(payload.encode("utf-8"))


def next_chunk_index(output_dir: Path, base_name: str) -> int:
    """First free index after existing rb_action_data_XXXXX.json files."""
    max_idx = 0
    for path in output_dir.glob(f"{base_name}_*.json"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


class ChunkBuffer:
    """Buffer examples and flush when estimated size exceeds max_bytes."""

    def __init__(
        self,
        *,
        output_dir: Path,
        base_name: str,
        max_bytes: int,
        pretty: bool,
        start_chunk_idx: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.max_bytes = max_bytes
        self.pretty = pretty
        self.rows: List[Dict[str, object]] = []
        self.estimated_bytes = 2  # "[]"
        if start_chunk_idx > 0:
            self.chunk_idx = start_chunk_idx
        else:
            self.chunk_idx = next_chunk_index(output_dir, base_name)
        self.chunks_written = 0

    def append(self, row: Dict[str, object]) -> bool:
        row_bytes = _row_json_bytes(row, pretty=self.pretty)
        if self.rows:
            self.estimated_bytes += 1 + row_bytes  # comma + row
        else:
            self.estimated_bytes = 2 + row_bytes
        self.rows.append(row)
        return self.estimated_bytes >= self.max_bytes

    def flush(self) -> Path | None:
        if not self.rows:
            return None
        out_path = write_chunk(
            chunk=self.rows,
            output_dir=self.output_dir,
            base_name=self.base_name,
            chunk_idx=self.chunk_idx,
            pretty_json=self.pretty,
        )
        nbytes = out_path.stat().st_size
        print(
            f"Wrote {len(self.rows)} examples ({nbytes / (1024 * 1024):.1f} MiB) to {out_path}",
            flush=True,
        )
        self.rows.clear()
        self.estimated_bytes = 2
        self.chunk_idx += 1
        self.chunks_written += 1
        return out_path
