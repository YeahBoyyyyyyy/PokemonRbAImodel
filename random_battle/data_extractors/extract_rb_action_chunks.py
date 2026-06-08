"""

Extract gen9randombattle training chunks from Pokéchamp (HF).



Only format: gen9randombattle. Partial observability via rb_visibility.py.

"""



from __future__ import annotations



import argparse

import sys

from dataclasses import asdict

from pathlib import Path

from typing import Dict



_SCRIPT_DIR = Path(__file__).resolve().parent

_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

for _path in (_PROJECT_ROOT, _SCRIPT_DIR, _PROJECT_ROOT / "shared" / "data_extractors"):

    _entry = str(_path)

    if _entry not in sys.path:

        sys.path.insert(0, _entry)



from chunk_io import ChunkBuffer  # noqa: E402

from extract_pokechamp_training_data import TrainingExample  # noqa: E402

from rb_training_extractor import RandomBattleTrainingExtractor  # noqa: E402

from random_battle.config import (  # noqa: E402

    ACTION_CHUNKS_DIR,

    FORMAT_ID,

    MIN_ELO_DEFAULT,

)



POKECHAMP_DATASET = "milkkarten/pokechamp"





def example_to_json(example: TrainingExample) -> Dict[str, object]:

    return asdict(example)





def parse_pokechamp_elo(elo_str: str) -> int:

    if not elo_str:

        return 0

    if "+" in elo_str:

        return int(elo_str.replace("+", ""))

    if "-" in elo_str:

        return int(elo_str.split("-")[0])

    try:

        return int(elo_str)

    except ValueError:

        return 0





def iter_gen9_rb_pokechamp(

    *,

    min_elo: int,

    max_replays: int,

    split: str,

    log_every_scanned: int = 5000,

):

    from datasets import load_dataset



    dataset = load_dataset(POKECHAMP_DATASET, split=split, streaming=True)

    accepted = 0

    scanned = 0

    for battle in dataset:

        scanned += 1

        if log_every_scanned and scanned % log_every_scanned == 0:

            print(

                f"  … scanned {scanned} rows on HF, "

                f"kept {accepted} {FORMAT_ID} replays (elo>={min_elo})",

                flush=True,

            )

        if max_replays and accepted >= max_replays:

            break

        if (battle.get("gamemode") or "").lower() != FORMAT_ID:

            continue

        if parse_pokechamp_elo(battle.get("elo", "0")) < min_elo:

            continue

        accepted += 1

        yield battle

    print(f"Pokechamp gen9randombattle: scanned={scanned}, accepted={accepted}")





def main() -> None:

    parser = argparse.ArgumentParser(

        description="Download & extract gen9randombattle from Pokéchamp."

    )

    parser.add_argument("--split", default="train")

    parser.add_argument("--min_elo", type=int, default=MIN_ELO_DEFAULT)

    parser.add_argument("--max_replays", type=int, default=0, help="0 = no limit")

    parser.add_argument("--output_dir", default=str(ACTION_CHUNKS_DIR))

    parser.add_argument("--base_name", default="rb_action_data")

    parser.add_argument(

        "--max_chunk_bytes",

        type=int,

        default=50 * 1024 * 1024,

        help="Flush each JSON chunk at ~50 MiB",

    )

    parser.add_argument("--pretty", action="store_true")

    parser.add_argument("--log_every_scanned", type=int, default=1000)

    parser.add_argument("--log_every_accepted", type=int, default=100)

    parser.add_argument("--filter_voluntary", action="store_true")

    parser.add_argument(
        "--start_chunk_idx",
        type=int,
        default=0,
        help="0 = auto-continue after last rb_action_data_XXXXX.json in output_dir",
    )

    args = parser.parse_args()



    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = RandomBattleTrainingExtractor()

    buffer = ChunkBuffer(

        output_dir=output_dir,

        base_name=args.base_name,

        max_bytes=args.max_chunk_bytes,

        pretty=args.pretty,

        start_chunk_idx=args.start_chunk_idx,

    )

    print(f"Writing chunks starting at index {buffer.chunk_idx:05d}", flush=True)

    total_examples = 0

    accepted = 0



    print(f"Streaming {POKECHAMP_DATASET} ({FORMAT_ID}, elo>={args.min_elo})")

    for battle in iter_gen9_rb_pokechamp(

        min_elo=args.min_elo,

        max_replays=args.max_replays,

        split=args.split,

        log_every_scanned=args.log_every_scanned,

    ):

        log_text = battle.get("text") or ""

        examples = extractor.extract_from_log(str(log_text))

        if not examples:

            continue

        accepted += 1

        if accepted == 1:

            print("First replay extracted, processing…", flush=True)

        if args.log_every_accepted and accepted % args.log_every_accepted == 0:

            print(

                f"Processed {accepted} replays, examples={total_examples}, "

                f"buffered={len(buffer.rows)}",

                flush=True,

            )



        for ex in examples:

            if args.filter_voluntary and not ex.is_voluntary:

                continue

            if buffer.append(example_to_json(ex)):

                buffer.flush()

            total_examples += 1



    buffer.flush()

    print(

        f"Done. format={FORMAT_ID} replays={accepted} "

        f"examples={total_examples} files={buffer.chunks_written} -> {output_dir}"

    )





if __name__ == "__main__":

    main()


