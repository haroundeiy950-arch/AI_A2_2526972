"""Recreate the individual dataset: python data/generate_data.py [--seed N].

The dataset is defined by the assignment identity rules:
  * random seed = LLLL = 6972 (last four digits of registration 25/26972)
  * base rows   = 240 + (6972 mod 41) = 242
  * deliberate defects: 12 missing soil moisture values, 12 missing
    fertilizer values and three exact duplicate rows appended
  * CSV values written with four decimal places
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, generate_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the individual CSV dataset.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="random seed (LLLL)")
    parser.add_argument("--out", type=str, default=str(config.DATA_PATH), help="output CSV path")
    args = parser.parse_args(argv)

    if args.seed != config.SEED:
        print(
            f"ERROR: seed {args.seed} is not allowed. Required seed is {config.SEED}.",
            file=sys.stderr,
        )
        return 2

    defects = generate_data.generate(seed=args.seed, path=args.out)
    print(f"Verification code : {config.VERIFICATION_CODE}")
    print(f"Seed              : {args.seed}")
    print(f"Base rows         : {defects['base_rows']}")
    print(f"Rows written      : {defects['rows_written']}")
    print(f"Missing values    : {defects['missing_counts']}")
    print(f"Duplicate rows    : {defects['duplicates_appended']}")
    print(f"SHA-256           : {defects['sha256']}")
    print(f"Dataset           : {defects['csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
