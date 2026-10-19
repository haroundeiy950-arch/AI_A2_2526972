"""Command-line prediction from saved artifacts.

Usage
-----
    python predict.py --json samples/valid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.inference import RecordValidationError, load_artifacts, predict_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict yield, risk and cluster for one JSON record.")
    parser.add_argument("--json", required=True, help="path to a JSON file with one farm record")
    parser.add_argument(
        "--models", default=str(config.MODELS_DIR), help="directory containing saved models"
    )
    args = parser.parse_args(argv)

    path = Path(args.json)
    if not path.exists():
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        return 2

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: {path} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    try:
        artifacts = load_artifacts(args.models)
        result = predict_record(record, artifacts)
    except (RecordValidationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
