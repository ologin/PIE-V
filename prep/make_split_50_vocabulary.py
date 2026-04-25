#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from piev.config import REPO_ROOT, load_settings

SETTINGS = load_settings()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a unique step-description vocabulary from split_50.json.")
    p.add_argument("--input", default=str(SETTINGS.split50_path))
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "resources" / "split_50_vocabulary.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)
    data = json.loads(in_path.read_text())

    seen = set()
    vocab = []  # [(id, step_description)]

    next_id = 1
    for ann in data.get("annotations", []):
        for seg in ann.get("segments", []):
            sd = seg.get("step_description")
            if not sd:
                continue
            sd = sd.strip()
            if not sd:
                continue
            if sd in seen:
                continue
            seen.add(sd)
            vocab.append((next_id, sd))
            next_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step_description_id", "step_description"])
        w.writerows(vocab)

    print(f"Written: {out_path}")
    print(f"Unique step_descriptions: {len(vocab)}")

if __name__ == "__main__":
    main()
