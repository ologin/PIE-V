#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from piev.config import REPO_ROOT
from piev.utils.semrep_utils import generate_semrep_items_openai, MODEL, BATCH_SIZE

MAX_BATCHES = int(os.getenv("MAX_BATCHES", "0"))  # 0 = no limit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate semantic representations for a step vocabulary.")
    p.add_argument("--vocab_csv", default=str(REPO_ROOT / "data" / "resources" / "split_50_vocabulary.csv"))
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "resources" / "semantic_representations_split_50.json"))
    return p.parse_args()

def load_vocab(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = (row.get("step_description_id") or "").strip()
            sd = (row.get("step_description") or "").strip()
            if sid and sd:
                rows.append((sid, sd))
    return rows

def chunk(items: List[Tuple[str, str]], n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]

def main():
    args = parse_args()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if OpenAI is None:
        raise RuntimeError("openai package is not available.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    vocab_csv = Path(args.vocab_csv)
    out_json = Path(args.out)

    vocab = load_vocab(vocab_csv)
    print(f"Model: {MODEL}")
    print(f"Vocab entries: {len(vocab)}")

    out: Dict[str, Dict[str, str]] = {}
    if out_json.exists():
        try:
            out = json.loads(out_json.read_text(encoding="utf-8"))
            if not isinstance(out, dict):
                out = {}
        except Exception:
            out = {}

    done_ids = set(out.keys())
    pending = [(sid, sd) for sid, sd in vocab if sid not in done_ids]
    print(f"Already in output: {len(done_ids)}")
    print(f"Pending: {len(pending)}")
    print(f"Batch size: {BATCH_SIZE}")

    batch_counter = 0
    for batch in chunk(pending, BATCH_SIZE):
        batch_counter += 1
        if MAX_BATCHES and batch_counter > MAX_BATCHES:
            print(f"MAX_BATCHES reached: {MAX_BATCHES}")
            break

        expected = {sid: sd for sid, sd in batch}

        try:
            generated = generate_semrep_items_openai(client, expected)
        except Exception as e:
            print(f"[Batch {batch_counter}] Failed: {e}")
            sys.exit(1)

        missing_ids = [sid for sid in expected.keys() if sid not in generated]
        if missing_ids:
            print(f"[Batch {batch_counter}] Missing ids: {len(missing_ids)} (showing up to 10): {missing_ids[:10]}")

        for sid, sd in expected.items():
            sem = generated.get(sid, "")
            if isinstance(sem, str) and sem.strip():
                out[sid] = {
                    "step_description": sd,
                    "semantic_representation": sem.strip(),
                }

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"[Batch {batch_counter}] Saved. Total saved: {len(out)}")

    print("Done.")
    print(f"Saved entries: {len(out)}")
    print(f"Output: {out_json}")

if __name__ == "__main__":
    main()
