#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extend_sem_rep_from_rewrites.py

Extend semantic_representations_split_50.json with NEW step texts found in rewrite outputs:
  - split_50_error_instructions_openai.json
  - split_50_error_instructions_qwen.json

Output format matches the base semrep JSON:
  {
    "<id>": {"step_description": "...", "semantic_representation": "..."},
    ...
  }

New steps get synthetic ids:
  <id_prefix>_<sha1_16>

This keeps judge simple: you can build a reverse map step_description -> id.

Usage:
  python extend_sem_rep_from_rewrites.py \
    --rewrite data/examples/split_50_error_instructions_openai.json

"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
import re
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

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
DEFAULT_BASE_SEMREP_JSON = REPO_ROOT / "data" / "resources" / "semantic_representations_split_50.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "resources"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))
MODEL = OPENAI_MODEL
MAX_BATCHES = int(os.getenv("MAX_BATCHES", "0"))  # 0 = no limit

SYSTEM_PROMPT = """You are a linguistic semantic analyzer.
For each procedural step description, generate the semantic representation in roles.

Hard constraints:
- Use compact single-line format: PREDICATE(Role: value, Role: value, ...)
- Predicates MUST be UPPERCASE (e.g., READ, INSERT, ADD, MOVE).
- Prefer the role name Object (NOT Theme) for concrete manipulated entities.
- Use Agent: you unless the sentence clearly specifies another agent.
- Prepositional phrases that modify a NOUN should be nested inside that noun:
  Example: "Read the instruction leaflet on the table"
  READ(Agent: you, Object: instruction_leaflet(Location: on(table)))
  (NOT: READ(..., Location: on(table)) unless the location is truly the action location.)
- Typical roles:
  Object, Destination, Coobject, Location, Instrument, Manner, Purpose, Temporal, Origin, Result
- Keep entities as lowercase_with_underscores; allow nested structures like of(...), in(...), on(...).

Examples:
1) As eggs begin to set, gently move spatula across bottom and side of skillet to form large, soft curds.
MOVE(Agent: you, Object: spatula, Path: across(bottom(of(skillet))) & across(side(of(skillet))), Manner: gently, Purpose: FORM(Agent: you, Result: large_soft_curds), Temporal: WHILE(BEGIN(SET(Object: eggs))))

2) Insert the test swab into her nostril
INSERT(Agent: you, Object: test_swab, Destination: into(nostril(of(her))))

3) Use a cloth to dry off any liquid from the chain lube on the chain while backpaddling with hand
USE(Agent: you, Object: cloth, Purpose: DRY_OFF(Agent: you, Object: any_liquid, Origin: from(chain_lube), Location: on(chain)), Temporal: WHILE(BACKPEDAL(Agent: you, Instrument: hand)))

4) Add coffee grounds from a bowl to the filter in the French press
ADD(Agent: you, Object: coffee_grounds, Origin: bowl, Destination: filter(Location: in(french_press)))

5) Add cut onions to the egg in the mixing bowl
ADD(Agent: you, Object: cut(onions), Coobject: egg(Location: in(mixing_bowl)))

Return only JSON that matches the schema.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "step_description": {"type": "string"},
                    "semantic_representation": {"type": "string"},
                },
                "required": ["id", "step_description", "semantic_representation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wrapper: extend base semantic representations with new steps from rewrites.")
    p.add_argument("--rewrite", required=True, help="Path to split_50_error_instructions_{openai|qwen}.json",)
    p.add_argument("--base_semrep", default=str(DEFAULT_BASE_SEMREP_JSON), help="Base semantic representation JSON.")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Directory for semantic_representations_split_50_{tag}_extended.json.")
    p.add_argument("--limit_steps", type=int, default=0, help="Optional limit of NEW steps to generate (0=no limit)")
    return p.parse_args()

def infer_tag(rewrite_path: Path) -> str:
    """
    Infer tag strictly from filename:
      split_50_error_instructions_openai.json -> openai
      split_50_error_instructions_qwen.json   -> qwen
    """
    name = rewrite_path.name
    m = re.search(r"split_50_error_instructions_(openai|qwen)\.json$", name)
    if not m:
        raise ValueError(f"Cannot infer tag from rewrite filename: {name}")
    return m.group(1)

def backoff_sleep(attempt: int) -> None:
    time.sleep(min(30, 2 ** attempt))

def is_fatal_request_error(e: Exception) -> bool:
    s = str(e)
    return ("Error code: 400" in s) or ("invalid_request_error" in s) or ("invalid_json_schema" in s)

def sha1_16(s: str) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:16]

def load_json_dict(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def normalize_step_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def iter_rewrite_steps(rewrite_obj: Dict) -> List[str]:
    steps: List[str] = []
    takes = rewrite_obj.get("takes") or {}
    if not isinstance(takes, dict):
        return steps
    for _tid, t in takes.items():
        if not isinstance(t, dict):
            continue
        rw = t.get("rewrite") or {}
        if not isinstance(rw, dict):
            continue
        fs = rw.get("final_steps") or []
        if isinstance(fs, list):
            for x in fs:
                if isinstance(x, str) and x.strip():
                    steps.append(normalize_step_text(x))
    return steps

def chunk(items: List[Tuple[str, str]], n: int):
    for i in range(0, len(items), n):
        yield items[i:i+n]

def main() -> None:
    args = parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if OpenAI is None:
        raise RuntimeError("openai package is not available.")

    client = OpenAI(api_key=openai_api_key)

    rewrite_path = Path(args.rewrite)
    if not rewrite_path.exists():
        raise FileNotFoundError(f"--rewrite not found: {rewrite_path}")

    # Infer tag from filename and derive output + id_prefix automatically (no flags)
    tag = infer_tag(rewrite_path)  # "openai" | "qwen"
    out_path = Path(args.out_dir) / f"semantic_representations_split_50_{tag}_extended.json"
    id_prefix = f"{tag}_ext"

    base = load_json_dict(Path(args.base_semrep))
    # allow resume from out (if exists), else start from base
    out = load_json_dict(out_path) if out_path.exists() else dict(base)

    # Build set of known step_descriptions from out
    known_steps = set()
    for _id, v in out.items():
        if isinstance(v, dict):
            sd = v.get("step_description")
            if isinstance(sd, str) and sd.strip():
                known_steps.add(normalize_step_text(sd))

    rewrite_obj = json.loads(rewrite_path.read_text(encoding="utf-8"))
    all_steps = iter_rewrite_steps(rewrite_obj)
    uniq_steps = sorted(set(all_steps))

    new_steps = [s for s in uniq_steps if s not in known_steps]
    if args.limit_steps and args.limit_steps > 0:
        new_steps = new_steps[: args.limit_steps]

    print(f"Tag: {tag}")
    print(f"Output: {out_path}")
    print(f"Rewrite steps (unique): {len(uniq_steps)}")
    print(f"Already covered in output: {len(known_steps)}")
    print(f"NEW steps to generate: {len(new_steps)}")
    print(f"Batch size: {BATCH_SIZE}")

    # prepare (id -> step_description)
    pending_pairs: List[Tuple[str, str]] = []
    used_ids = set(out.keys())

    for sd in new_steps:
        base_id = f"{id_prefix}_{sha1_16(sd)}"
        sid = base_id
        k = 1
        while sid in used_ids:
            # extremely unlikely, but keep deterministic
            sid = f"{base_id}_{k}"
            k += 1
        used_ids.add(sid)
        pending_pairs.append((sid, sd))

    batch_counter = 0
    for batch in chunk(pending_pairs, BATCH_SIZE):
        batch_counter += 1
        if MAX_BATCHES and batch_counter > MAX_BATCHES:
            print(f"MAX_BATCHES reached: {MAX_BATCHES}")
            break

        expected = {sid: sd for sid, sd in batch}

        user_content = (
            "Generate items for these id -> step_description pairs.\n"
            "Return one item per input pair.\n"
            "The 'id' field must equal the input key.\n"
            "The 'step_description' field must be copied exactly.\n\n"
            + json.dumps(expected, ensure_ascii=False, indent=2)
        )

        attempt = 0
        while True:
            try:
                resp = client.responses.create(
                    model=MODEL,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "semantic_representations_batch",
                            "strict": True,
                            "schema": OUTPUT_SCHEMA,
                        }
                    },
                )

                payload = json.loads(resp.output_text)
                items = payload.get("items", [])
                by_id = {it["id"]: it for it in items if isinstance(it, dict) and "id" in it}

                missing = [sid for sid in expected.keys() if sid not in by_id]
                if missing:
                    print(f"[Batch {batch_counter}] Missing ids: {len(missing)} (up to 10): {missing[:10]}")

                saved = 0
                for sid, sd in expected.items():
                    it = by_id.get(sid)
                    if not it:
                        continue
                    sem = it.get("semantic_representation")
                    if isinstance(sem, str) and sem.strip():
                        out[sid] = {
                            "step_description": sd,
                            "semantic_representation": sem.strip(),
                        }
                        saved += 1

                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(out, ensure_ascii=False, indent=4), encoding="utf-8")
                print(f"[Batch {batch_counter}] Saved {saved}/{len(batch)}. Total entries: {len(out)}")
                break

            except Exception as e:
                if is_fatal_request_error(e):
                    print(f"[Batch {batch_counter}] Fatal error: {e}", file=sys.stderr)
                    sys.exit(1)
                attempt += 1
                if attempt >= 6:
                    print(f"[Batch {batch_counter}] Failed after retries: {e}", file=sys.stderr)
                    break
                print(f"[Batch {batch_counter}] Error: {e} | retry {attempt}/5")
                backoff_sleep(attempt)

    print("Done.")
    print(f"Output: {out_path}")
    print(f"Total entries: {len(out)}")

if __name__ == "__main__":
    main()
