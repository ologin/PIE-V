# Semantic representation helpers used by PIE-V writer and judge stages.
from __future__ import annotations

import hashlib
import json
import os
import time
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# -------------------------
# API key
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# -------------------------
# SemRep generation settings (single source of truth)
# -------------------------
MODEL = "gpt-5.2"
BATCH_SIZE = 15
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 8000

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
  Object, Coobject, Destination, Location, Instrument, Manner, Purpose, Temporal, Origin, Result
- Keep entities as lowercase_with_underscores; allow nested structures like of(...), in(...), on(...).

Examples:
1) As eggs begin to set, gently move spatula across bottom and side of skillet to form large, soft curds.
MOVE(Agent: you, Object: spatula, Path: across(bottom(of(skillet)) & side(of(skillet))), Manner: gently,
     Purpose: FORM(Agent: you, Object: large_soft_curds), Temporal: WHILE(BEGIN(SET(Object: eggs))))

2) Insert the test swab into her nostril
INSERT(Agent: you, Object: test_swab, Destination: into(nostril(of(her))))

3) Use a cloth to dry off any liquid from the chain lube on the chain while backpaddling with hand
USE(Agent: you, Object: cloth, Purpose: DRY_OFF(Agent: you, Object: liquid(Origin: chain_lube(Location: on(chain)))),
    Temporal: WHILE(BACKPEDAL(Agent: you, Instrument: hand)))

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


def sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def normalize_step_text(s: str) -> str:
    s = " ".join((s or "").strip().split())
    # drop trailing punctuation that often differs between sources
    s = re.sub(r"[\.!\?:;]+$", "", s).strip()
    return s


def backoff_sleep(attempt: int) -> None:
    time.sleep(min(30, 2 ** attempt))


def is_fatal_request_error(e: Exception) -> bool:
    s = str(e)
    return ("Error code: 400" in s) or ("invalid_request_error" in s) or ("invalid_json_schema" in s)


def build_semrep_step_to_id(semrep_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Reverse map: step_description text -> semrep id.
    Stores multiple keys per step_description to be robust to whitespace/case differences.
    """
    out: Dict[str, str] = {}
    for sid, v in (semrep_map or {}).items():
        if not isinstance(sid, str) or not isinstance(v, dict):
            continue
        sd = v.get("step_description")
        if not isinstance(sd, str) or not sd.strip():
            continue
        raw = sd.strip()
        keys = {raw, raw.strip(), normalize_step_text(raw), raw.lower().strip()}
        for k in keys:
            if k and k not in out:
                out[k] = sid
    return out


def _chunk(items: List[Tuple[str, str]], n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _get_openai_client() -> OpenAI:
    """
    Prefer environment variable OPENAI_API_KEY. Fall back to OPENAI_API_KEY constant.
    """
    if OpenAI is None:
        raise RuntimeError("openai package is not available.")
    key = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY env var is not set. Do not hardcode keys in code.")
    return OpenAI(api_key=key)


def generate_semrep_items_openai(
    client: OpenAI,
    expected: Dict[str, str],
) -> Dict[str, str]:
    """
    expected: {id: step_description}
    returns:  {id: semantic_representation}
    """
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

            output_text = getattr(resp, "output_text", None)
            if not isinstance(output_text, str) or not output_text.strip():
                raise ValueError("OpenAI response did not contain output_text.")

            payload = json.loads(output_text)
            items = payload.get("items", [])
            out: Dict[str, str] = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                sid = it.get("id")
                sem = it.get("semantic_representation")
                if isinstance(sid, str) and isinstance(sem, str) and sem.strip():
                    out[sid] = sem.strip()
            # ensure 1:1 coverage; re-request missing ids
            missing_ids = [k for k in expected.keys() if k not in out]
            if missing_ids:
                raise ValueError(f"Model returned {len(out)}/{len(expected)} items; missing={missing_ids[:3]}...")
                
            return out

        except Exception as e:
            if is_fatal_request_error(e):
                raise
            attempt += 1
            if attempt >= 6:
                raise
            backoff_sleep(attempt)


@dataclass
class SemRepAutoExtender:
    """
    In-memory semrep map extender.
    - Detects step descriptions missing from semrep_map
    - Generates semantic representations via OpenAI
    - Updates semrep_map + reverse map step_to_id
    - Optionally writes semrep_map back to out_path on flush()
    """
    semrep_map: Dict[str, Dict[str, str]]
    out_path: Optional[str] = None
    id_prefix: str = "openai_ext"
    client: Optional[OpenAI] = None
    dirty: bool = False

    def __post_init__(self):
        if self.client is None:
            self.client = _get_openai_client()
        self.step_to_id = build_semrep_step_to_id(self.semrep_map)

    def ensure_for_texts(self, step_texts: Iterable[str]) -> None:
        """
        Ensure semantic representations exist for all provided step_texts.
        This updates semrep_map and step_to_id in-place.
        """
        missing: List[str] = []
        for s in step_texts:
            sd = normalize_step_text(s)
            if not sd:
                continue
            sid = self.step_to_id.get(sd) or self.step_to_id.get(sd.lower())
            if sid and sid in self.semrep_map and self.semrep_map[sid].get("semantic_representation"):
                continue
            missing.append(sd)

        missing = sorted(set(missing))
        if not missing:
            return

        used_ids = set(self.semrep_map.keys())
        pending_pairs: List[Tuple[str, str]] = []
        for sd in missing:
            base_id = f"{self.id_prefix}_{sha1_16(sd)}"
            sid = base_id
            k = 1
            while sid in used_ids:
                sid = f"{base_id}_{k}"
                k += 1
            used_ids.add(sid)
            pending_pairs.append((sid, sd))

        for batch in _chunk(pending_pairs, BATCH_SIZE):
            expected = {sid: sd for sid, sd in batch}
            gen = generate_semrep_items_openai(self.client, expected)
            for sid, sd in expected.items():
                sem = gen.get(sid, "")
                if not sem:
                    continue

                self.semrep_map[sid] = {
                    "step_description": sd,
                    "semantic_representation": sem,
                }

                for k in {sd, sd.strip(), normalize_step_text(sd), sd.lower().strip()}:
                    if k and k not in self.step_to_id:
                        self.step_to_id[k] = sid

                self.dirty = True

    def flush(self) -> None:
        """
        Persist updated semrep_map to out_path if it changed.
        """
        if self.dirty and self.out_path:
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump(self.semrep_map, f, ensure_ascii=False, indent=2)
            self.dirty = False
