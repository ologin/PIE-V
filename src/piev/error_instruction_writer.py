#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_instruction_writer.py

Generate an *erroneous but coherent* rewritten procedure from an error+correction plan.

Key outputs per take:
- rewrite.final_steps: list[str]
- rewrite.meta: list[dict] timeline meta with old/new indices and labels

Meta format (timeline order):
{
  "old": int | "",          # original step index (0-based) if derived from source; "" for insertion/correction-only steps
  "new": int | "",          # index into final_steps (0-based) or "" for deletions (no new step)
  "mod": "u"|"e"|"a"|"i"|"c"|"d"|"ms"|"mt",
  # u  = unchanged (verbatim)
  # e  = error step (wrong_execution/substitution/etc.)
  # a  = adjusted/cascade repair (must carry eid)
  # i  = insertion
  # c  = correction
  # d  = deletion (new="")
  # ms = moved_source (transposition)
  # mt = moved_target (transposition)
  "etype": str | null,      # label for the type of step: insertion|deletion|substitution|wrong_execution|transposition|correction
  "eid": str | null,        # eid must equal the plan's error_id
  "cid": str | null         # cid must equal the plan's correction_id
}

Rules:
- If meta[k]["new"] is an int N, then final_steps[N] MUST be the corresponding step text.
- Deletions are represented *only* in meta (mod="d") with "new": "".
- Do NOT output separate "del"/"ins"/"corrs" arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
import logging

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from piev.config import REPO_ROOT, load_settings

SETTINGS = load_settings()

# Module logger (configured in main()).
logger = logging.getLogger(__name__)

# -------------------------
# CLI (restore legacy flags: --model openai|qwen)
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

DEFAULT_OPENAI_MODEL_ID = SETTINGS.openai_model
DEFAULT_QWEN_MODEL_ID = SETTINGS.qwen_text_model


def _auto_suffix_out_path(base_out: str, model_choice: str) -> str:
    """
    If user left the base default out path unchanged, create a model-specific postfix:
      .../split_50_error_instructions_openai.json
      .../split_50_error_instructions_qwen.json
    """
    root, ext = os.path.splitext(base_out)
    if root.endswith("_openai") or root.endswith("_qwen"):
        return base_out
    return f"{root}_{model_choice}{ext or '.json'}"


# OpenAI backend
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Qwen backend (transformers)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


# -------------------------
# SemRep auto-extension
# -------------------------
try:
    from piev.utils.semrep_utils import (
        SemRepAutoExtender,
        build_semrep_step_to_id as _helper_build_semrep_step_to_id,
    )
except ImportError:
    try:
        from semrep_utils import (
            SemRepAutoExtender,
            build_semrep_step_to_id as _helper_build_semrep_step_to_id,
        )
    except Exception:
        SemRepAutoExtender = None
        _helper_build_semrep_step_to_id = None


def _build_semrep_step_to_id_fallback(semrep_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Reverse lookup: step_description -> semrep_id.
    Works for base vocab ids and synthetic *_ext_* ids.
    """
    m: Dict[str, str] = {}
    for sid, v in (semrep_map or {}).items():
        if not isinstance(sid, str) or not isinstance(v, dict):
            continue
        sd = v.get("step_description")
        if isinstance(sd, str) and sd.strip():
            raw = sd.strip()
            keys = {raw, raw.strip(), normalize_ws(raw), normalize_lookup_key(raw)}
            for k in keys:
                if k and k not in m:
                    m[k] = sid
    return m


build_semrep_step_to_id = _helper_build_semrep_step_to_id or _build_semrep_step_to_id_fallback

# -------------------------
# Small utilities
# -------------------------


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def normalize_lookup_key(s: str) -> str:
    """
    Normalization for step_description lookup across:
      - vocab CSV
      - semrep JSON (base + extended)
      - generated final_steps (often differ by punctuation/case)
    """
    s = normalize_ws((s or "").strip())
    s = s.rstrip(".!?:;")
    return s.lower()


def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(tokenize_simple(a)), set(tokenize_simple(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalize_step_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_vocab_csv(path: str) -> Dict[str, str]:
    """
    Load vocabulary CSV into a mapping:
    step_description lookup keys -> step_description_id (string)

    IMPORTANT:
    - We keep *one* id per key; if duplicates exist, first wins.
    - We preserve the exact raw step_description keys as they appear in the CSV.
    - Additionally, we also register a normalized lookup key (lowercased, trimmed,
      stripped of trailing punctuation) to improve matching when callers differ by
      punctuation/case. This normalized key is "first wins" and may collide.
    """
    m: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = str(row.get("step_description_id") or "").strip()
                txt = str(row.get("step_description") or "").strip()
                if not sid or not txt:
                    continue
                # Keep EXACT text (including double spaces / typos).
                # Only strip line-end whitespace safely (not internal spaces).
                raw = txt.rstrip("\n\r")
                if raw and raw not in m:
                    m[raw] = sid
                # Small safety: also allow matching when caller has accidental outer spaces.
                raw_strip = raw.strip()
                if raw_strip and raw_strip not in m:
                    m[raw_strip] = sid
                # Optional: add a normalized lookup key WITHOUT changing raw keys.
                nk = normalize_lookup_key(raw)
                if nk and nk not in m:
                    m[nk] = sid
    except Exception as e:
        raise RuntimeError(f"Failed to load vocab csv: {path} ({e})")
    return m


def load_semrep_json(path: str) -> Dict[str, Dict[str, str]]:
    """
    semrep json format:
      { "817": { "step_description": "...", "semantic_representation": "..." }, ... }
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("semrep json must be an object")
        # ensure dict[str, dict]
        out: Dict[str, Dict[str, str]] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, dict):
                continue
            sd = str(v.get("step_description") or "")
            sr = str(v.get("semantic_representation") or "")
            out[k.strip()] = {"step_description": sd, "semantic_representation": sr}
        return out
    except Exception as e:
        raise RuntimeError(f"Failed to load semrep json: {path} ({e})")


def resolve_step_id_from_text(
    step_txt: str,
    vocab_map: Optional[Dict[str, str]],
    extra_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Resolve step_description_id from raw text using:
      1) vocab_map (CSV)
      2) extra_map (semrep_step_to_id from extended JSON / auto-extender)
    We try raw, stripped, and normalized keys.
    """
    raw = (step_txt or "").rstrip("\n\r")
    nk = normalize_lookup_key(raw)

    if vocab_map:
        if raw in vocab_map:
            return vocab_map[raw]
        if raw.strip() in vocab_map:
            return vocab_map[raw.strip()]
        if nk in vocab_map:
            return vocab_map[nk]

    if extra_map:
        if raw in extra_map:
            return extra_map[raw]
        if raw.strip() in extra_map:
            return extra_map[raw.strip()]
        if nk in extra_map:
            return extra_map[nk]

    return None


# -------------------------
# Global reverse SemRep map for O(1) exact lookup by text
# -------------------------
_REVERSE_SEMREP_MAP: Dict[str, str] = {}


def init_reverse_semrep_map(semrep_map: Dict[str, Dict[str, str]]) -> None:
    """
    Build normalized_text -> semrep_string.
    IMPORTANT: Call this again after SemRepAutoExtender adds new entries.
    """
    _REVERSE_SEMREP_MAP.clear()
    for _sid, data in (semrep_map or {}).items():
        if not isinstance(data, dict):
            continue
        raw_text = data.get("step_description", "")
        sr = data.get("semantic_representation", "")
        if isinstance(raw_text, str) and raw_text.strip() and isinstance(sr, str) and sr.strip():
            key = normalize_step_text(raw_text)
            val = sr.strip()
            # Collision handling: keep the FIRST value for stability and warn if SR differs.
            if key in _REVERSE_SEMREP_MAP:
                if _REVERSE_SEMREP_MAP[key] != val:
                    preview = (
                        (raw_text.strip()[:140] + "…")
                        if len(raw_text.strip()) > 140
                        else raw_text.strip()
                    )
                    logger.warning(
                        "SemRep reverse-map collision for normalized step text (sid=%s). "
                        "Keeping the first SR and ignoring this one. Text preview='%s'",
                        str(_sid),
                        preview,
                    )
                continue
            _REVERSE_SEMREP_MAP[key] = val


def find_semrep_exact(step_text: str) -> Optional[str]:
    return _REVERSE_SEMREP_MAP.get(normalize_step_text(step_text))


def compute_semrep_focus_indices(take: Dict[str, Any]) -> Set[int]:
    """
    Which step indices deserve semrep in the prompt:
      - error src
      - transposition target
      - correction after_step_idx (detect position proxy)
      - plus +/-1 around each (small local window)
    """
    idxs: Set[int] = set()
    steps = take.get("steps", []) or []
    n = len(steps) if isinstance(steps, list) else 0

    for e in take.get("errors", []) or []:
        if not isinstance(e, dict):
            continue
        s = e.get("step_index")
        t = (
            e.get("spec", {}).get("transposition_target")
            if isinstance(e.get("spec"), dict)
            else None
        )
        for v in (s, t):
            if isinstance(v, int):
                for w in (v - 1, v, v + 1):
                    if 0 <= w < n:
                        idxs.add(w)

    for c in take.get("corrections", []) or []:
        if not isinstance(c, dict):
            continue
        a = c.get("detect_at_step_index")
        if isinstance(a, int):
            for w in (a - 1, a, a + 1):
                if 0 <= w < n:
                    idxs.add(w)

    return idxs


def is_too_similar_substitution(original: str, substituted: str) -> bool:
    a = set(tokenize_simple(original))
    b = set(tokenize_simple(substituted))
    if not a or not b:
        return False
    j = len(a & b) / max(1, len(a | b))
    o = normalize_step_text(original)
    s = normalize_step_text(substituted)
    containment = (o in s) or (s in o)
    return containment or (j >= 0.78)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction (legacy behavior):
    - try full parse
    - otherwise find ALL balanced {...} candidates and return the LAST valid JSON object
    """
    if not isinstance(text, str):
        raise ValueError("Response is not a string.")
    s = text.strip()
    if not s:
        raise ValueError("Empty response.")

    # Fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Collect all balanced {...} candidates
    candidates: List[str] = []
    depth = 0
    cur_start: Optional[int] = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                cur_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and cur_start is not None:
                    candidates.append(s[cur_start : i + 1])
                    cur_start = None

    for chunk in reversed(candidates):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    raise ValueError("No valid JSON object found in response.")


def safe_get(d: Dict[str, Any], k: str, default: Any) -> Any:
    return d[k] if k in d else default


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rewrite procedures with errors+corrections into coherent step-by-step instructions (minimal edits)."
    )
    p.add_argument(
        "--input",
        default=str(REPO_ROOT / "data" / "examples" / "split_50_error_plan_with_corrections.json"),
        help="Path to split_50_error_plan_with_corrections.json",
    )
    # Base default; we post-fix with _openai/_qwen if user didn't override.
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "local" / "outputs" / "split_50_error_instructions.json"),
        help=(
            "Output JSON path with rewrites per take. "
            "If left as default, suffix _openai/_qwen will be added automatically."
        ),
    )
    p.add_argument(
        "--model", required=True, choices=["openai", "qwen"], help="Which backend to use"
    )
    p.add_argument("--max_takes", type=int, default=0, help="If >0, process only first N takes")
    p.add_argument(
        "--take_name",
        nargs="+",
        default=[],
        help="If set, process only these take_name values (space-separated). Example: --take_name fair_bike_07_12 sfu_cooking015_4",
    )
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument(
        "--retry_temp_decay", type=float, default=0.15, help="Temperature decay per retry attempt"
    )
    p.add_argument("--max_new_tokens", type=int, default=8000)

    # Optional semrep support (external resources)
    p.add_argument(
        "--include_semrep",
        action="store_true",
        help="Include semantic representations (semrep) for all steps.",
    )
    p.add_argument(
        "--vocab_csv",
        default=str(REPO_ROOT / "data" / "resources" / "split_50_vocabulary.csv"),
        help="Path to split_50_vocabulary.csv (step_description_id <-> step_description).",
    )
    p.add_argument(
        "--semrep_json",
        default=str(REPO_ROOT / "data" / "resources" / "semantic_representations_split_50.json"),
        help="Path to semantic_representations_split_50.json (keyed by step_description_id).",
    )

    return p.parse_args()


# -------------------------
# Validation helpers
# -------------------------

# Only actions that behave like "possession/state toggles".
NON_REPEATABLE_ACTION_PREFIXES = (
    "return ",
    "put away ",
    "put back ",
    "dispose ",
)


def is_near_duplicate_step(a: str, b: str) -> bool:
    a_n, b_n = normalize_ws(a).lower(), normalize_ws(b).lower()
    if a_n == b_n:
        return True
    return jaccard_similarity(a_n, b_n) >= 0.85


def is_nonrepeatable_action(step: str) -> bool:
    s = normalize_ws(step).lower()
    return s.startswith(NON_REPEATABLE_ACTION_PREFIXES)


def _same_step_text_loose(a: str, b: str) -> bool:
    """
    Treat punctuation/casing/whitespace-only differences as "unchanged".
    """
    return normalize_step_text(a) == normalize_step_text(b)


# -------------------------
# Minimal SemRep parsing (ONLY for your existing SemRep format)
# -------------------------

_SEMREP_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*\((.*)\)\s*$")


def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _norm_semrep_value(v: str) -> str:
    s = (v or "").strip().strip('"').strip("'").lower()
    s = " ".join(s.split())
    s = s.replace(" ", "_")
    return s


def parse_semrep_minimal(semrep: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Parse: PRED(Role: value, Role2: value2, ...)
    Returns: (PRED_UPPER, roles_dict_raw_values)
    """
    if not isinstance(semrep, str) or not semrep.strip():
        return None
    m = _SEMREP_RE.match(semrep.strip())
    if not m:
        return None
    pred = (m.group(1) or "").strip().upper()
    inside = (m.group(2) or "").strip()
    if not pred:
        return None
    roles: Dict[str, str] = {}
    if inside:
        for chunk in _split_top_level_commas(inside):
            if ":" not in chunk:
                continue
            k, v = chunk.split(":", 1)
            k = (k or "").strip()
            v = (v or "").strip()
            if k:
                roles[k] = v
    return pred, roles


def _roles_norm_excluding(roles: Dict[str, str], exclude: Set[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in roles.items():
        kk = str(k).strip()
        if not kk or kk in exclude:
            continue
        out[kk] = _norm_semrep_value(str(v))
    return out


def _semrep_for_original_idx(
    take: Dict[str, Any],
    old_idx: int,
    vocab_map: Optional[Dict[str, str]],
    semrep_map: Optional[Dict[str, Dict[str, str]]],
) -> str:
    steps = take.get("steps") or take.get("raw_steps") or []
    if not isinstance(steps, list):
        return ""
    for s in steps:
        if not isinstance(s, dict):
            continue
        idx = s.get("idx")
        if not isinstance(idx, int):
            idx = s.get("index")
        if idx != old_idx:
            continue
        sr0 = s.get("semantic_representation")
        if isinstance(sr0, str) and sr0.strip():
            return sr0.strip()
        txt = s.get("txt")
        if not isinstance(txt, str) or not txt:
            txt = s.get("step_description")
        if not isinstance(txt, str):
            txt = ""
        if vocab_map is None or semrep_map is None:
            return ""
        sid = s.get("step_description_id")
        if not isinstance(sid, str) or not sid.strip():
            sid = resolve_step_id_from_text(txt, vocab_map) if txt else None
        if sid and sid in semrep_map:
            sr = (semrep_map[sid] or {}).get("semantic_representation") or ""
            return sr.strip() if isinstance(sr, str) else ""
        return ""
    return ""


def _semrep_for_step_text(
    step_text: str,
    vocab_map: Optional[Dict[str, str]],
    semrep_map: Optional[Dict[str, Dict[str, str]]],
    semrep_step_to_id: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Resolve SemRep for an arbitrary step text.
    This supports LLM-generated steps by using semrep_step_to_id produced by SemRepAutoExtender.
    """
    # 1) Fast exact lookup against reverse semrep map (covers extended entries too)
    sr = find_semrep_exact(step_text)
    if sr:
        return sr

    # 2) Resolve step id via vocab_map OR extended step_to_id map
    sid = resolve_step_id_from_text(step_text, vocab_map, extra_map=semrep_step_to_id)
    if not sid or not semrep_map:
        return None

    rec = semrep_map.get(str(sid))
    if not isinstance(rec, dict):
        return None
    out = (rec.get("semantic_representation") or "").strip()
    return out or None


def validate_location_continuity_semrep(
    final_steps: List[str],
    vocab_map: Dict[str, str],
    semrep_map: Dict[str, Dict[str, str]],
    semrep_step_to_id: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    SemRep-based Location Consistency.

    Logic:
    1. Tracks the last known location of an Object based on 'Destination' roles in previous steps.
    2. If a subsequent step explicitly specifies a 'Location' (or 'Container') role for that Object,
       it must match the tracked location.

    This detects: "Add noodles to sink" -> "Stir noodles in pot" (sink != pot).
    It does NOT use hardcoded lists of "bad" locations; it strictly checks continuity.
    """
    issues: List[str] = []

    # Map: object_name -> location_name
    # We use simple normalized strings for tracking.
    obj_loc_tracker: Dict[str, str] = {}

    # Roles that imply setting a new location
    # Note: We rely on standard SemRep roles.
    DEST_ROLES = {"Destination", "Into", "To", "On", "In"}

    # Roles that imply checking the current location
    LOC_ROLES = {"Location", "Container", "In", "At"}

    def get_first_entity(role_val: str) -> Optional[str]:
        # Helper to extract the main entity from a SemRep value (e.g. "pot" from "pot(large)")
        ents = _extract_required_entities_from_role_value(str(role_val))
        if ents:
            return sorted(list(ents))[0]  # deterministic pick
        return None

    for i, step_text in enumerate(final_steps):
        sr = _semrep_for_step_text(step_text, vocab_map, semrep_map, semrep_step_to_id)
        if not sr:
            continue

        parsed = parse_semrep_minimal(sr)
        if not parsed:
            continue

        pred, roles = parsed

        # 1. Identify the main Object(s) of this step
        # Usually defined by 'Object' or 'Theme' role
        objs_in_step = set()
        for r in ["Object", "Theme", "Patient"]:
            val = roles.get(r)
            if val:
                ents = _extract_required_entities_from_role_value(str(val))
                objs_in_step.update(ents)

        if not objs_in_step:
            continue

        # 2. Check for Location Mismatch (Validation)
        # Does this step imply the action happens AT a specific place?
        current_loc_val = None
        for r in LOC_ROLES:
            if r in roles:
                current_loc_val = get_first_entity(roles[r])
                if current_loc_val:
                    break

        if current_loc_val:
            for obj in objs_in_step:
                # If we tracked this object previously, and the location differs
                if obj in obj_loc_tracker:
                    prev_loc = obj_loc_tracker[obj]
                    # Simple string equality check (normalized)
                    if prev_loc != current_loc_val:
                        issues.append(
                            f"location_consistency_semrep: Object '{obj}' was previously moved to '{prev_loc}', "
                            f"but step {i} occurs at '{current_loc_val}'. "
                            f"(Pred={pred}). Step: '{step_text}'"
                        )

        # 3. Update Tracker (State Change)
        # Does this step move the object TO a new place?
        # Note: GET/TAKE usually moves object to "Agent" (hand), we clear location logic there.
        if pred in {"GET", "TAKE", "PICK", "RETRIEVE"}:
            for obj in objs_in_step:
                # Object is now in hand; previous location is irrelevant/cleared
                if obj in obj_loc_tracker:
                    del obj_loc_tracker[obj]

        else:
            # Check for explicitly setting a destination
            new_dest_val = None
            for r in DEST_ROLES:
                if r in roles:
                    new_dest_val = get_first_entity(roles[r])
                    if new_dest_val:
                        break

            if new_dest_val:
                for obj in objs_in_step:
                    obj_loc_tracker[obj] = new_dest_val

    return issues


def _inventory_missing_records_from_semrep_sequence(
    n_steps: int,
    get_semrep_at: Any,
    get_step_text_at: Any,
    check_indices: Optional[Set[int]] = None,
    require_roles: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Internal helper: runs SemRep-driven inventory over a step sequence and returns
    structured "missing entity" records per step.
    get_semrep_at(i) -> semrep string or ""
    get_step_text_at(i) -> step text string
    If check_indices is provided, we still PROCESS all steps to maintain HAVE state,
    but only RETURN records for indices in the reporting window (check_indices +/-1).
    """
    roles_to_require = (
        set(require_roles) if require_roles is not None else set(_DEFAULT_REQUIRE_ROLES)
    )
    have: Set[str] = set()
    records: List[Dict[str, Any]] = []
    report_idxs: Optional[Set[int]] = None
    if check_indices is not None:
        report_idxs = set()
        for i in check_indices:
            if not isinstance(i, int):
                continue
            for j in (i - 1, i, i + 1):
                if 0 <= j < n_steps:
                    report_idxs.add(j)

    def _should_report(i: int) -> bool:
        return (report_idxs is None) or (i in report_idxs)

    for i in range(n_steps):
        sr = get_semrep_at(i) or ""
        if not sr:
            continue
        parsed = parse_semrep_minimal(sr)
        if not parsed:
            continue
        pred, roles = parsed
        pred = (pred or "").upper()
        # acquire
        if pred in _ACQUIRE_PREDS:
            for role_name in ("Object", "Instrument"):
                v = roles.get(role_name)
                for ent in _extract_required_entities_from_role_value(str(v or "")):
                    have.add(ent)
            continue
        # reference checks
        for role_name, v in (roles or {}).items():
            if role_name == "Agent":
                continue
            if role_name not in roles_to_require:
                continue
            for ent in _extract_required_entities_from_role_value(str(v or "")):
                if ent not in have and _should_report(i):
                    records.append(
                        {
                            "idx": i,
                            "pred": pred,
                            "role": role_name,
                            "ent": ent,
                            "step": str(get_step_text_at(i) or ""),
                        }
                    )
        # release
        if pred in _RELEASE_PREDS:
            for role_name in ("Object", "Instrument"):
                v = roles.get(role_name)
                for ent in _extract_required_entities_from_role_value(str(v or "")):
                    have.discard(ent)
    return records


def validate_inventory_semrep_delta_against_original(
    take: Dict[str, Any],
    original_steps: List[str],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    vocab_map: Dict[str, str],
    semrep_map: Dict[str, Dict[str, str]],
    semrep_step_to_id: Optional[Dict[str, str]] = None,
    check_indices: Optional[Set[int]] = None,
    require_roles: Optional[Set[str]] = None,
) -> List[str]:
    """
    Inventory validation that ignores "baseline" missing-acquisition issues already present
    in ORIGINAL, but still flags NEW inconsistencies introduced by edits.
    Rule:
      - For steps with mod in {u, ms, mt}:
          if the same (ent, role) missing-issue exists at the corresponding ORIGINAL old index,
          we ignore it as a baseline issue.
      - Otherwise, we report as usual (subject to check_indices reporting window).
    This matches your requirement:
      - ignore milk mismatch if it already existed in ORIGINAL
      - DO flag sugar mismatch if it appears only after substitution changed acquisition
    """
    issues: List[str] = []
    # Map new index -> meta entry
    meta_by_new: Dict[int, Dict[str, Any]] = {}
    for m in meta:
        newv = m.get("new")
        if isinstance(newv, int):
            meta_by_new[int(newv)] = m
    # --- Baseline: compute missing records on ORIGINAL using original semreps (idx-based, no text->id ambiguity) ---
    n_orig = len(original_steps)
    baseline_records = _inventory_missing_records_from_semrep_sequence(
        n_steps=n_orig,
        get_semrep_at=lambda i: _semrep_for_original_idx(take, i, vocab_map, semrep_map),
        get_step_text_at=lambda i: original_steps[i] if 0 <= i < n_orig else "",
        check_indices=None,  # baseline: compute for all
        require_roles=require_roles,
    )
    baseline_by_old: Dict[int, Set[Tuple[str, str]]] = defaultdict(set)
    for r in baseline_records:
        oi = r.get("idx")
        ent = r.get("ent")
        role = r.get("role")
        if isinstance(oi, int) and isinstance(ent, str) and isinstance(role, str):
            baseline_by_old[int(oi)].add((ent, role))
    # --- Now compute missing records on FINAL (reporting window controlled by check_indices) ---
    n_final = len(final_steps)
    final_records = _inventory_missing_records_from_semrep_sequence(
        n_steps=n_final,
        get_semrep_at=lambda i: _semrep_for_step_text(
            final_steps[i], vocab_map, semrep_map, semrep_step_to_id
        ),
        get_step_text_at=lambda i: final_steps[i] if 0 <= i < n_final else "",
        check_indices=check_indices,
        require_roles=require_roles,
    )
    for r in final_records:
        ni = r.get("idx")
        ent = r.get("ent")
        role = r.get("role")
        pred = r.get("pred")
        step_txt = r.get("step")
        if not isinstance(ni, int) or not isinstance(ent, str) or not isinstance(role, str):
            continue
        m = meta_by_new.get(int(ni), {})
        mod = str(m.get("mod") or "")
        oldv = m.get("old")
        # Ignore baseline issues ONLY for truly-unchanged/moved-original steps
        if mod in {"u", "ms", "mt"} and isinstance(oldv, int):
            if (ent, role) in baseline_by_old.get(int(oldv), set()):
                continue
        issues.append(
            f"inventory_semrep_delta: final_steps[{ni}] references '{ent}' via role '{role}' "
            f"but it was never acquired (pred={pred}). Step: '{step_txt}'"
        )
    return issues


def validate_error_realization_minimal(
    take: Dict[str, Any],
    original_steps: List[str],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    changed_new: Set[int],
    include_semrep: bool,
    vocab_map: Optional[Dict[str, str]],
    semrep_map: Optional[Dict[str, Dict[str, str]]],
    semrep_step_to_id: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Catch "fake" error realizations like:
      - mod='e', etype='wrong_execution' but step text is unchanged
      - (optional SemRep) wrong_execution changes predicate or changes nothing in roles
    """
    issues: List[str] = []

    planned: Dict[str, str] = {}
    for e in take.get("errors") or []:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        if not eid:
            continue
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        if etype:
            planned[eid] = etype

    for m in meta:
        if m.get("mod") != "e":
            continue
        eid = (m.get("eid") or "").strip() if isinstance(m.get("eid"), str) else ""
        etype = str(m.get("etype") or "").strip().lower() if m.get("etype") is not None else ""
        if not eid or eid not in planned:
            continue
        if planned[eid] == "transposition":
            continue
        if etype not in {"wrong_execution", "substitution"}:
            continue

        old = m.get("old")
        new = m.get("new")
        if not isinstance(old, int) or not isinstance(new, int):
            continue
        if not (0 <= old < len(original_steps)) or not (0 <= new < len(final_steps)):
            continue

        # Must actually differ from original (loose text equality)
        if _same_step_text_loose(original_steps[old], final_steps[new]):
            issues.append(
                f"error_realization: mod='e' etype='{etype}' eid={eid} is unchanged vs ORIGINAL[{old}]"
            )
            continue

        # Optional SemRep sanity when available
        if include_semrep:
            sr_old = _semrep_for_original_idx(take, old, vocab_map, semrep_map)
            sr_new = _semrep_for_step_text(
                final_steps[new], vocab_map, semrep_map, semrep_step_to_id
            )
            p_old = parse_semrep_minimal(sr_old) if sr_old else None
            p_new = parse_semrep_minimal(sr_new) if sr_new else None
            if p_old and p_new:
                pred_old, roles_old = p_old
                pred_new, roles_new = p_new

                # Ignore Agent in comparisons (usually constant "you")
                ro = _roles_norm_excluding(roles_old, exclude={"Agent"})
                rn = _roles_norm_excluding(roles_new, exclude={"Agent"})

                if etype == "wrong_execution":
                    if pred_old != pred_new:
                        issues.append(
                            f"error_realization: wrong_execution eid={eid} changed predicate {pred_old}->{pred_new}"
                        )
                    elif ro == rn:
                        issues.append(
                            f"error_realization: wrong_execution eid={eid} did not change any roles (SemRep identical excluding Agent)"
                        )
                elif etype == "substitution":
                    if pred_old == pred_new and ro == rn:
                        issues.append(
                            f"error_realization: substitution eid={eid} is SemRep-identical to original (excluding Agent)"
                        )

    return issues


def validate_adjacent_duplicates(final_steps: List[str]) -> List[str]:
    issues: List[str] = []
    for i in range(1, len(final_steps)):
        if is_near_duplicate_step(final_steps[i - 1], final_steps[i]):
            issues.append(
                f"adjacent_duplicate: near-duplicate steps at {i - 1},{i}: "
                f"'{final_steps[i - 1]}' ~ '{final_steps[i]}'"
            )
    return issues


# -------------------------
# SemRep degree consistency (adjacent steps, SemRep-only)
# -------------------------

DEGREE_FULL_VALUES: Set[str] = {"fully", "well", "completely"}
DEGREE_PARTIAL_VALUES: Set[str] = {"partially", "a_little", "slightly", "incompletely"}


def _roles_norm_except_degree(roles: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in roles.items():
        kk = str(k).strip()
        if not kk or kk == "Degree":
            continue
        out[kk] = _norm_semrep_value(str(v))
    return out


def _extract_pred_obj_degree(semrep: str) -> Optional[Tuple[str, str, str, Dict[str, str]]]:
    parsed = parse_semrep_minimal(semrep)
    if not parsed:
        return None
    pred, roles = parsed
    obj_raw = roles.get("Object") or ""
    obj = _norm_semrep_value(str(obj_raw))
    if not obj:
        return None
    deg = _norm_semrep_value(str(roles.get("Degree") or ""))
    roles_no_deg = _roles_norm_except_degree(roles)
    return pred, obj, deg, roles_no_deg


# -------------------------
# Inventory via SemRep (NO word allowlists; roles-only)
# -------------------------

# These are SemRep PREDICATES (not raw text words)
_ACQUIRE_PREDS: Set[str] = {"GET", "TAKE"}
_RELEASE_PREDS: Set[str] = {"RETURN", "PUT_AWAY", "PUT_BACK", "DISPOSE"}

# By default: inventory is required only for things you HOLD/USE directly.
# If you want stricter checks, pass a bigger set at call site.
_DEFAULT_REQUIRE_ROLES: Set[str] = {"Object", "Instrument"}

# Structural tokens of the SemRep value grammar (NOT domain words)
_STRUCT_TOKENS: Set[str] = {"of", "from", "in", "into", "on", "onto", "to", "with", "at", "for"}


def _semrep_value_tokens(v: str) -> List[str]:
    s = (v or "").strip().strip('"').strip("'").lower()
    # keep underscores; just normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return re.findall(r"[a-z0-9_]+", s)


def _extract_required_entities_from_role_value(v: str) -> Set[str]:
    """
    Roles can contain nested structures like:
      tomato_stalk(of(tomato)) -> should require tomato (host), not tomato_stalk (derived part)
      from(tomato)            -> should require tomato
      knife                   -> should require knife

    Rule (structure-only, no lexicon):
      - If value has parentheses: require ONLY tokens coming from inside (...) arguments.
      - If value has NO parentheses: require the first token (head).
      - Ignore structural tokens (_STRUCT_TOKENS) and very short tokens (len<3).
    """
    if not isinstance(v, str) or not v.strip():
        return set()

    s = (v or "").strip().strip('"').strip("'")
    has_paren = "(" in s and ")" in s

    toks = _semrep_value_tokens(s)
    toks = [t for t in toks if t and t not in _STRUCT_TOKENS and len(t) >= 3 and not t.isdigit()]
    if not toks:
        return set()

    if not has_paren:
        # plain entity like "knife"
        return {toks[0]}

    # has parentheses: keep only tokens that appear inside the outermost "( ... )"
    m = re.search(r"\((.*)\)", s, flags=re.DOTALL)
    inside = m.group(1) if m else ""
    inside_toks = _semrep_value_tokens(inside)
    inside_toks = [
        t for t in inside_toks if t and t not in _STRUCT_TOKENS and len(t) >= 3 and not t.isdigit()
    ]

    return set(inside_toks)


def validate_inventory_semrep(
    final_steps: List[str],
    vocab_map: Dict[str, str],
    semrep_map: Dict[str, Dict[str, str]],
    semrep_step_to_id: Optional[Dict[str, str]] = None,
    check_indices: Optional[Set[int]] = None,
    require_roles: Optional[Set[str]] = None,
) -> List[str]:
    """
    Inventory validator driven ONLY by SemRep.

    Tracks HAVE-set of entities acquired via GET/TAKE(Object/Instrument...).
    Flags entity references in selected roles if entity not in HAVE.

    - No fixture allowlists.
    - No domain word tweaking.
    - Roles control strictness via require_roles.

    Reporting:
      - If check_indices is provided, only report near those indices (+/-1),
        BUT we still process all steps to keep state consistent.
    """
    issues: List[str] = []
    have: Set[str] = set()

    roles_to_require = (
        set(require_roles) if require_roles is not None else set(_DEFAULT_REQUIRE_ROLES)
    )

    n = len(final_steps)
    report_idxs: Optional[Set[int]] = None
    if check_indices is not None:
        report_idxs = set()
        for i in check_indices:
            if not isinstance(i, int):
                continue
            for j in (i - 1, i, i + 1):
                if 0 <= j < n:
                    report_idxs.add(j)

    def _should_report(i: int) -> bool:
        return (report_idxs is None) or (i in report_idxs)

    for i, step_text in enumerate(final_steps):
        sr = _semrep_for_step_text(step_text, vocab_map, semrep_map, semrep_step_to_id)
        if not sr:
            continue
        parsed = parse_semrep_minimal(sr)
        if not parsed:
            continue

        pred, roles = parsed
        pred = (pred or "").upper()

        # --- acquire ---
        if pred in _ACQUIRE_PREDS:
            # Acquire what GET/TAKE says as Object (and optionally Instrument if present)
            for role_name in ("Object", "Instrument"):
                v = roles.get(role_name)
                for ent in _extract_required_entities_from_role_value(str(v or "")):
                    have.add(ent)
            continue

        # --- reference checks (roles-only) ---
        for role_name, v in (roles or {}).items():
            if role_name == "Agent":
                continue
            if role_name not in roles_to_require:
                continue
            for ent in _extract_required_entities_from_role_value(str(v or "")):
                if ent not in have and _should_report(i):
                    issues.append(
                        f"inventory_semrep: final_steps[{i}] references '{ent}' via role '{role_name}' "
                        f"but it was never acquired (pred={pred}). Step: '{final_steps[i]}'"
                    )

        # --- release ---
        if pred in _RELEASE_PREDS:
            for role_name in ("Object", "Instrument"):
                v = roles.get(role_name)
                for ent in _extract_required_entities_from_role_value(str(v or "")):
                    # optional: warn if releasing something never acquired
                    if ent not in have and _should_report(i):
                        issues.append(
                            f"inventory_semrep: final_steps[{i}] releases '{ent}' (pred={pred}) "
                            f"but it was never acquired. Step: '{final_steps[i]}'"
                        )
                    have.discard(ent)

    return issues


def validate_degree_consistency_semrep_adjacent(
    final_steps: List[str],
    vocab_map: Dict[str, str],
    semrep_map: Dict[str, Dict[str, str]],
    semrep_step_to_id: Optional[Dict[str, str]] = None,
    check_indices: Optional[Set[int]] = None,
) -> List[str]:
    """
    Adjacent-only SemRep check:
      A) FULL -> PARTIAL for same (Predicate,Object)
      B) FULL -> Degree dropped for same (Predicate,Object) with identical other roles
    Uses SemRep resolved strictly via: final_step_text -> vocab_id -> semrep_json.
    """
    issues: List[str] = []
    n = len(final_steps)
    if n < 2:
        return issues

    for i in range(n - 1):
        j = i + 1
        if check_indices is not None and i not in check_indices and j not in check_indices:
            continue

        sr_i = _semrep_for_step_text(final_steps[i], vocab_map, semrep_map, semrep_step_to_id) or ""
        sr_j = _semrep_for_step_text(final_steps[j], vocab_map, semrep_map, semrep_step_to_id) or ""
        if not sr_i or not sr_j:
            continue

        pi = _extract_pred_obj_degree(sr_i)
        pj = _extract_pred_obj_degree(sr_j)
        if not pi or not pj:
            continue

        pred_i, obj_i, deg_i, roles_i_no_deg = pi
        pred_j, obj_j, deg_j, roles_j_no_deg = pj
        if pred_i != pred_j or obj_i != obj_j:
            continue

        if deg_i in DEGREE_FULL_VALUES and deg_j in DEGREE_PARTIAL_VALUES:
            issues.append(
                f"degree_consistency: FULL->PARTIAL for {pred_i}(Object:{obj_i}) at {i}->{j}"
            )
            continue

        if deg_i in DEGREE_FULL_VALUES and deg_j == "" and roles_i_no_deg == roles_j_no_deg:
            issues.append(
                f"degree_consistency: Degree dropped after FULL for {pred_i}(Object:{obj_i}) at {i}->{j}"
            )

    return issues


# -------------------------
# Ordering constraints (same object) — ported from error_simulator_new.py
# -------------------------

ORDERING_CONSTRAINTS_SAME_OBJECT = [
    ("CLOSE", "OPEN"),
    ("COVER", "OPEN"),
    ("TURN_OFF", "TURN_ON"),
    ("DEFLATE", "INFLATE"),
    ("PUT_AWAY", "GET"),
    ("PUT_AWAY", "TAKE"),
    ("PUT_BACK", "GET"),
    ("PUT_BACK", "TAKE"),
    ("FIT", "SEPARATE"),
    ("READ", "OPEN"),
    ("RETURN", "GET"),
    ("COOK", "ADD"),
    ("COMBINE", "ADD"),
    ("ADD", "GET"),
    ("STIR", "POUR"),
    ("CUT", "GET"),
]

ORDERING_ENFORCED_DEPENDENTS = {
    "RETURN",
    "PUT_AWAY",
    "PUT_BACK",
    "CLOSE",
    "COVER",
    "TURN_OFF",
    "DEFLATE",
    "READ",
    "FIT",
}

# Text triggers for predicates in ORDERING_CONSTRAINTS_SAME_OBJECT (heuristic).
# We intentionally keep this minimal + explicit.
_PRED_TO_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "OPEN": ("open ",),
    "CLOSE": ("close ",),
    "COVER": ("cover ",),
    "TURN_ON": ("turn on ",),
    "TURN_OFF": ("turn off ",),
    "INFLATE": ("inflate ",),
    "DEFLATE": ("deflate ",),
    "GET": ("get ", "take ", "grab ", "pick ", "pick up ", "retrieve "),
    "TAKE": ("take ", "grab ", "pick ", "pick up ", "remove "),
    "PUT_AWAY": ("put away ",),
    "PUT_BACK": ("put back ",),
    "RETURN": ("return ",),
    "SEPARATE": ("separate ",),
    "FIT": ("fit ",),
    "READ": ("read ",),
    "ADD": ("add ",),
    "COMBINE": ("combine ",),
    "POUR": ("pour ",),
    "STIR": ("stir ",),
    "CUT": ("cut ",),
    "COOK": ("cook ",),
}

_STOP_TOKENS = {"of", "in", "on", "from", "to", "with", "and", "or", "into", "onto", "at", "for"}


def _detect_predicate_and_object(step_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristic parse:
      predicate = based on leading verb phrase
      object    = first chunk after predicate until a stop token ("to", "with", ...)
    Example:
      "Return the wrenches to the table" -> ("RETURN", "wrenches")
    """
    if not isinstance(step_text, str):
        return None, None
    s = normalize_ws(step_text).lower()
    # normalize punctuation to spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = normalize_ws(s)

    pred: Optional[str] = None
    used_prefix: Optional[str] = None
    # longest-prefix-first to prefer "pick up " over "pick "
    candidates: List[Tuple[str, str]] = []
    for p, prefixes in _PRED_TO_PREFIXES.items():
        for pref in prefixes:
            candidates.append((p, pref))
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    for p, pref in candidates:
        if s.startswith(pref):
            pred = p
            used_prefix = pref
            break

    if pred is None or used_prefix is None:
        return None, None

    rest = s[len(used_prefix) :].strip()
    if not rest:
        return pred, None

    toks = [t for t in rest.split() if t]
    # drop articles
    while toks and toks[0] in {"the", "a", "an"}:
        toks = toks[1:]

    obj_tokens: List[str] = []
    for t in toks:
        if t in _STOP_TOKENS:
            break
        obj_tokens.append(t)

    obj = " ".join(obj_tokens).strip() if obj_tokens else None
    return pred, obj


def check_ordering_constraints_same_object(
    final_steps: List[str],
    check_indices: Optional[Set[int]] = None,
) -> List[str]:
    """
    Enforce ordering ONLY for selected dependents, and report ONLY when the
    dependent occurrence is in check_indices (i.e., model-changed steps).
    Rules:
      - first dependent(obj) is allowed
      - subsequent dependent(obj) requires ANY prerequisite(obj) in between (OR)
    """
    issues: List[str] = []

    dep_to_prereqs: Dict[str, Set[str]] = defaultdict(set)
    prereq_to_deps: Dict[str, Set[str]] = defaultdict(set)

    for dep, pre in ORDERING_CONSTRAINTS_SAME_OBJECT:
        if dep not in ORDERING_ENFORCED_DEPENDENTS:
            continue
        dep_to_prereqs[dep].add(pre)
        prereq_to_deps[pre].add(dep)

    seen_dep: Dict[Tuple[str, str], bool] = defaultdict(bool)  # (dep,obj)
    unlocked: Dict[Tuple[str, str], bool] = defaultdict(
        bool
    )  # (dep,obj) prereq seen since last dep

    for i, step in enumerate(final_steps):
        pred, obj = _detect_predicate_and_object(step)
        if not pred or not obj:
            continue

        # Any prerequisite unlocks its dependents (OR)
        if pred in prereq_to_deps:
            for dep in prereq_to_deps[pred]:
                unlocked[(dep, obj)] = True

        # Dependent logic
        if pred in dep_to_prereqs:
            key = (pred, obj)

            # Only report if this dependent occurrence is in a model-changed step
            should_check = (check_indices is None) or (i in check_indices)

            if should_check and seen_dep[key] and not unlocked[key]:
                prereqs = sorted(dep_to_prereqs[pred])
                issues.append(
                    f"ordering constraint violated at final_steps[{i}]: "
                    f"{pred}({obj}) repeats without any of {prereqs}({obj}) in between. "
                    f"Step: '{final_steps[i]}'"
                )

            # Update state for future checks regardless of should_check
            seen_dep[key] = True
            unlocked[key] = False

    return issues


def find_global_nonrepeatable_duplicates(final_steps: List[str]) -> List[str]:
    """
    Disallow repeats of non-repeatable actions anywhere in the procedure,
    not only adjacent. Uses normalized whitespace + lowercase equality.
    """
    seen: Dict[str, int] = {}
    issues: List[str] = []
    for i, s in enumerate(final_steps):
        if not isinstance(s, str):
            continue
        if not is_nonrepeatable_action(s):
            continue
        key = normalize_ws(s).lower()
        if key in seen:
            j = seen[key]
            issues.append(
                f"non-repeatable step repeated (global) at {j} and {i}: '{final_steps[i]}'"
            )
        else:
            seen[key] = i
    return issues


def validate_rewrite(
    original_steps: List[str],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    changed_new: Optional[Set[int]] = None,
) -> Tuple[bool, List[str]]:
    """
    Deterministic sanity checks. Returns (ok, issues).
    """
    issues: List[str] = []

    # Allowed mods + id formats
    allowed_mod = {"u", "e", "a", "i", "c", "d", "ms", "mt"}

    # --- etype consistency with mod ---
    for i, m in enumerate(meta):
        mod = m.get("mod")
        etype = m.get("etype")

        # normalize empty strings already done earlier, но на всякий:
        if isinstance(etype, str) and not etype.strip():
            etype = None

        if mod == "e":
            if etype not in {"wrong_execution", "substitution"}:
                issues.append(
                    f"meta[{i}] mod='e' requires etype in {{wrong_execution, substitution}} (got {etype})"
                )

        elif mod == "i":
            if etype != "insertion":
                issues.append(f"meta[{i}] mod='i' requires etype='insertion' (got {etype})")

        elif mod == "d":
            if etype != "deletion":
                issues.append(f"meta[{i}] mod='d' requires etype='deletion' (got {etype})")

        elif mod in {"ms", "mt"}:
            if etype != "transposition":
                issues.append(f"meta[{i}] mod='{mod}' requires etype='transposition' (got {etype})")

        elif mod == "c":
            # Correction steps carry an explicit etype to keep writer/judge semantics aligned.
            if etype != "correction":
                issues.append(f"meta[{i}] mod='c' requires etype='correction' (got {etype})")

        elif mod in {"u", "a"}:
            # Unchanged / adjusted steps should not carry an etype.
            if etype is not None:
                issues.append(f"meta[{i}] mod='{mod}' requires etype=null (got {etype})")

    def _nonempty_str(x: Any) -> bool:
        return isinstance(x, str) and x.strip() != ""

    move_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ms": 0, "mt": 0})

    for i, m in enumerate(meta):
        if "new" not in m or "old" not in m or "mod" not in m:
            issues.append(f"meta[{i}] missing required keys.")
            continue
        mod = m.get("mod")
        if mod not in allowed_mod:
            issues.append(f"meta[{i}].mod invalid: {mod}")

        # Require eid for any error-bearing / adjusted / moved / deletion steps
        if mod in {"e", "i", "a", "ms", "mt", "d"}:
            if not _nonempty_str(m.get("eid")):
                issues.append(
                    f"meta[{i}] mod='{mod}' requires non-empty eid (use plan error_id string)"
                )
        # Require cid for corrections
        if mod == "c":
            if not _nonempty_str(m.get("cid")):
                issues.append(
                    f"meta[{i}] mod='c' requires non-empty cid (use plan correction_id string)"
                )

        # Deletion must have new="" and old=int
        if mod == "d":
            if m.get("new") not in ("", None):
                issues.append(f"meta[{i}] mod='d' must have new=''")
            if not isinstance(m.get("old"), int):
                issues.append(f"meta[{i}] mod='d' must have int old")

        # Insertion/correction must have old=""
        if mod in {"i", "c"}:
            if m.get("old") not in ("", None):
                issues.append(f"meta[{i}] mod='{mod}' must have old='' (new step)")

        # Adjusted/error/unchanged/moves must preserve old index
        if mod in {"u", "e", "a", "ms", "mt"}:
            if not isinstance(m.get("old"), int):
                issues.append(f"meta[{i}] mod='{mod}' must have int old")

        # Non-deletions must have int new
        if mod != "d":
            if not isinstance(m.get("new"), int):
                issues.append(f"meta[{i}] mod='{mod}' must have int new")

        # ms/mt must be verbatim ORIGINAL[old] and count exactly once per eid
        if mod in {"ms", "mt"}:
            old = m.get("old")
            new = m.get("new")
            eid = m.get("eid")
            if _nonempty_str(eid):
                move_counts[str(eid).strip()][mod] += 1
            if not isinstance(old, int) or not isinstance(new, int):
                issues.append(f"meta[{i}] mod='{mod}' must have int old and int new")
            else:
                if 0 <= old < len(original_steps) and 0 <= new < len(final_steps):
                    if normalize_ws(final_steps[new]) != normalize_ws(original_steps[old]):
                        issues.append(f"meta[{i}] {mod} must be verbatim ORIGINAL[{old}]")
                else:
                    issues.append(
                        f"meta[{i}] {mod} old/new out of range (old={old}, new={new}, final_len={len(final_steps)})"
                    )

        newv = m["new"]
        if newv == "" or newv is None:
            continue
        if not isinstance(newv, int):
            issues.append(f"meta[{i}].new must be int or '' (got {type(newv)}).")
            continue

    # Each transposition eid must have exactly one ms and one mt
    for eid, cts in move_counts.items():
        if cts.get("ms", 0) != 1 or cts.get("mt", 0) != 1:
            issues.append(
                f"transposition eid={eid} must have exactly one ms and one mt (ms={cts.get('ms', 0)} mt={cts.get('mt', 0)})"
            )

    # Unchanged steps must match source text exactly (ignoring whitespace)
    for i, m in enumerate(meta):
        if m.get("mod") == "u":
            old = m.get("old")
            new = m.get("new")
            if not isinstance(old, int) or not isinstance(new, int):
                issues.append(f"meta[{i}] mod='u' must have int old and int new.")
                continue
            if not (0 <= new < len(final_steps)):
                issues.append(
                    f"meta[{i}] new index {new} out of range for final_steps (len={len(final_steps)})."
                )
                continue
            if old < 0 or old >= len(original_steps):
                issues.append(f"meta[{i}] old out of range for mod='u': {old}.")
                continue
            if normalize_ws(final_steps[new]) != normalize_ws(original_steps[old]):
                issues.append(f"meta[{i}] mod='u' but final_steps[{new}] != original_steps[{old}].")

    # Insertion must not be near-duplicate of adjacent steps
    for i, m in enumerate(meta):
        if m.get("mod") == "i":
            new = m.get("new")
            if not isinstance(new, int) or not (0 <= new < len(final_steps)):
                issues.append(f"meta[{i}] insertion index {new} out of range.")
                continue
            cur = final_steps[new]
            prev = final_steps[new - 1] if new - 1 >= 0 else None
            nxt = final_steps[new + 1] if new + 1 < len(final_steps) else None
            if prev and is_near_duplicate_step(cur, prev):
                issues.append(
                    f"insertion at final_steps[{new}] is near-duplicate of previous step."
                )
            if nxt and is_near_duplicate_step(cur, nxt):
                issues.append(f"insertion at final_steps[{new}] is near-duplicate of next step.")

    # Corrections must not be empty filler (Continue/Proceed/etc.)
    BAD_CORR_PREFIXES = (
        "continue",
        "proceed",
        "go on",
        "carry on",
        "keep going",
        "then continue",
        "next",
        "move on",
    )
    for i, m in enumerate(meta):
        if m.get("mod") != "c":
            continue
        new = m.get("new")
        # ADDED BOUNDS CHECK HERE
        if not isinstance(new, int) or not (0 <= new < len(final_steps)):
            issues.append(f"meta[{i}] correction index {new} out of range.")
            continue
        txt = normalize_ws(final_steps[new]).lower()
        if len(txt.split()) <= 2 or txt.startswith(BAD_CORR_PREFIXES):
            issues.append(
                f"correction at final_steps[{new}] looks like filler: '{final_steps[new]}'"
            )

    # Non-repeatable exact repetitions next to each other
    for i in range(1, len(final_steps)):
        a, b = final_steps[i - 1], final_steps[i]
        if is_nonrepeatable_action(a) and is_near_duplicate_step(a, b):
            issues.append(f"adjacent non-repeatable duplicate steps at {i - 1},{i}: '{a}' ~ '{b}'.")

    # Ordering constraints (same object) — heuristic text check
    changed_new = changed_new or set()

    issues.extend(find_global_nonrepeatable_duplicates(final_steps))

    # Substitution distance checks (legacy)
    all_orig_norm = {normalize_step_text(x) for x in original_steps}
    for i, m in enumerate(meta):
        if m.get("mod") != "e":
            continue
        if m.get("etype") != "substitution":
            continue
        old = m.get("old")
        new = m.get("new")
        if not isinstance(old, int) or not isinstance(new, int):
            issues.append(f"meta[{i}] substitution requires int old and int new")
            continue
        if not (0 <= old < len(original_steps)) or not (0 <= new < len(final_steps)):
            issues.append(f"meta[{i}] substitution indices out of range")
            continue
        orig = original_steps[old]
        sub = final_steps[new]
        if is_too_similar_substitution(orig, sub):
            issues.append(f"meta[{i}] substitution too similar to original step old={old}")
        sub_norm = normalize_step_text(sub)
        orig_norm = normalize_step_text(orig)
        if sub_norm in all_orig_norm and sub_norm != orig_norm:
            issues.append(f"meta[{i}] substitution is copy of a different original step")

    return (len(issues) == 0), issues


def check_ordering_constraints_same_object_semrep(
    final_steps: List[str],
    vocab_map: Dict[str, str],
    semrep_map: Dict[str, Dict[str, str]],
    semrep_step_to_id: Optional[Dict[str, str]] = None,
    check_indices: Optional[Set[int]] = None,
) -> List[str]:
    issues: List[str] = []

    dep_to_prereqs: Dict[str, Set[str]] = defaultdict(set)
    prereq_to_deps: Dict[str, Set[str]] = defaultdict(set)

    for dep, pre in ORDERING_CONSTRAINTS_SAME_OBJECT:
        if dep not in ORDERING_ENFORCED_DEPENDENTS:
            continue
        dep_to_prereqs[dep].add(pre)
        prereq_to_deps[pre].add(dep)

    seen_dep: Dict[Tuple[str, str], bool] = defaultdict(bool)  # (dep,obj_entity)
    unlocked: Dict[Tuple[str, str], bool] = defaultdict(bool)  # prereq seen since last dep

    for i, step_text in enumerate(final_steps):
        sr = _semrep_for_step_text(step_text, vocab_map, semrep_map, semrep_step_to_id)
        if not sr:
            continue
        parsed = parse_semrep_minimal(sr)
        if not parsed:
            continue
        pred, roles = parsed
        pred = (pred or "").upper()

        obj_raw = roles.get("Object") or ""
        if not obj_raw:
            continue

        objs = _extract_required_entities_from_role_value(str(obj_raw))
        if not objs:
            # fallback: хотя бы нормализованный Object как ключ
            o = _norm_semrep_value(str(obj_raw))
            if o:
                objs = {o}
            else:
                continue

        for obj in objs:
            # prerequisites unlock
            if pred in prereq_to_deps:
                for dep in prereq_to_deps[pred]:
                    unlocked[(dep, obj)] = True

            # dependent check
            if pred in dep_to_prereqs:
                key = (pred, obj)
                should_check = (check_indices is None) or (i in check_indices)

                if should_check and seen_dep[key] and not unlocked[key]:
                    prereqs = sorted(dep_to_prereqs[pred])
                    issues.append(
                        f"ordering_same_object: violation at final_steps[{i}]: "
                        f"{pred}(Object:{obj}) repeats without any of {prereqs}(Object:{obj}) in between. "
                        f"Step: '{final_steps[i]}'"
                    )

                seen_dep[key] = True
                unlocked[key] = False

    return issues


def validate_plan_coverage(take: Dict[str, Any], meta: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    errors = take.get("errors", []) or []
    corrs = take.get("corrections", []) or []

    out_eids: Set[str] = set()
    del_old_idxs: Set[int] = set()
    corr_counts: Dict[str, int] = defaultdict(int)
    msmt_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ms": 0, "mt": 0})
    del_old_by_eid: Dict[str, List[int]] = defaultdict(list)

    for m in meta:
        mod = m.get("mod")
        eid = m.get("eid")
        cid = m.get("cid")
        if mod in {"e", "a", "i", "ms", "mt", "d"} and isinstance(eid, str) and eid.strip():
            out_eids.add(eid.strip())
        if mod == "d" and isinstance(m.get("old"), int):
            del_old_idxs.add(int(m["old"]))
            if isinstance(eid, str) and eid.strip():
                del_old_by_eid[eid.strip()].append(int(m["old"]))
        if mod == "c" and isinstance(cid, str) and cid.strip():
            corr_counts[cid.strip()] += 1
        if mod in {"ms", "mt"} and isinstance(eid, str) and eid.strip():
            msmt_counts[eid.strip()][mod] += 1

    # error coverage (by eid)
    for e in errors:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        if not eid:
            continue
        # must appear somewhere as an eid in output (we don't force deletion form)
        if eid not in out_eids:
            issues.append(f"plan_coverage: missing error realization eid={eid}")

        # transposition must have exactly one ms and one mt
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        if etype == "transposition":
            cts = msmt_counts.get(eid, {"ms": 0, "mt": 0})
            if cts.get("ms", 0) != 1 or cts.get("mt", 0) != 1:
                issues.append(
                    f"plan_coverage: transposition eid={eid} must have exactly one ms and one mt (ms={cts.get('ms', 0)} mt={cts.get('mt', 0)})"
                )

        # deletion must be realized as mod='d' and must delete src or an allowed alternate
        if etype == "deletion":
            src = e.get("step_index")
            if not isinstance(src, int):
                src = e.get("src_step_idx")
            alt = e.get("alternate_src_indices")
            allowed: Set[int] = set()
            if isinstance(src, int):
                allowed.add(int(src))
            if isinstance(alt, list):
                for v in alt:
                    if isinstance(v, int):
                        allowed.add(int(v))

            realized = del_old_by_eid.get(eid, [])
            if not realized:
                issues.append(f"plan_coverage: deletion eid={eid} missing mod='d' meta entry")
            elif len(realized) != 1:
                issues.append(
                    f"plan_coverage: deletion eid={eid} must delete exactly one step (got olds={realized})"
                )
            else:
                old_del = realized[0]
                if allowed and old_del not in allowed:
                    issues.append(
                        f"plan_coverage: deletion eid={eid} deleted old={old_del} but expected one of {sorted(allowed)}"
                    )
    # correction coverage: exactly once
    for c in corrs:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("correction_id") or "").strip()
        if not cid:
            continue
        n = corr_counts.get(cid, 0)
        if n == 0:
            issues.append(f"plan_coverage: missing correction cid={cid}")
        elif n != 1:
            issues.append(f"plan_coverage: correction cid={cid} appears {n} times (must be 1)")

    return issues


def validate_old_index_coverage(original_steps: List[str], meta: List[Dict[str, Any]]) -> List[str]:
    """
    Require each original old index 0..N-1 to appear exactly once in meta,
    except insertions/corrections (i/c) which do not consume old.
    Deletions DO consume old (they count as covering that old index).
    """
    issues: List[str] = []
    n = len(original_steps)
    seen: Dict[int, int] = defaultdict(int)

    for i, m in enumerate(meta):
        mod = m.get("mod")
        if mod in {"i", "c"}:
            continue
        old = m.get("old")
        if not isinstance(old, int):
            issues.append(f"old_coverage: meta[{i}] mod={mod} requires int old")
            continue
        seen[int(old)] += 1

    for k in range(n):
        c = seen.get(k, 0)
        if c == 0:
            issues.append(f"old_coverage: missing old index {k}")
        elif c > 1:
            issues.append(f"old_coverage: old index {k} appears {c} times")

    return issues


def validate_transposition_realized(take: Dict[str, Any], meta: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    for e in take.get("errors") or []:
        if not isinstance(e, dict):
            continue
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        if etype != "transposition":
            continue
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        if not eid:
            continue
        src = e.get("step_index")
        if not isinstance(src, int):
            src = e.get("src_step_idx")

        tgt = None
        if isinstance(e.get("spec"), dict):
            tgt = e["spec"].get("transposition_target")
        if not isinstance(tgt, int):
            tgt = e.get("target_step_idx")
        if not isinstance(src, int) or not isinstance(tgt, int):
            continue

        msmt = [
            m for m in meta if (m.get("eid") or "").strip() == eid and m.get("mod") in {"ms", "mt"}
        ]
        if len(msmt) != 2:
            continue

        old_to_new: Dict[int, int] = {}
        for m in msmt:
            if isinstance(m.get("old"), int) and isinstance(m.get("new"), int):
                old_to_new[int(m["old"])] = int(m["new"])

        olds = sorted(old_to_new.keys())
        if len(olds) != 2:
            continue

        # LOG (not fail) if the realized swap pair differs from the planned one.
        planned_pair = (src, tgt) if isinstance(src, int) and isinstance(tgt, int) else None
        if planned_pair is not None and set(olds) != set(planned_pair):
            logger.warning(
                "transposition_shift: eid=%s planned_pair=%s realized_pair=%s",
                eid,
                planned_pair,
                tuple(olds),
            )

        # Require reversal of order for the realized pair (whatever it is).
        a, b = olds[0], olds[1]
        na, nb = old_to_new.get(a), old_to_new.get(b)
        if not isinstance(na, int) or not isinstance(nb, int):
            continue
        # If original order (a<b) is NOT reversed (na<nb), the swap wasn't realized.
        if (a - b) * (na - nb) >= 0:
            issues.append(
                f"transposition eid={eid} NOT realized: old_pair={tuple(olds)} new_positions=({na},{nb})"
            )

    return issues


def build_retry_prompt(user_prompt: str, last_raw: str, error_msg: str) -> str:
    """
    Minimal retry wrapper (ported from legacy):
    - keep task same
    - explicitly tell what failed
    - force JSON only
    """
    tail = (last_raw or "")[-2500:]
    missing_cids = re.findall(r"missing correction cid=([^\s;]+)", error_msg or "")
    missing_hint = ""
    if missing_cids:
        missing_hint = (
            "\nMISSING CORRECTIONS:\n"
            "You MUST include exactly one meta entry mod='c' for EACH of these correction_id values:\n"
            + "\n".join(f"- {c}" for c in missing_cids)
            + "\nEach such meta entry must set cid to the exact string above.\n"
        )
    return (
        user_prompt
        + "\n\nRETRY NOTICE:\n"
        + f"The previous output was invalid: {error_msg}\n"
        + "Fix the issues and return ONLY the JSON object.\n"
        + "Use the EXACT correction_id strings from the plan as cid values in meta entries with mod='c'.\n"
        + missing_hint
        + "Reminders:\n"
        + "- meta.new must cover 0..len(final_steps)-1 exactly once\n"
        + "- the number of NON-deletion meta entries must equal len(final_steps)\n"
        + "- mod='u' must be verbatim copy of ORIGINAL step (ignore whitespace only)\n"
        + "- transposition must be encoded as exactly one 'ms' and one 'mt' for that eid\n"
        + "- correction steps must not be empty fillers like 'Continue'/'Proceed'\n"
        + "\nPREVIOUS OUTPUT (debug only, do not repeat):\n"
        + tail
    )


def canonicalize_meta_new_indices(meta: List[Dict[str, Any]], final_steps: List[str]) -> None:
    """
    Hybrid canonicalizer.

    If non-deletion meta entries have a valid permutation of [0..len(final_steps)-1],
    we TRUST those indices and reorder final_steps into meta-order first.

    Otherwise, we assume final_steps is already in meta-order and just renumber meta.new sequentially.

    After this:
      - For mod='d': new=""
      - For others: new=0..len(final_steps)-1 in meta order
    """
    non_del = [m for m in meta if m.get("mod") != "d"]
    if len(non_del) != len(final_steps):
        raise ValueError(
            f"meta/final_steps length mismatch after canonicalize: "
            f"meta_non_del={len(non_del)} vs len(final_steps)={len(final_steps)}"
        )

    idxs = [m.get("new") for m in non_del]
    has_perm = all(isinstance(i, int) for i in idxs) and set(idxs) == set(range(len(final_steps)))

    # Case A: meta.new is a clean permutation -> reorder steps into meta order
    if has_perm:
        final_steps[:] = [final_steps[i] for i in idxs]

    # Canonical renumbering
    new_i = 0
    for m in meta:
        if m.get("mod") == "d":
            m["new"] = ""
        else:
            m["new"] = new_i
            new_i += 1

    if new_i != len(final_steps):
        # should never happen due to the earlier check, but keep for safety
        raise ValueError(
            f"meta/final_steps length mismatch after canonicalize: "
            f"meta_non_del={new_i} vs len(final_steps)={len(final_steps)}"
        )


def compute_changed_new_indices(meta: List[Dict[str, Any]]) -> Set[int]:
    """
    Indices in final_steps that were generated/modified by the model (not pure 'u').
    We will apply plausibility checks ONLY to these indices.
    """
    out: Set[int] = set()
    for m in meta:
        mod = m.get("mod")
        new = m.get("new")
        if mod in {"e", "a", "i", "c"} and isinstance(new, int):
            out.add(new)
    return out


def expand_inventory_check_indices_for_get_substitutions(
    take: Dict[str, Any],
    original_steps: List[str],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    changed_new: Set[int],
    vocab_map: Optional[Dict[str, str]],
    semrep_map: Optional[Dict[str, Dict[str, str]]],
    semrep_step_to_id: Optional[Dict[str, str]] = None,
    window: int = 80,
) -> Set[int]:
    """
    Expand SemRep inventory validation window after GET-like substitutions.
    Motivation:
      When the model substitutes a GET step (e.g., get lettuce instead of cucumber),
      downstream *unchanged* steps may still reference the original entity (cucumber).
      If inventory checks run only on changed indices, this inconsistency is never detected,
      and the model won't learn to cascade (mod='a') its entity swap.
    """
    out = set(changed_new or set())
    GETLIKE = {"GET", "TAKE", "PICK", "RETRIEVE"}

    for m in meta:
        if m.get("mod") != "e":
            continue
        if str(m.get("etype") or "").strip().lower() != "substitution":
            continue
        old = m.get("old")
        new = m.get("new")
        if not (isinstance(old, int) and isinstance(new, int)):
            continue
        if not (0 <= old < len(original_steps) and 0 <= new < len(final_steps)):
            continue

        # Determine if the ORIGINAL step is GET-like via SemRep when possible, else text fallback.
        is_getlike = False
        sr_old = ""
        if vocab_map is not None and semrep_map is not None:
            sr_old = _semrep_for_original_idx(take, old, vocab_map, semrep_map)
            p = parse_semrep_minimal(sr_old) if sr_old else None
            if p and p[0] in GETLIKE:
                is_getlike = True
        if not is_getlike:
            t = (original_steps[old] or "").strip().lower()
            is_getlike = t.startswith(("get ", "take ", "pick ", "retrieve "))

        if is_getlike:
            lo = max(0, new - 1)
            hi = min(len(final_steps), new + 1 + max(1, int(window)))
            out.update(range(lo, hi))

    return out


def enforce_verbatim_for_u_and_moves(
    original_steps: List[str],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
) -> None:
    """
    Hard-enforce that mod='u','ms','mt' steps are verbatim originals.
    This prevents needless failures when the model slightly rewrites unchanged steps.
    """
    for m in meta:
        mod = m.get("mod")
        if mod not in {"u", "ms", "mt"}:
            continue
        old = m.get("old")
        new = m.get("new")
        if not isinstance(old, int) or not isinstance(new, int):
            continue
        if 0 <= old < len(original_steps) and 0 <= new < len(final_steps):
            final_steps[new] = original_steps[old]


# -------------------------
# Prompting
# -------------------------


def format_steps_for_prompt(
    steps: List[Dict[str, Any]],
    semrep_by_idx: Optional[Dict[int, str]] = None,
) -> str:
    out: List[str] = []
    for s in steps:
        idx = s.get("idx")
        if not isinstance(idx, int):
            idx = s.get("index")
        if not isinstance(idx, int):
            idx = 0

        txt = s.get("txt")
        if not isinstance(txt, str) or not txt:
            txt = s.get("step_description")
        if not isinstance(txt, str):
            txt = ""

        line = f"{idx:02d} {txt}"
        if semrep_by_idx and idx in semrep_by_idx:
            sr = (semrep_by_idx.get(idx) or "").strip()
            if sr:
                line += f" | SR={sr}"
        out.append(line)
    return "\n".join(out)


def format_error_plan_for_prompt(
    errors: List[Dict[str, Any]], corrections: List[Dict[str, Any]]
) -> str:
    out = []
    corr_by_eid: Dict[str, List[Dict[str, Any]]] = {}
    for c in corrections:
        # Simulator uses 'targets_error_id'
        target_eid = c.get("targets_error_id")
        if target_eid:
            corr_by_eid.setdefault(target_eid, []).append(c)

    for e in errors:
        # Simulator fields: event_id, type, step_index
        eid = e.get("event_id")
        etype = e.get("type")
        src = e.get("step_index")
        if not isinstance(src, int):
            src = e.get("src_step_idx")
        extra = []
        if etype == "transposition":
            target = e.get("spec", {}).get("transposition_target")
            extra.append(f"swap_pair=[{src},{target}] (must output ms+mt with this eid)")
        if etype == "insertion":
            extra.append("insert_before=src")
        if etype == "deletion":
            alt = e.get("alternate_src_indices")
            if alt:
                extra.append(f"alt_del={alt}")

        extra_s = ("; " + ", ".join(extra)) if extra else ""
        out.append(f"{eid}: {etype} at src={src}{extra_s}")

        for c in corr_by_eid.get(eid, []):
            cid = c.get("correction_id")
            ctype = c.get("correction_type")
            # Simulator field: detect_at_step_index
            cpos = c.get("detect_at_step_index")
            intent = c.get("intent") or ""
            if intent:
                out.append(f"  - {cid}: {ctype} after src={cpos} | intent: {intent}")
            else:
                out.append(f"  - {cid}: {ctype} after src={cpos}")
    return "\n".join(out)


# SYSTEM_PROMPT = """You are a careful procedure rewriter.
# You will receive:
# (1) an ORIGINAL step list with indices and phases
# (2) an ERROR+CORRECTION plan (where and what type of error must occur, and what correction action is expected)
# Your job is to output a NEW procedure that is:
# - still WRONG in the planned ways (contains the planned errors),
# - but remains logically coherent and physically plausible enough to follow.
# If a plan instruction conflicts with basic feasibility (e.g., for transposition "pour mixture into a closed pan"), you MUST still keep an error of the requested type, but choose the closest feasible variant (e.g., transpose two different feasible steps, or adjust other steps with the new object if an object or instrument was substituted) while keeping the error near the requested location.

# The produced errors of each type must correspond to the following human behaviour and nature:
# - Substitution (wrong step / wrong object, but a plausible one) represents an erraneous step due to confusion between similar steps.
# - Wrong Execution represents execution slips due to unfamiliarity or motor learning, associative errors like using a wrong tool, container, etc.
# - Deletion represents omissions and skipped steps due to lapses from memory overload.
# - Insertion represents an extra step with a mistake because of the confusion. Sometimes it could be a partial repetition like adding the same ingredient twice.
# - Transposition (sequence/order error) represents a sequencing failure or a planning slip through the swapped order of two actions.
# Each change of the procedure by the plan (i, d, e, ms, mt), must contain an error of this type or be a correction (c) or a cascading adjustment (a) in order to keep the overall procedure feasible in real world.
# For instence, if in a particular step a person mistakenly used a knife instead of a spoon to scoop a jelly (wrong execution), then the knife (and not the spoon!) must be used in the following steps like spread the jelly on a toast with the knife (and not the spoon).

# Scenario framing (IMPORTANT):
# Imagine a person is genuinely following the ORIGINAL instructions in real life.
# They occasionally drift or make small mistakes without realizing it, then they may later notice and correct some of the mistakes.
# Your rewrite should read like a realistic, followable instruction list with natural human mistakes.

# Locality rule for any NEW/CHANGED step (mod in {e,a,i,c}):
# - The step must be plausible *at that moment* given the nearby context.
# - Do NOT “jump ahead” and use future ingredients or containers before they appear.

# Top priority order:
# 1) Coherent, feasible sequence (no impossible preconditions, no nonsensical odd steps).
# 2) Preserve the intended error types and keep them close to the instructed locations (change the location only if the current location contradicts the overall procedure flow).
# 3) Keep as much of the original wording as possible when mod='u', but adjust it if the procedure coherence requires it.

# Important constraints you MUST follow:
# - INSERTION (etype="insertion"):
# - The inserted step should be a small realistic extra action a person might do by mistake.
# - It MAY repeat something that happened earlier in the procedure, but it must make sense *now*.
# - It must NOT be a near-duplicate of the surrounding ssteps.
# - For insertion events: keep the src step (as u/a…), and add one new step immediately BEFORE it with mod='i'.
# - Avoid copying a random unrelated step from another phase just because it exists in the ORIGINAL list.
# - CORRECTION steps must actually FIX something that became wrong due to the error. Do NOT write narrative filler like "continue" or "proceed".
# - DELETION: if deleting the instructed step would break the rest (missing object/mixture/state), you may either (A) delete a nearby safer step in the SAME phase (use alt_del if provided), OR (B) keep the deletion but perform CASCADE edits to make subsequent steps refer only to what exists.
# - ADJUSTED (CASCADE) edits: if you change an object/tool in one step, you must propagate that change consistently to later steps (mark those later steps as mod='a' and set eid to the causing error id).
#   Mark every such cascade-repaired step as mod='a' (adjusted) and set eid to the SAME error id that caused the mismatch.
# - "mod='u'" means the final step text must match the original step text (ignore whitespace only).

# TRANSPOSITION encoding (STRICT):
# - If an error has etype="transposition", you MUST represent the swap using exactly two meta entries with the SAME eid:
#   * one entry with mod="ms" and old=<swap source index A>
#   * one entry with mod="mt" and old=<swap target index B>
# - The step texts for ms/mt must be verbatim copies of original_steps[old] (ignore whitespace only).
# - Any additional feasibility repairs caused by the swap must be mod="a" and reuse the SAME eid.

# ORDERING (same object): Follow ORDERING_CONSTRAINTS_SAME_OBJECT that will be provided in the user message.
#   The list is an array of [dependent_predicate, prerequisite_predicate].
#   For the SAME object: you must not produce dependent(Object) twice unless at least one prerequisite(Object)
#   occurred in between (since the last dependent(Object)).
#   If the same dependent appears with multiple prerequisites in the list, ANY of those prerequisites is sufficient.

# ADJUSTMENTS:
# Tool/Object consistency (HARD):
# If an error step changes or introduces a tool/object by substitution or wrong execuption, you MUST update every later step that refers to the old tool/object to match what the person now has/uses.
# Every such updated later step MUST be mod="a" and must reuse the SAME eid as the causing error.
# Do not “magically reintroduce” the original tool later unless there is an explicit step that gets it again.

# Hard output constraints:
# - Output ONLY one JSON object. No markdown, no commentary.
# - meta must be in correct timeline order; new indices will be re-numbered automatically, but must be present for non-deletions.
# - Do NOT add steps after the goal is achieved (no zombie steps).
# - Maintain temporal/physical feasibility (no tools/states before they exist).
# - Any cascade feasibility repair MUST be mod='a' with eid set to the causing error id.

# Output STRICTLY as JSON object with keys:
# {
#   "final_steps": [string, ...],
#   "meta": [
#     {"old": int|"" , "new": int|"" , "mod": "u|e|a|i|c|d|ms|mt", "etype": string|null, "eid": string|null, "cid": string|null},
#     ...
#   ]
# }

# Meta rules:
# - Meta is in timeline order (relative to the ORIGINAL), with insertions/corrections placed where they occur.
# - Every integer new index 0..len(final_steps)-1 must appear exactly once in meta.
# - For deletions: include a meta entry with mod='d', old=<original index>, new="".
# - For insertion: include a meta entry with mod='i', old="", new=<index into final_steps>, eid=<error id>, etype="insertion".
# - For correction: include a meta entry with mod='c', old="", new=<index into final_steps>,
#   etype="correction", cid=<EXACT correction_id string from the plan>.
# """

SYSTEM_PROMPT = """
You are a procedure rewriter.

You will receive:
(0) SCENARIO_NAME 
(1) ORIGINAL steps with 0-based indices and phases
(2) an ERROR+CORRECTION plan
(3) Semantic representations for ORIGINAL steps (predicate + roles)

Goal:
Output a new step list that is physically plausible and followable, but contains the planned errors near the planned locations.
Then output timeline meta mapping ORIGINAL -> NEW with required labels.

Scenario:
A person genuinely follows the ORIGINAL in real life, occasionally drifting into human procedural mistakes, then sometimes noticing and correcting them.
The rewrite must read like a realistic, followable sequence of steps (not a story).

Use SCENARIO_NAME only as high-level context (domain/style). Do not include it in the output JSON and do not change the output schema.

HARD PRIORITIES (in order):
1) Physical and temporal feasibility + locality (no impossible preconditions; no using tools/ingredients before they exist).
2) Realize every planned eid and cid, close to the plan locations.
3) Preserve original wording when mod="u" (verbatim, ignore whitespace only) and keep other edits minimal.

If the plan location/wording would make the sequence physically impossible, you must still realize the requested error type,
but choose the closest feasible variant near the same location (and then repair downstream references via cascade adjustments).

Treat the procedure as tracking what the person currently does or has and where it is (check the the semantic roles).
- If the person GET/TAKE a singular Object or Instrument, they must not GET/TAKE the same singular object again
  unless you explicitly RETURN/PUT_AWAY/PUT_BACK/DISPOSE it in between.
- If the task truly needs multiple instances, you say “another/second” and treat it as a distinct instance (e.g., “get a second Object).
- After RETURN/PUT_AWAY/PUT_BACK/DISPOSE(Object: X), you must not USE/APPLY(Object: X) in later steps unless X is explicitly gotten again.

Avoid creating irreversible/blocking states that make the next planned steps impossible, unless you also repair later steps (mod="a", same eid).
Examples of blocking: tighten/lock/seal/close-and-latch/drain/throw away/turn off when later steps require the prior state.

PHYSICAL STATE CONSISTENCY
If a DEGREE of the action X in a step is full ("fully"/"completely"/"well" X), the step cannot be followed by a partial DEGREE of the same action ("partially"/"incompletely"/"slightly"/"a little").
You cannot also insert a reverse step (e.g., "push back") before the original one (e.g., "pull...")
Do not create impossible physically inconsistent reversals.
PHYSICAL INTEGRITY OF NEW STEPS
Any new/changed step (mod in {e,a,i,c}) must be physically executable and linguistically faithful to the action.
- Do not invent impossible action–instrument pairs (e.g., "wash hands/stir with a napkin").
- Do not use "ADD/POUR" with a destination "in/into" that is not a container (e.g., countertop/floor/wall/air/hand are not containers). 
- For substitution steps, you may prefer physically plausible accident verbs: spill, drop, knock over, smear, splash, miss the container, pour some on the side.
- If the semantic roles imply a canonical predicate (e.g., WASH requires water/sink/soap), keep the predicate consistent with the required physical setup.

IMPORTANT SPECIAL CASE: GET-substitution propagation
If the planned error substitutes a GET-like step (get/take/pick/retrieve) so that you acquire Y instead of X,
then assume X is NOT available later unless it is explicitly acquired again (do NOT add new insertions unless the plan includes them).
Therefore, any later steps that refer to X should be cascade-adjusted to refer to Y:
- rewrite those downstream steps and mark them as mod="a" with the SAME eid as the GET-substitution.
This keeps the procedure executable while preserving the planned mistake.

CRITICAL LOOKAHEAD RULE:
Before writing a step that REMOVES a tool/object (e.g., "put back", "return", "put away", "throw away"), you MUST check the following steps.
If the next steps require using that same tool/object, you are creating a DEADLOCK.
To avoid this, you MUST either:
A) Choose a different error realization (e.g., instead of "put back", do "pick up" the object of the wrong size).
B) OR, if you must REMOVE, you must insert a cascade adjustment (mod='a') in the step that has this tool/object after it to "Get the object again and ..." before using it.

NON-BLOCKING ERROR RULE:
When realizing an error (especially substitution or wrong_execution), do NOT create a state that makes the next original steps physically impossible (Deadlock).
- Example of DEADLOCK: Replacing "Open the jar" with "Tighten the lid" when the next step is "Pour the sauce from the jar". (You cannot pour from a closed jar).
- Example of DEADLOCK: Replacing "Loosen the nut" with "Tighten the nut" when the next step is "Remove the wheel".

Instead, choose a "softer" error that preserves the possibility of attempting the next step, or an error that represents a failure to progress rather than a reversal.
If you MUST create a blocking state (like "Tighten"), you MUST insert a repair step (mod='a') immediately after to fix the state (e.g., "Realize mistake and loosen it again") before proceeding to the blocked step.

ERROR TYPES are human-like and must be meaningfully wrong:
Each planned error realization must be an actual mistake with a negative procedural effect (for instance, wrong state because of wrong action, wrong object, incomplete/wasted/undone work, wrong manner, degree, slips, lapses),
not merely a harmless alternative or a redundant but neutral repetition.

- substitution:
  A plausible confusion with a similar step/object or a completely different step within the procedure context. May change the main predicate and/or object.
  It must be wrong in context (causes mismatch or lost progress), not just a copy of another correct step.
  When the ORIGINAL step contains an explicit state or condition in the text or SemRep,
  a possible human-like substitutin is to make it under the wrong condition (wrong temperature/state/readiness/timing).

- wrong_execution:
  The intended predicate is the same, but performed incorrectly (wrong instrument/detail/location/destination/container/amount/manner/target).
  Prefer local slips that do NOT flip key state (e.g., do not replace “loosen” with “tighten” if the next steps require removal),
  unless you also cascade-repair later steps.
  Similar to substitution, "wrong condition" mistakes are allowed here too (wrong temperature/state/readiness),
  but ensure you follow the plan's requested error type at that location.

- deletion:
  Omit the src step (meta mod="d", new="").
  If deletion breaks feasibility, keep the deletion but cascade-repair later steps (mod="a", same eid) so they refer only to what exists.

- insertion:
  Add exactly one extra mistaken action immediately before the source step in the plan.
  The inserted step must be a mistake (spill/drop/wrong container/wrong amount/wrong small action/forget-to-finish),
  and must not be a harmless improvement or deviation from the procedure sequence.
  It must not be a near-duplicate of the previous or next steps.

- transposition:
  Swap two ORIGINAL steps only.
  Encode using exactly TWO meta entries with the SAME eid:
    * one entry mod="ms" old=A
    * one entry mod="mt" old=B
  The step texts for ms/mt must be verbatim copies of ORIGINAL[old] (ignore whitespace).
  For both ms and mt: set etype="transposition" and the same eid.
  Do not create any additional mod="e" for that transposition eid; ms+mt are the realization.
  Any extra feasibility repairs caused by the swap must be mod="a" with the same eid.

Use Semantic Representations SEMREP (if provided) as hints about the canonical predicate and roles.
- wrong_execution: keep the same predicate; change only roles/details.
- substitution: predicate may change.
Use SEMREP to avoid impossible object/state references and to implement consistent cascades.

CASCADE ADJUSTMENTS (HARD):
If an error step changes or introduces a tool/object/state, you must check and update every later step that refers to the old tool/object/state
to match what the person now has/uses.
- Every such repaired later step MUST be mod="a" and MUST reuse the SAME eid as the causing error.
- Edit minimally: change only the parts needed for consistency.
- Do not “magically reintroduce” the original tool/object unless a later step explicitly gets it again.

CORRECTIONS (HARD):
Add exactly one correction step per plan correction_id.
Each correction must actually fix a concrete earlier mistake (undo/rollback, then redo/perform missed step as required by the plan).
No filler like “continue/proceed”.

ORDERING CONSTRAINTS (same object):
You will be given ORDERING_CONSTRAINTS_SAME_OBJECT = [ [dependent_predicate, prerequisite_predicate], ... ].
For the SAME object: do not output dependent(Object) twice unless at least one prerequisite(Object) occurred in between.

OUTPUT FORMAT (HARD):
Output ONLY one JSON object (no markdown, no commentary) with keys:
{
  "final_steps": [string, ...],
  "meta": [
    {"old": int|"" , "new": int|"" , "mod": "u|e|a|i|c|d|ms|mt", "etype": string|null, "eid": string|null, "cid": string|null},
    ...
  ]
}

META RULES (HARD):
- meta is in timeline order relative to ORIGINAL, with insertions/corrections placed where they occur.
- Every integer new index 0..len(final_steps)-1 MUST appear exactly once in meta (except deletions which have new="").
- Every ORIGINAL old index 0..N-1 MUST appear exactly once in meta with mod in {"u","e","a","ms","mt","d"}.
- mod="u" means final step text MUST be verbatim ORIGINAL[old] (ignore whitespace only).

META LABELING (HARD):
- mod="e": the PRIMARY planned error realization for a non-transposition eid.
- mod="a": a FOLLOW-UP step changed only to keep feasibility/consistency after an earlier error; MUST reuse the causing eid.
- mod="i": inserted mistaken step (old=""), etype="insertion", eid required.
- mod="d": deleted original step (new=""), etype="deletion", eid required.
- mod="c": correction step (old=""), etype="correction", cid required and must match the plan exactly.
- mod="ms"/"mt": the two swapped original steps for transposition (etype="transposition", eid required), verbatim ORIGINAL[old].

FINAL SELF-CHECK BEFORE YOU OUTPUT (HARD):
1) Coverage: every planned eid and cid is realized exactly as required.
2) Inventory: no duplicate GET of the same singular tool/object without put-away/return between.
3) No use-after-put-away: if X is put away/returned, X is not used later unless gotten again.
4) Cascades: if a tool/object changes, later references are updated with mod="a" and same eid.
5) Transpositions: only ms+mt for that eid, texts verbatim.
"""

REWRITE_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["final_steps", "meta"],
    "properties": {
        "final_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "meta": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["old", "new", "mod", "etype", "eid", "cid"],
                "properties": {
                    "old": {
                        "oneOf": [
                            {"type": "integer", "minimum": 0},
                            {"type": "string", "enum": [""]},
                        ]
                    },
                    "new": {
                        "oneOf": [
                            {"type": "integer", "minimum": 0},
                            {"type": "string", "enum": [""]},
                        ]
                    },
                    "mod": {
                        "type": "string",
                        "enum": ["u", "e", "a", "i", "c", "d", "ms", "mt"],
                    },
                    "etype": {
                        "oneOf": [
                            {
                                "type": "string",
                                "enum": [
                                    "insertion",
                                    "deletion",
                                    "substitution",
                                    "wrong_execution",
                                    "transposition",
                                    "correction",
                                ],
                            },
                            {"type": "null"},
                        ]
                    },
                    "eid": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                    "cid": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                },
            },
        },
    },
}

SEMREP_PROMPT_ADDON = """
Semantic representations are hints that describe the canonical action and roles.
Use them to keep edits physically plausible and not breaking the overall procedure flow:
- For wrong_execution: usually keep the same main action (predicate) and change only small details/roles.
- For substitution: you MAY change the main action (predicate) and/or objects/roles as needed.
  Priority is feasibility: do not break the procedure's physical/temporal logic.
  If the substitution makes later steps inconsistent, apply cascade repairs and mark them as mod='a' with the SAME eid.
- For cascade repairs (mod='a'): the cascade repairs are needed to keep the sense of the whole procedure considering the main step changes; keep the cascaded step as original in all but the changes necessary for the repair; avoid rewriting the whole step.
"""


def build_user_prompt(
    take: Dict[str, Any],
    include_semrep: bool = False,
    vocab_map: Optional[Dict[str, str]] = None,
    semrep_map: Optional[Dict[str, Dict[str, str]]] = None,
    semrep_extender=None,
    semrep_step_to_id: Optional[Dict[str, str]] = None,
) -> str:

    # High-level domain hint for the model (must NOT appear in output JSON)
    scenario_name = str(
        take.get("take_name") or take.get("name") or take.get("takeName") or ""
    ).strip()
    scenario_name = normalize_ws(scenario_name) if scenario_name else ""
    steps = take.get("steps") or take.get("raw_steps") or []
    if not isinstance(steps, list):
        steps = []
    errors = take.get("errors", [])
    corrections = take.get("corrections", [])

    semrep_by_idx: Dict[int, str] = {}
    if include_semrep:
        # Provide semrep for ALL steps (if available).
        for s in steps:
            idx = s.get("idx")
            if not isinstance(idx, int):
                idx = s.get("index")
            if not isinstance(idx, int):
                continue

            txt = s.get("txt")
            if not isinstance(txt, str) or not txt:
                txt = s.get("step_description")
            if not isinstance(txt, str):
                txt = ""

            # Prefer semrep already present
            sr0 = s.get("semantic_representation")
            if isinstance(sr0, str) and sr0.strip():
                semrep_by_idx[int(idx)] = sr0.strip()
                continue

            # Otherwise: resolve id -> semrep via vocab + semrep_json
            sid = s.get("step_description_id")
            if not isinstance(sid, str) or not sid.strip():
                sid = None
            if sid is None and vocab_map is not None and txt:
                sid = resolve_step_id_from_text(str(txt), vocab_map)
            if sid and semrep_map is not None and sid in semrep_map:
                sr = semrep_map[sid].get("semantic_representation") or ""
                if sr:
                    semrep_by_idx[int(idx)] = sr

    steps_block = format_steps_for_prompt(steps, semrep_by_idx=semrep_by_idx)
    plan_block = format_error_plan_for_prompt(errors, corrections)

    ordering_pairs_json = json.dumps(ORDERING_CONSTRAINTS_SAME_OBJECT, ensure_ascii=False)
    ordering_block = f"""ORDERING_CONSTRAINTS_SAME_OBJECT (same object predicate pairs):
{ordering_pairs_json}

Interpretation:
- Each pair is [dependent, prerequisite].
- For the SAME object: do not output dependent(Object) twice unless prerequisite(Object) happened in between.
- If multiple prerequisites exist for the same dependent, any of them satisfies the prerequisite.
"""

    addon = SEMREP_PROMPT_ADDON if include_semrep else ""
    return f"""SCENARIO_NAME: {scenario_name}

ORIGINAL STEPS (0-based indices):
{steps_block}

ERROR+CORRECTION PLAN:
{plan_block}

{ordering_block}

{addon}

Now produce the rewritten procedure JSON.
"""


# -------------------------
# Backends
# -------------------------


@dataclass
class LLMBackend:
    name: str

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__(name="openai")
        if OpenAI is None:
            raise RuntimeError("openai package not available.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_id = DEFAULT_OPENAI_MODEL_ID

    def generate(
        self, messages, temperature, max_new_tokens, json_schema: Optional[Dict[str, Any]] = None
    ):
        try:
            text_format = None
            if json_schema is not None:
                # Structured Outputs via text.format (JSON Schema)
                # See: text: { format: { type: "json_schema", strict: true, schema: ... } }
                text_format = {
                    "format": {
                        "type": "json_schema",
                        "name": "rewrite_output",
                        "strict": True,
                        "schema": json_schema,
                    }
                }
            resp = self.client.responses.create(
                model=self.model_id,
                input=messages,
                temperature=float(temperature),
                max_output_tokens=int(max_new_tokens),
                **({"text": text_format} if text_format is not None else {}),
            )
            txt = getattr(resp, "output_text", None)
            if isinstance(txt, str) and txt.strip():
                return txt
            return str(resp)
        except Exception:
            # Fallback: Chat Completions API (no schema enforcement here)
            cc_kwargs = dict(
                model=self.model_id,
                messages=messages,
                temperature=float(temperature),
                # IMPORTANT: for gpt-5.* models Chat Completions uses max_completion_tokens, not max_tokens
                max_completion_tokens=int(max_new_tokens),
            )
            # If you want schema also in fallback:
            chat = self.client.chat.completions.create(**cc_kwargs)
            return chat.choices[0].message.content or ""


class QwenLocalBackend(LLMBackend):
    def __init__(self, model_id: str = DEFAULT_QWEN_MODEL_ID) -> None:
        super().__init__(name="qwen")
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise RuntimeError("transformers/torch not available; cannot use qwen backend.")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            dtype="auto",
        )
        self.model.eval()
        self.torch = torch

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Qwen/Transformers generation helper.

        Fixes two common issues:
        1) Prompt echo: model output appears to "repeat the prompt" because we decoded
           the full sequence (prompt + generated tokens). We must decode ONLY the
           newly generated tokens after the prompt length.
        2) Empty / non-parseable output: model sometimes stops immediately or outputs
           very short text. We enforce a minimum number of generated tokens and we
           optionally perform a light retry with slightly adjusted decoding settings.
        """

        # 1) Build the prompt using the chat template (preferred).
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation if chat template fails.
            prompt_text = "\n\n".join(
                [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages]
            )

        # 2) Tokenize and move to model device.
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3) Generation configuration.
        #    - min_new_tokens: prevents "empty answer" or 1-2 token outputs.
        #    - pad_token_id: avoids warnings/errors for some configs and improves stability.
        #    - do_sample: sampling only when temperature is meaningfully > 0.
        def _run_generation(temp: float, min_new: int) -> str:
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                min_new_tokens=int(min_new),
                repetition_penalty=1.02,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            if temp <= 0.01:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(temp)
                gen_kwargs["top_p"] = 0.95

            with self.torch.no_grad():
                out = self.model.generate(**gen_kwargs)

            # IMPORTANT:
            # Transformers returns prompt+generated tokens in `out[0]`.
            # Decode ONLY tokens generated after the prompt length to avoid prompt echo.
            prompt_len = inputs["input_ids"].shape[-1]
            gen_ids = out[0][prompt_len:]

            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            return decoded

        # 4) First attempt.
        decoded = _run_generation(float(temperature), min_new=32)

        # 5) If output is empty or too short, retry once with slightly safer settings.
        #    - Increase min_new_tokens
        #    - Reduce temperature (less chance to terminate instantly / go off-format)
        #    This is intentionally minimal to avoid heavy compute.
        if not decoded or len(decoded) < 10:
            decoded = _run_generation(max(0.0, float(temperature) * 0.5), min_new=64)

        return decoded.strip()


def make_backend(model_choice: str) -> LLMBackend:
    if model_choice == "openai":
        return OpenAIBackend()
    if model_choice == "qwen":
        return QwenLocalBackend()
    raise ValueError(f"Unknown model choice: {model_choice}")


# -------------------------
# Main generation loop
# -------------------------


def generate_for_take(
    take: Dict[str, Any],
    backend: LLMBackend,
    temperature: float,
    max_tokens: int,
    max_retries: int = 2,
    retry_temp_decay: float = 0.15,
    include_semrep: bool = False,
    vocab_map: Optional[Dict[str, str]] = None,
    semrep_map: Optional[Dict[str, Dict[str, str]]] = None,
    semrep_extender=None,
    semrep_step_to_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(
        take,
        include_semrep=include_semrep,
        vocab_map=vocab_map,
        semrep_map=semrep_map,
    )

    # Prepare ORIGINAL text list once
    original_steps_txt: List[str] = []
    for s in take.get("steps") or take.get("raw_steps") or []:
        if not isinstance(s, dict):
            original_steps_txt.append("")
            continue
        txt = s.get("txt")
        if not isinstance(txt, str) or not txt:
            txt = s.get("step_description")
        original_steps_txt.append(txt if isinstance(txt, str) else "")

    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_raw: str = ""
    last_err: str = ""
    last_parsed_rewrite: Optional[Dict[str, Any]] = None
    last_parsed_issues: List[str] = []

    for attempt in range(max_retries + 1):
        # Default: decay temperature on retries.
        t = max(0.0, float(temperature) - float(retry_temp_decay) * float(attempt))

        # If we hit structural mismatches that typically improve with more deterministic decoding,
        # clamp temperature harder on retries.
        if attempt > 0 and any(
            k in (last_err or "") for k in ("meta/final_steps length mismatch",)
        ):
            t = min(t, 0.05)

        # If we failed on "fake error" or adjacency duplication, prefer lower temperature retries.
        if attempt > 0 and any(
            k in (last_err or "") for k in ("error_realization:", "adjacent_duplicate:")
        ):
            t = min(t, 0.05)

        # If we specifically failed degree consistency, bump temperature a bit so the model "re-thinks"
        # rather than producing the same structure again.
        if attempt > 0 and "degree_consistency:" in (last_err or ""):
            t = min(0.95, float(temperature) + 0.15 * float(attempt))

        messages = base_messages
        if attempt > 0:
            retry_user = build_retry_prompt(
                user_prompt, last_raw, last_err or "unknown_validation_error"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": retry_user},
            ]

        schema = REWRITE_OUTPUT_JSON_SCHEMA if getattr(backend, "name", "") == "openai" else None
        raw = backend.generate(
            messages=messages, temperature=t, max_new_tokens=max_tokens, json_schema=schema
        )
        last_raw = raw

        try:
            obj = extract_json_object(raw)
            final_steps = obj.get("final_steps")
            meta = obj.get("meta")

            if not isinstance(final_steps, list) or not all(
                isinstance(x, str) for x in final_steps
            ):
                raise ValueError("final_steps must be list[str].")
            if not isinstance(meta, list) or not all(isinstance(x, dict) for x in meta):
                raise ValueError("meta must be list[dict].")

            # Ensure required keys exist to simplify normalization and reduce KeyError risk
            for m in meta:
                m.setdefault("old", "")
                m.setdefault("new", "")
                m.setdefault("mod", "u")
                m.setdefault("etype", None)
                m.setdefault("eid", None)
                m.setdefault("cid", None)

            # Normalize meta fields
            for m in meta:
                mod = str(m.get("mod") or "").strip().lower()
                m["mod"] = mod
                if m.get("old") is None:
                    m["old"] = ""
                if m.get("new") is None:
                    m["new"] = ""

                if isinstance(m.get("old"), str) and m["old"].strip() == "":
                    m["old"] = ""
                if isinstance(m.get("new"), str) and m["new"].strip() == "":
                    m["new"] = ""

                if m.get("etype") == "":
                    m["etype"] = None
                if m.get("eid") == "":
                    m["eid"] = None
                if m.get("cid") == "":
                    m["cid"] = None

                if isinstance(m.get("etype"), str):
                    _x = m["etype"].strip()
                    m["etype"] = _x.lower() if _x else None
                if isinstance(m.get("eid"), str):
                    _x = m["eid"].strip()
                    m["eid"] = _x if _x else None
                if isinstance(m.get("cid"), str):
                    _x = m["cid"].strip()
                    m["cid"] = _x if _x else None

                # Ensure correction steps always carry an explicit etype.
                # This keeps downstream consumers simple (no special-casing etype=null for mod='c').
                if mod == "c":
                    m["etype"] = "correction"

                # insertion/correction are NEW steps (no old index)
                if mod in {"i", "c"}:
                    m["old"] = ""

            # Save the last successfully parsed version EARLY (even before canonicalize/validation),
            # so we don't end up with rewrite=null on structural exceptions.
            last_parsed_rewrite = {"final_steps": list(final_steps), "meta": list(meta)}
            last_parsed_issues = []

            # Canonicalize meta.new indices to match meta timeline order
            try:
                canonicalize_meta_new_indices(meta, final_steps)
            except Exception as e:
                # Treat as a validation failure (retryable), not a parse failure.
                # Keep the parsed rewrite for debugging/output.
                last_err = str(e)
                last_parsed_issues = [last_err]
                continue

            # Enforce verbatim for unchanged/move steps (reduces pointless failures)
            enforce_verbatim_for_u_and_moves(original_steps_txt, final_steps, meta)

            changed_new = compute_changed_new_indices(meta)
            inv_check_new = changed_new

            # -------------------------
            # SemRep auto-extension BEFORE any semrep-dependent validation
            # -------------------------
            if include_semrep and semrep_extender is not None and semrep_map is not None:
                # Ensure SemRep exists for generated/modified steps (or just do all final_steps for simplicity)
                semrep_extender.ensure_for_texts(final_steps)
                semrep_step_to_id = semrep_extender.step_to_id
                init_reverse_semrep_map(semrep_map)

            ok, issues = validate_rewrite(
                original_steps_txt, final_steps, meta, changed_new=changed_new
            )
            issues.extend(validate_plan_coverage(take, meta))
            issues.extend(validate_old_index_coverage(original_steps_txt, meta))
            issues.extend(validate_transposition_realized(take, meta))

            # Catch "fake wrong_execution/substitution" (unchanged error step) + SemRep sanity when available
            issues.extend(
                validate_error_realization_minimal(
                    take=take,
                    original_steps=original_steps_txt,
                    final_steps=final_steps,
                    meta=meta,
                    changed_new=changed_new,
                    include_semrep=bool(include_semrep),
                    vocab_map=vocab_map,
                    semrep_map=semrep_map,
                    semrep_step_to_id=semrep_step_to_id,
                )
            )

            # Catch adjacent duplicates introduced by the model
            issues.extend(validate_adjacent_duplicates(final_steps))

            # SemRep-only degree consistency (requires include_semrep + vocab/semrep resources)
            if include_semrep and vocab_map is not None and semrep_map is not None:
                # Expand inventory checks after GET-substitution errors so downstream unchanged steps
                # that still mention the original entity are validated too (forces cascade as mod='a').
                inv_check_new = expand_inventory_check_indices_for_get_substitutions(
                    take=take,
                    original_steps=original_steps_txt,
                    final_steps=final_steps,
                    meta=meta,
                    changed_new=changed_new,
                    vocab_map=vocab_map,
                    semrep_map=semrep_map,
                    semrep_step_to_id=semrep_step_to_id,
                    window=80,
                )
                issues.extend(
                    validate_degree_consistency_semrep_adjacent(
                        final_steps=final_steps,
                        vocab_map=vocab_map,
                        semrep_map=semrep_map,
                        semrep_step_to_id=semrep_step_to_id,
                        check_indices=inv_check_new,
                    )
                )

                # Inventory via SemRep
                issues.extend(
                    validate_inventory_semrep_delta_against_original(
                        take=take,
                        original_steps=original_steps_txt,
                        final_steps=final_steps,
                        meta=meta,
                        vocab_map=vocab_map,
                        semrep_map=semrep_map,
                        semrep_step_to_id=semrep_step_to_id,
                        # Report only near model-changed indices (+/-1) to avoid surfacing unrelated baseline issues.
                        check_indices=changed_new,
                        require_roles={"Object", "Instrument"},
                    )
                )

            if include_semrep and vocab_map is not None and semrep_map is not None:
                issues.extend(
                    check_ordering_constraints_same_object_semrep(
                        final_steps=final_steps,
                        vocab_map=vocab_map,
                        semrep_map=semrep_map,
                        semrep_step_to_id=semrep_step_to_id,
                        # Avoid flagging pre-existing issues in the original procedure.
                        check_indices=changed_new,
                    )
                )

            # Save the last parsed+canonicalized version (even if it does not pass validation)
            last_parsed_rewrite = {
                "final_steps": final_steps,
                "meta": meta,
                "changed_new_indices": sorted(list(changed_new)),
            }
            last_parsed_issues = list(issues)

            if not issues:
                return {
                    "status": "ok",
                    "validation_issues": [],
                    "rewrite": {
                        "final_steps": final_steps,
                        "meta": meta,
                        "changed_new_indices": sorted(list(changed_new)),
                    },
                }

            last_err = " ; ".join(issues)

        except Exception as e:
            last_err = str(e)

    # exhausted retries: return the best-effort result
    if last_parsed_rewrite is not None:
        return {
            "status": "invalid_schema",
            "error": last_err or "exhausted_retries",
            "validation_issues": last_parsed_issues,
            "raw_output_preview": (last_raw or "")[:8000],
            "rewrite": last_parsed_rewrite,
        }

    return {
        "status": "parse_failed",
        "error": last_err or "exhausted_retries",
        "raw_output_preview": (last_raw or "")[:8000],
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    random.seed(args.seed)

    vocab_map: Optional[Dict[str, str]] = None
    semrep_map: Optional[Dict[str, Dict[str, str]]] = None
    semrep_step_to_id: Optional[Dict[str, str]] = None
    semrep_extender = None
    if bool(getattr(args, "include_semrep", False)):
        vocab_map = load_vocab_csv(str(args.vocab_csv))
        semrep_map = load_semrep_json(str(args.semrep_json))
        init_reverse_semrep_map(semrep_map)
        semrep_step_to_id = build_semrep_step_to_id(semrep_map)

    # Enable SemRep auto-extension when SemRep resources are loaded (i.e., --include_semrep)
    # and SemRepAutoExtender is available, so we can validate semrep-dependent constraints for generated steps.
    if (
        bool(getattr(args, "include_semrep", False))
        and semrep_map is not None
        and SemRepAutoExtender is not None
        and args.semrep_json
    ):
        # Prefix ids by backend to avoid confusing mixed provenance in semrep ids
        id_prefix = f"{args.model}_ext"
        semrep_extender = SemRepAutoExtender(
            semrep_map=semrep_map,
            out_path=args.semrep_json,
            id_prefix=id_prefix,
        )
        semrep_step_to_id = semrep_extender.step_to_id
        # light sanity ping
        if not vocab_map:
            print("WARNING: vocab_map is empty; semrep will not be used.", file=sys.stderr)
        if not semrep_map:
            print("WARNING: semrep_map is empty; semrep will not be used.", file=sys.stderr)

    base_default_out = str(REPO_ROOT / "local" / "outputs" / "split_50_error_instructions.json")
    out_path = args.out
    if out_path == base_default_out:
        out_path = _auto_suffix_out_path(out_path, args.model)

    backend = make_backend(args.model)

    with open(args.input, "r", encoding="utf-8") as f:
        plan = json.load(f)
    takes: Dict[str, Any] = plan["takes"]

    out_obj: Dict[str, Any] = {"takes": {}}

    take_ids = list(takes.keys())
    take_ids.sort(key=lambda x: str(x))
    if getattr(args, "take_name", None):
        want = {str(x).strip() for x in (args.take_name or []) if str(x).strip()}
        if want:

            def _take_name_of(t: Dict[str, Any]) -> str:
                return str(t.get("take_name") or t.get("name") or t.get("takeName") or "").strip()

            take_ids = [tid for tid in take_ids if _take_name_of(takes.get(tid, {}) or {}) in want]

    if (getattr(args, "take_name", None) and (args.take_name or [])) and not take_ids:
        print(
            "WARNING: --take_name filter matched 0 takes. "
            "Make sure the provided names exactly match take['take_name'] in the input plan JSON.",
            file=sys.stderr,
        )
    if args.max_takes and int(args.max_takes) > 0:
        take_ids = take_ids[: int(args.max_takes)]

    for tid in take_ids:
        take = takes[tid]

        # Try common name fields; keep empty string if missing.
        take_name = take.get("take_name") or take.get("name") or take.get("takeName") or ""

        result = generate_for_take(
            take=take,
            backend=backend,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            max_retries=args.max_retries,
            retry_temp_decay=args.retry_temp_decay,
            include_semrep=bool(args.include_semrep),
            vocab_map=vocab_map,
            semrep_map=semrep_map,
            semrep_extender=semrep_extender,
            semrep_step_to_id=semrep_step_to_id,
        )

        status = result.get("status", "unknown")

        if status == "ok":
            out_obj["takes"][tid] = {
                "take_name": take_name,
                "status": "ok",
                "rewrite": result.get("rewrite"),
            }
        else:
            out_obj["takes"][tid] = {
                "take_name": take_name,
                "status": status,  # invalid_schema / parse_failed / ...
                "error": result.get("error", ""),
                "validation_issues": result.get("validation_issues", []),
                "raw_output_preview": result.get("raw_output_preview", ""),
                "rewrite": result.get("rewrite", None),
            }

    # Persist any newly generated SemRep entries
    if semrep_extender is not None:
        semrep_extender.flush()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
