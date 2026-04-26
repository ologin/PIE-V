#!/usr/bin/env python3
"""
correction_simulator.py

Adds a structured correction plan to an existing error plan JSON produced by error_simulator.py.

Design goals:
- Keep the input JSON schema intact: we only add/overwrite take-level field "corrections".
- Also inject concise correction lines into "simulation_log_lines" and "simulation_log".
- Do not generate human-readable correction sentences. Produce compact, readable structured specs.
- No "ignore" corrections: if an error is detected but not acted upon, no correction entry is created.
- At most one correction per detect_at_step_index to keep the plan readable and cheap for LLM prompting.
- Corrections are plausibility-oriented but avoid a full world model.
"""

import argparse
import json
import random
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Config (base priors)
# -----------------------------

# Detection priors by error type and phase bucket (early/mid/late)
BASE_DETECT: Dict[str, Dict[str, float]] = {
    "wrong_execution": {"early": 0.85, "mid": 0.80, "late": 0.75},
    # "deletion": {"early": 0.35, "mid": 0.30, "late": 0.25},
    "deletion": {"early": 0.85, "mid": 0.80, "late": 0.75},
    "substitution": {"early": 0.45, "mid": 0.50, "late": 0.55},
    "insertion": {"early": 0.60, "mid": 0.55, "late": 0.50},
    # "transposition": {"early": 0.55, "mid": 0.50, "late": 0.45},
    "transposition": {"early": 0.85, "mid": 0.80, "late": 0.75},
}

# Must match error_simulator
UNLOCK_PREDS = {"GET", "TAKE", "RETRIEVE", "PICK_UP", "GRAB", "REMOVE"}

# For object token matching
STOP_TOKENS = {"of", "in", "on", "from", "to", "with", "and", "or"}

# “Prep” actions: if deleted and the same object is used soon after, we should fix before first use.
PREP_PREDS = {
    "CUT",
    "CHOP",
    "SLICE",
    "DICE",
    "MINCE",
    "SHRED",
    "GRATE",
    "PEEL",
    "TRIM",
    "SEED",
    "CORE",
}

# Roles in which “use of an object” is meaningful downstream
FUTURE_USE_ROLES = ("Object", "Instrument", "Destination", "Origin", "Coobject")

# Latency priors (cheap hazard model): P(L=0)=pi0 else geometric tail with parameter q
LATENCY: Dict[str, Dict[str, float]] = {
    "wrong_execution": {"pi0": 0.70, "q": 0.60},
    "insertion": {"pi0": 0.35, "q": 0.45},
    "transposition": {"pi0": 0.25, "q": 0.40},
    "substitution": {"pi0": 0.15, "q": 0.30},
    "deletion": {"pi0": 0.05, "q": 0.25},
}

# Action priors: probability to actually perform a correction once detected
# (If not acted upon -> no correction entry, per requirement.)
BASE_ACT: Dict[str, Dict[str, float]] = {
    "wrong_execution": {"early": 0.85, "mid": 0.80, "late": 0.75},
    # "deletion": {"early": 0.65, "mid": 0.60, "late": 0.50},
    "deletion": {"early": 0.85, "mid": 0.80, "late": 0.75},
    "substitution": {"early": 0.70, "mid": 0.65, "late": 0.55},
    "insertion": {"early": 0.45, "mid": 0.40, "late": 0.35},
    # "transposition": {"early": 0.60, "mid": 0.55, "late": 0.45},
    "transposition": {"early": 0.85, "mid": 0.80, "late": 0.75},
}

# Predicates that tend to have stronger preconditions/outcome salience.
DANGEROUS_PREDICATES = {
    "OPEN",
    "CLOSE",
    "COVER",
    "UNCOVER",
    "TURN_ON",
    "TURN_OFF",
    "COOK",
    "BOIL",
    "HEAT",
    "COMBINE",
    "MIX",
    "ADD",
    "PUT_AWAY",
    "PUT_BACK",
    "REMOVE",
    "INSERT",
    "APPLY",
}

# Heuristic precondition triggers for faster detection (no world model, just local patterns).
# Maps "trigger predicate" -> set of "likely prerequisite predicates".
PRECONDITION_MAP: Dict[str, set] = {
    "CLOSE": {"OPEN"},
    "COVER": {"OPEN", "UNCOVER"},
    "TURN_OFF": {"TURN_ON"},
    "COMBINE": {"ADD"},
    "MIX": {"ADD"},
    "COOK": {"ADD", "COMBINE", "MIX"},
    "PUT_AWAY": {"GET", "TAKE", "OPEN"},
    "PUT_BACK": {"GET", "TAKE"},
}

# Transpositions between commutative predicates are often not worth correcting.
# Per your request: if transposition between ADD and ADD -> do not generate correction.
COMMUTATIVE_TRANSPOSE_PAIRS = {
    ("ADD", "ADD"),
    ("STIR", "STIR"),
    ("CLEAN", "CLEAN"),
    ("WIPE", "WIPE"),
    ("CHECK", "CHECK"),
    ("GET", "GET"),
}


# Actions that are minor/repetitive and usually don't require corrections
# if they are slightly delayed, skipped, or swapped.
TRIVIAL_PREDICATES = {
    "CLEAN",
    "WIPE",
    "STIR",
    "PLACE",
    "ROLL",
    "CHECK",
    "DRY",
    "WASH_HANDS",
    "CLEAN_HANDS",
    "WIPE_HANDS",
    "DISPOSE",
    "ARRANGE",
}

# -----------------------------
# Helpers
# -----------------------------


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def phase_bucket(phase: str) -> str:
    # Keep consistent with error_simulator output
    if phase == "phase_1":
        return "early"
    if phase == "phase_2":
        return "mid"
    return "late"


def geom_ge_1(rng: random.Random, p: float) -> int:
    """Geometric RV with support {1,2,3,...} with success probability p."""
    k = 1
    while rng.random() > p:
        k += 1
    return k


def sample_latency_steps(rng: random.Random, err_type: str) -> int:
    params = LATENCY.get(err_type)
    if not params:
        return 0
    pi0 = params["pi0"]
    q = params["q"]
    if rng.random() < pi0:
        return 0
    return 1 + geom_ge_1(rng, q)


def is_dangerous_predicate(pred: Optional[str]) -> bool:
    if not pred:
        return False
    return pred.upper() in DANGEROUS_PREDICATES


def severity_rank(sev: str) -> int:
    s = (sev or "").lower()
    if s == "high":
        return 3
    if s == "medium":
        return 2
    return 1


def normalize_step_text(s: Any) -> str:
    """Lowercase + strip + collapse whitespace for robust equality checks."""
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().lower().split())


def _tokens(val: Any) -> set:
    """Tokenize (lowercase alpha/_), drop stop tokens. Mirrors error_simulator style."""
    if not isinstance(val, str):
        return set()
    toks = set(re.findall(r"[a-z_]+", val.lower()))
    return {t for t in toks if t and t not in STOP_TOKENS}


def _object_tokens(obj: Optional[str]) -> set:
    if not isinstance(obj, str):
        return set()
    return _tokens(obj.strip())


def _step_uses_object_tokens(step: Dict[str, Any], obj: str) -> bool:
    """
    True if obj tokens overlap with:
      - step['object_value'] tokens
      - role_to_value in FUTURE_USE_ROLES
    This catches singular/plural and small surface variations.
    """
    ot = _object_tokens(obj)
    if not ot:
        return False

    ov = step.get("object_value")
    if isinstance(ov, str) and (ot & _object_tokens(ov)):
        return True

    rtv = step.get("role_to_value", {}) or {}
    if isinstance(rtv, dict):
        for role in FUTURE_USE_ROLES:
            v = rtv.get(role)
            if isinstance(v, str) and (ot & _tokens(v)):
                return True

    return False


def find_first_future_use_index_tokens(
    steps: List[Dict[str, Any]],
    idx: int,
    obj: Optional[str],
    max_lookahead: int = 10,
) -> Optional[int]:
    """
    Find first future step that uses 'obj' (token overlap) within lookahead.
    Returns index or None.
    """
    if not isinstance(obj, str) or not obj.strip():
        return None
    if idx < 0 or idx >= len(steps):
        return None
    end = min(len(steps), idx + 1 + max(1, int(max_lookahead)))
    for j in range(idx + 1, end):
        if _step_uses_object_tokens(steps[j], obj):
            return j
    return None


def _canonical_obj_string(step: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort object string:
    1) step['object_value']
    2) step['role_to_value']['Object']
    Strips nested "(...)" tail to match error_simulator's extract_main_object behavior.
    """
    ov = step.get("object_value")
    if isinstance(ov, str) and ov.strip():
        s = ov.strip()
    else:
        rtv = step.get("role_to_value", {}) or {}
        v = rtv.get("Object") if isinstance(rtv, dict) else None
        if not (isinstance(v, str) and v.strip()):
            return None
        s = v.strip()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    return s or None


def _step_uses_object_in_roles(
    step: Dict[str, Any], obj: Any, roles: Tuple[str, ...] = FUTURE_USE_ROLES
) -> bool:
    """
    True if obj matches step['object_value'] exactly OR appears token-wise in role_to_value for roles.
    This is the “object used” notion we want to align with error_simulator.
    """
    if not isinstance(obj, str):
        return False
    obj = obj.strip()
    if not obj:
        return False
    ot = _tokens(obj)
    if not ot:
        return False

    ov = step.get("object_value")
    if isinstance(ov, str) and ov.strip() == obj:
        return True

    rtv = step.get("role_to_value", {}) or {}
    if not isinstance(rtv, dict):
        return False
    for role in roles:
        v = rtv.get(role)
        if isinstance(v, str) and (ot & _tokens(v)):
            return True
    return False


def object_reacquired_before_index(
    steps: List[Dict[str, Any]],
    obj: Optional[str],
    start_exclusive: int,
    end_exclusive: int,
) -> bool:
    """
    True if between (start_exclusive, end_exclusive) there is an UNLOCK step that
    re-introduces the same object.
    We check:
      - exact object_value match
      - token-wise match inside role_to_value (Object first, then FUTURE_USE_ROLES)
    This prevents generating a redundant forced correction.
    """
    if not obj or not isinstance(obj, str):
        return False
    obj = obj.strip()
    if not obj:
        return False
    lo = max(0, int(start_exclusive) + 1)
    hi = min(len(steps), int(end_exclusive))
    for i in range(lo, hi):
        pred = (steps[i].get("predicate") or "").upper()
        if pred not in UNLOCK_PREDS:
            continue
        st = steps[i]
        ov = st.get("object_value")
        if isinstance(ov, str) and ov.strip() == obj:
            return True
        rtv = st.get("role_to_value", {}) or {}
        if isinstance(rtv, dict):
            v_obj = rtv.get("Object")
            if isinstance(v_obj, str) and (_tokens(obj) & _tokens(v_obj)):
                return True
            for role in FUTURE_USE_ROLES:
                v = rtv.get(role)
                if isinstance(v, str) and (_tokens(obj) & _tokens(v)):
                    return True
    return False


def compute_detect_prob(
    err_type: str,
    phase_b: str,
    severity: str,
    is_essential: bool,
    predicate: Optional[str],
    step_load: float,
) -> float:
    base = BASE_DETECT.get(err_type, {}).get(phase_b, 0.3)

    # Severity: higher severity tends to be more detectable.
    sev = (severity or "low").lower()
    if sev == "high":
        sev_factor = 1.15
    elif sev == "medium":
        sev_factor = 1.00
    else:
        sev_factor = 0.85

    # Essentiality: essential steps often have more downstream consequences.
    # Deletion is a special case: missing an essential step may be noticed later.
    if err_type == "deletion" and is_essential:
        if phase_b == "early":
            essential_factor = 0.90
        elif phase_b == "mid":
            essential_factor = 1.00
        else:
            essential_factor = 1.10
    else:
        essential_factor = 1.05 if is_essential else 0.95

    # Predicate salience
    pred_factor = 1.15 if is_dangerous_predicate(predicate) else 1.00

    # Load effect: high cognitive load reduces detection reliability.
    load_factor = 1.0 - 0.25 * clamp01(step_load)
    load_factor = max(0.70, load_factor)

    return clamp01(base * sev_factor * essential_factor * pred_factor * load_factor)


def compute_act_prob(
    err_type: str,
    phase_b: str,
    severity: str,
    step_load: float,
) -> float:
    base = BASE_ACT.get(err_type, {}).get(phase_b, 0.4)

    sev = (severity or "low").lower()
    if sev == "high":
        sev_factor = 1.10
    elif sev == "medium":
        sev_factor = 1.00
    else:
        sev_factor = 0.90

    # Under high load people are less likely to execute a full correction.
    load_factor = 1.0 - 0.20 * clamp01(step_load)
    load_factor = max(0.75, load_factor)

    return clamp01(base * sev_factor * load_factor)


def within_same_taxonomy_block(
    steps: List[Dict[str, Any]],
    a_idx: int,
    b_idx: int,
) -> bool:
    """Keep corrections local to the same taxonomy block when possible."""
    if a_idx < 0 or b_idx < 0 or a_idx >= len(steps) or b_idx >= len(steps):
        return False
    a = steps[a_idx].get("taxonomy_block_id")
    b = steps[b_idx].get("taxonomy_block_id")
    if a is None or b is None:
        return True
    return a == b


def precondition_trigger_min_latency(
    steps: List[Dict[str, Any]],
    err_step_idx: int,
    err_predicate: Optional[str],
    err_object: Optional[str],
    lookahead: int = 5,
) -> Optional[int]:
    """
    Heuristic: if soon after the error we see an action that usually depends on some prerequisite,
    detection tends to happen sooner (latency small). Uses only predicate + object_value locality.
    """
    if err_step_idx < 0 or err_step_idx >= len(steps):
        return None

    if is_dangerous_predicate(err_predicate):
        return 1

    obj = err_object if isinstance(err_object, str) and err_object.strip() else None
    for j in range(err_step_idx + 1, min(len(steps), err_step_idx + 1 + lookahead)):
        next_pred = (steps[j].get("predicate") or "").upper()
        if obj:
            # keep locality only if we can establish this step uses the same object in any relevant role
            if not _step_uses_object_in_roles(steps[j], obj, roles=FUTURE_USE_ROLES):
                continue

        prereqs = PRECONDITION_MAP.get(next_pred)
        if prereqs and err_predicate and err_predicate.upper() in prereqs:
            return 1

    return None


def get_error_step_index(err: Dict[str, Any]) -> Optional[int]:
    """Robustly extract the anchor step index for an error event."""
    v = err.get("step_index")
    if isinstance(v, int):
        return v
    v = err.get("index")
    if isinstance(v, int):
        return v
    return None


def get_transposition_pair(err: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract (a_idx, b_idx) for transposition from various possible specs:
    - spec.swap_step_index_a / spec.swap_step_index_b
    - spec.transposition_source / spec.transposition_target
    - spec.swap_with_step_index / spec.swapped_with_step_index (legacy)
    Returns indices as ints when present.
    """
    spec = err.get("spec", {})
    if not isinstance(spec, dict):
        spec = {}

    a = spec.get("swap_step_index_a")
    b = spec.get("swap_step_index_b")

    if isinstance(a, int) and isinstance(b, int):
        return a, b

    a2 = spec.get("transposition_source")
    b2 = spec.get("transposition_target")
    if isinstance(a2, int) and isinstance(b2, int):
        return a2, b2

    # Legacy: one-sided "swap_with_step_index" (assume current error step is the other endpoint).
    other = spec.get("swap_with_step_index") or spec.get("swapped_with_step_index")
    err_idx = get_error_step_index(err)
    if isinstance(other, int) and isinstance(err_idx, int):
        return err_idx, other

    return None, None


def deletion_is_redundant(steps: List[Dict[str, Any]], err_step_idx: int) -> bool:
    """
    Checks if a deleted step is effectively redundant because the
    same action occurs in the immediate vicinity (regional or future).
    """
    if err_step_idx < 0 or err_step_idx >= len(steps):
        return False

    s0 = steps[err_step_idx]
    pred0 = (s0.get("predicate") or "").upper()
    obj0 = s0.get("object_value")
    obj0_str = obj0.strip() if isinstance(obj0, str) and obj0.strip() else _canonical_obj_string(s0)
    sid0 = s0.get("step_description_id")

    # REGIONAL SEARCH: Look +/- 5 steps.
    # If the same action exists nearby, skipping one instance is not a critical error.
    search_range = range(max(0, err_step_idx - 5), min(len(steps), err_step_idx + 6))

    for j in search_range:
        if j == err_step_idx:
            continue
        sj = steps[j]
        if (sj.get("predicate") or "").upper() != pred0:
            continue
        if obj0_str:
            ovj = sj.get("object_value")
            if isinstance(ovj, str) and ovj.strip() == obj0_str:
                return True
            # fallback: same object appears in roles
            if _step_uses_object_in_roles(sj, obj0_str, roles=FUTURE_USE_ROLES):
                return True

    # FUTURE SEARCH: Check if the exact same step ID appears later in the procedure.
    for j in range(err_step_idx + 1, len(steps)):
        sj = steps[j]
        sidj = sj.get("step_description_id")
        if sid0 is not None and sidj is not None and str(sidj) == str(sid0):
            return True
        # Fallback to text check if ID is missing
        if normalize_step_text(sj.get("step_description")) == normalize_step_text(
            s0.get("step_description")
        ):
            return True

    return False


def transposition_is_commutative_add_add(steps: List[Dict[str, Any]], a: int, b: int) -> bool:
    """Guard: transposition between ADD and ADD is often harmless; skip correction."""
    if a < 0 or b < 0 or a >= len(steps) or b >= len(steps):
        return False
    pa = (steps[a].get("predicate") or "").upper()
    pb = (steps[b].get("predicate") or "").upper()
    return (pa, pb) in COMMUTATIVE_TRANSPOSE_PAIRS


# -----------------------------
# Candidate selection / one correction per detect step
# -----------------------------


def pick_one_correction_per_detect_step(
    candidates: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enforce: at most one correction per detect_at_step_index.
    If multiple candidates map to the same detect step, choose one by priority.
    """
    by_detect: Dict[int, List[Dict[str, Any]]] = {}
    for c in candidates:
        d = int(c["detect_at_step_index"])
        by_detect.setdefault(d, []).append(c)

    selected: List[Dict[str, Any]] = []
    for d_idx, group in sorted(by_detect.items(), key=lambda x: x[0]):

        def key_fn(c: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
            forced = (
                1
                if c.get("correction_type")
                in {
                    "reacquire_missing_object_before_use",
                    "perform_missed_step_before_first_use",
                }
                else 0
            )
            sev_r = severity_rank(c.get("severity", "low"))
            ess = 1 if c.get("is_essential", False) else 0
            pred = c.get("predicate")
            dang = 1 if is_dangerous_predicate(pred) else 0
            lat = int(c.get("latency_steps", 0))
            err_id = str(c.get("targets_error_id", ""))
            return (forced, sev_r, ess, dang, -lat, err_id)

        best = sorted(group, key=key_fn, reverse=True)[0]

        # Forced corrections must survive even if they violate the block-locality guard.
        # Otherwise we can end up with a physically impossible plan (missing required object/step).
        forced_types = {
            "reacquire_missing_object_before_use",
            "perform_missed_step_before_first_use",
        }
        is_forced = best.get("correction_type") in forced_types

        # Locality constraint for rollback-heavy corrections:
        # Compare redo_step_index with the original error step index (not with detect index),
        # to avoid rejecting substitutions just because detection happened in another block.
        ctype = best.get("correction_type")
        if ctype == "rollback_and_redo" and not is_forced:
            redo_idx = best.get("redo_step_index")
            err_idx = best.get("_error_step_index")
            if isinstance(redo_idx, int) and isinstance(err_idx, int):
                if not within_same_taxonomy_block(steps, redo_idx, err_idx):
                    continue

        selected.append(best)

    return selected


# Maximum steps allowed between error and detection for a "fix" to remain logical.
# If a missed step isn't caught within 3 steps, performing it 'now' usually makes no sense.
MAX_DELETION_REPAIR_LATENCY = 3

# For transpositions, if too many steps passed, the sequence is too corrupted to 'restore order'.
MAX_TRANSPOSITION_REPAIR_LATENCY = 4

# -----------------------------
# Correction synthesis
# -----------------------------


def propose_correction_for_error(
    rng: random.Random, take_uid: str, err: Dict[str, Any], steps: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Evaluates an error and synthesizes a structured correction,
    applying semantic, temporal, and redundancy guards.
    """
    err_type = str(err.get("type") or err.get("error_type") or "").lower().strip()
    if err_type not in BASE_DETECT:
        return None

    err_step_idx = get_error_step_index(err)
    if not isinstance(err_step_idx, int) or not (0 <= err_step_idx < len(steps)):
        return None

    step = steps[err_step_idx]
    phase_b = phase_bucket(str(step.get("phase") or err.get("phase") or "phase_2"))
    severity = str(err.get("severity") or "low")
    is_essential = bool(step.get("is_essential", False))
    predicate = (step.get("predicate") or err.get("predicate") or "").upper()
    step_load = float(step.get("step_load", 0.0) or 0.0)
    object_value = step.get("object_value")
    obj_for_matching = _canonical_obj_string(step)

    # --- 0) FORCED CORRECTION FROM ERROR SIMULATOR (UNLOCK deletion) ---
    # error_simulator may mark deletion events with:
    #   requires_correction=true, correction_before_step_index=j
    # meaning: allow deletion BUT must reacquire missing object before first use.
    if err_type == "deletion" and bool(err.get("requires_correction", False)):
        j = err.get("correction_before_step_index")
        if isinstance(j, int) and 0 <= j < len(steps):
            # If object already reacquired before j, no forced correction needed.
            if object_reacquired_before_index(
                steps=steps,
                obj=obj_for_matching,
                start_exclusive=err_step_idx,
                end_exclusive=j,
            ):
                return None

            detect_idx = max(
                0, j - 1
            )  # log line “after step detect_idx”, correction happens before step j
            L = max(0, detect_idx - err_step_idx)
            targets_error_id = str(err.get("event_id") or "").strip()

            return {
                "correction_id": "",  # filled by caller
                "targets_error_id": targets_error_id,
                "targets_error_type": err_type,
                "detect_at_step_index": int(detect_idx),
                "latency_steps": int(L),
                "correction_type": "reacquire_missing_object_before_use",
                "insert_before_step_index": int(j),
                "missing_object": obj_for_matching
                if obj_for_matching is not None
                else object_value,
                "missing_from_step_index": int(err_step_idx),
                "_error_step_index": int(err_step_idx),
                "_priority_meta": {
                    # forced corrections should win when grouped by detect step
                    "severity": "high",
                    "is_essential": bool(step.get("is_essential", False)),
                    "predicate": predicate,
                },
            }
        # if malformed index, fall through to normal logic (best-effort)

    # --- 0b) FORCED-ish CORRECTION: deletion of PREP step and object is used soon after ---
    # This fixes exactly the “CUT carrot deleted, then ADD carrots happens” class of bugs.
    if err_type == "deletion":
        pred0 = predicate.upper() if isinstance(predicate, str) else ""
        if pred0 in PREP_PREDS:
            first_use = find_first_future_use_index_tokens(
                steps=steps,
                idx=int(err_step_idx),
                obj=object_value,
                max_lookahead=10,
            )
            if isinstance(first_use, int):
                # If “use” is too far, keep your temporal constraint logic (avoid zombie fixes)
                detect_idx = max(0, first_use - 1)
                L = max(0, detect_idx - err_step_idx)
                if L <= MAX_DELETION_REPAIR_LATENCY:
                    targets_error_id = str(err.get("event_id") or "").strip()
                    return {
                        "correction_id": "",
                        "targets_error_id": targets_error_id,
                        "targets_error_type": err_type,
                        "detect_at_step_index": int(detect_idx),
                        "latency_steps": int(L),
                        "correction_type": "perform_missed_step_before_first_use",
                        "redo_step_index": int(err_step_idx),
                        "insert_before_step_index": int(first_use),
                        "missing_object": object_value,
                        "_error_step_index": int(err_step_idx),
                        "_priority_meta": {
                            "severity": "high",
                            "is_essential": bool(step.get("is_essential", False)),
                            "predicate": predicate,
                        },
                    }

    # --- 1. SEMANTIC & REDUNDANCY GUARDS ---

    # Filter out trivial actions (cleaning, stirring, etc.) for sequence errors
    if predicate in TRIVIAL_PREDICATES:
        if err_type in {"deletion", "transposition"}:
            return None  # Do not correct minor sequencing shifts for low-stakes actions
        if err_type == "wrong_execution" and rng.random() > 0.3:
            return None  # Reduce corrections for minor slips on trivial actions

    # Redundant deletions (repeating an action nearby)
    if err_type == "deletion" and deletion_is_redundant(steps, err_step_idx):
        return None

    # Harmless transpositions
    if err_type == "transposition":
        a, b = get_transposition_pair(err)
        if a is not None and b is not None:
            # Check for commutative actions (e.g., ADD vs ADD)
            if transposition_is_commutative_add_add(steps, a, b):
                return None
            # Check if any step in the swap pair is trivial
            pa, pb = (
                (steps[a].get("predicate") or "").upper(),
                (steps[b].get("predicate") or "").upper(),
            )
            if pa in TRIVIAL_PREDICATES or pb in TRIVIAL_PREDICATES:
                return None

    # --- 2. DETECTION & LATENCY LOGIC ---

    # Calculate probability of detection based on severity and load
    p_detect = compute_detect_prob(
        err_type=err_type,
        phase_b=phase_b,
        severity=severity,
        is_essential=is_essential,
        predicate=predicate,
        step_load=step_load,
    )
    if rng.random() > p_detect:
        return None

    # Latency calculation
    L = sample_latency_steps(rng, err_type)

    # Precondition triggers (does the next step force us to notice the error?)
    min_L = precondition_trigger_min_latency(
        steps, err_step_idx, predicate, object_value, lookahead=5
    )
    if min_L is not None:
        L = min(L, min_L)

    detect_idx = min(err_step_idx + L, len(steps) - 1)

    # --- 3. TEMPORAL & BLOCK CONSTRAINTS ---

    # Temporal Window: Prevent "zombie corrections" happening too late
    if err_type == "deletion" and L > MAX_DELETION_REPAIR_LATENCY:
        return None  # Too late to perform the missed step logically

    if err_type == "transposition" and L > MAX_TRANSPOSITION_REPAIR_LATENCY:
        return None  # Sequence is too far gone to restore order simply

    # Block Guard: Detection must stay within the same functional block for sequence fixes
    if err_type in {"deletion", "transposition"}:
        if not within_same_taxonomy_block(steps, err_step_idx, detect_idx):
            return None

    # Decision to act
    p_act = compute_act_prob(err_type, phase_b, severity, step_load)
    if rng.random() > p_act:
        return None

    # --- 4. CORRECTION SYNTHESIS ---

    c_type: str
    rb: Optional[int] = 0
    redo: Optional[int] = None

    if err_type == "wrong_execution":
        if phase_b == "late" and step_load > 0.6:
            c_type, redo = "stop_and_fix", err_step_idx
        else:
            if rng.random() < 0.65:
                c_type, redo = "stop_and_fix", err_step_idx
            else:
                c_type = "rollback_and_redo"
                rb = min(geom_ge_1(rng, 0.65) - 1, 2)
                redo = max(err_step_idx - rb, 0)

    elif err_type == "deletion":
        c_type, redo = "perform_missed_step_now", err_step_idx

    elif err_type == "insertion":
        c_type, rb = "undo_extra_step", (0 if rng.random() < 0.7 else 1)

    elif err_type == "substitution":
        c_type = "rollback_and_redo"
        rb = min(geom_ge_1(rng, 0.55), 2)
        redo = max(err_step_idx - rb, 0)

    elif err_type == "transposition":
        c_type = "restore_order"
    else:
        return None

    # Assemble correction record
    targets_error_id = str(err.get("event_id") or "").strip()
    correction = {
        "correction_id": "",  # Filled by caller
        "targets_error_id": targets_error_id,
        "targets_error_type": err_type,
        "detect_at_step_index": int(detect_idx),
        "latency_steps": int(L),
        "correction_type": c_type,
        "_error_step_index": int(err_step_idx),
    }
    if rb is not None:
        correction["rollback_steps"] = int(rb)
    if redo is not None:
        correction["redo_step_index"] = int(redo)

    # For transpositions, store the pair for the LLM
    if err_type == "transposition" and c_type == "restore_order":
        pair = get_transposition_pair(err)
        if pair[0] is not None:
            correction["restore_order_pair"] = [int(pair[0]), int(pair[1])]

    correction["_priority_meta"] = {
        "severity": severity,
        "is_essential": is_essential,
        "predicate": predicate,
    }

    # Add a compact intent string to guide LLM writer/judge.
    ctype = correction.get("correction_type")
    intent = None
    if ctype == "undo_extra_step":
        intent = "Undo the extra inserted step so the procedure can continue."
    elif ctype == "stop_and_fix":
        intent = "Stop, fix the wrong execution, and redo the intended action correctly."
    elif ctype == "rollback_and_redo":
        intent = "Rollback the substituted action and redo the original intended step."
    elif ctype == "restore_order":
        intent = "Restore a logically feasible order for the transposed steps."
    else:
        intent = "Apply a minimal fix so later steps remain feasible."
    correction["intent"] = intent
    return correction


def simulate_corrections_for_take(
    rng: random.Random,
    take_uid: str,
    take_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    steps = take_payload.get("steps", [])
    errors = take_payload.get("errors", [])
    if not isinstance(steps, list) or not isinstance(errors, list):
        return []

    candidates: List[Dict[str, Any]] = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        cand = propose_correction_for_error(rng, take_uid, err, steps)
        if cand is None:
            continue

        # Flatten priority meta for sorting/group selection, then remove later.
        meta = cand.pop("_priority_meta", {})
        cand["severity"] = meta.get("severity", "low")
        cand["is_essential"] = bool(meta.get("is_essential", False))
        cand["predicate"] = meta.get("predicate")
        candidates.append(cand)

    selected = pick_one_correction_per_detect_step(candidates, steps)

    # Assign stable correction ids per take
    for idx, c in enumerate(
        sorted(selected, key=lambda x: (x["detect_at_step_index"], str(x["targets_error_id"])))
    ):
        c["correction_id"] = f"{take_uid}_corr_{idx}"

        # Strip helper fields to keep output minimal
        c.pop("severity", None)
        c.pop("is_essential", None)
        c.pop("predicate", None)
        c.pop("_error_step_index", None)

    return sorted(selected, key=lambda x: (x["detect_at_step_index"], x["correction_id"]))


# -----------------------------
# Log injection
# -----------------------------


def format_correction_log_line(
    corr_number_1based: int,
    detect_idx: int,
    phase: str,
    correction: Dict[str, Any],
    steps: List[Dict[str, Any]],
) -> str:
    cid = str(correction.get("targets_error_id", ""))
    etype = str(correction.get("targets_error_type", "unknown"))
    ctype = str(correction.get("correction_type", ""))

    suffix = ""
    if ctype in {"stop_and_fix", "rollback_and_redo"}:
        redo = correction.get("redo_step_index", None)
        if isinstance(redo, int):
            suffix = f"-> redo Step {redo + 1:02d}"
        else:
            suffix = "-> redo step"

    elif ctype == "perform_missed_step_now":
        redo = correction.get("redo_step_index", None)
        if isinstance(redo, int):
            suffix = f"-> do missed Step {redo + 1:02d} now"
        else:
            suffix = "-> do missed step now"

    elif ctype == "undo_extra_step":
        suffix = "-> undo extra step"

    elif ctype == "restore_order":
        pair = correction.get("restore_order_pair")
        if isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, int) for x in pair):
            a, b = pair[0], pair[1]
            suffix = f"-> restore order between Steps {a + 1:02d} and {b + 1:02d}"
        else:
            suffix = "-> restore correct step order"

    elif ctype == "reacquire_missing_object_before_use":
        ins = correction.get("insert_before_step_index", None)
        if isinstance(ins, int):
            suffix = f"-> reacquire before Step {ins + 1:02d}"
        else:
            suffix = "-> reacquire before first use"
    elif ctype == "perform_missed_step_before_first_use":
        ins = correction.get("insert_before_step_index", None)
        if isinstance(ins, int):
            suffix = f"-> do missed step before Step {ins + 1:02d}"
        else:
            suffix = "-> do missed step before first use"

    return (
        f"[Corr {corr_number_1based:02d} after Step {detect_idx + 1:02d}] "
        f"({phase}) CORRECTION for {cid} {etype}: {ctype} {suffix}"
    )


def inject_corrections_into_logs(
    take_payload: Dict[str, Any],
    corrections: List[Dict[str, Any]],
) -> None:
    """
    Inject correction lines into take_payload['simulation_log_lines'] and rebuild take_payload['simulation_log'].
    This modifies take_payload in-place.
    """
    lines = take_payload.get("simulation_log_lines", [])
    steps = take_payload.get("steps", [])
    if not isinstance(lines, list) or not isinstance(steps, list):
        return
    if not corrections:
        # Ensure simulation_log matches lines if needed.
        take_payload["simulation_log"] = "\n".join([str(x) for x in lines])
        return

    # Group corrections by detect step index
    by_detect: Dict[int, List[Dict[str, Any]]] = {}
    for c in corrections:
        d = c.get("detect_at_step_index")
        if isinstance(d, int):
            by_detect.setdefault(d, []).append(c)

    # Stable ordering within each detect step
    for d in list(by_detect.keys()):
        by_detect[d] = sorted(by_detect[d], key=lambda x: str(x.get("correction_id", "")))

    out_lines: List[str] = []
    corr_counter = 0

    for i, line in enumerate(lines):
        out_lines.append(str(line))

        if i in by_detect:
            # Use the phase of the detect step for readability
            phase = str(steps[i].get("phase", "phase_2")) if 0 <= i < len(steps) else "phase_2"
            for c in by_detect[i]:
                corr_counter += 1
                out_lines.append(
                    format_correction_log_line(
                        corr_number_1based=corr_counter,
                        detect_idx=i,
                        phase=phase,
                        correction=c,
                        steps=steps,
                    )
                )

    take_payload["simulation_log_lines"] = out_lines
    take_payload["simulation_log"] = "\n".join(out_lines)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add structured correction plans to an error plan JSON."
    )
    p.add_argument("--input", required=True, help="Path to split_50_error_plan.json")
    p.add_argument("--out", required=True, help="Output path for extended JSON with corrections")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_data = deepcopy(data)

    takes = out_data.get("takes", {})
    if not isinstance(takes, dict):
        raise ValueError("Input JSON must contain a dict at key 'takes'.")

    n_corr_total = 0
    for take_uid, take_payload in takes.items():
        if not isinstance(take_payload, dict):
            continue

        corrections = simulate_corrections_for_take(rng, str(take_uid), take_payload)
        take_payload["corrections"] = corrections
        n_corr_total += len(corrections)

        # Inject correction lines directly into existing logs
        inject_corrections_into_logs(take_payload, corrections)

    # Update meta
    meta = out_data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        out_data["meta"] = meta

    meta["corrections_meta"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "config": {
            "base_detect": BASE_DETECT,
            "latency": LATENCY,
            "base_act": BASE_ACT,
            "dangerous_predicates": sorted(list(DANGEROUS_PREDICATES)),
            "max_one_correction_per_detect_step": True,
            "no_ignore_records": True,
            "guards": {
                "deletion_anti_redundant": True,
                "transposition_commutative_add_add": True,
                "rollback_locality_compared_to_error_step": True,
                "transposition_spec_supported": [
                    "swap_step_index_a/b",
                    "transposition_source/target",
                    "swap_with_step_index (legacy)",
                ],
            },
        },
        "stats": {
            "n_takes": len(takes),
            "n_corrections_total": n_corr_total,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print("=== Correction Simulation ===")
    print(f"Input:  {args.input}")
    print(f"Output: {args.out}")
    print(f"Takes: {len(takes)}")
    print(f"Total corrections: {n_corr_total}")


if __name__ == "__main__":
    main()


# Example:
# piev-corrections \
#   --input local/outputs/split_50_error_plan.json \
#   --out   local/outputs/split_50_error_plan_with_corrections.json \
#   --seed 123
