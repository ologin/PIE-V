#!/usr/bin/env python3
"""
PIE-V Error Simulator (policy-only)

Injects structured error events into procedural step sequences.
Does NOT rewrite step text; outputs a JSON "error plan" usable by a later LLM stage.

Key principles:
- Use semantic representations to estimate cognitive load per step.
- Use step duration to approximate time pressure and attentional demands.
- Use taxonomy blocks (from keystep_train.json) to constrain transpositions
  (and *prefer* “same block” substitutions) without building a world model.
- Use role impact (high/medium/low) and predicate-specific role priors
  to parameterize wrong-execution errors in a psychologically plausible way.
- Enforce hard constraints to avoid obviously broken procedures.
"""

import argparse
import csv
import json
import math
import random
import sys
import re
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set


# Error taxonomy used throughout PIE-V.
ERROR_TYPES_ORDER = ["wrong_execution", "deletion", "substitution", "insertion", "transposition"]

# Phase-level error-rate model (controls WHERE errors land, not which TYPE they are).
# These are relative phase rates from your table:
#   Phase 1: 10%, Phase 2: 19%, Phase 3: 14%
# We normalize them to multipliers with mean=1.0 so total error count k remains governed by proc risk.
PHASE_ERROR_RATE_MODEL = {
    "phase_1": 0.10,
    "phase_2": 0.19,
    "phase_3": 0.14,
}

def phase_rate_multipliers(model: Dict[str, float]) -> Dict[str, float]:
    vals = [float(v) for v in model.values() if isinstance(v, (int, float))]
    if not vals:
        return {"phase_1": 1.0, "phase_2": 1.0, "phase_3": 1.0}
    mean = sum(vals) / float(len(vals))
    if mean <= 0:
        return {"phase_1": 1.0, "phase_2": 1.0, "phase_3": 1.0}
    out: Dict[str, float] = {}
    for k, v in model.items():
        try:
            out[str(k)] = max(0.01, float(v) / mean)
        except Exception:
            out[str(k)] = 1.0
    # Ensure all three phases exist
    out.setdefault("phase_1", 1.0)
    out.setdefault("phase_2", 1.0)
    out.setdefault("phase_3", 1.0)
    return out


# Phase priors: distribution over error types within a phase.
PHASE_ERROR_TYPE_PRIORS = {
    "phase_1": {"PRIOR_WEIGHTS": [3.5, 1.0, 2.5, 2.0, 1.0]},  # order matches ERROR_TYPES_ORDER
    "phase_2": {"PRIOR_WEIGHTS": [2.0, 2.0, 1.5, 2.5, 2.0]},
    "phase_3": {"PRIOR_WEIGHTS": [3.5, 2.5, 1.0, 2.0, 1.0]},
}

# Hard constraints on transposition.
FORBIDDEN_TRANSPOSE_PREDICATES = {"REPEAT"}

# Ordering constraints: dependent_pred should not occur before prerequisite_pred for same Object.
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
    ("CUT", "GET")
]

UNLOCK_PREDS = {"GET","TAKE","RETRIEVE","PICK_UP","GRAB","REMOVE"}
STOP_TOKENS = {"of","in","on","from","to","with","and","or"}


ROLE_VALUE_PRED_RE = re.compile(r"\b([A-Z][a-zA-Z_]*)\s*:\s*([A-Z_]+)\s*\(")
OPERATOR_RE = re.compile(r"\b(WHILE|UNTIL|BEFORE|AFTER|WHEN|IF|DURING)\b")

# Roles in which we consider an object "used" downstream for unlock deletion correction.
FUTURE_USE_ROLES = ("Object", "Instrument", "Destination", "Origin", "Coobject")


# -----------------------------
# Utility: JSON / CSV loading
# -----------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab_csv(path: Path) -> Dict[str, str]:
    """
    Loads split_50_vocabulary.csv into mapping:
      step_description -> step_description_id (string).
    """
    mapping: Dict[str, str] = {}
    duplicate_count = 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sd = (row.get("step_description") or "").strip()
            sid = (row.get("step_description_id") or "").strip()
            if sd and sid:
                if sd in mapping and mapping[sd] != sid:
                    duplicate_count += 1
                mapping[sd] = sid
    if duplicate_count > 0:
        print(
            f"[WARN] load_vocab_csv: found {duplicate_count} duplicate step_description rows with differing ids; "
            f"last value wins. File: {path}",
            file=sys.stderr,
        )
    return mapping


def load_role_impact_csv(path: Path) -> Dict[str, str]:
    """Loads semantic_roles.csv mapping role -> impact category (high/medium/low)."""
    role_to_impact: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = (row.get("role") or "").strip()
            impact = (row.get("impact") or "").strip().lower()
            if role and impact:
                role_to_impact[role] = impact
    return role_to_impact


def load_roles_by_predicate_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Loads semantic_roles_by_predicate.csv into mapping:
      predicate -> role -> share_in_predicate (float).
    """
    prior: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = (row.get("predicate") or "").strip().upper()
            role = (row.get("role") or "").strip()
            share = row.get("share_in_predicate")
            if not pred or not role or share is None:
                continue
            try:
                prior[pred][role] = float(share)
            except Exception:
                continue
    return prior


def count_structural_relations(rep: str) -> Tuple[int, int]:
    role_pred_pairs = ROLE_VALUE_PRED_RE.findall(rep)
    structural_roles = {"Purpose", "Temporal", "Condition", "Result"}
    n_comp = sum(1 for role, _pred in role_pred_pairs if role in structural_roles)
    n_ops = len(OPERATOR_RE.findall(rep))
    return n_comp, n_ops


def normalize_step_text(s: Any) -> str:
    """Lowercase + strip + collapse whitespace for robust equality checks."""
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().lower().split())

def tokens(val: str) -> Set[str]:
    toks = set(re.findall(r"[a-z_]+", val.lower()))
    return {t for t in toks if t not in STOP_TOKENS}

def object_tokens(obj: Any) -> Set[str]:
    if not isinstance(obj, str):
        return set()
    return tokens(obj.strip())

def required_entities(step: dict) -> Set[str]:
    rtv = step.get("role_to_value") or {}
    out = set()
    if isinstance(rtv, dict):
        for role in ("Instrument","Object","Coobject"):
            v = rtv.get(role)
            if isinstance(v, str):
                out |= tokens(v)
    return out

def introduced_entities(step: dict) -> Set[str]:
    pred = (step.get("predicate") or "").upper()
    if pred not in UNLOCK_PREDS:
        return set()
    rtv = step.get("role_to_value") or {}
    v = rtv.get("Object") if isinstance(rtv, dict) else None
    return tokens(v) if isinstance(v, str) else set()


def is_step_essential(
    steps: List[Dict[str, Any]],
    idx: int,
    *,
    min_future_refs: int = 2,
    roles: Tuple[str, ...] = FUTURE_USE_ROLES,
) -> bool:
    """
    Heuristic "essential" guard for deletion retargeting.

    A step is treated as essential if entities mentioned in its Object/Coobject
    (and/or object_value) are referenced multiple times in *future* steps' semantic roles.

    This intentionally uses only semrep-derived signals (role_to_value tokens),
    no world model.
    """
    if idx < 0 or idx >= len(steps):
        return False

    s = steps[idx]

    # Candidate entities from this step:
    ent_tokens: Set[str] = set()
    rtv = s.get("role_to_value") or {}
    if isinstance(rtv, dict):
        for r in ("Object", "Coobject"):
            v = rtv.get(r)
            if isinstance(v, str):
                ent_tokens |= tokens(v)

    ov = s.get("object_value")
    if isinstance(ov, str):
        ent_tokens |= tokens(ov)

    # If we can't extract anything, do not block deletion via "essential".
    if not ent_tokens:
        return False

    # Count how often these tokens appear in future role values.
    future_refs = 0
    for j in range(idx + 1, len(steps)):
        sj = steps[j]
        rtvj = sj.get("role_to_value") or {}
        if not isinstance(rtvj, dict):
            continue
        for r in roles:
            v = rtvj.get(r)
            if isinstance(v, str) and (ent_tokens & tokens(v)):
                future_refs += 1
                if future_refs >= int(min_future_refs):
                    return True

        # also count direct object_value matches as a strong reference
        ovj = sj.get("object_value")
        if isinstance(ovj, str) and (ent_tokens & tokens(ovj)):
            future_refs += 1
            if future_refs >= int(min_future_refs):
                return True

    return False


# -----------------------------
# Semantic parsing & complexity
# -----------------------------

def semantic_complexity(representation: str) -> float:
    """
    Complexity proxy based on:
    - number of predicates
    - number of roles
    - nesting depth
    - number of explicit relations
    - number of distinct entities/tokens
    """
    if not representation or not isinstance(representation, str):
        return 0.0

    predicates = re.findall(r"\b[A-Z_]+\s*\(", representation)
    n_pred = len(predicates)

    roles = re.findall(r"\b[A-Z][a-zA-Z_]*:", representation)
    n_roles = len(roles)

    entities = re.findall(r"\b[a-z_]+\b", representation)
    n_entities = len(set(entities))

    depth, max_depth = 0, 0
    for ch in representation:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)

    n_comp, n_ops = count_structural_relations(representation)
    n_rel = n_comp + n_ops

    a, b, c, d, e = 1.0, 1.0, 2.0, 1.0, 0.5
    return round(a * n_pred + b * n_roles + c * max_depth + d * n_rel + e * n_entities, 3)


def parse_semantic_representation(rep: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Shallow parse:
      PREDICATE(ROLE: value, ROLE2: value2(...))

    Returns:
      predicate, role_to_value
    """
    if not rep or not isinstance(rep, str):
        return None, {}

    m = re.match(r"\s*([A-Z_]+)\s*\(", rep)
    predicate = m.group(1) if m else None

    start = rep.find("(")
    end = rep.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return predicate, {}

    inside = rep[start + 1:end].strip()
    if not inside:
        return predicate, {}

    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in inside:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)

        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    role_to_value: Dict[str, str] = {}
    for p in parts:
        if ":" not in p:
            continue
        role, value = p.split(":", 1)
        role = role.strip()
        value = value.strip()
        if role and value:
            role_to_value[role] = value

    return predicate, role_to_value


def extract_main_object(role_to_value: Dict[str, str]) -> Optional[str]:
    """Extracts shallow Object value (token before any nested structure)."""
    if "Object" not in role_to_value:
        return None
    raw = role_to_value["Object"].strip()
    if "(" in raw:
        raw = raw.split("(", 1)[0].strip()
    return raw


def minmax_normalize(values: Sequence[float]) -> List[float]:
    """Min-max normalize to [0, 1]. If constant -> zeros."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


# -----------------------------
# Taxonomy block resolution
# -----------------------------

@dataclass
class TaxNode:
    node_id: int
    name: str
    parent_id: Optional[int]
    is_leaf: bool
    unique_id: Optional[int]


def build_taxonomy_index(taxonomy_for_scenario: Dict[str, Any]) -> Tuple[Dict[int, TaxNode], Dict[int, int]]:
    """Build indices for a single scenario taxonomy."""
    nodes_by_id: Dict[int, TaxNode] = {}
    leaf_unique_to_node_id: Dict[int, int] = {}

    for _, node in taxonomy_for_scenario.items():
        if not isinstance(node, dict) or "id" not in node:
            continue
        try:
            node_id = int(node["id"])
        except Exception:
            continue

        name = str(node.get("name", ""))
        parent_id = node.get("parent_id", None)
        if parent_id is not None:
            try:
                parent_id = int(parent_id)
            except Exception:
                parent_id = None

        is_leaf = bool(node.get("is_leafnode", False))
        unique_id = node.get("unique_id", None)
        if unique_id is not None:
            try:
                unique_id = int(unique_id)
            except Exception:
                unique_id = None

        tn = TaxNode(node_id=node_id, name=name, parent_id=parent_id, is_leaf=is_leaf, unique_id=unique_id)
        nodes_by_id[node_id] = tn

        if is_leaf and unique_id is not None:
            leaf_unique_to_node_id[unique_id] = node_id

    return nodes_by_id, leaf_unique_to_node_id


def resolve_top_level_block(node_id: int, nodes_by_id: Dict[int, TaxNode]) -> Tuple[Optional[int], Optional[str], List[str]]:
    """
    Resolve leaf node to top-level block under scenario root (parent_id == 0).
    Returns (block_id, block_name, path_names_root_to_leaf)
    """
    path: List[str] = []
    cur = nodes_by_id.get(node_id)
    visited = set()
    while cur is not None and cur.node_id not in visited:
        visited.add(cur.node_id)
        path.append(cur.name)
        if cur.parent_id is None:
            break
        parent = nodes_by_id.get(cur.parent_id)
        if parent is None:
            break
        cur = parent

    path_rev = list(reversed(path))

    cur = nodes_by_id.get(node_id)
    visited.clear()
    while cur is not None and cur.node_id not in visited:
        visited.add(cur.node_id)
        if cur.parent_id == 0:
            return cur.node_id, cur.name, path_rev
        if cur.parent_id is None:
            break
        cur = nodes_by_id.get(cur.parent_id)

    return None, None, path_rev


# -----------------------------
# Phase assignment using load
# -----------------------------

def assign_phases_by_load(step_loads: List[float]) -> List[str]:
    """Assign phase_1/2/3 by splitting cumulative load into thirds."""
    if not step_loads:
        return []
    total = sum(step_loads)
    if total <= 0:
        return ["phase_1" for _ in step_loads]

    cum = [0.0]
    for v in step_loads:
        cum.append(cum[-1] + v)

    t1 = total / 3.0
    t2 = 2.0 * total / 3.0

    phases: List[str] = []
    for i in range(len(step_loads)):
        mid = (cum[i] + cum[i + 1]) / 2.0
        if mid <= t1:
            phases.append("phase_1")
        elif mid <= t2:
            phases.append("phase_2")
        else:
            phases.append("phase_3")
    return phases


# -----------------------------
# Procedure-level risk -> K errors
# -----------------------------

@dataclass
class ProcRiskFeatures:
    n_steps: int
    total_duration_s: float
    total_complexity: float
    density_steps_per_min: float


def compute_proc_features(step_durations: List[float], step_complexities: List[float]) -> ProcRiskFeatures:
    n_steps = len(step_durations)
    total_duration_s = float(sum(step_durations)) if step_durations else 0.0
    total_complexity = float(sum(step_complexities)) if step_complexities else 0.0
    density_steps_per_min = (60.0 * n_steps / total_duration_s) if total_duration_s > 0 else 0.0
    return ProcRiskFeatures(
        n_steps=n_steps,
        total_duration_s=total_duration_s,
        total_complexity=total_complexity,
        density_steps_per_min=density_steps_per_min,
    )


def proc_risk_score(
    feat: ProcRiskFeatures,
    norm_total_complexity: float,
    norm_total_duration: float,
    norm_density: float,
) -> float:
    w_c, w_d, w_t = 0.5, 0.3, 0.2
    r = w_c * norm_total_complexity + w_d * norm_density + w_t * norm_total_duration
    return float(min(max(r, 0.0), 1.0))


def sample_k_errors(
    rng: np.random.Generator,
    risk: float,
    n_steps: int,
    max_errors: int,
) -> Tuple[int, float]:
    length_boost = min(1.0, max(0.0, (n_steps - 6) / 24.0))
    lam = 0.8 + 3.2 * risk + 0.6 * length_boost
    k = int(rng.poisson(lam))
    k = max(1, min(int(max_errors), k))
    return k, float(lam)


# -----------------------------
# Wrong execution parameterization
# -----------------------------

def impact_weight(impact: str) -> float:
    impact = (impact or "").lower()
    if impact == "low":
        return 1.0
    if impact == "medium":
        return 0.8
    if impact == "high":
        return 0.25
    return 0.6


def choose_roles_to_mutate(
    rng: random.Random,
    predicate: Optional[str],
    role_to_value: Dict[str, str],
    role_to_impact: Dict[str, str],
    roles_prior_by_predicate: Dict[str, Dict[str, float]],
) -> List[str]:
    if not role_to_value:
        return []

    roles_present = [r for r in role_to_value.keys() if r != "Agent"]
    if not roles_present:
        return []

    pred = (predicate or "").upper()
    prior = roles_prior_by_predicate.get(pred, {})

    scored: List[Tuple[str, float]] = []
    for role in roles_present:
        imp = role_to_impact.get(role, "medium")
        w_imp = impact_weight(imp)
        w_prior = prior.get(role, 0.05)
        score = w_imp * (0.2 + w_prior)
        scored.append((role, score))

    n = 1
    if len(scored) >= 2 and rng.random() < 0.20:
        n = 2

    chosen: List[str] = []
    candidates = scored[:]
    for _ in range(n):
        total = sum(w for _, w in candidates)
        if total <= 0:
            break
        r = rng.random() * total
        acc = 0.0
        pick_idx = 0
        for i, (_, w) in enumerate(candidates):
            acc += w
            if acc >= r:
                pick_idx = i
                break
        chosen_role = candidates[pick_idx][0]
        chosen.append(chosen_role)
        candidates.pop(pick_idx)

    return chosen


def choose_wrong_execution_scope(rng: random.Random, role: str) -> str:
    if role in {"Location", "Destination", "Origin", "Path", "Temporal"}:
        probs = [0.10, 0.25, 0.65]  # attribute, entity, substructure
    elif role in {"Instrument", "Object", "Coobject"}:
        probs = [0.35, 0.65, 0.0]
    else:
        probs = [0.25, 0.75, 0.0]

    x = rng.random()
    if x < probs[0]:
        return "attribute"
    if x < probs[0] + probs[1]:
        return "entity"
    return "substructure"


# -----------------------------
# Transposition + Deletion guards
# -----------------------------

def violates_ordering_constraints(
    pred_a: Optional[str],
    obj_a: Optional[str],
    pred_b: Optional[str],
    obj_b: Optional[str],
) -> bool:
    if not pred_a or not pred_b or not obj_a or not obj_b:
        return False
    if obj_a != obj_b:
        return False

    pa = pred_a.upper()
    pb = pred_b.upper()
    for dependent_pred, prerequisite_pred in ORDERING_CONSTRAINTS_SAME_OBJECT:
        # dependent_pred should NOT appear before prerequisite_pred
        if pa == dependent_pred and pb == prerequisite_pred:
            return True
    return False


def swap_respects_constraints(steps: List[Dict[str, Any]], i: int, j: int) -> bool:
    if i == j:
        return False
    a, b = (i, j) if i < j else (j, i)

    seg = steps[a:b+1]
    seg2 = seg[:]  # shallow copy
    seg2[i-a], seg2[j-a] = seg2[j-a], seg2[i-a]

    preds = [((s.get("predicate") or "").upper(), s.get("object_value")) for s in seg2]

    for x in range(len(preds)):
        pred_x, obj_x = preds[x]
        for y in range(x + 1, len(preds)):
            pred_y, obj_y = preds[y]
            if violates_ordering_constraints(pred_x, obj_x, pred_y, obj_y):
                return False
    
    first_unlock = {}
    for idx, st in enumerate(seg2):
        for ent in introduced_entities(st):
            first_unlock.setdefault(ent, idx)
    
    for idx, st in enumerate(seg2):
        req = required_entities(st)
        for ent in req:
            if ent in first_unlock and idx < first_unlock[ent]:
                return False
                
    return True


# def is_ordering_guarded_deletion_step(
#     steps: List[Dict[str, Any]],
#     idx: int,
#     max_lookahead: Optional[int] = None,
# ) -> bool:
#     if idx < 0 or idx >= len(steps):
#         return False

#     s = steps[idx]
#     pred = (s.get("predicate") or "").upper()
#     obj = s.get("object_value")
#     if not pred or not isinstance(obj, str) or not obj.strip():
#         return False
#     obj = obj.strip()

#     end = len(steps)
#     if isinstance(max_lookahead, int) and max_lookahead > 0:
#         end = min(end, idx + 1 + max_lookahead)

#     # If current step is a prerequisite_pred, check later dependent_pred.
#     for dependent_pred, prerequisite_pred in ORDERING_CONSTRAINTS_SAME_OBJECT:
#         if pred != prerequisite_pred:
#             continue
#         for j in range(idx + 1, end):
#             sj = steps[j]
#             pj = (sj.get("predicate") or "").upper()
#             oj = sj.get("object_value")
#             if pj == dependent_pred and isinstance(oj, str) and oj.strip() == obj:
#                 return True

#     return False


# def find_deletion_retarget_index(
#     steps: List[Dict[str, Any]],
#     idx: int,
#     taken_indices: Set[int],
#     max_lookahead: int,
# ) -> Optional[int]:
#     if idx < 0 or idx >= len(steps):
#         return None

#     s = steps[idx]
#     pred = (s.get("predicate") or "").upper()
#     obj = s.get("object_value")
#     if not pred or not isinstance(obj, str) or not obj.strip():
#         return None
#     obj = obj.strip()

#     end = min(len(steps), idx + 1 + max(1, int(max_lookahead)))

#     for dependent_pred, prerequisite_pred in ORDERING_CONSTRAINTS_SAME_OBJECT:
#         if pred != prerequisite_pred:
#             continue
#         for j in range(idx + 1, end):
#             if j in taken_indices:
#                 continue
#             sj = steps[j]
#             pj = (sj.get("predicate") or "").upper()
#             oj = sj.get("object_value")
#             if pj == dependent_pred and isinstance(oj, str) and oj.strip() == obj:
#                 return j

#     return None


def get_transposition_candidates(i: int, steps: List[Dict[str, Any]], window: int) -> List[int]:
    n = len(steps)
    if i < 0 or i >= n:
        return []

    si = steps[i]
    pred_i = (si.get("predicate") or "").upper()
    block_i = si.get("taxonomy_block_name")
    
    # 1. Prepare identity data for Step I
    id_i = str(si.get("step_description_id")).strip() if si.get("step_description_id") is not None else None
    raw_text_i = si.get("step_description", "")
    text_i = raw_text_i.strip().lower() if isinstance(raw_text_i, str) else ""
    sem_i = si.get("semantic_representation", None)
    sem_i = sem_i.strip() if isinstance(sem_i, str) else None

    if pred_i in FORBIDDEN_TRANSPOSE_PREDICATES:
        return []

    lo = max(0, i - int(window))
    hi = min(n - 1, i + int(window))

    candidates: List[int] = []
    for j in range(lo, hi + 1):
        if j == i:
            continue
        
        sj = steps[j]
        
        # 2. Prepare identity data for Step J
        id_j = str(sj.get("step_description_id")).strip() if sj.get("step_description_id") is not None else None
        raw_text_j = sj.get("step_description", "")
        text_j = raw_text_j.strip().lower() if isinstance(raw_text_j, str) else ""
        sem_j = sj.get("semantic_representation", None)
        sem_j = sem_j.strip() if isinstance(sem_j, str) else None

        # --- CRITICAL: IDENTITY CHECK ---
        # If IDs match OR text matches OR semantic representation matches, they are "the same"
        # We don't use 'is not None' here to avoid skipping if mapping failed.
        is_same = False
        if id_i and id_j and id_i == id_j:
            is_same = True
        if text_i and text_j and text_i == text_j:
            is_same = True
        if sem_i and sem_j and sem_i == sem_j:
            is_same = True
            
        if is_same:
            continue # Skip identical steps

        # 3. Taxonomy and safety
        pred_j = (sj.get("predicate") or "").upper()
        block_j = sj.get("taxonomy_block_name")

        if pred_j in FORBIDDEN_TRANSPOSE_PREDICATES:
            continue

        if block_i != block_j:
            # allow only if both None (unknown block)
            if not (block_i is None and block_j is None):
                continue
        # Extra guard: if both blocks are unknown, keep swaps very local
        if block_i is None and block_j is None:
            if abs(i - j) > 3:
                continue

        # --- 4. ADVANCED RANGE CHECK (Fixes GET/RETURN jump issue) ---
        if swap_respects_constraints(steps, i, j):
            candidates.append(j)

    return candidates

def choose_transposition_partner_from_candidates(rng: random.Random, candidates: List[int]) -> Optional[int]:
    if not candidates:
        return None
    return rng.choice(candidates)


def _step_uses_object_in_roles(step: Dict[str, Any], obj: str, roles: Tuple[str, ...] = FUTURE_USE_ROLES) -> bool:
    """
    True if 'obj' appears as a token match inside specified semantic roles or matches object_value.
    Token matching avoids substring false positives.
    """
    if not isinstance(obj, str) or not obj.strip():
        return False
    ot = object_tokens(obj)
    if not ot:
        return False
    
    # direct object_value equality (kept as a strong signal)
    ov = step.get("object_value")
    if isinstance(ov, str) and ov.strip() == obj.strip():
        return True
    
    rtv = step.get("role_to_value", {}) or {}
    if not isinstance(rtv, dict):
        return False
    
    for role in roles:
        v = rtv.get(role)
        if isinstance(v, str) and (ot & tokens(v)):
            return True
    return False


def _object_used_soon_after(steps: List[Dict[str, Any]], idx: int, max_lookahead: int) -> bool:
    if idx < 0 or idx >= len(steps):
        return False

    obj = steps[idx].get("object_value")
    if not isinstance(obj, str) or not obj.strip():
        return False

    obj = obj.strip()

    end = min(len(steps), idx + 1 + max(1, int(max_lookahead)))
    for j in range(idx + 1, end):
        sj = steps[j]

        if _step_uses_object_in_roles(sj, obj, roles=FUTURE_USE_ROLES):
            return True

    return False

def find_first_future_use_index(
    steps: List[Dict[str, Any]],
    idx: int,
    max_lookahead: int,
) -> Optional[int]:
    """
    For unlock deletion: find the first future step that uses the same object
    (as object_value or within FUTURE_USE_ROLES). Correction must be inserted before it.
    """
    if idx < 0 or idx >= len(steps):
        return None
    obj = steps[idx].get("object_value")
    if not isinstance(obj, str) or not obj.strip():
        return None
    obj = obj.strip()

    end = min(len(steps), idx + 1 + max(1, int(max_lookahead)))
    for j in range(idx + 1, end):
        if _step_uses_object_in_roles(steps[j], obj, roles=FUTURE_USE_ROLES):
            return j
    return None

def compute_unlock_prior_by_predicate(
    takes_intermediate: List[Dict[str, Any]],
    max_lookahead: int,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    total: Dict[str, int] = defaultdict(int)
    reused: Dict[str, int] = defaultdict(int)

    for take in takes_intermediate:
        steps = take.get("raw_steps", [])
        if not isinstance(steps, list):
            continue
        for i, s in enumerate(steps):
            pred = (s.get("predicate") or "").upper()
            obj = s.get("object_value")
            if not pred or not isinstance(obj, str) or not obj.strip():
                continue
            total[pred] += 1
            if _object_used_soon_after(steps, i, max_lookahead=max_lookahead):
                reused[pred] += 1

    prior: Dict[str, float] = {}
    for pred, n in total.items():
        if n > 0:
            prior[pred] = float(reused.get(pred, 0) / n)
    return prior, dict(total)


def is_guarded_deletion_step(
    steps: List[Dict[str, Any]],
    idx: int,
    unlock_prior_by_predicate: Dict[str, float],
    unlock_support_by_predicate: Dict[str, int],
    threshold: float,
    max_lookahead: int,
    min_support: int,
) -> bool:
    if idx < 0 or idx >= len(steps):
        return False

    s = steps[idx]

    pred = (s.get("predicate") or "").upper()
    obj = s.get("object_value")
    if not pred or not isinstance(obj, str) or not obj.strip():
        return False

    support = int(unlock_support_by_predicate.get(pred, 0))
    if support < int(min_support):
        return False

    prior = float(unlock_prior_by_predicate.get(pred, 0.0))
    if prior < float(threshold):
        return False

    return _object_used_soon_after(steps, idx, max_lookahead=max_lookahead)


# -----------------------------
# Substitution (IMPORTANT: not constrained)
# -----------------------------

# def choose_substitution_replacement(
#     rng_py: random.Random,
#     scenario: str,
#     taxonomy_block_name: Optional[str],
#     current_step_description_id: Optional[str],
#     scenario_step_pool: Dict[str, List[Dict[str, Any]]],
#     scenario_block_step_pool: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]],
#     forbidden_step_description_ids: Optional[Set[str]] = None,
#     forbidden_step_texts_norm: Optional[Set[str]] = None,
#     forbidden_semreps: Optional[Set[str]] = None,
#     p_prefer_same_block: float = 0.75,
#     max_tries: int = 50,
# ) -> Optional[Dict[str, Any]]:
#     """
#     Substitution is conceptually always feasible (can be any other step).
#     Here we optionally sample a concrete replacement from pools.
#     Preference only:
#       - same scenario
#       - probabilistically same taxonomy block if available
#     """
#     pool_any = scenario_step_pool.get(scenario, [])
#     if not pool_any:
#         return None

#     pool_block = scenario_block_step_pool.get((scenario, taxonomy_block_name), [])
#     use_block = bool(pool_block) and (rng_py.random() < float(p_prefer_same_block))
#     pool = pool_block if use_block else pool_any

#     cur = str(current_step_description_id) if current_step_description_id is not None else None

#     forbidden_step_description_ids = forbidden_step_description_ids or set()
#     forbidden_step_texts_norm = forbidden_step_texts_norm or set()
#     forbidden_semreps = forbidden_semreps or set()

#     for _ in range(int(max_tries)):
#         cand = rng_py.choice(pool)
#         sid = cand.get("step_description_id")
#         if sid is None:
#             continue
#         sid_str = str(sid).strip()
#         if cur is not None and sid_str == cur:
#             continue
#         # forbid candidates that already appear in this take (by id/text/semrep).
#         if sid_str and sid_str in forbidden_step_description_ids:
#             continue
#         txt_norm = normalize_step_text(cand.get("step_description", ""))
#         if txt_norm and txt_norm in forbidden_step_texts_norm:
#             continue
#         sem = cand.get("semantic_representation")
#         sem = sem.strip() if isinstance(sem, str) else ""
#         if sem and sem in forbidden_semreps:
#             continue
#         return cand

#     if pool is pool_block:
#         for _ in range(int(max_tries)):
#             cand = rng_py.choice(pool_any)
#             sid = cand.get("step_description_id")
#             if sid is None:
#                 continue
#             sid_str = str(sid).strip()
#             if cur is not None and sid_str == cur:
#                 continue
#             if sid_str and sid_str in forbidden_step_description_ids:
#                 continue
#             txt_norm = normalize_step_text(cand.get("step_description", ""))
#             if txt_norm and txt_norm in forbidden_step_texts_norm:
#                 continue
#             sem = cand.get("semantic_representation")
#             sem = sem.strip() if isinstance(sem, str) else ""
#             if sem and sem in forbidden_semreps:
#                 continue
#             return cand

#     return None


# def build_substitution_spec(
#     rng_py: random.Random,
#     scenario: str,
#     taxonomy_block_name: Optional[str],
#     current_step_description_id: Optional[str],
#     scenario_step_pool: Dict[str, List[Dict[str, Any]]],
#     scenario_block_step_pool: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]],
#     forbidden_step_description_ids: Optional[Set[str]] = None,
#     forbidden_step_texts_norm: Optional[Set[str]] = None,
#     forbidden_semreps: Optional[Set[str]] = None,
#     p_prefer_same_block: float = 0.75,
# ) -> Dict[str, Any]:
#     spec: Dict[str, Any] = {
#         "replace_step": True,
#         "replacement_policy": "prefer_same_scenario_prefer_same_taxonomy_block",
#         "notes": "replace_step_prefer_same_block_but_not_required",
#     }

#     replacement = choose_substitution_replacement(
#         rng_py=rng_py,
#         scenario=scenario,
#         taxonomy_block_name=taxonomy_block_name,
#         current_step_description_id=current_step_description_id,
#         forbidden_step_description_ids=forbidden_step_description_ids,
#         forbidden_step_texts_norm=forbidden_step_texts_norm,
#         forbidden_semreps=forbidden_semreps,
#         p_prefer_same_block=float(p_prefer_same_block),
#     )
#     if replacement is not None:
#         spec["replacement_step_description_id"] = replacement.get("step_description_id")
#         spec["replacement_step_description"] = replacement.get("step_description", "")
#         spec["replacement_taxonomy_block_name"] = replacement.get("taxonomy_block_name")

#     return spec


# -----------------------------
# Error event specification
# -----------------------------

def severity_from_roles(roles: List[str], role_to_impact: Dict[str, str]) -> str:
    impacts = [role_to_impact.get(r, "medium").lower() for r in roles]
    if any(i == "high" for i in impacts):
        return "high"
    if any(i == "medium" for i in impacts):
        return "medium"
    return "low"


def choose_error_type_for_step(
    rng_np: np.random.Generator,
    rng_py: random.Random,
    phase: str,
    is_essential: bool,
    n_steps: int,
    disallow_types: Optional[Set[str]] = None,
) -> str:

    if phase not in PHASE_ERROR_TYPE_PRIORS:
        raise KeyError(f"Unknown phase='{phase}'. Expected one of: {list(PHASE_ERROR_TYPE_PRIORS.keys())}")

    prior = np.array(PHASE_ERROR_TYPE_PRIORS[phase]["PRIOR_WEIGHTS"], dtype=float)
    props = prior / float(prior.sum())    
    modifiers = np.ones_like(props)

    if disallow_types:
        for t in disallow_types:
            if t in ERROR_TYPES_ORDER:
                modifiers[ERROR_TYPES_ORDER.index(t)] = 0.0

    if n_steps <= 4:
        modifiers[ERROR_TYPES_ORDER.index("deletion")] = 0.0
        modifiers[ERROR_TYPES_ORDER.index("transposition")] = 0.0
    else:
        if not is_essential:
            modifiers[ERROR_TYPES_ORDER.index("deletion")] *= 1.25

    if not is_essential:
        modifiers[ERROR_TYPES_ORDER.index("insertion")] *= 1.35

    weights = props * modifiers
    total = float(weights.sum())
    if total <= 0:
        allowed = [t for t in ERROR_TYPES_ORDER if not disallow_types or t not in disallow_types]
        return rng_py.choice(allowed) if allowed else "wrong_execution"

    weights = weights / total
    return rng_py.choices(ERROR_TYPES_ORDER, weights=weights.tolist(), k=1)[0]


def choose_error_type_with_retries(
    rng_np: np.random.Generator,
    rng_py: random.Random,
    phase: str,
    is_essential: bool,
    n_steps: int,
    feasible_types: Set[str],
    max_tries: int = 10,
) -> str:
    feasible_types = set(t for t in feasible_types if t in ERROR_TYPES_ORDER)
    if not feasible_types:
        return "wrong_execution"

    disallow = set(ERROR_TYPES_ORDER) - feasible_types


    for _ in range(int(max_tries)):
        t = choose_error_type_for_step(
            rng_np=rng_np,
            rng_py=rng_py,
            phase=phase,
            is_essential=is_essential,
            n_steps=n_steps,
            disallow_types=disallow,
        )
        if t in feasible_types:
            return t

    for t in ERROR_TYPES_ORDER:
        if t in feasible_types:
            return t
    return "wrong_execution"


# -----------------------------
# Main simulation per take
# -----------------------------

def simulate_take(
    rng_np: np.random.Generator,
    rng_py: random.Random,
    annotation: Dict[str, Any],
    vocab_map: Dict[str, str],
    semrep_map: Dict[str, Any],
    role_to_impact: Dict[str, str],
    roles_prior_by_predicate: Dict[str, Dict[str, float]],
    taxonomy_root: Dict[str, Any],
    wc: float,
    wt: float,
) -> Dict[str, Any]:
    scenario = str(annotation.get("scenario", "unknown"))
    take_uid = str(annotation.get("take_uid", "unknown"))
    take_name = str(annotation.get("take_name", "unknown"))

    segments = annotation.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    # Always enforce chronological order by start_time (stable tie-breaker by original position).
    indexed: List[Tuple[int, Dict[str, Any]]] = []
    for k, seg in enumerate(segments):
        if isinstance(seg, dict):
            indexed.append((k, seg))
    indexed.sort(key=lambda x: (float(x[1].get("start_time", 0.0) or 0.0), x[0]))
    segments = [seg for _k, seg in indexed]


    scenario_tax = taxonomy_root.get(scenario, {})
    nodes_by_id, leaf_unique_to_node_id = build_taxonomy_index(scenario_tax if isinstance(scenario_tax, dict) else {})

    steps_out: List[Dict[str, Any]] = []
    complexities: List[float] = []
    durations: List[float] = []

    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue

        step_desc = str(seg.get("step_description", ""))
        step_desc_id = vocab_map.get(step_desc)  # string id
        sem_entry = semrep_map.get(str(step_desc_id), {}) if step_desc_id is not None else {}
        rep = sem_entry.get("semantic_representation")

        start_t = float(seg.get("start_time", 0.0) or 0.0)
        end_t = float(seg.get("end_time", 0.0) or 0.0)
        dur = max(0.0, end_t - start_t)

        comp = semantic_complexity(rep) if isinstance(rep, str) else 0.0

        predicate, role_to_value = parse_semantic_representation(rep) if isinstance(rep, str) else (None, {})
        predicate = predicate.upper() if isinstance(predicate, str) else predicate
        obj_value = extract_main_object(role_to_value)

        step_unique_id = seg.get("step_unique_id", None)
        block_id, block_name, path_names = None, None, []
        leaf_node_id = None
        if step_unique_id is not None:
            try:
                uid_int = int(step_unique_id)
                leaf_node_id = leaf_unique_to_node_id.get(uid_int)
            except Exception:
                leaf_node_id = None

        if leaf_node_id is not None:
            block_id, block_name, path_names = resolve_top_level_block(leaf_node_id, nodes_by_id)

        step_record = {
            "index": idx,
            "start_time": start_t,
            "end_time": end_t,
            "duration_s": dur,
            "step_description": step_desc,
            "step_description_id": step_desc_id,
            "step_unique_id": step_unique_id,
            "is_essential": bool(seg.get("is_essential", False)),
            "semantic_representation": rep,
            "predicate": predicate,
            "roles_present": sorted(list(role_to_value.keys())),
            "role_to_value": role_to_value,
            "object_value": obj_value,
            "taxonomy_block_id": block_id,
            "taxonomy_block_name": block_name,
            "taxonomy_path": path_names,
        }

        steps_out.append(step_record)
        complexities.append(comp)
        durations.append(dur)

    norm_comp = minmax_normalize(complexities)
    norm_dur = minmax_normalize(durations)

    step_loads: List[float] = []
    for i in range(len(steps_out)):
        load = wc * norm_comp[i] + wt * norm_dur[i]
        step_loads.append(float(load))
        steps_out[i]["complexity"] = complexities[i]
        steps_out[i]["norm_complexity"] = float(norm_comp[i])
        steps_out[i]["norm_duration"] = float(norm_dur[i])
        steps_out[i]["step_load"] = float(load)

    phases = assign_phases_by_load(step_loads)
    for i, ph in enumerate(phases):
        steps_out[i]["phase"] = ph

    feat = compute_proc_features(durations, complexities)

    return {
        "take_uid": take_uid,
        "take_name": take_name,
        "scenario": scenario,
        "raw_steps": steps_out,
        "proc_features": {
            "n_steps": feat.n_steps,
            "total_duration_s": feat.total_duration_s,
            "total_complexity": feat.total_complexity,
            "density_steps_per_min": feat.density_steps_per_min,
        },
        "step_loads": step_loads,
    }


def enforce_max_consecutive(selected_indices: List[int], max_consecutive: int) -> bool:
    if not selected_indices:
        return True
    s = sorted(selected_indices)
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            run += 1
            if run > int(max_consecutive):
                return False
        else:
            run = 1
    return True

def max_consecutive_run_from_set(idxs: Set[int]) -> int:
    """Max length of consecutive run in a set of ints."""
    if not idxs:
        return 0
    s = sorted(i for i in idxs if isinstance(i, int))
    if not s:
        return 0
    best = 1
    cur = 1
    for a, b in zip(s, s[1:]):
        if b == a + 1:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def would_violate_consecutive_cap(existing: Set[int], add: Set[int], cap: int) -> bool:
    """True if adding 'add' to 'existing' would exceed consecutive run cap."""
    if cap <= 0:
        return False
    merged = set(existing)
    merged.update(i for i in add if isinstance(i, int))
    return max_consecutive_run_from_set(merged) > int(cap)


def pick_error_step_indices(
    rng_py: random.Random,
    steps: List[Dict[str, Any]],
    k: int,
    max_consecutive: int,
    phase_multipliers: Optional[Dict[str, float]] = None,
) -> List[int]:
    n = len(steps)
    if n == 0:
        return []

    phase_multipliers = phase_multipliers or {"phase_1": 1.0, "phase_2": 1.0, "phase_3": 1.0}
    weights = []
    for s in steps:
        load = float(s.get("step_load", 0.0) or 0.0)
        load = max(0.0, load)
        # Base: higher load => more likely error.
        w = 0.15 + 0.85 * load
        # phase intensity multiplier (controls error rates across phases).
        ph = str(s.get("phase", "phase_2"))
        w *= float(phase_multipliers.get(ph, 1.0))
        weights.append(w)

    selected: List[int] = []
    attempts = 0
    max_attempts = 5000

    while len(selected) < min(int(k), n) and attempts < max_attempts:
        attempts += 1
        idx = rng_py.choices(list(range(n)), weights=weights, k=1)[0]
        if idx in selected:
            continue
        trial = selected + [idx]
        if not enforce_max_consecutive(trial, max_consecutive):
            continue
        selected.append(idx)

    return sorted(selected)


def pick_nonviolating_type(
    rng_np: np.random.Generator,
    rng_py: random.Random,
    phase: str,
    is_essential: bool,
    n_steps: int,
    feasible_types: Set[str],
    affected_set: Set[int],
    anchor_idx: int,
    cap: int,
) -> Optional[str]:
    feasible = set(t for t in feasible_types if t in ERROR_TYPES_ORDER)
    if not feasible:
        return None

    ok_types: Set[str] = set()
    for t in feasible:
        # Any event affects the anchor index at minimum; if adding anchor alone
        # violates the consecutive cap, we must reject ALL types including transposition.
        if would_violate_consecutive_cap(affected_set, {int(anchor_idx)}, cap):
            continue
         # Anchor feasibility already ensured here; for transposition the partner is checked later.
        ok_types.add(t)

    if not ok_types:
        return None

    # choose_error_type_with_retries must already exist in your file
    return choose_error_type_with_retries(
        rng_np=rng_np,
        rng_py=rng_py,
        phase=phase,
        is_essential=is_essential,
        n_steps=n_steps,
        feasible_types=ok_types,
    )


def affected_indices_for_event(etype: str, anchor: int, partner: Optional[int]) -> Set[int]:
    out = {int(anchor)}
    if etype == "transposition" and partner is not None:
        out.add(int(partner))
    return out


def apply_event_spec(
    *,
    etype: str,
    event: Dict[str, Any],
    rng_py: random.Random,
    predicate: Optional[str],
    role_to_value: Dict[str, str],
    object_value: Optional[str],
    role_to_impact: Dict[str, str],
    roles_prior_by_predicate: Dict[str, Dict[str, float]],
    scenario: str,
    taxonomy_block_name: Optional[str],
    current_step_description_id: Optional[str],
    step_idx: int,
    partner_index: Optional[int],
    is_essential: bool,
) -> None:
    if etype == "wrong_execution":
        roles = choose_roles_to_mutate(
            rng=rng_py,
            predicate=predicate,
            role_to_value=role_to_value,
            role_to_impact=role_to_impact,
            roles_prior_by_predicate=roles_prior_by_predicate,
        )
        scopes = {r: choose_wrong_execution_scope(rng_py, r) for r in roles}
        sev = severity_from_roles(roles, role_to_impact)
        event["severity"] = sev
        event["spec"] = {
            "keep_predicate": True,
            "predicate": predicate,
            "mutated_roles": roles,
            "mutation_scopes": scopes,
            "notes": "keep_predicate; mutate_roles; agent_fixed",
        }
        return

    if etype == "deletion":
        event["severity"] = "low" if not is_essential else "medium"
        event["spec"] = {"delete_step": True, "notes": "skip_step"}
        return

    if etype == "insertion":
        event["severity"] = "low"
        event["spec"] = {
            "insert_before_step_index": int(step_idx),
            "candidate_policy": "same_scenario_prefer_nonessential",
            "notes": "insert_nonessential_before_step",
        }
        return

    if etype == "substitution":
        event["severity"] = "medium"
        has_predicate = bool(predicate)
        has_object = bool(object_value)

        # Substitution = "do a different step than intended".
        # Keep this high-level: downstream LLM will instantiate the concrete substitute.
        strategies = []
        if has_predicate and has_object:
            strategies += [
                # e.g., GET tortilla -> DROP tortilla (same object, different action)
                "change_predicate_keep_object",
                # e.g., GET tortilla -> GET spoon (same action, different object)
                "change_object_keep_predicate",
                # e.g., GET tortilla -> DROP spoon (different action and different object)
                "change_predicate_and_object",
            ]
        elif has_object and not has_predicate:
            strategies += [
                "change_object_keep_frame",
                "replace_step_semantically_related",
            ]
        else:
            strategies += ["replace_step_semantically_related"]

        event["spec"] = {
           "original_predicate": predicate,
           "original_object": object_value,          
           "substitution_strategies": strategies,
           "preference": {
               # optional: you already enforce same-block for transposition;
               # for substitution you can *prefer* same taxonomy block, but not hard constrain
               "prefer_same_taxonomy_block": True
           },
           "notes": "policy-only substitution; no role-argument swaps here (those belong to wrong_execution)"
        }
        return

    if etype == "transposition":
        event["severity"] = "medium"
        event["spec"] = {
            "transposition_source": int(step_idx),
            "transposition_target": int(partner_index) if partner_index is not None else None,
            "notes": "swap_steps_within_same_taxonomy_block",
        }
        return


def build_error_events_for_take(
    rng_np: np.random.Generator,
    rng_py: random.Random,
    take_payload: Dict[str, Any],
    norm_proc_complexity: float,
    norm_proc_duration: float,
    norm_proc_density: float,
    role_to_impact: Dict[str, str],
    roles_prior_by_predicate: Dict[str, Dict[str, float]],
    unlock_prior_by_predicate: Dict[str, float],
    unlock_support_by_predicate: Dict[str, int],
    deletion_guard_threshold: float,
    deletion_guard_max_lookahead: int,
    deletion_guard_min_support: int,
    max_errors: int,
    max_consecutive_errors: int,
    transposition_window: int,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = take_payload["raw_steps"]
    proc_feat: Dict[str, Any] = take_payload["proc_features"]
    n_steps = int(proc_feat.get("n_steps", 0))

    risk = proc_risk_score(
        feat=ProcRiskFeatures(
            n_steps=n_steps,
            total_duration_s=float(proc_feat.get("total_duration_s", 0.0)),
            total_complexity=float(proc_feat.get("total_complexity", 0.0)),
            density_steps_per_min=float(proc_feat.get("density_steps_per_min", 0.0)),
        ),
        norm_total_complexity=float(norm_proc_complexity),
        norm_total_duration=float(norm_proc_duration),
        norm_density=float(norm_proc_density),
    )

    k, lam = sample_k_errors(rng=rng_np, risk=risk, n_steps=n_steps, max_errors=int(max_errors))

    ph_mult = phase_rate_multipliers(PHASE_ERROR_RATE_MODEL)
    selected_indices = pick_error_step_indices(
        rng_py=rng_py,
        steps=steps,
        k=k,
        max_consecutive=int(max_consecutive_errors),
        phase_multipliers=ph_mult,
    )
    if not selected_indices and steps:
        selected_indices = [int(rng_py.randrange(len(steps)))]

    selected_set = set(selected_indices)

    affected_set: Set[int] = set()
    used_anchor_indices: Set[int] = set()
    transposition_involved_indices: Set[int] = set()
    transposition_pairs: Set[Tuple[int, int]] = set()

    events: List[Dict[str, Any]] = []
    event_counter = 0

    for step_idx in selected_indices:
        step_idx = int(step_idx)
        s = steps[step_idx]
        phase = str(s.get("phase", "phase_1"))
        is_essential = bool(s.get("is_essential", False))

        feasible = set(ERROR_TYPES_ORDER)
        if n_steps <= 4:
            feasible.discard("deletion")
            feasible.discard("transposition")
        if step_idx in transposition_involved_indices:
            feasible.discard("transposition")

        transposition_candidates: List[int] = []
        if "transposition" in feasible:
            transposition_candidates = get_transposition_candidates(
                i=step_idx,
                steps=steps,
                window=int(transposition_window),
            )
            transposition_candidates = [
                j for j in transposition_candidates
                if j not in selected_set
                and j not in used_anchor_indices
                and j not in transposition_involved_indices
            ]
            if not transposition_candidates:
                feasible.discard("transposition")

        etype = pick_nonviolating_type(
            rng_np=rng_np,
            rng_py=rng_py,
            phase=phase,
            is_essential=is_essential,
            n_steps=n_steps,
            feasible_types=feasible,
            affected_set=affected_set,
            anchor_idx=step_idx,
            cap=int(max_consecutive_errors),
        )
        if etype is None:
            continue

        # Deletion policy:
        # - If deletion removes an UNLOCK step (GET/TAKE/...), and the object is used later,
        #   DO NOT forbid deletion. Instead, REQUIRE a correction before first future use.
        # - Keep learned deletion-guard for non-UNLOCK deletions.
        requires_correction = False
        correction_before_index: Optional[int] = None
        if etype == "deletion":
            pred_now = (s.get("predicate") or "").upper()
            if pred_now in UNLOCK_PREDS:
                first_use = find_first_future_use_index(
                    steps=steps,
                    idx=int(step_idx),
                    max_lookahead=int(deletion_guard_max_lookahead),
                )
                if first_use is not None:
                    requires_correction = True
                    correction_before_index = int(first_use)
            else:
                if is_guarded_deletion_step(
                    steps=steps,
                    idx=step_idx,
                    unlock_prior_by_predicate=unlock_prior_by_predicate,
                    unlock_support_by_predicate=unlock_support_by_predicate,
                    threshold=float(deletion_guard_threshold),
                    max_lookahead=int(deletion_guard_max_lookahead),
                    min_support=int(deletion_guard_min_support),
                ):
                    feasible_no_del = feasible - {"deletion"}
                    etype = choose_error_type_with_retries(
                        rng_np=rng_np,
                        rng_py=rng_py,
                        phase=phase,
                        is_essential=is_essential,
                        n_steps=n_steps,
                        feasible_types=feasible_no_del,
                    )

        partner_index: Optional[int] = None
        if etype == "transposition":
            for _ in range(10):
                cand = choose_transposition_partner_from_candidates(rng_py, transposition_candidates)
                if cand is None:
                    break

                add_set = affected_indices_for_event("transposition", step_idx, int(cand))
                if would_violate_consecutive_cap(affected_set, add_set, int(max_consecutive_errors)):
                    continue

                a, b = int(step_idx), int(cand)
                pair = (min(a, b), max(a, b))
                if a in transposition_involved_indices or b in transposition_involved_indices or pair in transposition_pairs:
                    continue

                partner_index = int(cand)
                transposition_involved_indices.add(a)
                transposition_involved_indices.add(b)
                transposition_pairs.add(pair)
                break

            if partner_index is None:
                feasible2 = set(feasible)
                feasible2.discard("transposition")
                etype2 = pick_nonviolating_type(
                    rng_np=rng_np,
                    rng_py=rng_py,
                    phase=phase,
                    is_essential=is_essential,
                    n_steps=n_steps,
                    feasible_types=feasible2,
                    affected_set=affected_set,
                    anchor_idx=step_idx,
                    cap=int(max_consecutive_errors),
                )
                if etype2 is None:
                    continue
                etype = etype2

        # Final cap guard with the actual affected set
        add_set_final = affected_indices_for_event(str(etype), step_idx, partner_index)
        if would_violate_consecutive_cap(affected_set, add_set_final, int(max_consecutive_errors)):
            feasible3 = set(feasible)
            feasible3.discard(str(etype))
            feasible3.discard("transposition")
            etype3 = pick_nonviolating_type(
                rng_np=rng_np,
                rng_py=rng_py,
                phase=phase,
                is_essential=is_essential,
                n_steps=n_steps,
                feasible_types=feasible3,
                affected_set=affected_set,
                anchor_idx=step_idx,
                cap=int(max_consecutive_errors),
            )
            if etype3 is None:
                continue
            etype = etype3
            partner_index = None
            add_set_final = affected_indices_for_event(str(etype), step_idx, partner_index)
            if would_violate_consecutive_cap(affected_set, add_set_final, int(max_consecutive_errors)):
                continue

        # Commit event
        event_counter += 1
        used_anchor_indices.add(int(step_idx))
        affected_set.update(add_set_final)

        predicate = s.get("predicate")
        role_to_value = s.get("role_to_value", {}) or {}
        object_value = s.get("object_value")
        taxonomy_block_name = s.get("taxonomy_block_name")
        scenario = str(take_payload.get("scenario", "unknown"))
        current_sid = s.get("step_description_id")

        event: Dict[str, Any] = {
            "event_id": f"E{event_counter:02d}",
            "step_index": int(step_idx),
            "type": str(etype),
            "phase": str(phase),
            "taxonomy_block_name": taxonomy_block_name,
            "is_essential": bool(is_essential),
        }

        if etype == "deletion" and requires_correction:
            event["requires_correction"] = True
            event["correction_before_step_index"] = correction_before_index

        apply_event_spec(
            etype=str(etype),
            event=event,
            rng_py=rng_py,
            predicate=predicate,
            role_to_value=role_to_value,
            object_value=object_value,
            role_to_impact=role_to_impact,
            roles_prior_by_predicate=roles_prior_by_predicate,
            scenario=scenario,
            taxonomy_block_name=taxonomy_block_name,
            current_step_description_id=current_sid,
            step_idx=int(step_idx),
            partner_index=partner_index,
            is_essential=bool(is_essential),
        )

        # Populate alternate deletion candidates (same phase, nearby, preferably non-essential).
        if etype == 'deletion':
            src_phase = steps[step_idx].get('phase')
            cand = []
            for j in (step_idx-2, step_idx-1, step_idx+1, step_idx+2):
                if 0 <= j < n_steps and steps[j].get('phase') == src_phase and j != step_idx:
                     # Avoid proposing deletion of steps that are heavily referenced later.
                     if not is_step_essential(steps, j, min_future_refs=2):
                         cand.append(int(j))
            event['alternate_src_indices'] = cand[:4]
        events.append(event)

    # Debug log
    log_lines: List[str] = []
    total_steps = len(steps)
    error_by_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for ev in events:
        a = ev.get("step_index")
        if isinstance(a, int):
            error_by_index[a].append(ev)
        if ev.get("type") == "transposition":
            spec = ev.get("spec", {}) or {}
            b = spec.get("transposition_target")
            if isinstance(b, int):
                error_by_index[b].append(ev)

    for i, st in enumerate(steps):
        ph = st.get("phase", "phase_1")
        sd = st.get("step_description", "")
        prefix = f"[Step {i+1:02d} / {total_steps}] ({ph})"
        if i in error_by_index:
            labels: List[str] = []
            seen: Set[str] = set()
            for ev in error_by_index[i]:
                t = str(ev.get("type", "unknown"))
                if t != "transposition":
                    if t not in seen:
                        labels.append(t)
                        seen.add(t)
                else:
                    spec = ev.get("spec", {}) or {}
                    src = spec.get("transposition_source")
                    tgt = spec.get("transposition_target")
                    lab = f"transposition({(src+1) if isinstance(src,int) else '?'}->{(tgt+1) if isinstance(tgt,int) else '?'})"
                    if lab not in seen:
                        labels.append(lab)
                        seen.add(lab)
            log_lines.append(f"{prefix} ERROR: '{sd}' -> {', '.join(labels)}")
        else:
            log_lines.append(f"{prefix} OK: '{sd}'")

    return {
        "procedure_stats": {
            "n_steps": int(n_steps),
            "total_duration_s": float(proc_feat.get("total_duration_s", 0.0)),
            "total_complexity": float(proc_feat.get("total_complexity", 0.0)),
            "density_steps_per_min": float(proc_feat.get("density_steps_per_min", 0.0)),
            "risk_score": float(risk),
            "expected_errors_lambda": float(lam),
            "k_errors": int(len(events)),
            "hard_caps": {
                "max_errors": int(max_errors),
                "max_consecutive_errors": int(max_consecutive_errors),
            },
        },
        "steps": steps,
        "errors": events,
        "simulation_log_lines": log_lines,
        "simulation_log": "\n".join(log_lines),
    }


# -----------------------------
# Entry point
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PIE-V error simulator (policy-only).")
    parser.add_argument("--split50", required=True, help="Path to split_50.json")
    parser.add_argument("--vocab_csv", required=True, help="Path to split_50_vocabulary.csv")
    parser.add_argument("--semrep_json", required=True, help="Path to semantic_representations_split_50.json")
    parser.add_argument("--roles_csv", required=True, help="Path to semantic_roles.csv")
    parser.add_argument("--roles_by_predicate_csv", required=True, help="Path to semantic_roles_by_predicate.csv")
    parser.add_argument("--keystep", required=True, help="Path to keystep_train.json")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--wc", type=float, default=0.6, help="Weight for normalized complexity in step_load")
    parser.add_argument("--wt", type=float, default=0.4, help="Weight for normalized duration in step_load")
    parser.add_argument("--max_errors", type=int, default=5, help="Hard cap on number of errors per take")
    parser.add_argument("--max_consecutive_errors", type=int, default=3, help="Max consecutive error steps")
    parser.add_argument("--transposition_window", type=int, default=6, help="Local window size for transposition partner search")
    parser.add_argument("--deletion_guard_threshold", type=float, default=0.8, help="Guard: disallow deletion if learned unlock prior >= threshold and object used soon after")
    parser.add_argument("--deletion_guard_max_lookahead", type=int, default=10, help="Lookahead for deletion guard")
    parser.add_argument("--deletion_guard_min_support", type=int, default=50, help="Min predicate support for applying deletion guard")
    args = parser.parse_args()

    if args.wc < 0.0 or args.wt < 0.0:
        raise ValueError(f"wc and wt must be non-negative (got wc={args.wc}, wt={args.wt})")
    if math.isclose(args.wc + args.wt, 0.0):
        raise ValueError(f"wc + wt must be > 0 (got wc={args.wc}, wt={args.wt})")

    split50 = load_json(Path(args.split50))
    vocab_map = load_vocab_csv(Path(args.vocab_csv))
    semrep_map = load_json(Path(args.semrep_json))
    role_to_impact = load_role_impact_csv(Path(args.roles_csv))
    roles_prior_by_predicate = load_roles_by_predicate_csv(Path(args.roles_by_predicate_csv))
    keystep = load_json(Path(args.keystep))

    taxonomy_root = keystep.get("taxonomy", {})
    if not isinstance(taxonomy_root, dict):
        raise ValueError("keystep_train.json: expected 'taxonomy' to be a dict")

    annotations = split50.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("split_50.json: expected 'annotations' to be a list")

    rng_np = np.random.default_rng(args.seed)
    rng_py = random.Random(args.seed)

    # Pass 1: build step-level features per take and collect procedure features for normalization.
    takes_intermediate: List[Dict[str, Any]] = []
    proc_complexities: List[float] = []
    proc_durations: List[float] = []
    proc_densities: List[float] = []

    for ann in annotations:
        take_payload = simulate_take(
            rng_np=rng_np,
            rng_py=rng_py,
            annotation=ann,
            vocab_map=vocab_map,
            semrep_map=semrep_map,
            role_to_impact=role_to_impact,
            roles_prior_by_predicate=roles_prior_by_predicate,
            taxonomy_root=taxonomy_root,
            wc=float(args.wc),
            wt=float(args.wt),
        )
        takes_intermediate.append(take_payload)

        pf = take_payload["proc_features"]
        proc_complexities.append(float(pf["total_complexity"]))
        proc_durations.append(float(pf["total_duration_s"]))
        proc_densities.append(float(pf["density_steps_per_min"]))

    # Learn predicate-level unlock prior for deletion guard.
    unlock_prior_by_predicate, unlock_support_by_predicate = compute_unlock_prior_by_predicate(
        takes_intermediate=takes_intermediate,
        max_lookahead=int(args.deletion_guard_max_lookahead),
    )

    # Global normalization across takes.
    norm_proc_complexities = minmax_normalize(proc_complexities)
    norm_proc_durations = minmax_normalize(proc_durations)
    norm_proc_densities = minmax_normalize(proc_densities)

    # Pass 2: sample errors per take.
    output_takes: Dict[str, Any] = {}
    for idx, take_payload in enumerate(takes_intermediate):
        take_uid = str(take_payload["take_uid"])
        take_name = str(take_payload["take_name"])
        scenario = str(take_payload["scenario"])

        take_out = build_error_events_for_take(
            rng_np=rng_np,
            rng_py=rng_py,
            take_payload=take_payload,
            norm_proc_complexity=float(norm_proc_complexities[idx]),
            norm_proc_duration=float(norm_proc_durations[idx]),
            norm_proc_density=float(norm_proc_densities[idx]),
            role_to_impact=role_to_impact,
            roles_prior_by_predicate=roles_prior_by_predicate,
            unlock_prior_by_predicate=unlock_prior_by_predicate,
            unlock_support_by_predicate=unlock_support_by_predicate,
            deletion_guard_threshold=float(args.deletion_guard_threshold),
            deletion_guard_max_lookahead=int(args.deletion_guard_max_lookahead),
            deletion_guard_min_support=int(args.deletion_guard_min_support),
            max_errors=int(args.max_errors),
            max_consecutive_errors=int(args.max_consecutive_errors),
            transposition_window=int(args.transposition_window),
        )

        output_takes[take_uid] = {
            "take_name": take_name,
            "scenario": scenario,
            **take_out,
        }

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "inputs": {
            "split50": str(Path(args.split50)),
            "vocab_csv": str(Path(args.vocab_csv)),
            "semrep_json": str(Path(args.semrep_json)),
            "roles_csv": str(Path(args.roles_csv)),
            "roles_by_predicate_csv": str(Path(args.roles_by_predicate_csv)),
            "keystep": str(Path(args.keystep)),
        },
        "config": {
            "phase_error_rate_model": {
                "rates": PHASE_ERROR_RATE_MODEL,
                "notes": "Used as normalized multipliers when sampling error step indices (WHERE errors land).",
            },
            "step_load": {
                "wc": float(args.wc),
                "wt": float(args.wt),
                "description": "step_load = wc * norm_complexity + wt * norm_duration",
            },
            "caps": {
                "max_errors": int(args.max_errors),
                "max_consecutive_errors": int(args.max_consecutive_errors),
            },
            "transposition": {
                "window": int(args.transposition_window),
                "no_cross_taxonomy_block": True,
            },
            "deletion_guard": {
                "enabled": True,
                "threshold": float(args.deletion_guard_threshold),
                "max_lookahead": int(args.deletion_guard_max_lookahead),
                "min_support": int(args.deletion_guard_min_support),
                "note": "Data-driven guard using learned P(object reused soon | predicate)",
            },
            "hard_constraints": {
                "no_deletion_if_n_steps_leq_4": True,
                "agent_never_mutated": True,
            },
        },
    }

    out_obj = {"meta": meta, "takes": output_takes}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


# piev-error-plan \
#   --split50 local/egoexo4d/split_50.json \
#   --vocab_csv data/resources/split_50_vocabulary.csv \
#   --semrep_json data/resources/semantic_representations_split_50.json \
#   --roles_csv data/resources/semantic_roles.csv \
#   --roles_by_predicate_csv data/resources/semantic_roles_by_predicate.csv \
#   --keystep local/egoexo4d/keystep_train.json \
#   --out local/outputs/split_50_error_plan.json \
#   --seed 123
