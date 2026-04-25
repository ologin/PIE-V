#!/usr/bin/env python3
"""
diagnose_llm_freeform_outputs.py

Goal:
- Validate and diagnose outputs of llm_error_plan_freeform_writer.py
- Produce a per-take CSV with schema/logical flags (lightweight, no world model)

Inputs:
- --results: split_50_llm_freeform_errors_{openai|qwen}.json
- --split50: original split_50.json (for verbatim checks and n_input_steps)
Output:
- --out_csv: diagnostics CSV

Notes:
- Comments are English-only (project rule).
"""

import argparse
import csv
import json
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to model output JSON")
    p.add_argument("--split50", required=True, help="Path to split_50.json")
    p.add_argument("--out_csv", required=True, help="Output CSV path")
    return p.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_take_uid_to_steps(split50: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    anns = split50.get("annotations", [])
    if not isinstance(anns, list):
        return out
    for a in anns:
        if not isinstance(a, dict):
            continue
        tuid = str(a.get("take_uid", "")).strip()
        segs = a.get("segments", [])
        if not tuid or not isinstance(segs, list):
            continue
        steps = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            steps.append(str(seg.get("step_description", "")))
        out[tuid] = steps
    return out


def safe_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        return int(x)
    except Exception:
        return None


def validate_compact_schema(obj: Dict[str, Any], n_input_steps: int) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Output must be a JSON object"
    final_steps = obj.get("final_steps")
    meta = obj.get("meta")
    deletions = obj.get("del", [])

    if not isinstance(final_steps, list) or not all(isinstance(x, str) for x in final_steps):
        return False, "Missing/invalid final_steps"
    if not isinstance(meta, list) or not all(isinstance(x, list) for x in meta):
        return False, "Missing/invalid meta"
    if len(final_steps) != len(meta):
        return False, "final_steps/meta length mismatch"

    allowed_mod = {"u", "e", "m", "c", "i"}
    m_eid_counts = defaultdict(int)

    for i, m in enumerate(meta):
        if len(m) < 4:
            return False, f"meta[{i}] must have 4 elements"
        src_idx, mod, eid, cid = m[0], m[1], m[2], m[3]

        if not isinstance(src_idx, int):
            return False, f"meta[{i}].src_idx must be int"
        if not (0 <= src_idx < n_input_steps):
            return False, f"meta[{i}].src_idx out of range: {src_idx}"
        if mod not in allowed_mod:
            return False, f"meta[{i}].mod invalid: {mod}"
        if eid is not None and not isinstance(eid, str):
            return False, f"meta[{i}].eid must be str|null"
        if cid is not None and not isinstance(cid, str):
            return False, f"meta[{i}].cid must be str|null"

        if mod == "m":
            if not isinstance(eid, str) or not eid.strip():
                return False, f"meta[{i}] mod='m' requires non-empty eid"
            m_eid_counts[eid.strip()] += 1

    for eid, cnt in m_eid_counts.items():
        if cnt != 2:
            return False, f"transposition eid {eid} has {cnt} moved steps (must be 2)"

    if deletions is None:
        return True, "ok"
    if not isinstance(deletions, list) or not all(isinstance(x, list) for x in deletions):
        return False, "Invalid del"
    for j, d in enumerate(deletions):
        if len(d) < 2:
            return False, f"del[{j}] must have >=2 elements"
        if not isinstance(d[0], int) or not (0 <= d[0] < n_input_steps):
            return False, f"del[{j}].src_idx out of range"
        if not isinstance(d[1], str) or not d[1].strip():
            return False, f"del[{j}].eid must be non-empty string"

    return True, "ok"


def diagnose_take(
    take_payload: Dict[str, Any],
    input_steps: List[str],
) -> Dict[str, Any]:
    n_input = len(input_steps)
    status = str(take_payload.get("status", "missing"))
    error = str(take_payload.get("error", "")) if status != "ok" else ""

    rewrite = take_payload.get("rewrite", {}) if isinstance(take_payload.get("rewrite", {}), dict) else {}
    final_steps = rewrite.get("final_steps", [])
    meta = rewrite.get("meta", [])
    deletions = rewrite.get("del", [])

    # Basic fields
    out: Dict[str, Any] = {
        "take_uid": "",
        "scenario": str(take_payload.get("scenario", "")),
        "take_name": str(take_payload.get("take_name", "")),
        "status": status,
        "error": error,
        "n_input_steps": n_input,
        "n_final_steps": len(final_steps) if isinstance(final_steps, list) else -1,
        "n_meta": len(meta) if isinstance(meta, list) else -1,
        "n_del": len(deletions) if isinstance(deletions, list) else 0,
    }

    # If not ok, still compute shallow parse availability
    if not isinstance(rewrite, dict) or not isinstance(final_steps, list) or not isinstance(meta, list):
        out.update({
            "mods_u": 0, "mods_e": 0, "mods_m": 0, "mods_c": 0, "mods_i": 0,
            "u_not_verbatim": -1,
            "e_no_change": -1,
            "deleted_src_still_used": -1,
        })
        return out

    # Re-run validator for an objective schema flag
    ok_schema, schema_msg = validate_compact_schema(rewrite, n_input_steps=n_input)
    out["schema_ok"] = int(ok_schema)
    out["schema_msg"] = schema_msg

    mods = defaultdict(int)
    u_not_verbatim = 0
    e_no_change = 0
    deleted_src_still_used = 0

    used_src_idxs = set()
    for i, m in enumerate(meta):
        if not (isinstance(m, list) and len(m) >= 4):
            continue
        src_idx, mod = m[0], m[1]
        if isinstance(mod, str):
            mods[mod] += 1
        if isinstance(src_idx, int):
            used_src_idxs.add(src_idx)

        # u must be verbatim
        if mod == "u" and isinstance(src_idx, int) and 0 <= src_idx < n_input and i < len(final_steps):
            if final_steps[i] != input_steps[src_idx]:
                u_not_verbatim += 1

        # e should usually differ from original (diagnostic only)
        if mod == "e" and isinstance(src_idx, int) and 0 <= src_idx < n_input and i < len(final_steps):
            if final_steps[i] == input_steps[src_idx]:
                e_no_change += 1

    # If a step is deleted, ideally it should not appear as a src_idx in meta
    del_srcs = set()
    if isinstance(deletions, list):
        for d in deletions:
            if isinstance(d, list) and len(d) >= 2 and isinstance(d[0], int):
                del_srcs.add(d[0])
    for s in del_srcs:
        if s in used_src_idxs:
            deleted_src_still_used += 1

    out.update({
        "mods_u": mods["u"],
        "mods_e": mods["e"],
        "mods_m": mods["m"],
        "mods_c": mods["c"],
        "mods_i": mods["i"],
        "u_not_verbatim": u_not_verbatim,
        "e_no_change": e_no_change,
        "deleted_src_still_used": deleted_src_still_used,
    })
    return out


def main() -> None:
    args = parse_args()
    split50 = load_json(args.split50)
    uid2steps = build_take_uid_to_steps(split50)

    results = load_json(args.results)
    takes = results.get("takes", {})
    if not isinstance(takes, dict):
        raise ValueError("results JSON must contain dict at key 'takes'")

    rows: List[Dict[str, Any]] = []
    for take_uid, payload in takes.items():
        if not isinstance(payload, dict):
            continue
        inp = uid2steps.get(str(take_uid), [])
        row = diagnose_take(payload, input_steps=inp)
        row["take_uid"] = str(take_uid)
        rows.append(row)

    # Write CSV
    fieldnames = [
        "take_uid", "scenario", "take_name", "status", "error",
        "schema_ok", "schema_msg",
        "n_input_steps", "n_final_steps", "n_meta", "n_del",
        "mods_u", "mods_e", "mods_m", "mods_c", "mods_i",
        "u_not_verbatim", "e_no_change", "deleted_src_still_used",
    ]
    # Make sure missing keys exist
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, "")

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Console summary
    by_status = defaultdict(int)
    for r in rows:
        by_status[str(r.get("status", "missing"))] += 1

    print("=== Diagnostics ===")
    print(f"Results: {args.results}")
    print(f"Split50:  {args.split50}")
    print(f"Out CSV:  {args.out_csv}")
    print("Status counts:")
    for k in sorted(by_status.keys()):
        print(f"  {k}: {by_status[k]}")


if __name__ == "__main__":
    main()

# python3 diagnose_llm_freeform_outputs.py \
#   --results data/examples/split_50_llm_freeform_errors_qwen.json \
#   --split50 local/egoexo4d/split_50.json \
#   --out_csv local/outputs/qwen_diagnostics.csv
