#!/usr/bin/env python3
"""
llm_error_plan_freeform_writer.py

Goal:
- Take existing procedures from split_50.json
- Ask an LLM to inject plausible human-like errors directly into the procedure text
  (wrong_execution / substitution / insertion / deletion / transposition)
- Optionally insert correction steps
- Output per take:
  * final_steps: list[str]
  * meta: list[list] mapping each final step to origin: [src_idx, mod, error_id, correction_id]
  * del: list[list] deletions: [src_idx, error_id]

Backends:
- --model openai: OpenAI Python SDK
- --model qwen: Qwen/Qwen2.5-32B-Instruct locally via transformers (device_map="auto")

Notes:
- No explicit error simulator. The LLM chooses where and which errors occur.
- Comments are English-only (project rule).
"""

import argparse
import json
import os
import re
import sys

# from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from piev.config import REPO_ROOT, load_settings

SETTINGS = load_settings()
DEFAULT_OPENAI_MODEL_ID = SETTINGS.openai_model
DEFAULT_QWEN_MODEL_ID = SETTINGS.qwen_text_model

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# -----------------------------
# CLI
# -----------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rewrite procedures by injecting plausible errors directly via an LLM (no simulator)."
    )
    p.add_argument(
        "--input",
        default=str(SETTINGS.split50_path),
        help="Path to split_50.json",
    )
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "local" / "outputs" / "split_50_llm_freeform_errors.json"),
        help="Output JSON path with rewrites per take.",
    )
    p.add_argument(
        "--model", required=True, choices=["openai", "qwen"], help="Which backend to use"
    )
    p.add_argument("--max_takes", type=int, default=0, help="If >0, process only first N takes")
    p.add_argument("--take_uid", default="", help="If set, process only this take_uid")
    p.add_argument(
        "--seed", type=int, default=123, help="Random seed (used only for ordering/subsampling)"
    )
    p.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=3000,
        help="Max tokens to generate for local Qwen (and a soft target for OpenAI).",
    )
    p.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="How many extra attempts to make if parsing/validation fails (default: 2).",
    )
    p.add_argument(
        "--retry_temp_decay",
        type=float,
        default=0.15,
        help="Temperature decay per retry attempt (default: 0.15).",
    )
    return p.parse_args()


# -----------------------------
# Small utilities
# -----------------------------

# def safe_int(x: Any) -> Optional[int]:
#     try:
#         if isinstance(x, bool):
#             return None
#         return int(x)
#     except Exception:
#         return None


# def compact_text(s: Any) -> str:
#     if not isinstance(s, str):
#         return ""
#     return re.sub(r"\s+", " ", s).strip()


def norm_text(s: str) -> str:
    s = s if isinstance(s, str) else ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9']+", norm_text(a)))
    tb = set(re.findall(r"[a-z0-9']+", norm_text(b)))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def build_retry_prompt(base_prompt: str, failure_reason: str) -> str:
    """
    Adds a strict retry notice to the prompt.
    Keep it short: local models often get worse when we add long extra text.
    """
    failure_reason = str(failure_reason or "").strip()
    if not failure_reason:
        failure_reason = "Your previous output did not match the required JSON schema."
    retry_notice = (
        "\n\n### RETRY NOTICE (CRITICAL)\n"
        f"Your previous output was REJECTED: {failure_reason}\n"
        "Regenerate the answer.\n"
        "HARD RULES:\n"
        "- Output ONLY ONE JSON object.\n"
        "- NO markdown, NO commentary, NO extra keys.\n"
        "- 'final_steps' and 'meta' same length.\n"
        "- If mod in ['we','s'], the text MUST differ meaningfully from the source step.\n"
        "- If mod in ['ms','mt'], the text MUST be verbatim of the source step.\n"
        "- If mod='i', inserted text MUST NOT duplicate the anchor step.\n"
    )
    return base_prompt + retry_notice


def _iter_json_object_candidates(text: str) -> List[Dict[str, Any]]:
    """
    Return all parseable JSON objects found as {...} blocks in the text.
    This helps when the model echoes the prompt or includes examples.
    """
    if not isinstance(text, str):
        return []
    s = text.strip()
    if not s:
        return []

    # If the entire text is JSON, take it as a single candidate.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    starts = [i for i, ch in enumerate(s) if ch == "{"]  # best-effort
    out: List[Dict[str, Any]] = []
    for start in starts:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:
                        pass
                    break
    return out


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction:
    - Prefer the LAST candidate that looks like our expected schema.
    - Fallback to the last parseable JSON object.
    """
    candidates = _iter_json_object_candidates(text)
    if not candidates:
        return None

    def looks_like_output(obj: Dict[str, Any]) -> bool:
        fs = obj.get("final_steps")
        meta = obj.get("meta")
        d = obj.get("del", [])
        return (
            isinstance(fs, list)
            and isinstance(meta, list)
            and "final_steps" in obj
            and "meta" in obj
            and ("del" in obj)
            and (d is None or isinstance(d, list))
        )

    for obj in reversed(candidates):
        if looks_like_output(obj):
            return obj
    return candidates[-1]


def estimate_output_budget(n_steps: int, hard_cap: int) -> int:
    """
    Rough heuristic:
    - Longer takes need more room for meta and potential corrections.
    """
    est = 900 + 45 * max(1, n_steps)
    return int(min(max(est, 1200), hard_cap))


def _auto_suffix_out_path(base_out: str, model_choice: str) -> str:
    """
    If the user provides a generic output path, append a model suffix:
      .../split_50_llm_freeform_errors_openai.json
      .../split_50_llm_freeform_errors_qwen.json
    If the filename already ends with _openai/_qwen, keep it as-is.
    """
    root, ext = os.path.splitext(base_out)
    if root.endswith("_openai") or root.endswith("_qwen"):
        return base_out
    ext = ext or ".json"
    return f"{root}_{model_choice}{ext}"


# -----------------------------
# Input loading (split_50.json)
# -----------------------------


def load_split50_annotations(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("split_50.json must contain a list at key 'annotations'.")
    return [a for a in annotations if isinstance(a, dict)]


def segments_to_steps(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Always produce steps in chronological order.
    Stable sort by (start_time, end_time, original_position) to avoid tie jitter.
    """
    segs = [(i, s) for i, s in enumerate(segments) if isinstance(s, dict)]
    segs.sort(
        key=lambda it: (
            float(it[1].get("start_time", 0.0) or 0.0),
            float(it[1].get("end_time", 0.0) or 0.0),
            int(it[0]),
        )
    )

    steps: List[Dict[str, Any]] = []
    for idx, (_orig_i, seg) in enumerate(segs):
        steps.append(
            {
                "index": idx,
                "text": str(seg.get("step_description", "")),
                "is_essential": bool(seg.get("is_essential", False)),
                "start_time": float(seg.get("start_time", 0.0) or 0.0),
                "end_time": float(seg.get("end_time", 0.0) or 0.0),
            }
        )
    return steps


# -----------------------------
# Prompt building (no simulator)
# -----------------------------


def build_prompt(
    take_uid: str, scenario: str, steps: List[Dict[str, Any]], model_choice: str
) -> str:
    """
    Prompt designed for "procedure editing", not story writing.
    No planned error plan is provided. The LLM selects 1-3 errors itself.
    """
    step_ctx = []
    for s in steps:
        step_ctx.append(
            {
                "index": int(s.get("index", 0)),
                "text": s.get("text", ""),
                "is_essential": bool(s.get("is_essential", False)),
            }
        )

    # Keep a JSON schema example only for OpenAI. Some local chat models echo it,
    # and naive extractors may mistakenly parse the example instead of the final answer.
    out_schema_example = {
        "final_steps": ["Step 1 text", "Step 2 text"],
        "meta": [[0, "u", None, None], [1, "we", "E01", None], [1, "c", None, "C01"]],
        "del": [[7, "E02"]],
    }

    instructions = (
        f"### ROLE: Procedure Editor for '{scenario}'\n"
        f"You will receive a step-by-step procedure.\n"
        f"Your task is to produce an edited final procedure that contains 1 to 3 plausible human mistakes.\n"
        f"You may also add 0 to 2 explicit correction steps.\n\n"
        f"### ERROR TYPES (HIGH-LEVEL)\n"
        f"- wrong_execution (mod='we'): Keep the same general goal, but execute it slightly wrong (wrong amount, messy action, wrong orientation).\n"
        f"- substitution (mod='s'): Replace the step with a different plausible action caused by confusion.\n"
        f"  Do not copy a later step verbatim.\n"
        f"- insertion (mod='i'): Insert one extra plausible step (unnecessary repetition, extra cleaning, checking, etc.).\n"
        f"- deletion: Remove the step completely (do not mention it was skipped). Add it to 'del'.\n"
        f"- transposition: Swap two steps.\n"
        f"  Use mod='ms' for the moved SOURCE step and mod='mt' for the moved TARGET step (both verbatim text, only order changes).\n\n"
        f"### HARD CONSTRAINTS\n"
        f"1) VERBATIM PRESERVATION: Keep most steps unchanged unless directly edited.\n"
        f"2) IMPERATIVE STYLE: Use direct imperative commands. No story.\n"
        f"3) SOURCE INDEX RANGE: Every meta source_idx MUST be a valid original index in [0, {len(steps) - 1}]. Never use -1.\n"
        f"4) PHYSICAL PLAUSIBILITY: Do not use tools/ingredients/objects before they appear earlier in the procedure.\n"
        f"5) METADATA ALIGNMENT: 'final_steps' and 'meta' must have the exact same length.\n\n"
        f"6) UNCHANGED MEANS VERBATIM: If mod='u', the final step text MUST match the original step text exactly.\n"
        f"7) TRANSPOSITION RULE: If using transposition, error_id must be non-null and exactly TWO steps must share that error_id:\n"
        f"   - one with mod='ms' and one with mod='mt'.\n\n"
        f"8) NO FAKE ERRORS: If mod in ['we','s'], the text MUST be meaningfully different from the original step at source_idx.\n"
        f"9) MOVE IS VERBATIM: If mod in ['ms','mt'], the text MUST be identical to the original step at source_idx (only moved).\n"
        f"10) INSERTION IS NEW: If mod='i', the inserted text MUST NOT be identical to the anchor step at source_idx.\n"
        f"11) CORRECTION NEEDS ID: If mod='c', correction_id must be non-null like 'C01' and the text must be new.\n"
        f"12) ERROR IDS: For mod in ['we','s','i','ms','mt'] use error_id like 'E01'. For deletions also.\n"
        f"### OUTPUT FORMAT (STRICT JSON ONLY)\n"
        f"- final_steps: list[str]\n"
        f"- meta: list[list], one per final step: [source_idx, mod, error_id, correction_id]\n"
        f"  mod_type: 'u'(unchanged), 'we'(wrong_execution), 's'(substitution), 'ms'(moved_source), 'mt'(moved_target), 'c'(correction), 'i'(insertion)\n"
        f"- del: list[list] for deleted steps: [source_idx, error_id]\n\n"
        f"For every deletion, error_id must be a string like 'E01' (never null).\n"
        f"Return ONLY a single JSON object. No markdown. No extra text.\n"
        f"Do NOT repeat the prompt. Output ONLY JSON.\n"
    )

    if str(model_choice).lower().strip() == "openai":
        instructions += f"Schema example: {json.dumps(out_schema_example)}\n"

    payload = {
        "take_uid": take_uid,
        "scenario": scenario,
        "steps": step_ctx,
    }

    return instructions + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# -----------------------------
# Backends
# -----------------------------


@dataclass
class LLMBackend:
    name: str

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__(name="openai")

        if OpenAI is None:
            raise RuntimeError("openai package is not available.")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_id = DEFAULT_OPENAI_MODEL_ID

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        try:
            resp = self.client.responses.create(
                model=self.model_id,
                input=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
                max_output_tokens=int(max_new_tokens),
            )
            txt = getattr(resp, "output_text", None)
            if isinstance(txt, str) and txt.strip():
                return txt
            return str(resp)
        except Exception:
            chat = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
                max_tokens=int(max_new_tokens),
            )
            return chat.choices[0].message.content or ""


class QwenLocalBackend(LLMBackend):
    def __init__(self, model_id: str = DEFAULT_QWEN_MODEL_ID) -> None:
        super().__init__(name="qwen")
        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("torch/transformers are not available; install the qwen extra.")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()
        self.torch = torch

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        try:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=0.95,
                repetition_penalty=1.02,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if decoded.startswith(text):
            decoded = decoded[len(text) :].strip()
        return decoded.strip()


def make_backend(model_choice: str) -> LLMBackend:
    if model_choice == "openai":
        return OpenAIBackend()
    if model_choice == "qwen":
        return QwenLocalBackend()
    raise ValueError(f"Unknown model choice: {model_choice}")


# -----------------------------
# Output validation (compact format)
# -----------------------------


def validate_model_output(
    obj: Dict[str, Any],
    input_steps: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Output must be a JSON object"

    n_input_steps = len(input_steps)
    final_steps = obj.get("final_steps")
    meta = obj.get("meta")
    deletions = obj.get("del", [])

    if not isinstance(final_steps, list) or not all(isinstance(x, str) for x in final_steps):
        return False, "Missing or invalid 'final_steps' (must be list[str])"
    if not isinstance(meta, list) or not all(isinstance(x, list) for x in meta):
        return False, "Missing or invalid 'meta' (must be list[list])"
    if len(final_steps) != len(meta):
        return False, "'final_steps' and 'meta' must have the same length"

    has_any_error = False
    allowed_mod = {"u", "we", "s", "ms", "mt", "c", "i"}
    move_eid_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ms": 0, "mt": 0})

    EID_RE = re.compile(r"^E\d{2}$")
    CID_RE = re.compile(r"^C\d{2}$")

    for i, m in enumerate(meta):
        if len(m) < 4:
            return False, f"meta[{i}] must have 4 elements: [src_idx, mod, error_id, correction_id]"

        src_idx, mod, eid, cid = m[0], m[1], m[2], m[3]

        if not isinstance(src_idx, int):
            return False, f"meta[{i}][0] src_idx must be int"
        if src_idx < 0 or src_idx >= n_input_steps:
            return False, f"meta[{i}][0] src_idx out of range: {src_idx}"
        if mod not in allowed_mod:
            return False, f"meta[{i}][1] invalid mod '{mod}'"
        if eid is not None and not isinstance(eid, str):
            return False, f"meta[{i}][2] error_id must be str|null"
        if cid is not None and not isinstance(cid, str):
            return False, f"meta[{i}][3] correction_id must be str|null"

        original = str(input_steps[src_idx].get("text", ""))
        final = str(final_steps[i])

        # optional but makes the dataset cleaner:
        # If there is an error-related mod, require an error_id (except 'c' which uses correction_id).
        if mod in {"we", "s", "i", "ms", "mt"}:
            if eid is None or not eid.strip() or not EID_RE.match(eid.strip()):
                return False, f"meta[{i}] mod='{mod}' requires error_id like E01"
        if mod == "c":
            if cid is None or not cid.strip() or not CID_RE.match(cid.strip()):
                return False, f"meta[{i}] mod='c' requires correction_id like C01"

        # Unchanged must be verbatim.
        if mod == "u":
            if final != original:
                return (
                    False,
                    f"meta[{i}] mod='u' but final_steps[{i}] is not verbatim of INPUT.steps[{src_idx}]",
                )

        # Transposition: moved steps should remain verbatim (they are moved, not edited).
        if mod in {"ms", "mt"}:
            if final != original:
                return (
                    False,
                    f"meta[{i}] mod='{mod}' but text differs from source step {src_idx} (moved steps must be verbatim)",
                )
            move_eid_counts[eid.strip()][mod] += 1

        # Edited must actually differ (at least a bit).
        if mod in {"we", "s"}:
            if norm_text(final) == norm_text(original):
                return (
                    False,
                    f"meta[{i}] mod='{mod}' but text is IDENTICAL to original step {src_idx}",
                )
            # optional: require meaningful difference (prevents tiny punctuation tweaks)
            if token_jaccard(final, original) > 0.92:
                return (
                    False,
                    f"meta[{i}] mod='{mod}' but change is too small (token Jaccard > 0.92)",
                )

        # Insertion must not be identical to the anchor step text.
        # It MAY repeat some other earlier step (that's a plausible insertion),
        # but it must not claim to be inserted while copying the anchor verbatim.
        if mod == "i":
            if norm_text(final) == norm_text(original):
                return (
                    False,
                    f"meta[{i}] mod='i' but inserted text duplicates anchor source step {src_idx}",
                )

        # Correction step should not be a verbatim copy of the anchor either.
        if mod == "c":
            if norm_text(final) == norm_text(original):
                return False, f"meta[{i}] mod='c' but correction duplicates source step {src_idx}"

        if mod != "u":
            has_any_error = True

    # transposition: each error_id must have exactly one 'ms' and one 'mt'
    for eid, counts in move_eid_counts.items():
        if counts.get("ms", 0) != 1 or counts.get("mt", 0) != 1:
            return (
                False,
                f"transposition_eid_must_have_exactly_one_ms_and_one_mt: {eid} has ms={counts.get('ms', 0)} mt={counts.get('mt', 0)}",
            )

    # require at least 1 actual error (since prompt demands 1-3)
    if not has_any_error and (not deletions):
        return False, "No errors found: all steps are 'u' and no deletions"

    # deletions
    if deletions is None:
        return True, "ok"
    if not isinstance(deletions, list) or not all(isinstance(x, list) for x in deletions):
        return False, "Invalid 'del' (must be list[list])"
    for j, d in enumerate(deletions):
        if len(d) < 2:
            return False, f"del[{j}] must have at least 2 elements: [src_idx, error_id]"
        if not isinstance(d[0], int) or not (0 <= d[0] < n_input_steps):
            return False, f"del[{j}][0] src_idx out of range"
        if not isinstance(d[1], str) or not EID_RE.match(d[1].strip()):
            return False, f"del[{j}][1] error_id must be like E01"
        has_any_error = True

    return True, "ok"


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    args = parse_args()

    take_uid_filter = str(args.take_uid or "").strip()

    # Auto-suffix output path by model unless the user already provided a suffixed path.
    args.out = _auto_suffix_out_path(str(args.out), str(args.model))
    backend = make_backend(args.model)

    annotations = load_split50_annotations(args.input)

    # Deterministic ordering.
    ann_items = annotations[:]
    ann_items.sort(key=lambda a: str(a.get("take_uid", "")))

    if take_uid_filter:
        ann_items = [a for a in ann_items if str(a.get("take_uid", "")) == take_uid_filter]

    if args.max_takes and int(args.max_takes) > 0:
        ann_items = ann_items[: int(args.max_takes)]

    out_obj: Dict[str, Any] = {
        "meta": {
            "created_with": "llm_error_plan_freeform_writer.py",
            "model_backend": backend.name,
            "model_choice": args.model,
            "temperature": float(args.temperature),
            "seed": int(args.seed),
            "input_path": args.input,
            "out_path": args.out,
        },
        "takes": {},
    }

    for ann in ann_items:
        take_uid = str(ann.get("take_uid", "")).strip()
        scenario = str(ann.get("scenario", "unknown"))
        take_name = str(ann.get("take_name", "unknown"))
        segments = ann.get("segments", []) if isinstance(ann.get("segments", []), list) else []

        steps = segments_to_steps(segments)
        n_steps = len(steps)

        if n_steps == 0:
            out_obj["takes"][take_uid] = {
                "scenario": scenario,
                "take_name": take_name,
                "status": "empty_procedure",
            }
            continue

        base_prompt = build_prompt(
            take_uid=take_uid,
            scenario=scenario,
            steps=steps,
            model_choice=str(args.model),
        )

        budget = estimate_output_budget(n_steps, int(args.max_new_tokens))

        max_retries = max(0, int(args.max_retries))
        temp0 = float(args.temperature)
        decay = float(args.retry_temp_decay)

        last_raw: str = ""
        last_obj: Optional[Dict[str, Any]] = None
        last_fail_status: str = ""
        last_fail_reason: str = ""

        success_obj: Optional[Dict[str, Any]] = None

        for attempt in range(max_retries + 1):
            # decay temperature on retries to reduce "creative cheating"
            t = max(0.0, temp0 - decay * float(attempt))

            prompt = base_prompt
            if attempt > 0:
                prompt = build_retry_prompt(base_prompt, f"{last_fail_status}: {last_fail_reason}")

            raw = backend.generate(
                prompt=prompt,
                temperature=float(t),
                max_new_tokens=budget,
            )
            last_raw = raw

            obj = extract_json_object(raw)
            last_obj = obj

            if obj is None:
                last_fail_status = "parse_failed"
                last_fail_reason = (
                    "Could not extract a valid JSON object with keys: final_steps, meta, del"
                )
                continue

            # Attach identifiers (do not affect validation)
            obj["take_uid"] = take_uid
            obj["scenario"] = scenario

            ok, msg = validate_model_output(obj, input_steps=steps)
            if not ok:
                last_fail_status = "invalid_schema"
                last_fail_reason = msg
                continue

            success_obj = obj
            break

        if success_obj is None:
            # final failure after retries
            payload: Dict[str, Any] = {
                "scenario": scenario,
                "take_name": take_name,
                "status": last_fail_status or "failed",
                "error": last_fail_reason or "Unknown failure",
                "raw_output_preview": (last_raw or "")[:8000],
            }
            if last_obj is not None:
                payload["parsed_output"] = last_obj
            out_obj["takes"][take_uid] = payload
            continue

        out_obj["takes"][take_uid] = {
            "scenario": scenario,
            "take_name": take_name,
            "status": "ok",
            "n_input_steps": n_steps,
            "rewrite": success_obj,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("=== LLM Freeform Errors ===")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.out}")
    print(f"Backend: {backend.name}")
    if take_uid_filter:
        print(f"Filtered take_uid: {take_uid_filter}")


if __name__ == "__main__":
    main()
