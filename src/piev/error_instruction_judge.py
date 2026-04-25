#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_instruction_judge.py (v3)

Validate + repair LLM-generated erroneous procedures.

Inputs:
  --plan      : error plan JSON (output of correction_simulator; contains takes->steps/errors/corrections)
  --rewrite   : JSON produced by error_instruction_writer.py (takes->...->rewrite->{final_steps,meta})
Outputs:
  --out       : judged JSON (structure close to writer output; ensures take_name is filled)
  --report_base : base path (without ext) for CSV/JSON reports of detected issues

Model flags (same style as writer):
  --model openai|qwen

Filtering:
  --take_name <name1> <name2> ... : process only these takes (matches take["take_name"])

LLM (primary pass):
  We call LLM first to minimally repair hard illogicalities; then run deterministic cleanup.
  Disable with --disable_llm_fallback.

Optional semrep support:
  --vocab_csv, --semrep_json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from piev.config import REPO_ROOT, load_settings

SETTINGS = load_settings()

try:
    from piev.utils.semrep_utils import SemRepAutoExtender, build_semrep_step_to_id as _helper_build_semrep_step_to_id
except ImportError:
    try:
        from semrep_utils import SemRepAutoExtender, build_semrep_step_to_id as _helper_build_semrep_step_to_id
    except Exception:
        SemRepAutoExtender = None
        _helper_build_semrep_step_to_id = None

try:
    from piev.utils.frame_utils import EgoVideoFrameCache, FrameRequest
except Exception:
    EgoVideoFrameCache = None  # type: ignore
    FrameRequest = None  # type: ignore

# -------------------------
# API key
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

DEFAULT_OPENAI_MODEL_ID = SETTINGS.openai_model
DEFAULT_QWEN_MODEL_ID = SETTINGS.qwen_vl_model

# -------------------------
# Backends
# -------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    import torch
    from transformers import AutoTokenizer, AutoProcessor
    from transformers import AutoModelForCausalLM
    # Prefer Qwen3-VL when available; fall back to Qwen2-VL for older transformers.
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except Exception:
        Qwen3VLForConditionalGeneration = None
    try:
        from transformers import Qwen2VLForConditionalGeneration
    except Exception:
        Qwen2VLForConditionalGeneration = None        
except Exception:
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    Qwen3VLForConditionalGeneration = None
    Qwen2VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:
    process_vision_info = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

import base64
import io

@dataclass
class LLMBackend:
    name: str

    def generate(self, messages: List[Dict[str, str]], temperature: float, max_new_tokens: int, image_data_urls: Optional[List[str]] = None) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self, model_id: str = DEFAULT_OPENAI_MODEL_ID) -> None:
        super().__init__(name="openai")
        if OpenAI is None:
            raise RuntimeError("openai package not available.")
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "") or OPENAI_API_KEY)
        self.model_id = model_id

    def generate(self, messages: List[Dict[str, str]], temperature: float, max_new_tokens: int, image_data_urls: Optional[List[str]] = None) -> str:
        # Prefer Responses API when available; fallback to chat.completions
        try:
            resp = self.client.responses.create(
                model=self.model_id,
                input=messages,
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
                messages=messages,
                temperature=float(temperature),
                max_completion_tokens=int(max_new_tokens),
            )
            return chat.choices[0].message.content or ""


class QwenLocalBackend(LLMBackend):
    """
    Qwen3-VL backend with real image support via AutoProcessor + process_vision_info.
    Falls back to text-only if VL deps are missing.
    """
    def __init__(self, model_id: str = DEFAULT_QWEN_MODEL_ID) -> None:
        super().__init__(name="qwen")

        if torch is None:
            raise RuntimeError("torch not available; cannot use qwen backend.")

        # VL path requires these
        self.has_vl = (
            AutoProcessor is not None
            and (Qwen3VLForConditionalGeneration is not None or Qwen2VLForConditionalGeneration is not None)
            and Image is not None
        )

        self.model_id = model_id
        self.torch = torch

        if self.has_vl:
            # Recommended path for Qwen3-VL
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
                self.model_id, use_fast=True, trust_remote_code=True
            )

            # Qwen3-VL uses Qwen3VLForConditionalGeneration in newer transformers.
            # Some environments still expose only Qwen2VLForConditionalGeneration.
            _vl_cls = Qwen3VLForConditionalGeneration or Qwen2VLForConditionalGeneration
            if _vl_cls is None:
                raise RuntimeError("No Qwen VL ConditionalGeneration class available in transformers.")
            try:
                self.model = _vl_cls.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    dtype="auto",
                    trust_remote_code=True,
                )
            except TypeError:
                # Backward compatible fallback.
                self.model = _vl_cls.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
        else:
            # text-only fallback (keeps your current behavior, but explicit)
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError("transformers not available; cannot use qwen backend.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=True, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            self.processor = None

        self.model.eval()

    def _data_url_to_pil(self, url: str):
        """
        Decode data:image/...;base64,.... to PIL.Image
        """
        if Image is None:
            return None
        if not isinstance(url, str) or not url.startswith("data:") or "base64," not in url:
            return None
        try:
            b64 = url.split("base64,", 1)[1]
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            return img
        except Exception:
            return None

    def _inject_images_into_messages(self, messages, image_data_urls):
        """
        Put images into the first user message as Qwen VL content parts:
          [{"type":"image","image": <PIL or path>}, {"type":"text","text": "..."}]
        """
        new_messages = []
        injected = False

        pil_images = []
        for u in (image_data_urls or []):
            im = self._data_url_to_pil(u)
            if im is not None:
                pil_images.append(im)

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user" and not injected and pil_images:
                parts = []
                for im in pil_images:
                    parts.append({"type": "image", "image": im})

                if isinstance(content, str):
                    parts.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    parts.extend(content)
                else:
                    parts.append({"type": "text", "text": str(content)})

                new_messages.append({"role": "user", "content": parts})
                injected = True
            else:
                new_messages.append(msg)

        if not injected and pil_images:
            parts = [{"type": "image", "image": im} for im in pil_images]
            parts.append({"type": "text", "text": "Use these images as context."})
            new_messages.append({"role": "user", "content": parts})

        return new_messages

    def generate(
        self,
        messages,
        temperature: float,
        max_new_tokens: int,
        image_data_urls=None,
    ) -> str:
        temp = float(temperature)
        do_sample = temp > 0.01

        # -------------------------
        # VL path (real images)
        # -------------------------
        if self.has_vl and image_data_urls:
            new_messages = self._inject_images_into_messages(messages, image_data_urls)

            try:
                inputs = self.processor.apply_chat_template(
                    new_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)

                device = next(self.model.parameters()).device
                inputs = inputs.to(device)

                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    repetition_penalty=1.02,
                    do_sample=do_sample,
                )
                if do_sample:
                    gen_kwargs["temperature"] = temp
                    gen_kwargs["top_p"] = 0.95

                with self.torch.no_grad():
                    generated_ids = self.model.generate(**gen_kwargs)

                # Trim the prompt tokens (important for VL models)
                in_ids = inputs.input_ids
                trimmed = []
                for i0, o0 in zip(in_ids, generated_ids):
                    trimmed.append(o0[len(i0):])

                out_text = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return (out_text[0] if out_text else "").strip()

            except Exception:
                #Fallback: process_vision_info (Qwen2-style)
                try:
                    if process_vision_info is None:
                        raise RuntimeError("process_vision_info is not available")
                    text = self.processor.apply_chat_template(
                        new_messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(new_messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.model.device)

                    gen_kwargs = dict(
                        **inputs,
                        max_new_tokens=int(max_new_tokens),
                        repetition_penalty=1.02,
                        do_sample=do_sample,
                    )
                    if do_sample:
                        gen_kwargs["temperature"] = temp
                        gen_kwargs["top_p"] = 0.95

                    with self.torch.no_grad():
                        generated_ids = self.model.generate(**gen_kwargs)

                    in_ids = inputs.input_ids
                    trimmed = []
                    for i0, o0 in zip(in_ids, generated_ids):
                        trimmed.append(o0[len(i0):])
                    out_text = self.processor.batch_decode(
                        trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    return (out_text[0] if out_text else "").strip()
                except Exception:
                    pass 
                pass

        # -------------------------
        # Text-only fallback
        # -------------------------
        try:
            if self.processor is not None:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], padding=True, return_tensors="pt")
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    repetition_penalty=1.02,
                    do_sample=do_sample,
                )
                if do_sample:
                    gen_kwargs["temperature"] = temp
                    gen_kwargs["top_p"] = 0.95
                with self.torch.no_grad():
                    generated_ids = self.model.generate(**gen_kwargs)

                # Trim prompt
                trimmed = []
                for i0, o0 in zip(inputs.input_ids, generated_ids):
                    trimmed.append(o0[len(i0):])

                out_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)
                return (out_text[0] if out_text else "").strip()

            # legacy tokenizer-only
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        except Exception:
            text = "\n\n".join([f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages])

        inputs = self.tokenizer(text, return_tensors="pt")
        try:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=1.02,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.95

        with self.torch.no_grad():
            out = self.model.generate(**gen_kwargs)

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # rough trim
        if decoded.startswith(text):
            decoded = decoded[len(text):].strip()
        return decoded.strip()

def make_backend(model_choice: str) -> LLMBackend:
    if model_choice == "openai":
        return OpenAIBackend()
    if model_choice == "qwen":
        return QwenLocalBackend()
    raise ValueError(f"Unknown model choice: {model_choice}")


# -------------------------
# Utilities
# -------------------------
ALLOWED_MOD = {"u", "e", "a", "i", "c", "d", "ms", "mt"}

def now_ms() -> int:
    return int(time.time() * 1000)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def normalize_lookup_key(s: str) -> str:
    """
    Normalization for step_description lookup across:
      - vocab CSV
      - semrep JSON (base + extended)
      - rewrite final_steps (often differ by punctuation/case)
    """
    s = normalize_ws((s or "").strip())
    s = s.rstrip(".!?:;")
    return s.lower()

def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())

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

def expand_entity_aliases(ent: str) -> Set[str]:
    """
    Minimal alias expansion to bridge container/content naming conventions.
    Example: sugar_bag -> {sugar_bag, sugar, bag}
             milk_container -> {milk_container, milk, container}
    No domain lexicon: only splits on '_' and '-'.
    """
    out: Set[str] = set()
    e = normalize_entity_token(ent or "")
    if not e:
        return out
    out.add(e)
    # split compounds
    for p in re.split(r"[_\-]+", e):
        p = (p or "").strip()
        if not p:
            continue
        # keep only non-trivial tokens
        if len(p) < 3 or p.isdigit():
            continue
        out.add(p)
    return out


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction:
      - try full parse
      - else return the last balanced {...} that parses as dict
    """
    if not isinstance(text, str):
        raise ValueError("Response is not a string.")
    s = text.strip()
    if not s:
        raise ValueError("Empty response.")

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    candidates: List[str] = []
    depth = 0
    start: Optional[int] = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(s[start:i+1])
                    start = None

    for chunk in reversed(candidates):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    raise ValueError("No valid JSON object found in response.")


def _find_meta_by_new(meta: List[Dict[str, Any]], new_idx: int) -> Optional[Dict[str, Any]]:
    for m in meta:
        if isinstance(m.get("new"), int) and int(m["new"]) == int(new_idx):
            return m
    return None

def is_near_duplicate_step(a: str, b: str) -> bool:
    a_n, b_n = normalize_ws(a).lower(), normalize_ws(b).lower()
    if a_n == b_n:
        return True
    return jaccard_similarity(a_n, b_n) >= 0.85

# -------------------------
# Global Cache for Exact SemRep Lookup
# -------------------------
# Map: normalized_step_text -> semantic_representation_string
_REVERSE_SEMREP_MAP: Dict[str, str] = {}

def init_reverse_semrep_map(semrep_map: Dict[str, Dict[str, str]]) -> None:
    """
    Builds a reverse lookup dictionary: normalized text -> SemRep string.
    This allows O(1) exact matching for steps generated by the LLM, 
    assuming they exist in the extended JSON.
    """
    global _REVERSE_SEMREP_MAP
    _REVERSE_SEMREP_MAP.clear()
    
    for sid, data in semrep_map.items():
        if not isinstance(data, dict):
            continue
        
        raw_text = data.get("step_description", "")
        sem_rep = data.get("semantic_representation", "")
        
        if raw_text and sem_rep:
            # Normalize key to ensure case-insensitive exact match
            # matches the logic in classify_step
            norm_key = normalize_step_text(raw_text)
            _REVERSE_SEMREP_MAP[norm_key] = sem_rep

def find_semrep_exact(text: str) -> Optional[str]:
    """
    Look up SemRep by exact text match (normalized).
    """
    norm_text = normalize_step_text(text)
    return _REVERSE_SEMREP_MAP.get(norm_text)

# -------------------------
# Optional vocab/semrep loading
# -------------------------
def load_vocab_csv(path: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("step_description_id") or "").strip()
            txt = str(row.get("step_description") or "").rstrip("\n\r")
            if sid and txt and txt not in m:
                m[txt] = sid
            if sid and txt.strip() and txt.strip() not in m:
                m[txt.strip()] = sid
            # normalized lookup keys (helps with punctuation/case mismatches)
            if sid:
                nk = normalize_lookup_key(txt)
                if nk and nk not in m:
                    m[nk] = sid
    return m

def load_semrep_json(path: str) -> Dict[str, Dict[str, str]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            out[k.strip()] = {
                "step_description": str(v.get("step_description") or ""),
                "semantic_representation": str(v.get("semantic_representation") or ""),
            }
    return out

def _build_semrep_step_to_id_fallback(semrep_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Reverse lookup: exact step_description -> semrep id.
    Works for base vocab ids and synthetic *_ext_* ids.
    """
    # Prefer helper implementation when available; fallback otherwise
    m: Dict[str, str] = {}
    for sid, v in (semrep_map or {}).items():
        if not isinstance(sid, str) or not isinstance(v, dict):
            continue
        sd = v.get("step_description")
        if isinstance(sd, str) and sd.strip():
            # store multiple keys for robust matching
            raw = sd.strip()
            keys = {
                raw,
                raw.strip(),
                normalize_ws(raw),
                normalize_lookup_key(raw),
            }
            for key in keys:
                if key and key not in m:
                    m[key] = sid
    return m

build_semrep_step_to_id = _helper_build_semrep_step_to_id or _build_semrep_step_to_id_fallback

def resolve_step_id_from_text(step_txt: str, vocab_map: Dict[str, str], extra_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    raw = (step_txt or "").rstrip("\n\r")
    if raw in vocab_map:
        return vocab_map[raw]
    if raw.strip() in vocab_map:
        return vocab_map[raw.strip()]
    nk = normalize_lookup_key(raw)
    if nk in vocab_map:
        return vocab_map[nk]
    if extra_map is not None:
        if raw in extra_map:
            return extra_map[raw]
        if raw.strip() in extra_map:
            return extra_map[raw.strip()]
        if nk in extra_map:
            return extra_map[nk]
    return None

def sanitize_nonverbatim_step_texts(final_steps: List[str], meta: List[Dict[str, Any]]) -> None:
    """
    Replace '_' with spaces in *generated* step texts only.
    IMPORTANT: Do NOT touch verbatim steps (u/ms/mt), otherwise schema/verbatim constraints break.
    """
    for m in meta:
        if m.get("mod") in {"u","ms","mt","d"}:
            continue
        new = m.get("new")
        if isinstance(new, int) and 0 <= new < len(final_steps):
            final_steps[new] = final_steps[new].replace("_", " ")

# -------------------------
# Step helpers (plan schema compatibility)
# -------------------------
def _step_idx(s: Dict[str, Any]) -> int:
    v = s.get("idx")
    if isinstance(v, int):
        return v
    v = s.get("index")
    if isinstance(v, int):
        return v
    return 0

def _step_text(s: Dict[str, Any]) -> str:
    v = s.get("txt")
    if isinstance(v, str) and v:
        return v
    v = s.get("step_description")
    if isinstance(v, str) and v:
        return v
    return ""

def normalize_object_key(obj_phrase: str) -> str:
    """
    Fallback normalization when semrep is missing.
    Keep it *minimal* (no big stopword tables); semrep should be the main path.
    """
    s = normalize_step_text(obj_phrase)
    if not s:
        return ""
    toks = re.findall(r"[a-z0-9]+", s)
    # tiny generic filter only
    tiny = {"the","a","an","to","from","into","onto","on","in","at","with","and","or","of","for","by"}
    toks = [t for t in toks if t and t not in tiny]
    if not toks:
        return ""
    toks = sorted(toks)
    return " ".join(toks)

def openai_repair_strict_json(
    client, model_id: str, system_prompt: str, user_prompt: str,
    temperature: float, max_output_tokens: int,
    image_data_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    user_parts = []
    for u in (image_data_urls or []):
        if isinstance(u, str) and u.strip():
            user_parts.append({"type": "input_image", "image_url": u, "detail": "low"})
    user_parts.append({"type": "input_text", "text": user_prompt})

    resp = client.responses.create(
        model=model_id,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_parts},
        ],
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
        text={
            "format": {
                "type": "json_schema",
                "name": "procedure_repair",
                "strict": True,
                "schema": REPAIR_OUTPUT_SCHEMA,
            }
        },
    )
    txt = getattr(resp, "output_text", None)
    if not isinstance(txt, str) or not txt.strip():
        raise ValueError("No output_text in OpenAI response.")
    return json.loads(txt)

# -------------------------
# Action classification (heuristic + optional semrep)
# -------------------------
# GET_VERBS = {"get","take","grab","pick","retrieve","fetch","remove"}
# PUT_BACK_VERBS = {"put","place","set","return","putback"}  # "put back" handled by phrase
# OPEN_VERBS = {"open","unseal","unscrew"}
# CLOSE_VERBS = {"close","seal","cover"}
# WASH_VERBS = {"wash","rinse","clean"}
# DRY_VERBS = {"dry","wipe"}
# ADD_VERBS = {"add","pour","sprinkle","mix","stir","combine"}
# CUT_VERBS = {"cut","chop","dice","slice","mince","remove","peel"}

def _starts_with_any(s: str, prefixes: Tuple[str, ...]) -> bool:
    return any(s.startswith(p) for p in prefixes)

def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in (s or ""):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]

def parse_semrep_one(semrep: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Parse single-line semrep: PRED(Role: value, Role: value, ...)
    Returns (PRED, {Role->value}) or None.
    """
    sr = (semrep or "").strip()
    m = re.match(r"^([A-Z][A-Z0-9_]*)\((.*)\)$", sr)
    if not m:
        return None
    pred = m.group(1).strip()
    body = m.group(2).strip()
    roles: Dict[str, str] = {}
    for part in _split_top_level_commas(body):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        roles[k.strip()] = v.strip()
    return pred, roles

def _head_entity(value: str) -> str:
    """
    Best-effort head extraction for value strings like:
      inside(of(cup)) -> cup
      work_table -> work_table
      filter(Location: in(french_press)) -> filter
      from(chain_lube) -> chain_lube
    """
    v = (value or "").strip()
    if not v:
        return ""
    # take first chunk before '&' (we only need identity for checks)
    v = v.split("&", 1)[0].strip()

    m = re.match(r"^([a-z][a-z0-9_]*)\s*(?:\(|$)", v)
    if not m:
        # fallback: first identifier
        toks = re.findall(r"[a-z][a-z0-9_]*", v.lower())
        return toks[0] if toks else ""
    head = m.group(1)

    # unwrap wrappers like of(...), in(...), on(...), inside(...), bottom(...)
    if v.startswith(head + "(") and v.endswith(")"):
        inner = v[len(head) + 1 : -1].strip()
        if inner:
            # if inner contains "of(...)" etc, drilling usually yields the real object identity
            if head in {"of","in","on","into","onto","from","to","with","inside","outside","bottom","top","side","surface","interior","exterior","next_to","beside","near","between","under","over","above","below","around","across","along"}:
                return _head_entity(inner)
            # also handle common pattern: X(of(Y)) -> Y
            if "of(" in inner:
                return _head_entity(inner)
    return normalize_entity_token(head)

_PREP_ACTIONS = {"WASH", "CHOP", "CUT", "SLICE", "PEEL"}

def _collect_baseline_affordances(semrep_by_old: Dict[int, str]) -> Dict[str, Set[str]]:
    """
    Build a per-entity action set from the ORIGINAL procedure (baseline).
    This is used to detect implausible cascades like WASH/CHOP on an entity that,
    in the baseline of this take, only ever appears as a seasoning (e.g., GET/ADD).
    No domain lexicon: purely derived from baseline SemRep.
    """
    aff: Dict[str, Set[str]] = {}
    for _old_idx, sr in (semrep_by_old or {}).items():
        if not sr:
            continue
        parsed = parse_semrep_one(sr)
        if not parsed:
            continue
        pred, roles = parsed
        act = _action_from_semrep(pred, roles)
        if not act:
            continue
        objh = _head_entity(roles.get("Object") or roles.get("Theme") or roles.get("Patient") or "")
        if not objh:
            continue
        aff.setdefault(objh, set()).add(act)
    return aff

def validate_affordance_mismatch_against_baseline_semrep(
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    semrep_by_new: Dict[int, str],
    semrep_by_old: Dict[int, str],
) -> List[Dict[str, Any]]:
    """
    Flag cases where an entity that already exists in baseline is used with a prep-like action
    (WASH/CHOP/CUT/SLICE/PEEL) that never occurs for that entity in baseline.

    This catches implausible cascades such as 'Wash black pepper with water' when black_pepper
    is only ever used as a seasoning in the original procedure.
    """
    issues: List[Dict[str, Any]] = []
    base_aff = _collect_baseline_affordances(semrep_by_old)
    if not base_aff:
        return issues

    # Map new index -> meta row (skip deletions)
    meta_by_new: Dict[int, Dict[str, Any]] = {}
    for m in meta:
        if m.get("mod") == "d":
            continue
        if isinstance(m.get("new"), int):
            meta_by_new[int(m["new"])] = m

    for i, step in enumerate(final_steps):
        sr = semrep_by_new.get(i) or ""
        parsed = parse_semrep_one(sr) if sr else None
        if not parsed:
            continue
        pred, roles = parsed
        act = _action_from_semrep(pred, roles)
        if act not in _PREP_ACTIONS:
            continue
        objh = _head_entity(roles.get("Object") or roles.get("Theme") or roles.get("Patient") or "")
        if not objh:
            continue
        if objh not in base_aff:
            # Unknown-in-baseline entities are allowed (don't over-penalize).
            continue
        if act in base_aff.get(objh, set()):
            continue

        mm = meta_by_new.get(i, {})
        issues.append({
            "code": "AFFORDANCE_MISMATCH_BASELINE",
            "step_index": i,
            "step": step,
            "object": objh,
            "detail": f"Action '{act}' was never used with '{objh}' in baseline for this take; likely implausible cascade.",
            "meta_mod": mm.get("mod"),
            "meta_eid": mm.get("eid"),
        })
    return issues

def _embedded_of_head(value: str) -> str:
    """Extract head entity from the first `of(...)` occurrence inside a raw SemRep role value.

    Examples:
      "jar(of(sugar))" -> "sugar"
      "pack(of(brown_sugar))" -> "brown_sugar"

    Returns "" if nothing is found.
    """
    v = (value or "").strip()
    if not v:
        return ""
    m = re.search(r"\bof\(([^)]+)\)", v)
    if not m:
        return ""
    inner = (m.group(1) or "").strip()
    return _head_entity(inner)

def _container_like_core(ent_head: str, ent_raw: str = "") -> str:
    """Best-effort content/core extraction for container-like entities.

    We intentionally keep this generic (no domain-specific word lists):
    - Prefer `of(...)` payloads when present (jar(of(sugar)) -> sugar).
    - Otherwise, for underscore compounds, drop the last token (sugar_bag -> sugar; soy_sauce_bottle -> soy_sauce).
    - Fallback to the head itself.
    """
    ofh = _embedded_of_head(ent_raw)
    if ofh:
        return ofh
    h = (ent_head or "").strip()
    if "_" in h:
        parts = [p for p in h.split("_") if p]
        if len(parts) >= 2:
            return "_".join(parts[:-1])
    return h

def _replace_entity_surface(text: str, from_ent: str, to_ent: str) -> str:
    """Replace an entity mention in free text, handling underscore vs space variants.

    This is intentionally conservative: we only replace whole-word-ish matches.
    """
    if not isinstance(text, str) or not text:
        return text
    if not from_ent or not to_ent or from_ent == to_ent:
        return text
    from_space = from_ent.replace("_", " ").strip()
    to_space = to_ent.replace("_", " ").strip()
    out = re.sub(rf"\b{re.escape(from_space)}\b", to_space, text, flags=re.IGNORECASE)
    out = re.sub(rf"\b{re.escape(from_ent)}\b", to_ent, out, flags=re.IGNORECASE)
    out = re.sub(rf"\b{re.escape(from_space.replace(' ', '-'))}\b", to_space, out, flags=re.IGNORECASE)
    return out

def validate_get_substitution_cascade_semrep(
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    original_steps: List[str],
    semrep_by_old: Dict[int, str],
    semrep_by_new: Dict[int, str],
    window: int = 12,
) -> List[Dict[str, Any]]:
    """Detect downstream steps that still refer to the ORIGINAL object/substance after a GET-substitution.

    Motivation (generic, no word hardcoding):
      If the plan says we substituted a GET (e.g., got X instead of Y), later steps that
      *use Y* but never re-acquire Y are internally inconsistent. A minimal cascade is
      to swap Y->X (or, for container-like entities, Y_core->X_core) in the downstream step.

    We emit soft plausibility issues of type SUBSTITUTION_CASCADE_ENTITY.
    Deterministic repair can then rewrite those steps and mark them as 'a' cascades.
    """
    issues: List[Dict[str, Any]] = []

    def _prev_entities_in_rewrite(new_limit: int) -> set[str]:
        """Entities (Object heads) that appear BEFORE new_limit in the CURRENT rewrite."""
        out: set[str] = set()
        for i in range(0, max(0, int(new_limit))):
            sr = (semrep_by_new or {}).get(i) or ""
            if sr:
                p = parse_semrep_one(sr)
                if p:
                    _pred, roles = p
                    objh = _head_entity(roles.get("Object") or roles.get("Theme") or roles.get("Patient") or "")
                    if objh:
                        out.add(objh)
                    continue
            # fallback if semrep missing: substring match against step text
            t = (final_steps[i] or "").lower()
            # store nothing here; we'll do fallback check directly where needed
        return out

    def _baseline_entities_before(old_limit: int) -> Set[str]:
        """Entities (Object/Theme/Patient heads) appearing in baseline BEFORE old_limit."""
        out: Set[str] = set()
        for oi, sr in (semrep_by_old or {}).items():
            if not isinstance(oi, int) or oi >= old_limit:
                continue
            if not sr:
                continue
            p = parse_semrep_one(sr)
            if not p:
                continue
            _pred, roles = p
            objh = _head_entity(roles.get("Object") or roles.get("Theme") or roles.get("Patient") or "")
            if objh:
                out.add(objh)
        return out


    meta_by_new: Dict[int, Dict[str, Any]] = {}
    for m in meta:
        if m.get("mod") == "d":
            continue
        if isinstance(m.get("new"), int):
            meta_by_new[int(m["new"])] = m

    def _is_u_step(j: int) -> bool:
        mm = meta_by_new.get(j)
        return isinstance(mm, dict) and mm.get("mod") == "u"

    subs: List[Tuple[int, int, str, str, str]] = []
    # (new_idx, old_idx, eid, orig_obj_raw, cand_obj_raw)
    for m in meta:
        if m.get("mod") != "e":
            continue
        if (m.get("etype") or "").strip().lower() != "substitution":
            continue
        old = m.get("old")
        new = m.get("new")
        if not (isinstance(old, int) and isinstance(new, int)):
            continue
        if not (0 <= old < len(original_steps) and 0 <= new < len(final_steps)):
            continue
        eid = (m.get("eid") or "").strip()

        orig_sr = semrep_by_old.get(old) or ""
        cand_sr = semrep_by_new.get(new) or ""
        orig_parsed = parse_semrep_one(orig_sr) if orig_sr else None
        cand_parsed = parse_semrep_one(cand_sr) if cand_sr else None
        if not orig_parsed:
            continue
        op, oroles = orig_parsed
        if _action_from_semrep(op, oroles) != "GET":
            continue

        if cand_parsed:
            cp, croles = cand_parsed
            if _action_from_semrep(cp, croles) != "GET":
                continue
            cand_obj_raw = (croles.get("Object") or croles.get("Theme") or croles.get("Patient") or "")
        else:
            cand_sr2 = find_semrep_exact(final_steps[new]) or ""
            cand_parsed2 = parse_semrep_one(cand_sr2) if cand_sr2 else None
            if not cand_parsed2:
                continue
            cp, croles = cand_parsed2
            if _action_from_semrep(cp, croles) != "GET":
                continue
            cand_obj_raw = (croles.get("Object") or croles.get("Theme") or croles.get("Patient") or "")

        orig_obj_raw = (oroles.get("Object") or oroles.get("Theme") or oroles.get("Patient") or "")
        subs.append((new, old, eid, orig_obj_raw, cand_obj_raw))

    for new_idx, _old_idx, eid, orig_obj_raw, cand_obj_raw in subs:
        orig_obj = _head_entity(orig_obj_raw)
        cand_obj = _head_entity(cand_obj_raw)
        if not orig_obj or not cand_obj or orig_obj == cand_obj:
            continue

        # Guard: if the substituted GET acquires an entity already present earlier in the rewrite,
        # cascading will often create nonsense (e.g., tea_bag => tea).
        prev_ents = _prev_entities_in_rewrite(new_idx)
        already_present = (cand_obj in prev_ents)
        if not already_present:
            # fallback: raw text contains "tea bag" etc.
            cand_phrase = cand_obj.replace("_", " ")
            already_present = any(cand_phrase in (final_steps[i] or "").lower() for i in range(0, new_idx))

        if already_present and cand_obj != orig_obj:
            issues.append({
                "code": "SUBSTITUTION_TARGET_ALREADY_PRESENT",
                "step_index": new_idx,
                "step": final_steps[new_idx],
                "object": cand_obj,
                "eid": eid or None,
                "detail": f"GET-substitution target '{cand_obj}' already appears earlier in the rewrite; choose a different substitution target.",
            })
            # IMPORTANT: do NOT cascade from this substitution; it must be repaired first.
            continue

        # If the substituted GET acquires an entity that is already present earlier in the baseline,
        # cascading tends to create nonsense (e.g., sugar -> tea from tea_bag). Treat this as a
        # plausibility issue and let LLM repair pick a different substitution target.
        prev_ents = _baseline_entities_before(_old_idx)
        if cand_obj in prev_ents and cand_obj != orig_obj:
            issues.append({
                "code": "SUBSTITUTION_TARGET_ALREADY_PRESENT",
                "step_index": new_idx,
                "step": final_steps[new_idx],
                "object": cand_obj,
                "eid": eid or None,
                "detail": (
                    f"GET-substitution target '{cand_obj}' already appears earlier in the original take; "
                    "choose a different substitution target to avoid implausible cascades."
                ),
            })
            # Do NOT cascade from this substitution; it will be repaired upstream.
            continue

        # If the original entity (or its content-like alias) is explicitly acquired again later,
        # we stop cascading: the procedure can legitimately switch back to the original entity.
        orig_cands = _content_candidates(orig_obj)
        stop_ents = {orig_obj}
        stop_ents.update(orig_cands[:2])  # keep conservative to avoid spurious early stopping

        # Always map the actual acquired object.
        pairs: List[Tuple[str, str]] = [(orig_obj, cand_obj)]

        # Also map content-like aliases (sugar_bag -> sugar, salt_shaker -> salt, etc.)
        # This avoids hardcoding words like "sugar/salt" and works across *_container, *_bag, *_shaker, ...
        orig_cands = _content_candidates(orig_obj)
        cand_cands = _content_candidates(cand_obj)
        # Add a small number of top candidate pairs (keep it conservative to avoid weird mappings).
        for oc, cc in zip(orig_cands[:2], cand_cands[:2]):
            if oc and cc and oc != cc and (oc, cc) not in pairs:
                pairs.append((oc, cc))

        j_end = min(len(final_steps), new_idx + 1 + max(1, int(window)))
        for j in range(new_idx + 1, j_end):
            srj = semrep_by_new.get(j) or ""
            parsed_j = parse_semrep_one(srj) if srj else None
            if not parsed_j:
                continue
            pj, rj = parsed_j
            actj = _action_from_semrep(pj, rj)
            objh_j = _head_entity(rj.get("Object") or rj.get("Theme") or rj.get("Patient") or "")
            if actj == "GET" and objh_j and objh_j in stop_ents:
                break
            if not _is_u_step(j):
                continue

            instrh_j = _head_entity(rj.get("Instrument") or "")
            origh_j = _head_entity(rj.get("Origin") or "")
            coobjh_j = _head_entity(rj.get("Coobject") or "")
            desth_j = _head_entity(rj.get("Destination") or rj.get("Goal") or rj.get("Location") or "")

            for from_ent, to_ent in pairs:
                hit = (
                    (objh_j == from_ent) or (instrh_j == from_ent) or (origh_j == from_ent)
                    or (coobjh_j == from_ent) or (desth_j == from_ent)
                )
                if not hit:
                    continue
                issues.append({
                    "code": "SUBSTITUTION_CASCADE_ENTITY",
                    "step_index": j,
                    "step": final_steps[j],
                    "object": from_ent,
                    "from_ent": from_ent,
                    "to_ent": to_ent,
                    "eid": eid or None,
                    "detail": (
                        f"downstream 'u' step still references '{from_ent}' after GET-substitution "
                        f"(eid={eid or 'unknown'}) replaced it with '{to_ent}'"
                    ),
                })
    return issues

def _content_candidates(ent_head: str) -> List[str]:
    """
    Derive content-like aliases for container/content naming conventions without a domain lexicon.
    Examples:
      sugar_bag -> ["sugar"]
      milk_container -> ["milk"]
      cooking_oil_container -> ["cooking_oil", "oil", "cooking"]
      soy_sauce_bottle -> ["soy_sauce", "sauce", "soy"]

    Heuristics (generic):
      - treat the LAST underscore/dash token as packaging/type and avoid using it
      - prefer the compound core (all tokens except the last)
      - also include the token right before the last (often the "main" noun: oil in cooking_oil_container)
      - finally fall back to expand_entity_aliases(ent) excluding the last token
    """
    e = normalize_entity_token(ent_head or "")
    if not e:
        return []

    parts = [p for p in re.split(r"[_\-]+", e) if p]
    pack = parts[-1] if len(parts) >= 2 else ""
    core = "_".join(parts[:-1]) if len(parts) >= 2 else ""
    pre_last = parts[-2] if len(parts) >= 2 else ""

    out: List[str] = []
    def _add(x: str) -> None:
        x = normalize_entity_token(x)
        if x and x not in out and x != e and x != pack:
            out.append(x)

    # Prefer the compound core (sugar_bag -> sugar; cooking_oil_container -> cooking_oil).
    _add(core)
    # Also include the main noun right before packaging (cooking_oil_container -> oil).
    _add(pre_last)

    # Use existing alias expansion, but ignore packaging token.
    try:
        aliases = expand_entity_aliases(e)
    except Exception:
        aliases = set()
    for a in sorted(aliases, key=len, reverse=True):
        _add(a)

    return out

def _embedded_location_head(value: str) -> str:
    """
    Extract an embedded Location from role values like:
      - butter(Location: in(skillet))
      - butter(in(skillet))
    and return the *head* of that location (e.g., 'skillet').

    This is important because many Ego-Exo4D SemReps encode Location inside Object
    rather than as a top-level role (Destination/Location/Goal).
    """
    v = (value or "").strip()
    if not v:
        return ""

    # 1) Preferred: explicit "Location: ...".
    idx = v.find("Location:")
    loc_expr = ""
    if idx >= 0:
        tail = v[idx + len("Location:"):].strip()
        # Read until comma at top-level (respect parentheses nesting).
        buf: List[str] = []
        depth = 0
        for ch in tail:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            if ch == "," and depth == 0:
                break
            buf.append(ch)
        loc_expr = "".join(buf).strip()
        # Trim extra trailing ')' if present (common in "... in(skillet))").
        while loc_expr.count("(") < loc_expr.count(")"):
            loc_expr = loc_expr[:-1].rstrip()
    else:
        # 2) Fallback: wrapper-like location mention inside the object string.
        m = re.search(r"\b(?:in|on|inside|into|onto)\([^()]*\)", v)
        if m:
            loc_expr = m.group(0).strip()

    if not loc_expr:
        return ""
    return _head_entity(loc_expr)

def normalize_entity_token(tok: str) -> str:
    t = (tok or "").strip().lower()
    if not t:
        return ""
    # minimal plural -> singular (avoid overdoing)
    # (keeps 'glass' intact, etc.)
    if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
        t2 = t[:-1]
        # tiny exceptions (extend if needed)
        if t2 not in {"ga", "clas"}:
            return t2
    return t


def _action_from_semrep(pred: str, roles: Dict[str, str]) -> Optional[str]:
    p = (pred or "").strip().upper()
    manner = (roles.get("Manner") or "").lower()
    if p in {"GET","TAKE","GRAB","PICK","PICK_UP","RETRIEVE","FETCH","REMOVE"}:
        return "GET"
    if p in {"RETURN"}:
        return "PUT_BACK"
    if p in {"PUT","PLACE","SET"}:
        # semrep generator for "put ... back" tends to include Manner: back
        if "back" in manner or "away" in manner:
            return "PUT_BACK"
        return None
    if p in {"OPEN","UNSEAL","UNSCREW"}:
        return "OPEN"
    if p in {"CLOSE","SEAL","COVER"}:
        return "CLOSE"
    if p in {"WASH","RINSE","CLEAN"}:
        return "WASH"
    if p in {"DRY","WIPE","DRY_OFF"}:
        return "DRY"
    if p in {"ADD","POUR","SPRINKLE","MIX","STIR","COMBINE","INSERT","TRANSFER"}:
        return "ADD"
    if p in {"CUT","CHOP","DICE","SLICE","MINCE","PEEL"}:
        return "CUT"
    if p in {"DRAIN","STRAIN"}:
        return "OTHER"
    if p in {"DISPOSE","DISCARD","THROW","TOSS","TRASH"}:
        return "DISPOSE"
    return None


def classify_step(step_text: str, semrep: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify the action based STRICTLY on Semantic Representation.
    
    1. If 'semrep' is provided (from meta/plan), use it.
    2. If not, look up the text in the loaded SemRep JSON (Exact Match).
    3. If not found, return 'OTHER' (no heuristic guessing).
    """
    # 1. Resolve SemRep string
    sr_string = semrep
    if not sr_string:
        sr_string = find_semrep_exact(step_text)
    
    # If we still don't have a SemRep, we cannot safely classify strictly.
    # Returning default safe values.
    if not sr_string:
        return {"action": "OTHER", "objects": set(), "dests": set()}

    # 2. Parse the SemRep: PREDICATE(Role: val, ...)
    parsed = parse_semrep_one(sr_string)
    if not parsed:
        return {"action": "OTHER", "objects": set(), "dests": set()}

    pred, roles = parsed
    action = "OTHER"
    p_upper = pred.strip().upper()

    # 3. Map PREDICATE to Action Category
    # Based on the vocabulary found in Ego-Exo4D SemReps
    
    # GET / TAKE
    if p_upper in {"GET", "TAKE", "GRAB", "PICK", "PICK_UP", "RETRIEVE", "FETCH", "REMOVE", "COLLECT", "FISH_OUT"}:
        action = "GET"
        
    # PUT BACK / RETURN
    # "RETURN" is explicit. "PUT_AWAY" usually implies freeing hands.
    # "PUT" can be ambiguous, but often implies placing something down. 
    # Logic: If it's explicitly "PUT_BACK" or "RETURN", mark as PUT_BACK.
    elif p_upper in {"PUT_BACK", "RETURN", "PUT_AWAY", "REPLACE"}:
        action = "PUT_BACK"
    elif p_upper in {"PUT", "PLACE", "SET", "SET_DOWN", "POSITION"}:
        # Check manner/direction for "back"
        manner = (roles.get("Manner") or "").lower()
        direction = (roles.get("Direction") or "").lower()
        if "back" in manner or "back" in direction:
            action = "PUT_BACK"
        else:
            # Simple placement usually frees hands without implying "returned to original location".
            action = "PUT_DOWN"

    # OPEN / CLOSE
    elif p_upper in {"OPEN", "UNSEAL", "UNSCREW", "UNBOX", "UNWRAP", "UNCORK"}:
        action = "OPEN"
    elif p_upper in {"CLOSE", "SEAL", "COVER", "SCREW"}:
        action = "CLOSE"

    # WASH / DRY
    elif p_upper in {"WASH", "RINSE", "CLEAN", "SCRUB"}:
        action = "WASH"
    elif p_upper in {"DRY", "WIPE", "DRY_OFF"}:
        action = "DRY"

    # ADD / MIX / POUR
    elif p_upper in {"ADD", "POUR", "SPRINKLE", "MIX", "STIR", "COMBINE", "INSERT", "TRANSFER", "FILL", "DIP", "SPREAD", "WHISK", "BEAT", "CRACK"}:
        action = "ADD"

    # CUT
    elif p_upper in {"CUT", "CHOP", "DICE", "SLICE", "MINCE", "PEEL", "CUT_OFF", "CUT_OUT", "CARVE"}:
        action = "CUT"

    # DISPOSE
    elif p_upper in {"DISPOSE", "DISCARD", "THROW", "TOSS", "TRASH", "DUMP_OUT"}:
        action = "DISPOSE"
        
    # USE (General interaction)
    elif p_upper in {"USE", "OPERATE", "TURN_ON", "TURN_OFF", "PRESS", "CHECK", "ADJUST", "MEASURE", "KNEAD", "ROLL", "FLATTEN", "SQUEEZE"}:
        action = "USE"

    # 4. Extract Objects
    objs: Set[str] = set()
    dests: Set[str] = set()

    # Extract Primary Object
    # In your scheme: Object, Theme, Patient are main targets
    raw_obj = roles.get("Object") or roles.get("Theme") or roles.get("Patient")
    if raw_obj:
        head = _head_entity(raw_obj)
        if head: objs.add(head)

    # Extract Instrument (tools usually need to be held)
    raw_instr = roles.get("Instrument")
    if raw_instr:
        head = _head_entity(raw_instr)
        if head: objs.add(head)

    # Extract Destination (important for ADD/POUR logic logic: "Pour X into Y")
    raw_dest = roles.get("Destination") or roles.get("Coobject")
    if raw_dest:
        head = _head_entity(raw_dest)
        if head: dests.add(head)

    return {"action": action, "objects": objs, "dests": dests}

def primary_object_from_step(step_text: str, semrep: Optional[str] = None) -> str:
    """
    Extract the primary (non-tool) object for simple state checks.
    This intentionally ignores trailing "with <tool>" fragments.
    """
    if isinstance(semrep, str) and semrep.strip():
        parsed = parse_semrep_one(semrep)
        if parsed:
            _pred, roles = parsed
            obj = _head_entity(roles.get("Object") or roles.get("Theme") or roles.get("Patient") or "")
            # unwrap inside(of(cup)) etc already handled in _head_entity
            if obj:
                return obj
    s_ws = normalize_ws(normalize_step_text(step_text))
    s_clean = re.sub(r"[^a-z0-9\s]", " ", s_ws)
    parts = s_clean.split()

    if parts[:2] == ["pick", "up"]:
        parts = parts[2:]
    else:
        parts = parts[1:] if parts else []

    while parts and parts[0] in {"the", "a", "an"}:
        parts = parts[1:]

    stop = {"to", "from", "into", "onto", "with", "in", "on", "at", "and", "or", "using", "by"}
    obj_tokens: List[str] = []
    for t in parts:
        if t in stop:
            break
        obj_tokens.append(t)

    if not obj_tokens:
        return ""
    return normalize_object_key(" ".join(obj_tokens))


def looks_like_drain_step(s_ws: str) -> bool:
    """
    Heuristic: draining typically mentions sieve/colander/strainer, or starts with "drain".
    """
    if not s_ws:
        return False
    if s_ws.startswith("drain "):
        return True
    if "drain" in s_ws and any(x in s_ws for x in ("sieve", "colander", "strainer")):
        return True
    return False


def looks_like_cook_into_pot_or_water_step(s_ws: str) -> bool:
    """
    Heuristic: putting an ingredient into the pot/water (precondition for draining).
    """
    if not s_ws:
        return False

    verbs = ("add ", "put ", "place ", "transfer ", "pour ")
    if not any(s_ws.startswith(v) for v in verbs):
        # Also accept explicit "cook/boil" mentions in pot/water
        if s_ws.startswith(("cook ", "boil ")) and ("pot" in s_ws or "water" in s_ws):
            return True
        return False

    if ("pot" in s_ws or "water" in s_ws) and (" into " in f" {s_ws} " or " in " in f" {s_ws} "):
        return True

    if "boiling water" in s_ws:
        return True

    return False


def looks_like_stir_in_pot_step(s_ws: str) -> bool:
    """
    Heuristic: stirring explicitly in the pot implies the object is still in the pot.
    """
    if not s_ws:
        return False
    if s_ws.startswith("stir ") and (" in the pot" in s_ws or " in pot" in s_ws):
        return True
    return False

# -------------------------
# Substance/contents normalization (for container-state plausibility)
# -------------------------
GENERIC_SUBSTANCE_WORDS = {
    "mixture", "mix",
    "contents", "content",
    "batter", "dough", "paste",
    "solution", "liquid",
    "ingredients", "stuff", "material",
}

def substance_tokens(ent: str) -> List[str]:
    """
    Convert entity text like 'egg_mixture' / 'egg mixture' / 'contents of bowl'
    into a set of meaningful tokens (dropping generic substance words).
    """
    s = (ent or "").lower().replace("_", " ")
    toks = re.findall(r"[a-z0-9]+", s)
    # tiny plural normalization
    toks = [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in toks]
    meaningful = [t for t in toks if t not in GENERIC_SUBSTANCE_WORDS]
    return meaningful or toks

def same_substance(a: str, b: str) -> bool:
    """
    Treat 'egg' and 'egg mixture' as the same substance (after dropping generic words).
    Strong rule: token-subset match.
    """
    ta, tb = set(substance_tokens(a)), set(substance_tokens(b))
    if not ta or not tb:
        return False
    if ta.issubset(tb) or tb.issubset(ta):
        return True
    # fallback (still on meaningful tokens)
    j = len(ta & tb) / len(ta | tb)
    return j >= 0.67

def looks_like_back_transfer(s_ws: str) -> bool:
    """
    Heuristic for "pour/transfer ... back into ..." style steps.
    """
    if not s_ws:
        return False
    s = f" {s_ws} "
    if " back " not in s and not s_ws.startswith("back "):
        return False
    # focus on transfer verbs
    return s_ws.startswith(("pour ", "transfer ", "put ", "place ", "return "))

# -------------------------
# Baseline from original (we assume original is logical)
# -------------------------
def build_original_baseline(original_steps: List[str], semrep_by_old: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """
    Derive:
      first_get[obj] = first index where obj is GET
      ambient = objects used without ever being GET in original
      best_get_text[obj] = exact original GET step text for that object
      portable = objects that appear in GET steps (treated as 'needs holding' for USE/ADD/CUT/Open etc)
      first_cook[obj] = first index where obj is placed/cooked in pot/water (heuristic)
      best_cook_text[obj] = original cook-into-pot/water step text for that object
    """
    first_get: Dict[str, int] = {}
    best_get_text: Dict[str, str] = {}
    ever_get: Set[str] = set()
    ever_use: Set[str] = set()

    first_cook: Dict[str, int] = {}
    best_cook_text: Dict[str, str] = {}

    # --- NEW: derive "heat surfaces" + "storage locations" from ORIGINAL semrep ---
    heat_surfaces: Set[str] = set()
    storage_locations: Set[str] = set()
    # edges from placement steps: (obj_head, dest_head, text)
    place_edges: List[Tuple[str, str, str]] = []
    best_place_on_heat: Dict[str, str] = {}

    def _pred_roles_from_sr(sr: str) -> Optional[Tuple[str, Dict[str, str]]]:
        if not isinstance(sr, str) or not sr.strip():
            return None
        return parse_semrep_one(sr.strip())

    HEAT_PREDS = {
        "HEAT","BOIL","COOK","SIMMER",
        "ADJUST","REGULATE",
        "TURN_ON","TURN_OFF",
    }
    PLACE_PREDS = {"PLACE","PUT","SET","SET_DOWN","POSITION"}
    STORAGE_PREDS = {"PUT_AWAY","RETURN","PUT_BACK","REPLACE"}

    for i, t in enumerate(original_steps):
        sem = semrep_by_old.get(i) if semrep_by_old else None
        c = classify_step(t, semrep=sem)
        action = c["action"]
        objs = c["objects"]
        if action == "GET":
            for o in objs:
                if o and o not in first_get:
                    first_get[o] = i
                    best_get_text[o] = t
                ever_get.add(o)
        if action in {"USE","ADD","CUT","OPEN","CLOSE","WASH","DRY","PUT_BACK"}:
            for o in objs:
                if o:
                    ever_use.add(o)

        # --- semrep-driven baseline location semantics ---
        pr = _pred_roles_from_sr(sem or "")
        if pr:
            pred, roles = pr
            pu = (pred or "").strip().upper()
            raw_obj = (roles.get("Object") or roles.get("Theme") or roles.get("Patient") or "")
            objh = _head_entity(raw_obj)
            desth = _head_entity(roles.get("Destination") or roles.get("Location") or roles.get("Goal") or "")

            if pu in HEAT_PREDS:
                # Whatever the ORIGINAL mentions as Location/Destination for heat-like predicates
                # becomes "heat surface" in THIS take (no global assumptions).
                if desth:
                    heat_surfaces.add(desth)

            if pu in STORAGE_PREDS:
                if desth:
                    storage_locations.add(desth)

            if pu in PLACE_PREDS and objh and desth:
                place_edges.append((objh, desth, t))

        # Track "cook into pot/water" baseline (needed for plausibility repairs)
        s_ws = normalize_ws(normalize_step_text(t))
        if looks_like_cook_into_pot_or_water_step(s_ws):
            obj0 = primary_object_from_step(t, semrep=sem)
            if obj0 and obj0 not in first_cook:
                first_cook[obj0] = i
                best_cook_text[obj0] = t

    # choose best placement-to-heat step per object (based on ORIGINAL heat_surfaces)
    for objh, desth, txt in place_edges:
        if desth in heat_surfaces and objh not in best_place_on_heat:
            best_place_on_heat[objh] = txt

    ambient = set([o for o in ever_use if o not in ever_get])
    portable = set([o for o in ever_get if o])

    return {
        "first_get": first_get,
        "ambient": ambient,
        "best_get_text": best_get_text,
        "portable": portable,
        "first_cook": first_cook,
        "best_cook_text": best_cook_text,
        "heat_surfaces": heat_surfaces,
        "storage_locations": storage_locations,
        "best_place_on_heat": best_place_on_heat,
    }


# -------------------------
# Meta normalization + alignment logic (as agreed)
# -------------------------
def normalize_meta(meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in meta:
        if not isinstance(m, dict):
            continue
        mm = dict(m)
        # Key warranty
        mm.setdefault("old", "")
        mm.setdefault("new", "")
        mm.setdefault("mod", "u")     
        mm.setdefault("etype", None)
        mm.setdefault("eid", None)
        mm.setdefault("cid", None)

        # Corrections ('c') are NEW steps (like insertions): they do not exist in ORIGINAL.
        # We enforce old="" (and NEVER an int old) to avoid breaking old-index coverage.
        if str(mm.get("mod") or "").strip() == "c":
            mm["old"] = ""

        for k in ("old", "new"):
            if mm.get(k) is None:
                mm[k] = ""
            if isinstance(mm.get(k), str) and mm[k].strip() == "":
                mm[k] = ""
        for k in ("etype", "eid", "cid"):
            if mm.get(k) == "":
                mm[k] = None
        if isinstance(mm.get("etype"), str):
            mm["etype"] = mm["etype"].strip().lower() or None
        if isinstance(mm.get("eid"), str):
            mm["eid"] = mm["eid"].strip() or None
        if isinstance(mm.get("cid"), str):
            mm["cid"] = mm["cid"].strip() or None
        out.append(mm)
    return out

def dedupe_corrections_in_place(meta: List[Dict[str, Any]]) -> None:
    """
    Ensure each correction cid appears exactly once.
    If duplicates exist, keep the first as mod='c' and convert the rest to insertion mod='i'
    (with eid='E_UNKNOWN' if missing). This makes plan_coverage pass.
    Also enforces old="" for corrections.
    """
    seen: Set[str] = set()
    for m in meta:
        if not isinstance(m, dict):
            continue
        mod = str(m.get("mod") or "").strip()
        if mod != "c":
            continue
        cid = str(m.get("cid") or "").strip()
        # correction must be a new step
        m["old"] = ""
        if not cid:
            continue
        if cid in seen:
            # convert duplicate correction into insertion
            m["mod"] = "i"
            m["etype"] = None
            m["cid"] = None
            if not (isinstance(m.get("eid"), str) and str(m.get("eid")).strip()):
                m["eid"] = "E_UNKNOWN"
        else:
            seen.add(cid)

def force_realize_transposition(final_steps, meta, eid, src, tgt):
    pos = {}
    idx_in_meta = {}
    for k, m in enumerate(meta):
        if (
            m.get("eid") == eid
            and m.get("mod") in {"ms","mt"}
            and isinstance(m.get("old"), int)
            and isinstance(m.get("new"), int)
        ):
            pos[int(m["old"])] = int(m["new"])
            idx_in_meta[int(m["new"])] = k

    if set(pos.keys()) != {src, tgt}:
        return False

    a, b = pos[src], pos[tgt]
    if a == b:
        return False

    # Guard against short final_steps
    if not (0 <= a < len(final_steps) and 0 <= b < len(final_steps)):
        return False

    final_steps[a], final_steps[b] = final_steps[b], final_steps[a]

    # swap blocks in meta timeline by new indices (preserve deletions adjacency)
    _meta_swap_blocks_by_new(meta, a, b)
    canonicalize_meta_new_indices_in_place(meta)
    return True

def meta_non_del_count(meta: List[Dict[str, Any]]) -> int:
    return sum(1 for m in meta if m.get("mod") != "d")

def canonicalize_meta_new_indices_in_place(meta: List[Dict[str, Any]]) -> None:
    """
    Renumber meta.new sequentially in meta order (excluding deletions).
    Does NOT invent/mutate final_steps length (final_steps is handled separately).
    """
    new_i = 0
    for m in meta:
        if m.get("mod") == "d":
            m["new"] = ""
            continue
        m["new"] = new_i
        new_i += 1

def _meta_pos_for_new(meta: List[Dict[str, Any]], new_idx: int) -> Optional[int]:
    for k, m in enumerate(meta):
        if m.get("mod") == "d":
            continue
        if isinstance(m.get("new"), int) and int(m["new"]) == int(new_idx):
            return k
    return None
def _meta_block_end(meta: List[Dict[str, Any]], start: int) -> int:
    """
    Block = head non-deletion meta entry + all immediately-following deletions.
    This preserves: 'deletion stands right after the previous step'.
    Returns slice end (exclusive).
    """
    end = start + 1
    while end < len(meta) and meta[end].get("mod") == "d":
        end += 1
    return end
def _meta_swap_blocks_by_new(meta: List[Dict[str, Any]], a_new: int, b_new: int) -> bool:
    pa = _meta_pos_for_new(meta, a_new)
    pb = _meta_pos_for_new(meta, b_new)
    if pa is None or pb is None or pa == pb:
        return False
    if pa > pb:
        pa, pb = pb, pa
        a_new, b_new = b_new, a_new
    qa = _meta_block_end(meta, pa)
    qb = _meta_block_end(meta, pb)
    block_a = meta[pa:qa]
    mid = meta[qa:pb]  # typically empty, but keep robust
    block_b = meta[pb:qb]
    meta[pa:qb] = block_b + mid + block_a
    return True
def _meta_insert_before_new(meta: List[Dict[str, Any]], new_idx: int, m_new: Dict[str, Any]) -> None:
    p = _meta_pos_for_new(meta, new_idx)
    if p is None:
        # insert at end (after last block)
        meta.append(m_new)
    else:
        meta.insert(p, m_new)
def _meta_remove_block_by_new(meta: List[Dict[str, Any]], new_idx: int) -> bool:
    """
    Remove the non-deletion head with new=new_idx.
    If it has trailing deletions, re-attach them to the previous block to preserve ordering.
    """
    p = _meta_pos_for_new(meta, new_idx)
    if p is None:
        return False
    q = _meta_block_end(meta, p)
    trailing_dels = [x for x in meta[p+1:q] if x.get("mod") == "d"]
    del meta[p:q]
    if trailing_dels:
        # attach to previous block (at position p, which is now "after previous block")
        if p <= 0:
            meta[0:0] = trailing_dels
        else:
            meta[p:p] = trailing_dels
    return True

def enforce_verbatim_for_u_and_moves(original_steps: List[str], final_steps: List[str], meta: List[Dict[str, Any]]) -> None:
    for m in meta:
        if m.get("mod") not in {"u", "ms", "mt"}:
            continue
        old = m.get("old")
        new = m.get("new")
        if isinstance(old, int) and isinstance(new, int) and 0 <= old < len(original_steps) and 0 <= new < len(final_steps):
            final_steps[new] = original_steps[old]

def align_final_to_meta_length(final_steps: List[str], meta: List[Dict[str, Any]]) -> List[str]:
    """
    Our agreed default priority:
      plan -> meta -> final_steps.
    So we *truncate* extra final_steps that meta doesn't describe.
    If meta expects MORE steps than exist in final_steps, we keep final_steps as-is and
    let deterministic/LLM repair handle it (we do not invent empty steps).
    """
    need = meta_non_del_count(meta)
    if need <= 0:
        return final_steps
    if len(final_steps) > need:
        del final_steps[need:]
    return final_steps


# -------------------------
# Schema validation + plan coverage
# -------------------------
def validate_rewrite_schema(original_steps: List[str], final_steps: List[str], meta: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    new_indices: List[int] = []
    move_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ms": 0, "mt": 0})

    # Corrections must not be empty filler (Continue/Proceed/etc.)
    BAD_CORR_PREFIXES = (
        "continue", "proceed", "go on", "carry on", "keep going",
        "then continue", "next", "move on"
    )

    def _nonempty(x: Any) -> bool:
        return isinstance(x, str) and x.strip() != ""

    for i, m in enumerate(meta):
        if m.get("mod") not in ALLOWED_MOD:
            issues.append(f"meta[{i}].mod invalid: {m.get('mod')}")
            continue
        mod = m.get("mod")
        et = m.get("etype")
        et_s = str(et).strip().lower() if isinstance(et, str) and et.strip() else None

        # Enforce etype consistency (writer-style strictness).
        if mod == "i":
            if et_s != "insertion":
                issues.append(f"meta[{i}] mod='i' requires etype='insertion'")
        elif mod == "d":
            if et_s not in {None, "deletion"}:
                issues.append(f"meta[{i}] mod='d' requires etype='deletion' or null")
        elif mod in {"ms","mt"}:
            if et_s not in {None, "transposition"}:
                issues.append(f"meta[{i}] mod='{mod}' requires etype='transposition' or null")
        elif mod == "e":
            if et_s not in {"substitution", "wrong_execution"}:
                issues.append(f"meta[{i}] mod='e' requires etype in {{substitution,wrong_execution}}")
        elif mod in {"u","a"}:
            if et_s is not None:
                issues.append(f"meta[{i}] mod='{mod}' requires etype=null")
        elif mod == "c":
            # Allow both legacy (null) and explicit 'correction' label.
            if et_s not in {None, "correction"}:
                issues.append(f"meta[{i}] mod='c' requires etype null or 'correction'")

        # Allow insertion without eid ONLY if it's an explicit repair insertion.
        is_repair_insertion = (mod == "i" and _nonempty(m.get("repair_reason")))
        if mod in {"e", "a", "i", "ms", "mt", "d"} and not _nonempty(m.get("eid")) and not is_repair_insertion:
            issues.append(f"meta[{i}] mod='{mod}' requires non-empty eid")

        if mod == "c" and not _nonempty(m.get("cid")):
            issues.append(f"meta[{i}] mod='c' requires non-empty cid")

        if mod == "c":
            # corrections are NEW steps; they must not consume an old index
            if m.get("old") not in ("", None):
                issues.append(f"meta[{i}] mod='c' must have old='' (correction is a new step)")

        if mod == "d":
            if m.get("new") not in ("", None):
                issues.append(f"meta[{i}] deletion must have new=''")
            if not isinstance(m.get("old"), int):
                issues.append(f"meta[{i}] deletion must have int old")

        if mod in {"ms", "mt"}:
            eid = m.get("eid")
            if _nonempty(eid):
                move_counts[str(eid).strip()][mod] += 1
            old = m.get("old")
            new = m.get("new")
            if not (isinstance(old, int) and isinstance(new, int)):
                issues.append(f"meta[{i}] {mod} must have int old and int new")
            else:
                if 0 <= old < len(original_steps) and 0 <= new < len(final_steps):
                    if normalize_ws(final_steps[new]) != normalize_ws(original_steps[old]):
                        issues.append(f"meta[{i}] {mod} must be verbatim ORIGINAL[{old}]")
                else:
                    issues.append(f"meta[{i}] {mod} old/new out of range")

        newv = m.get("new")
        if newv == "" or newv is None:
            continue
        if not isinstance(newv, int):
            issues.append(f"meta[{i}].new must be int or ''")
            continue
        new_indices.append(newv)
        if newv < 0 or newv >= len(final_steps):
            issues.append(f"meta[{i}].new out of range: {newv}")

        # Filler correction check (must reference a real step and be non-trivial).
        if mod == "c" and isinstance(newv, int) and 0 <= newv < len(final_steps):
            txt = normalize_ws(final_steps[newv]).lower()
            if len(txt.split()) <= 2 or txt.startswith(BAD_CORR_PREFIXES):
                issues.append(f"correction at final_steps[{newv}] looks like filler: '{final_steps[newv]}'")

    # Explicit structural check: number of non-deletion meta entries must equal final_steps length.
    non_del = [m for m in meta if m.get("mod") in ALLOWED_MOD and m.get("mod") != "d"]
    if len(non_del) != len(final_steps):
        issues.append(f"meta/final_steps length mismatch: meta_non_del={len(non_del)} vs len(final_steps)={len(final_steps)}")

    if sorted(new_indices) != list(range(len(final_steps))):
        issues.append("meta.new indices must cover 0..len(final_steps)-1 exactly once")

    for eid, cts in move_counts.items():
        if cts["ms"] != 1 or cts["mt"] != 1:
            issues.append(f"transposition eid={eid} must have exactly one ms and one mt (ms={cts['ms']} mt={cts['mt']})")

    # strict 'u'
    for i, m in enumerate(meta):
        if m.get("mod") != "u":
            continue
        old = m.get("old")
        new = m.get("new")
        if not (isinstance(old, int) and isinstance(new, int)):
            issues.append(f"meta[{i}] u must have int old and int new")
            continue
        if not (0 <= old < len(original_steps)) or not (0 <= new < len(final_steps)):
            issues.append(f"meta[{i}] u indices out of range")
            continue
        if normalize_ws(final_steps[new]) != normalize_ws(original_steps[old]):
            issues.append(f"meta[{i}] u mismatch ORIGINAL[{old}]")
    return issues

def validate_plan_coverage(take: Dict[str, Any], meta: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    errors = take.get("errors", []) or []
    corrs = take.get("corrections", []) or []

    out_eids: Set[str] = set()
    corr_counts: Dict[str, int] = defaultdict(int)
    msmt_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ms": 0, "mt": 0})

    for m in meta:
        mod = m.get("mod")
        eid = m.get("eid")
        cid = m.get("cid")
        if mod in {"e", "a", "i", "ms", "mt", "d"} and isinstance(eid, str) and eid.strip():
            out_eids.add(eid.strip())
        if mod == "c" and isinstance(cid, str) and cid.strip():
            corr_counts[cid.strip()] += 1
        if mod in {"ms", "mt"} and isinstance(eid, str) and eid.strip():
            msmt_counts[eid.strip()][mod] += 1

    for e in errors:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        if not eid:
            continue
        if eid not in out_eids:
            issues.append(f"plan_coverage: missing error realization eid={eid}")

        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        if etype == "transposition":
            cts = msmt_counts.get(eid, {"ms": 0, "mt": 0})
            if cts.get("ms", 0) != 1 or cts.get("mt", 0) != 1:
                issues.append(
                    f"plan_coverage: transposition eid={eid} must have exactly one ms and one mt "
                    f"(ms={cts.get('ms',0)} mt={cts.get('mt',0)})"
                )

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


def validate_transposition_realized(take: Dict[str, Any], meta: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    errors = take.get("errors", []) or []

    # index eid -> (src, tgt)
    trans: List[Tuple[str, int, int]] = []
    for e in errors:
        if not isinstance(e, dict):
            continue
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        if etype != "transposition":
            continue
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        if not eid:
            continue
        src = e.get("step_index") if isinstance(e.get("step_index"), int) else e.get("src_step_idx")
        spec = e.get("spec") if isinstance(e.get("spec"), dict) else {}
        tgt = spec.get("transposition_target") if isinstance(spec.get("transposition_target"), int) else e.get("target_step_idx")
        if not (isinstance(src, int) and isinstance(tgt, int)):
            continue
        trans.append((eid, src, tgt))

    # build helper: old->new for this eid
    for eid, src, tgt in trans:
        msmt = [m for m in meta if m.get("eid") == eid and m.get("mod") in {"ms", "mt"}]
        if len(msmt) != 2:
            continue  # Already covered by plan coverage check.

        # map old -> new
        old_to_new: Dict[int, int] = {}
        for m in msmt:
            if isinstance(m.get("old"), int) and isinstance(m.get("new"), int):
                old_to_new[int(m["old"])] = int(m["new"])

        if set(old_to_new.keys()) != {src, tgt}:
            # Allow intentional plan override, but still require a real swap of some pair.
            has_override = any(bool(m.get("plan_override")) for m in msmt)
            if not has_override:
                issues.append(f"transposition eid={eid} has ms/mt old={sorted(old_to_new.keys())}, expected {{{src},{tgt}}}")
                continue
            olds = sorted(old_to_new.keys())
            if len(olds) != 2:
                continue
            src2, tgt2 = olds[0], olds[1]
            if not (old_to_new[src2] > old_to_new[tgt2]):
                issues.append(
                    f"transposition eid={eid} override NOT realized: new[{old_to_new[src2]}] should be after new[{old_to_new[tgt2]}]"
                )
            continue

        # REQUIRE reversal
        if src < tgt and not (old_to_new[src] > old_to_new[tgt]):
            issues.append(
                f"transposition eid={eid} NOT realized: original order old[{src}]<old[{tgt}] "
                f"but final keeps order new[{old_to_new[src]}] < new[{old_to_new[tgt]}]"
            )
        if src > tgt and not (old_to_new[src] < old_to_new[tgt]):
            issues.append(
                f"transposition eid={eid} NOT realized: original order old[{src}]>old[{tgt}] "
                f"but final keeps wrong order"
            )

    return issues

def iter_plan_transpositions(take: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for e in (take.get("errors") or []):
        if not isinstance(e, dict):
            continue
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        if etype != "transposition":
            continue
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        if not eid:
            continue
        src = e.get("step_index") if isinstance(e.get("step_index"), int) else e.get("src_step_idx")
        spec = e.get("spec") if isinstance(e.get("spec"), dict) else {}
        tgt = spec.get("transposition_target") if isinstance(spec.get("transposition_target"), int) else e.get("target_step_idx")
        if isinstance(src, int) and isinstance(tgt, int):
            out.append((eid, int(src), int(tgt)))
    return out

def plan_insertion_text_for_eid(take: Dict[str, Any], eid: str) -> Optional[str]:
    for e in (take.get("errors") or []):
        if not isinstance(e, dict):
            continue
        eeid = str(e.get("event_id") or e.get("error_id") or "").strip()
        if eeid != eid:
            continue
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        if etype != "insertion":
            continue

        for k in ("step_description", "txt", "insert_step_description", "inserted_step", "insert_step"):
            v = e.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        spec = e.get("spec")
        if isinstance(spec, dict):
            for k in ("insert_step_description", "inserted_step", "insert_step"):
                v = spec.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    return None

def _new_index_of_old(meta: List[Dict[str, Any]], old_idx: int) -> Optional[int]:
    for m in meta:
        if m.get("mod") == "d":
            continue
        if m.get("old") == old_idx and isinstance(m.get("new"), int):
            return int(m["new"])
    return None

def validate_and_repair_insertions_from_plan(
    take: Dict[str, Any],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    semrep_by_new: Optional[Dict[int, str]] = None,
    jaccard_min: float = 0.70,
) -> Tuple[bool, List[str]]:
    """
    For each planned insertion error:
      - ensure there is meta mod='i' for that eid
      - ensure inserted step text matches plan (or is sufficiently similar)
    Deterministic repair:
      - if missing: insert planned text at position corresponding to src (insert_before src)
      - if mismatch: replace final_steps at that insertion with planned text
    """
    issues: List[str] = []
    changed = False

    # map eid -> planned insert text + src
    planned: List[Tuple[str, int, str]] = []
    for e in (take.get("errors") or []):
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        if etype != "insertion":
            continue
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        src = e.get("step_index") if isinstance(e.get("step_index"), int) else e.get("src_step_idx")
        if not eid or not isinstance(src, int):
            continue
        txt = plan_insertion_text_for_eid(take, eid) or ""
        if txt.strip():
            planned.append((eid, int(src), txt.strip()))

    if not planned:
        return False, []

    # index existing insertion metas by eid
    ins_meta_by_eid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in meta:
        if m.get("mod") == "i" and isinstance(m.get("eid"), str) and m["eid"].strip():
            ins_meta_by_eid[m["eid"].strip()].append(m)

    for eid, src, ptxt in planned:
        lst = ins_meta_by_eid.get(eid, [])
        if not lst:
            # missing insertion -> insert before src (best-effort mapping old->new)
            pos = _new_index_of_old(meta, src)
            if pos is None:
                pos = len(final_steps)
            template = {"old": "", "new": pos, "mod": "i", "etype": "insertion", "eid": eid, "cid": None}
            _insert_step(final_steps, meta, semrep_by_new, pos, ptxt, template)
            changed = True
            issues.append(f"INSERTION_MISSING: inserted planned insertion eid={eid} before src(old={src}) at new={pos}")
            continue

        # check first insertion realization for this eid (plan expects 1)
        m0 = lst[0]
        newi = m0.get("new")
        if not isinstance(newi, int) or not (0 <= newi < len(final_steps)):
            issues.append(f"INSERTION_BAD_META: eid={eid} has invalid new index")
            continue

        cand = final_steps[newi]
        sim = jaccard_similarity(normalize_ws(cand).lower(), normalize_ws(ptxt).lower())
        if sim < jaccard_min:
            # replace text deterministically
            final_steps[newi] = ptxt
            changed = True
            issues.append(f"INSERTION_TEXT_MISMATCH: eid={eid} jaccard={sim:.2f} replaced with plan text")

    if changed:
        canonicalize_meta_new_indices_in_place(meta)
        final_steps[:] = align_final_to_meta_length(final_steps, meta)
    return changed, issues

def validate_old_index_coverage(original_steps: List[str], meta: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    n = len(original_steps)
    seen: Dict[int, int] = defaultdict(int)
    for i, m in enumerate(meta):
        mod = m.get("mod")
        old = m.get("old")
        # insertions AND corrections do not consume old indices
        if mod in {"i", "c"}:
            continue
        if mod == "d":
            # deletions DO cover old indices (the step is accounted for)
            if not isinstance(old, int):
                issues.append(f"old_coverage: meta[{i}] deletion old must be int")
            else:
                seen[int(old)] += 1
            continue
        if isinstance(old, int):
            seen[old] += 1
        else:
            issues.append(f"old_coverage: meta[{i}] mod={mod} requires int old")

    for k in range(n):
        c = seen.get(k, 0)
        if c == 0:
            issues.append(f"old_coverage: missing old index {k}")
        elif c > 1:
            issues.append(f"old_coverage: old index {k} appears {c} times")
    return issues

def validate_planned_errors_still_realized(
    take: Dict[str, Any],
    original_steps: List[str],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
) -> List[str]:
    issues: List[str] = []
    errors = take.get("errors") or []

    # helper: find meta entries for eid
    def metas_for_eid(eid: str) -> List[Dict[str, Any]]:
        return [m for m in meta if (m.get("eid") or "").strip() == eid]

    for e in errors:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        src = e.get("step_index") if isinstance(e.get("step_index"), int) else e.get("src_step_idx")
        if not eid or not isinstance(src, int):
            continue

        ms = metas_for_eid(eid)
        if not ms:
            issues.append(f"planned_error_missing: eid={eid} not present in meta")
            continue

        if etype == "deletion":
            # must have a deletion meta with old=src
            alt = e.get("alternate_src_indices") or []
            alt_set = {int(x) for x in alt if isinstance(x, int)}
            dels = [
                m for m in ms
                if m.get("mod") == "d"
                and (m.get("new") in {"", None})
                and (
                    m.get("old") == src
                    or (isinstance(m.get("old"), int) and m.get("old") in alt_set)
                    or bool(m.get("plan_override"))
                )
            ]
            if not dels:
                issues.append(f"planned_error_not_realized: deletion eid={eid} must have meta d old={src} new=''")
            # and must NOT have any non-deletion referencing the same old
            # Only enforce "old=src must disappear" when the deletion was realized on src without override.
            realized_on_src = any(
                m.get("mod") == "d" and m.get("old") == src and not bool(m.get("plan_override"))
                for m in ms
            )
            if realized_on_src:
                kept = [m for m in meta if m.get("old") == src and m.get("mod") != "d"]
                if kept:
                    issues.append(f"planned_error_healed: deletion eid={eid} but old={src} still used in meta")

        elif etype == "substitution":
            subs = [m for m in ms if m.get("mod") == "e" and m.get("old") == src and isinstance(m.get("new"), int)]
            if not subs:
                issues.append(f"planned_error_not_realized: substitution eid={eid} needs meta e old={src}")
            else:
                m0 = subs[0]
                new = int(m0["new"])
                if 0 <= src < len(original_steps) and 0 <= new < len(final_steps):
                    if normalize_ws(final_steps[new]) == normalize_ws(original_steps[src]):
                        issues.append(f"planned_error_healed: substitution eid={eid} but final equals original at old={src}")

        elif etype == "insertion":
            ins = [m for m in ms if m.get("mod") == "i" and (m.get("old") in {"", None}) and isinstance(m.get("new"), int)]
            if not ins:
                issues.append(f"planned_error_not_realized: insertion eid={eid} needs meta i old=''")

        elif etype == "transposition":
            # already handled elsewhere
            pass

    return issues

# -------------------------
# Plausibility checks (core)
# -------------------------
def plausibility_issues(
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    original_baseline: Dict[str, Any],
    semrep_by_new: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of issue dicts:
      {code, step_index, step, detail, object?}
    """
    issues: List[Dict[str, Any]] = []

    ambient: Set[str] = set(original_baseline.get("ambient") or set())
    first_get: Dict[str, int] = dict(original_baseline.get("first_get") or {})
    best_get_text: Dict[str, str] = dict(original_baseline.get("best_get_text") or {})
    portable: Set[str] = set(original_baseline.get("portable") or set())

    # Track substance locations in containers to catch nonsense like:
    # "Pour the egg mixture back into the mixing bowl" when it's already there.
    container_contents: Dict[str, Set[str]] = defaultdict(set)
    substance_loc: Dict[str, str] = {}
    held: Set[str] = set()
    introduced: Set[str] = set()
    filled: Set[str] = set()
    last_get_idx: Dict[str, int] = {}

    # --- semrep-derived location state ---
    obj_loc: Dict[str, str] = {}
    heat_surfaces: Set[str] = set(original_baseline.get("heat_surfaces") or set())
    # taint: location state changed by non-'u' step (ms/mt/e/a/i/c)
    tainted_loc: Set[str] = set()

    HEAT_PREDS = {
        "HEAT","BOIL","COOK","SIMMER",
        "ADJUST","REGULATE",
        "TURN_ON","TURN_OFF",
    }
    PLACE_PREDS = {"PLACE","PUT","SET","SET_DOWN","POSITION"}
    STORAGE_PREDS = {"PUT_AWAY","RETURN","PUT_BACK","REPLACE"}

    def _pred_roles(step_text: str, semrep: Optional[str]) -> Optional[Tuple[str, Dict[str, str]]]:
        sr = semrep or find_semrep_exact(step_text) or ""
        if not sr:
            return None
        return parse_semrep_one(sr)

    # If model-changed steps alter "holding" state (GET/PUT_BACK), downstream verbatim 'u' steps
    # can become physically impossible. We track those objects and allow flagging even on 'u'.
    tainted_by_changed_state: Set[str] = set()

    # dispose tracking (for USE_AFTER_DISPOSE)
    disposed: Set[str] = set()
    disposed_at: Dict[str, int] = {}
    disposed_changed: Dict[str, bool] = {}

    # State for "in pot/water" vs "drained" ordering checks
    in_pot_or_water: Dict[str, bool] = {}
    drained: Dict[str, bool] = {}
    last_drain_idx: Dict[str, int] = {}

    # Avoid spamming the same logical violation for the same object repeatedly
    flagged_drain_before_cook: Set[str] = set()
    flagged_stir_after_drain: Set[str] = set()

    def _pick_one_dest(dests: Any) -> str:
        if not isinstance(dests, set) or not dests:
            return ""
        ds = sorted([d for d in dests if isinstance(d, str) and d.strip()])
        return ds[0] if ds else ""

    def _canon_substance_in_dest(sub_raw: str, dest: str) -> str:
        """
        Canonicalize a substance name against what we already believe is in a container.
        Example: sub_raw='egg mixture', existing in dest={'egg'} -> returns 'egg'.
        """
        if not isinstance(sub_raw, str) or not sub_raw.strip():
            return ""
        sr = sub_raw.strip()
        if dest:
            for ex in container_contents.get(dest, set()):
                if same_substance(sr, ex):
                    return ex
        # global unique match
        matches = [ex for ex in substance_loc.keys() if same_substance(sr, ex)]
        if len(matches) == 1:
            return matches[0]
        return sr

    meta_by_new: Dict[int, Dict[str, Any]] = {}
    for m in meta:
        if m.get("mod") == "d":
            continue
        if isinstance(m.get("new"), int):
            meta_by_new[int(m["new"])] = m
    def is_model_changed(i: int) -> bool:
        m = meta_by_new.get(int(i))
        if not isinstance(m, dict):
            return True
        return m.get("mod") != "u"

    def _canon_substance_simple(ent: str) -> str:
        toks = substance_tokens(ent)
        return toks[0] if toks else ""

    def _has_back(step_text: str, semrep: Optional[str]) -> bool:
        # text cue
        s = normalize_ws(normalize_step_text(step_text))
        if " back " in f" {s} ":
            return True
        # semrep cue (preferred when present)
        sr = semrep or find_semrep_exact(step_text) or ""
        parsed = parse_semrep_one(sr) if sr else None
        if parsed:
            _pred, roles = parsed
            md = ((roles.get("Manner") or "") + " " + (roles.get("Direction") or "")).lower()
            if "back" in md:
                return True
        return False

    def should_flag(i: int, obj: str) -> bool:
        # Flag if the current step is model-changed OR if this object state was changed earlier
        # by a model-changed holding-state step (GET/PUT_BACK), making a later 'u' step impossible.
        return is_model_changed(i) or (obj in tainted_by_changed_state)

    for i, step in enumerate(final_steps):
        sem = semrep_by_new.get(i) if semrep_by_new else None
        c = classify_step(step, semrep=sem)
        action = c["action"]
        objs = set([o for o in c["objects"] if o])

        s_ws = normalize_ws(normalize_step_text(step))
        obj0 = primary_object_from_step(step, semrep=sem)

        # --- location tracking + heat-while-in-storage check (SemRep-only) ---
        pr = _pred_roles(step, sem)
        if pr:
            pred, roles = pr
            pu = (pred or "").strip().upper()
            objh = _head_entity(roles.get("Object") or roles.get("Theme") or roles.get("Patient") or "")
            desth = _head_entity(roles.get("Destination") or roles.get("Location") or roles.get("Goal") or "")

            if pu in PLACE_PREDS and objh and desth:
                obj_loc[objh] = desth
                if is_model_changed(i):
                    tainted_loc.add(objh)                

            if pu in STORAGE_PREDS and objh and desth:
                obj_loc[objh] = desth
                if is_model_changed(i):
                    tainted_loc.add(objh)

            if pu in HEAT_PREDS and objh:
                loc = obj_loc.get(objh, "")
                # Flag even on 'u' heat-step if the object's location was changed by the model earlier
                if loc and heat_surfaces and (loc not in heat_surfaces):
                    if is_model_changed(i) or (objh in tainted_loc):
                        issues.append({
                            "code": "HEAT_WHILE_NOT_ON_HEAT_SURFACE",
                            "step_index": i,
                            "step": step,
                            "object": objh,
                            "detail": f"heat/adjust on '{objh}' while it is located at non-heat location '{loc}' (heat surfaces from ORIGINAL)",
                            "location": loc,
                        })

        # ---------- "BACK INTO SAME CONTAINER" check ----------
        # If we say "pour/transfer back into X" but the substance is already in X (never left),
        # that's physically nonsensical wording.
        # We only flag when the step is model-changed (so we don't punish verbatim originals).
        if action == "ADD":
            dests = c.get("dests") or set()
            dest1 = _pick_one_dest(dests)

            sub_raw = obj0 or (sorted(list(objs))[0] if objs else "")
            sub = ""
            if dest1 and sub_raw:
                sub = _canon_substance_in_dest(sub_raw, dest1)
            if not sub and sub_raw:
                sub = _canon_substance_simple(sub_raw)

            if dest1 and sub:
                if _has_back(step, sem) and substance_loc.get(sub) == dest1:
                    if is_model_changed(i):
                        issues.append({
                            "code": "BACK_TO_SAME_CONTAINER",
                            "step_index": i,
                            "step": step,
                            "object": sub,
                            "detail": f"'{sub}' is already in '{dest1}', so 'back into {dest1}' is nonsensical",
                        })
                prev = substance_loc.get(sub)
                if prev and prev != dest1:
                    container_contents[prev].discard(sub)
                container_contents[dest1].add(sub)
                substance_loc[sub] = dest1
        # ------------------------------------------------------

        # If something was disposed, using/checking it later is implausible unless reintroduced.
        # IMPORTANT: we flag even if the later step is 'u', as long as the disposal step was model-changed.
        if action == "DISPOSE" and obj0:
            disposed.add(obj0)
            disposed_at[obj0] = i
            disposed_changed[obj0] = bool(disposed_changed.get(obj0, False) or is_model_changed(i))

        # Reintroduction resets disposal state (e.g., "get a new test plate")
        if action == "GET" and obj0 and obj0 in disposed:
            disposed.remove(obj0)
            disposed_at.pop(obj0, None)
            disposed_changed.pop(obj0, None)

        if obj0 and obj0 in disposed and action in {"USE","ADD","OPEN","CLOSE","CUT","WASH","DRY","OTHER"}:
            if is_model_changed(i) or disposed_changed.get(obj0, False):
                issues.append({
                    "code": "USE_AFTER_DISPOSE",
                    "step_index": i,
                    "step": step,
                    "object": obj0,
                    "detail": f"'{obj0}' is used after it was disposed at step {disposed_at.get(obj0)}",
                })

        # Mark "object placed/cooked in pot/water"
        if obj0 and looks_like_cook_into_pot_or_water_step(s_ws):
            in_pot_or_water[obj0] = True
            drained[obj0] = False
        # Drain must happen only after object has been in pot/water
        if obj0 and looks_like_drain_step(s_ws):
            if not in_pot_or_water.get(obj0, False):
                if is_model_changed(i) and obj0 not in flagged_drain_before_cook:
                    issues.append({
                        "code": "DRAIN_BEFORE_COOK",
                        "step_index": i,
                        "step": step,
                        "object": obj0,
                        "detail": f"drain happens before '{obj0}' was ever placed/cooked in pot/water",
                    })
                    flagged_drain_before_cook.add(obj0)

            # Update state only for actual drain steps
            drained[obj0] = True
            in_pot_or_water[obj0] = False
            last_drain_idx[obj0] = i

        # Stirring "in the pot" after draining is inconsistent unless the object was put back into pot
        if obj0 and looks_like_stir_in_pot_step(s_ws):
            if drained.get(obj0, False):
                if is_model_changed(i) and obj0 not in flagged_stir_after_drain:
                    issues.append({
                        "code": "STIR_IN_POT_AFTER_DRAIN",
                        "step_index": i,
                        "step": step,
                        "object": obj0,
                        "detail": f"stirring '{obj0}' in the pot happens after it was drained",
                        "drain_index": last_drain_idx.get(obj0),
                    })
                    flagged_stir_after_drain.add(obj0)
            elif not in_pot_or_water.get(obj0, False):
                if should_flag(i, obj0):
                    issues.append({
                        "code": "STIR_IN_POT_NOT_IN_POT",
                        "step_index": i,
                        "step": step,
                        "object": obj0,
                        "detail": f"stirring '{obj0}' in the pot happens before it was placed/cooked in pot/water",
                    })

        # mark intro by GET
        if action == "GET":
            for o in objs:
                if not o:
                    continue

                # duplicate GET without any put-back in between
                if o in portable:
                    if o in held:
                        prev = last_get_idx.get(o)

                        # Flag if either the current GET or the previous GET was model-changed.
                        # This catches cases where an erroneous inserted/substituted GET creates
                        # a duplicate against a later verbatim 'u' step.
                        prev_changed = isinstance(prev, int) and is_model_changed(prev)
                        cur_changed = is_model_changed(i)

                        if cur_changed or prev_changed:
                            issues.append({
                                "code": "DUP_GET_NO_PUT",
                                "step_index": i,
                                "step": step,
                                "object": o,
                                "detail": (
                                    f"duplicate GET/PICK of '{o}' while it's already held "
                                    f"(no put back / set aside in between)"
                                ),
                                "prev_index": prev,
                            })

                    held.add(o)

                introduced.add(o)
                last_get_idx[o] = i

                # A model-changed GET affects state; keep a taint marker for downstream consistency.
                if is_model_changed(i):
                    tainted_by_changed_state.add(o)

        # Put-back / put-down => unhold portable (hands become free).
        if action in {"PUT_BACK", "PUT_DOWN"}:
            for o in objs:
                if o in held:
                    held.remove(o)
                # If a hold-state change was introduced/changed by the model, it can break later verbatim 'u' uses.
                if is_model_changed(i) and o:
                    tainted_by_changed_state.add(o)

        # OPEN: if object is portable, should be held OR ambient OR introduced already
        if action == "OPEN":
            for o in objs:
                if o in ambient:
                    continue
                if o in portable and o not in held:
                    # only flag if it looks like model created a new contradiction
                    if should_flag(i, o):
                        issues.append({
                            "code": "TOOL_NOT_HELD",
                            "step_index": i,
                            "step": step,
                            "object": o,
                            "detail": f"object '{o}' opened but not held (missing GET or should not have been put back)",
                        })

        # USE-ish actions: object should not appear "too late"
        if action in {"USE", "ADD", "CUT", "DRY", "WASH", "OPEN", "CLOSE"}:
            for o in objs:
                if not o:
                    continue
                if o in ambient:
                    continue
                # introduced?
                if o not in introduced and o in first_get:
                    # original required GET at some point; now we used before ever getting
                    if should_flag(i, o):
                        issues.append({
                            "code": "USE_BEFORE_INTRO",
                            "step_index": i,
                            "step": step,
                            "object": o,
                            "detail": f"object used before introduction (original had a GET for it): '{o}'",
                        })
                # portable needs holding for use/add/cut/dry/open/close (not for wash necessarily)
                if o in portable and action in {"USE","ADD","CUT","DRY","OPEN","CLOSE"}:
                    if o not in held and o not in ambient:
                        if should_flag(i, o):
                            issues.append({
                                "code": "PORTABLE_NOT_HELD",
                                "step_index": i,
                                "step": step,
                                "object": o,
                                "detail": f"portable item used but not held (missing GET or put back immediately before): '{o}'",
                            })

        # Detect GET-after-use (if object is used before, then later GET shows up)
        # We flag only when the GET is model-changed or when it creates obvious contradiction.
        if action == "GET":
            for o in objs:
                if not o:
                    continue
                # if already introduced earlier, repeated GET can be ok,
                # but if we saw a USE issue for same object before, repair tries to move GET earlier
                pass

        # DRY after fill contradiction (cup cleaned after coffee poured etc.)
        if action == "ADD":
            # crude: consider first object as "target container" if mentioned
            # Prefer semrep Destination if available, else fallback to objects
            dests = c.get("dests") or set()
            if isinstance(dests, set) and dests:
                for d in dests:
                    if d:
                        filled.add(d)
            else:
                for o in objs:
                    if o:
                        filled.add(o)

        if action == "DRY":
            for o in objs:
                if o and o in filled:
                    if should_flag(i, o):
                        issues.append({
                            "code": "DRY_AFTER_FILL",
                            "step_index": i,
                            "step": step,
                            "object": o,
                            "detail": f"dry/wipe happens after filling the same container/object: '{o}'",
                        })

        # PUT_BACK before use: if model added PUT_BACK then immediately uses same portable without GET
        if action in {"USE","ADD","CUT","OPEN","CLOSE"}:
            for o in objs:
                if o in portable and o not in held and o not in ambient:
                    # if there exists a near GET soon after, the sequence is likely swapped
                    # report a dedicated code for deterministic swap
                    if should_flag(i, o):
                        issues.append({
                            "code": "PUT_BACK_BEFORE_USE",
                            "step_index": i,
                            "step": step,
                            "object": o,
                            "detail": f"object '{o}' appears required here but is not held (may need to move nearby GET earlier)",
                        })

    # De-duplicate issues (same code+step+object)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for it in issues:
        key = (it.get("code"), it.get("step_index"), it.get("object"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq

def validate_location_continuity_semrep(
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    semrep_by_new: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    SemRep-only location continuity validation.

    Goals:
    - Track object -> location updates using Destination/Location roles.
    - Detect contradictions when a step claims the object is at/comes from a location
      that conflicts with tracked state.

    Notes:
    - We only flag issues when the current step is model-changed (mod != 'u')
      OR when the object's location state was changed earlier by a model-changed step.
    - Destination is treated as a location update (Destination -> Location equivalence).
    """
    issues: List[Dict[str, Any]] = []

    meta_by_new: Dict[int, Dict[str, Any]] = {}
    for m in meta:
        if m.get("mod") == "d":
            continue
        if isinstance(m.get("new"), int):
            meta_by_new[int(m["new"])] = m

    def is_model_changed(i: int) -> bool:
        mm = meta_by_new.get(int(i))
        if not isinstance(mm, dict):
            return True
        return mm.get("mod") != "u"

    def sr_for_step(i: int, txt: str) -> str:
        if semrep_by_new is not None and isinstance(semrep_by_new.get(i), str) and semrep_by_new.get(i).strip():
            return semrep_by_new[i].strip()
        sr = find_semrep_exact(txt) or ""
        return sr.strip()

    # object head -> last known location head
    obj_loc: Dict[str, str] = {}
    # objects whose location state has been affected by model-changed steps
    tainted_loc: Set[str] = set()

    # Predicates that imply a location update (place/put/transfer/insert/etc.)
    LOC_UPDATE_PREDS = {
        "PLACE", "PUT", "SET", "SET_DOWN", "POSITION",
        "INSERT", "TRANSFER", "MOVE", "STORE",
        "RETURN", "PUT_BACK", "PUT_AWAY", "REPLACE",
    }
    # Predicates that can encode an explicit source/location claim for GET/REMOVE.
    GET_PREDS = {"GET", "TAKE", "GRAB", "PICK", "PICK_UP", "RETRIEVE", "FETCH", "REMOVE"}

    def _extract_obj_loc(parsed: Tuple[str, Dict[str, str]]) -> Tuple[str, str, str]:
        pred, roles = parsed
        # Cache the raw object string once; we also use it to extract embedded Location.
        raw_obj = (roles.get("Object") or roles.get("Theme") or roles.get("Patient") or "")
        objh = _head_entity(raw_obj)
        # Treat Destination and Location as interchangeable "target/location" cues.
        loch = _head_entity(
            roles.get("Destination")
            or roles.get("Location")
            or roles.get("Goal")
            or ""
        )
        # Many SemReps encode Location inside Object (e.g., butter(Location: in(skillet))).
        if not loch and raw_obj:
            loch = _embedded_location_head(raw_obj)
            
        # Common "source" cues for GET-like actions.
        srch = _head_entity(
            roles.get("Source")
            or roles.get("Origin")
            or roles.get("Initial_Location")
            or roles.get("Location")
            or ""
        )
        return objh, loch, srch

    for i, step in enumerate(final_steps):
        sr = sr_for_step(i, step)
        parsed = parse_semrep_one(sr) if sr else None
        if not parsed:
            continue

        pred, roles = parsed
        pu = (pred or "").strip().upper()
        objh, loch, srch = _extract_obj_loc(parsed)
        if not objh:
            continue

        # Location update: Destination/Location assigns a new place for the object.
        if pu in LOC_UPDATE_PREDS and loch:
            obj_loc[objh] = loch
            if is_model_changed(i):
                tainted_loc.add(objh)
            continue

        # GET-like step: if it claims a Source/Location, it must match tracked location.
        if pu in GET_PREDS and srch:
            known = obj_loc.get(objh, "")
            if known and known != srch:
                if is_model_changed(i) or (objh in tainted_loc):
                    issues.append({
                        "code": "LOCATION_MISMATCH_ON_GET",
                        "step_index": i,
                        "step": step,
                        "object": objh,
                        "detail": f"SemRep claims '{objh}' comes from '{srch}', but tracked location is '{known}'",
                        "expected_location": known,
                        "claimed_location": srch,
                    })

        # Explicit Location mention on a non-update, non-GET predicate:
        # - If it contradicts tracked state, flag it (only when model-changed or tainted).
        # - Also "soft-update" tracked location from this mention to enable cascade fixes
        #   like: CUT on cutting_board -> later STIR expects in(skillet).
        if loch and pu not in LOC_UPDATE_PREDS and pu not in GET_PREDS:
            known = obj_loc.get(objh, "")
            if known and known != loch:
                if is_model_changed(i) or (objh in tainted_loc):
                    issues.append({
                        "code": "LOCATION_MISMATCH",
                        "step_index": i,
                        "step": step,
                        "object": objh,
                        "detail": f"SemRep mentions location '{loch}' for '{objh}', but tracked location is '{known}'",
                        "expected_location": known,
                        "claimed_location": loch,
                    })
            # Soft update (current claim becomes the new tracked location).
            if not known or known != loch:
                obj_loc[objh] = loch
                # If this step was model-changed, downstream verbatim 'u' steps may become impossible.
                if is_model_changed(i):
                    tainted_loc.add(objh)
    # De-duplicate by (code, step_index, object)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for it in issues:
        key = (it.get("code"), it.get("step_index"), it.get("object"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq

# -------------------------
# Deterministic repair (minimal local edits)
# -------------------------
def _swap_adjacent(final_steps: List[str], meta: List[Dict[str, Any]], semrep_by_new: Optional[Dict[int, str]], i: int) -> None:
    final_steps[i], final_steps[i+1] = final_steps[i+1], final_steps[i]
    # meta is a timeline (may include deletions). Swap BLOCKS by new index.
    _meta_swap_blocks_by_new(meta, i, i + 1)
    canonicalize_meta_new_indices_in_place(meta)
    if semrep_by_new is not None:
        si, sj = semrep_by_new.get(i), semrep_by_new.get(i+1)
        if si is None and sj is None:
            return
        semrep_by_new[i], semrep_by_new[i+1] = sj, si

def _insert_step(final_steps: List[str], meta: List[Dict[str, Any]], semrep_by_new: Optional[Dict[int, str]],
                 idx: int, step_text: str, template_meta: Dict[str, Any]) -> None:
    final_steps.insert(idx, step_text)
    # Insert as a new step BEFORE the block that currently has new==idx.
    _meta_insert_before_new(meta, idx, dict(template_meta))
    if semrep_by_new is not None:
        # shift right
        for k in sorted(list(semrep_by_new.keys()), reverse=True):
            if k >= idx:
                semrep_by_new[k+1] = semrep_by_new.pop(k)
        semrep_by_new[idx] = ""
    canonicalize_meta_new_indices_in_place(meta)
    # do NOT reorder meta; it is a timeline

def _delete_step(final_steps: List[str], meta: List[Dict[str, Any]], semrep_by_new: Optional[Dict[int, str]], idx: int) -> None:
    del final_steps[idx]
    # Remove the block for new==idx; keep any trailing deletions attached to previous block.
    _meta_remove_block_by_new(meta, idx)
    if semrep_by_new is not None:
        if idx in semrep_by_new:
            del semrep_by_new[idx]
        # shift left
        for k in sorted(list(semrep_by_new.keys())):
            if k > idx:
                semrep_by_new[k-1] = semrep_by_new.pop(k)
    canonicalize_meta_new_indices_in_place(meta)

def deterministic_repair(
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
    original_baseline: Dict[str, Any],
    semrep_by_new: Optional[Dict[int, str]] = None,
    take: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[Dict[str, Any]], bool]:
    """
    Try to fix common local contradictions without LLM:
      - PUT_BACK_BEFORE_USE: swap nearby GET earlier
      - USE_BEFORE_INTRO: insert best original GET for that object (as 'a' cascade under nearest eid)
      - DRY_AFTER_FILL: delete that dry step if it's model-changed
      - STIR_IN_POT_NOT_IN_POT / DRAIN_BEFORE_COOK: ensure the baseline cook-into-pot step happens
        before any stir/drain, WITHOUT deleting the planned substitution step.
    Returns (final_steps, meta, changed?)
    """
    changed = False
    best_get_text: Dict[str, str] = dict(original_baseline.get("best_get_text") or {})
    portable: Set[str] = set(original_baseline.get("portable") or set())
    best_cook_text: Dict[str, str] = dict(original_baseline.get("best_cook_text") or {})
    first_cook: Dict[str, int] = dict(original_baseline.get("first_cook") or {})
    best_place_on_heat: Dict[str, str] = dict(original_baseline.get("best_place_on_heat") or {})

    # helper: choose eid to attach to inserted cascade step
    def choose_eid_near(i_new: int) -> Optional[str]:
        """
        Choose a nearby eid in NEW-index space (final_steps indices).
        meta is a TIMELINE, so meta positions do not align with new indices.
        """
        # 1) exact step
        m0 = _find_meta_by_new(meta, i_new)
        if isinstance(m0, dict):
            eid0 = m0.get("eid")
            if isinstance(eid0, str) and eid0.strip():
                return eid0.strip()

        # 2) search neighbors by new-index distance
        for d in (1, 2, 3):
            for j_new in (i_new - d, i_new + d):
                if j_new < 0:
                    continue
                mj = _find_meta_by_new(meta, j_new)
                if isinstance(mj, dict):
                    eid = mj.get("eid")
                    if isinstance(eid, str) and eid.strip():
                        return eid.strip()
        return None

    def _make_nonverbatim_variant(orig: str) -> str:
        """
        Create a cook-like variant that will NOT equal the original after normalize_ws,
        so a planned substitution does not get "healed".
        """
        base = (orig or "").rstrip().rstrip(".")
        cand = base + " in the pot."
        if normalize_ws(cand).lower() == normalize_ws(orig).lower():
            cand = base + " in the pot, quickly."
        return cand

    def _new_index_of_old(old_idx: int) -> Optional[int]:
        for m in meta:
            if m.get("mod") == "d":
                continue
            if m.get("old") == old_idx and isinstance(m.get("new"), int):
                return int(m["new"])
        return None

    def _earliest_need_index_for_obj(obj: str) -> Optional[int]:
        """
        Find earliest index where obj is stirred in pot OR drained.
        We want cook-step to happen before that, even if those steps are 'u'.
        """
        obj = (obj or "").strip()
        if not obj:
            return None
        best: Optional[int] = None
        for k, st in enumerate(final_steps):
            s_ws = normalize_ws(normalize_step_text(st))
            o0 = primary_object_from_step(st)
            if o0 != obj:
                continue
            if looks_like_stir_in_pot_step(s_ws) or looks_like_drain_step(s_ws):
                best = k if best is None else min(best, k)
        return best

    def _ensure_cook_before_need(obj: str) -> bool:
        """
        Deterministically ensure that the baseline cook-into-pot step for obj occurs
        before the earliest stir/drain for that obj.

        Strategy:
          1) Identify baseline cook old-index for obj (first_cook[obj]).
          2) Find where that old-index currently sits in the candidate (meta index).
          3) Move it up (adjacent swaps) to just before earliest need.
          4) If that step is a planned substitution and not cook-like, rewrite it
             to a cook-like NON-verbatim variant of the original cook step.

        This avoids deleting the planned substitution and repairs STIR/DRAIN state.
        """
        if not obj or obj not in first_cook or obj not in best_cook_text:
            return False

        cook_old = int(first_cook[obj])
        cook_txt = str(best_cook_text[obj])
        pos = _new_index_of_old(cook_old)
        need = _earliest_need_index_for_obj(obj)
        if need is None:
            return False

        moved = False
        if pos is not None and pos > need:
            # bubble cook step upward to position 'need'
            for k in range(pos - 1, need - 1, -1):
                _swap_adjacent(final_steps, meta, semrep_by_new, k)
            moved = True

        # Recompute current position after moves
        pos2 = _new_index_of_old(cook_old)
        if pos2 is None:
            return moved

        # If the cook step is a planned substitution, keep it cook-like but non-verbatim.
        m2 = _find_meta_by_new(meta, pos2) or {}
        if (m2.get("mod") == "e") and (str(m2.get("etype") or "").strip().lower() == "substitution"):
            # If it's not already cook-like, rewrite it.
            s_ws = normalize_ws(normalize_step_text(final_steps[pos2]))
            if not looks_like_cook_into_pot_or_water_step(s_ws):
                final_steps[pos2] = _make_nonverbatim_variant(cook_txt)
                moved = True

        return moved

    # Iterate issues in order; apply small fixes and recompute later externally
    for it in issues:
        code = it.get("code")
        i = int(it.get("step_index"))
        obj = it.get("object") or ""

        if code in {"STIR_IN_POT_NOT_IN_POT", "DRAIN_BEFORE_COOK"} and isinstance(obj, str) and obj:
            if _ensure_cook_before_need(obj):
                changed = True
                continue

        if code == "BACK_TO_SAME_CONTAINER":
            # Minimal fix: remove "back" wording but keep the wrong destination (planned error may rely on it).
            if 0 <= i < len(final_steps):
                mi = _find_meta_by_new(meta, i)
                if isinstance(mi, dict) and mi.get("mod") != "u":
                    before = final_steps[i]
                    after = re.sub(r"\bback\s+(into|in|to)\b", r"\1", before, flags=re.IGNORECASE)
                    after = re.sub(r"\s+", " ", after).strip()
                    if after and after != before:
                        final_steps[i] = after
                        changed = True
            continue

        if code == "SUBSTITUTION_CASCADE_ENTITY":
            # Deterministically apply a minimal cascade:
            # - rewrite the downstream 'u' step by swapping the referenced entity
            # - mark it as a cascade ('a') under the same eid
            if 0 <= i < len(final_steps):
                from_ent = (it.get("from_ent") or "").strip()
                to_ent = (it.get("to_ent") or "").strip()
                eid = it.get("eid")

                before = final_steps[i]
                after = _replace_entity_surface(before, from_ent, to_ent)
                if after and after != before:
                    final_steps[i] = after
                    mi = _find_meta_by_new(meta, i)
                    # Only convert truly unchanged steps; don't override other planned edits.
                    if isinstance(mi, dict) and mi.get("mod") == "u":
                        mi["mod"] = "a"
                        mi["etype"] = None
                        if isinstance(eid, str) and eid.strip():
                            mi["eid"] = eid.strip()
                    if semrep_by_new is not None:
                        semrep_by_new[i] = ""
                    changed = True
            continue

        if code == "DRAIN_BEFORE_COOK" and isinstance(obj, str) and obj:
            # Try to pull up a nearby "put/add/transfer <obj> into pot/water" step from the following steps.
            found = None
            for j in range(i + 1, min(len(final_steps), i + 9)):
                s2 = normalize_ws(normalize_step_text(final_steps[j]))
                if not looks_like_cook_into_pot_or_water_step(s2):
                    continue
                if primary_object_from_step(final_steps[j]) == obj:
                    found = j
                    break

            if found is not None:
                for k in range(found - 1, i - 1, -1):
                    _swap_adjacent(final_steps, meta, semrep_by_new, k)
                    changed = True
                continue

        if code == "STIR_IN_POT_AFTER_DRAIN" and isinstance(obj, str) and obj:
            # If we know where the drain happened, push that drain step down after the current stir step.
            d = it.get("drain_index")
            if isinstance(d, int) and 0 <= d < i and d < len(final_steps):
                for k in range(d, i):
                    _swap_adjacent(final_steps, meta, semrep_by_new, k)
                    changed = True
                continue

        if code == "DUP_GET_NO_PUT" and isinstance(obj, str) and obj:
            prev = it.get("prev_index")
            if not isinstance(prev, int):
                continue
            if not (0 <= prev < len(final_steps) and 0 <= i < len(final_steps)):
                continue

            m_prev = _find_meta_by_new(meta, prev)

            replaced_insertion = False
            if (
                take is not None
                and m_prev is not None
                and m_prev.get("mod") == "i"
                and isinstance(m_prev.get("eid"), str)
                and m_prev["eid"].strip()
            ):
                eid0 = m_prev["eid"].strip()
                ins_txt = plan_insertion_text_for_eid(take, eid0)
                if isinstance(ins_txt, str) and ins_txt.strip():
                    ins_txt = ins_txt.strip()
                    if not is_near_duplicate_step(ins_txt, final_steps[prev]):
                        final_steps[prev] = ins_txt
                        replaced_insertion = True
                        changed = True

            if replaced_insertion:
                continue

            mi = _find_meta_by_new(meta, i)
            eid = choose_eid_near(i) or (mi.get("eid") if isinstance(mi, dict) else None)
            eid = eid.strip() if isinstance(eid, str) and eid.strip() else "E_UNKNOWN"

            # Try to put it back on the SAME surface it came from (countertop/cutting board/etc.)
            def _infer_surface(s: str) -> str:
                s2 = normalize_step_text(s)
                if "countertop" in s2:
                    return "countertop"
                if "cutting board" in s2:
                    return "cutting board"
                if "work table" in s2:
                    return "work table"
                if "table" in s2:
                    return "table"
                return ""

            surface = _infer_surface(final_steps[prev]) or _infer_surface(final_steps[i]) or "work table"
            put_txt = f"Put the {obj} back on the {surface}."
            # This is a physics repair insertion (NOT a planned error).
            template = {
                "old": "", "new": i, "mod": "i", "etype": "insertion",
                "eid": "", "cid": None,
                "repair_reason": "repair insertion: PUT_BACK_ON_SURFACE",
            }
            _insert_step(final_steps, meta, semrep_by_new, i, put_txt, template)
            changed = True
            continue

        if code == "PUT_BACK_BEFORE_USE" and obj in portable:
            # look ahead for a GET of the same object within next 3 steps; if found, bubble it up by swaps
            found = None
            for j in range(i+1, min(len(final_steps), i+4)):
                sem = semrep_by_new.get(j) if semrep_by_new else None
                cj = classify_step(final_steps[j], semrep=sem)
                if cj["action"] == "GET" and obj in cj["objects"]:
                    found = j
                    break
            if found is not None:
                for k in range(found-1, i-1, -1):
                    _swap_adjacent(final_steps, meta, semrep_by_new, k)
                    changed = True
                continue

        if code == "USE_BEFORE_INTRO" and isinstance(obj, str) and obj:
            # insert original GET for that object just before this step (as cascade 'a')
            get_txt = best_get_text.get(obj)
            if get_txt:
                mi = _find_meta_by_new(meta, i)
                eid = choose_eid_near(i) or (mi.get("eid") if isinstance(mi, dict) else None)
                eid = eid if isinstance(eid, str) and eid else (choose_eid_near(i) or "E_UNKNOWN")
                # This is a physics repair insertion (NOT a planned error).
                template = {
                    "old": "", "new": i, "mod": "i", "etype": "insertion",
                    "eid": "", "cid": None,
                    "repair_reason": "repair insertion: GET_BEFORE_USE",
                }
                _insert_step(final_steps, meta, semrep_by_new, i, get_txt, template)
                changed = True
                continue

        if code == "DRY_AFTER_FILL":
            # delete the step if model-changed (mod != 'u')
            mi = _find_meta_by_new(meta, i)
            if isinstance(mi, dict) and mi.get("mod") != "u":
                _delete_step(final_steps, meta, semrep_by_new, i)
                changed = True
                continue
 
        if code == "LOCATION_MISMATCH" and isinstance(obj, str) and obj:
            # Cascade fix: if a downstream step expects object at Loc_2 but we track Loc_1,
            # minimally adjust THIS step to include a transfer/move between locations.
            # This preserves the original error while keeping the overall procedure executable.
            if not (0 <= i < len(final_steps)):
                continue
            to_loc = it.get("claimed_location") or ""
            from_loc = it.get("expected_location") or ""
            if not isinstance(to_loc, str) or not to_loc.strip():
                continue

            mi = _find_meta_by_new(meta, i)
            if not isinstance(mi, dict):
                continue

            # Convert verbatim step to a cascade adjustment (mod='a') and attach the causing eid.
            if mi.get("mod") == "u":
                eid = choose_eid_near(i) or "E_UNKNOWN"
                mi["mod"] = "a"
                mi["eid"] = eid
                mi["etype"] = None  # schema requires etype=null for 'a'
                mi["repair_reason"] = "cascade adjustment: MOVE_OBJECT_TO_EXPECTED_LOCATION"
            else:
                # If already non-verbatim, still ensure an eid exists.
                if not (isinstance(mi.get("eid"), str) and mi.get("eid").strip()):
                    mi["eid"] = choose_eid_near(i) or "E_UNKNOWN"

            # Build a minimal "move" prefix and keep the original action after it.
            obj_txt = obj.replace("_", " ").strip()
            from_txt = (from_loc.replace("_", " ").strip() if isinstance(from_loc, str) else "")
            to_txt = to_loc.replace("_", " ").strip()

            prefix = ""
            if from_txt:
                prefix = f"Move the {obj_txt} from the {from_txt} to the {to_txt}, then "
            else:
                prefix = f"Move the {obj_txt} to the {to_txt}, then "

            orig = final_steps[i].strip()
            if orig:
                # make the continuation read naturally after "then"
                cont = orig[0].lower() + orig[1:] if orig[0].isupper() else orig
                new_txt = (prefix + cont).strip()
            else:
                new_txt = prefix.strip()

            # Avoid double-prefixing if we already injected a move here.
            if "Move the" not in orig[:30]:
                final_steps[i] = new_txt
                if semrep_by_new is not None:
                    semrep_by_new[i] = ""  # force refresh on next semrep recompute
                changed = True
            continue

        if code in {"HEAT_WHILE_IN_STORAGE_LOCATION","HEAT_WHILE_NOT_ON_HEAT_SURFACE"} and isinstance(obj, str) and obj:
            if 0 <= i <= len(final_steps):
                mi = _find_meta_by_new(meta, i)
                eid = choose_eid_near(i) or (mi.get("eid") if isinstance(mi, dict) else None)
                eid = eid.strip() if isinstance(eid, str) and eid.strip() else "E_UNKNOWN"

                # Prefer ORIGINAL placement-to-heat step for this object (SemRep-derived baseline)
                fix = best_place_on_heat.get(obj, "")
                if not fix:
                    fix = f"Place the {obj} back where it can be heated."

                template = {
                    "old": "", "new": i, "mod": "i", "etype": "insertion",
                    "eid": "", "cid": None,
                    "repair_reason": "repair insertion: PLACE_ON_HEAT_SURFACE",
                }
                _insert_step(final_steps, meta, semrep_by_new, i, fix, template)
                changed = True
            continue

    if changed:
        canonicalize_meta_new_indices_in_place(meta)
    return final_steps, meta, changed



# -------------------------
# LLM repair prompt (strengthened, with explicit examples)
# -------------------------
SYSTEM_REPAIR = """You are a strict judge+repairer for erroneous procedures grounded in the real world.

You receive:
0) SCENARIO / TASK GOAL (what the person is trying to accomplish; e.g., "Install a Wheel")
1) ORIGINAL steps (assumed logically valid in the real world)
2) ERROR+CORRECTION plan (which errors must exist and which corrections must exist)
3) Candidate rewrite: final_steps + meta (may contain mistakes)

Interpretation:
- The SCENARIO defines the intended real-world task and its required physical preconditions.
- The procedure may contain planned mistakes, BUT it must remain physically executable and logically coherent as a whole.
- You must preserve the planned errors/corrections, while ensuring the overall procedure still makes sense for the SCENARIO.

Global coherence rule (very important):
- Evaluate the *entire* sequence under the SCENARIO, not just isolated steps.
- If a planned error would break downstream feasibility (e.g., prevents later actions from being possible),
  adapt the error into a plausible mistake that preserves downstream meaning 
  or adjust the following steps so that the entire procedure sequence is plausible and feasible in real world.

Your job:
- Fix the candidate with MINIMAL edits so that the resulting procedure becomes physically plausible and coherent.
- Preserve the planned errors/corrections as much as possible (keep them near intended locations), but DO NOT keep an impossible sequence.

CRITICAL alignment rules (priority):
PLAN > META > FINAL_STEPS.
- If meta contradicts the plan, fix meta to match plan IF POSSIBLE.
- If final_steps contradict meta, fix final_steps to match meta (not vice versa), then update meta.new indices.

World/physics rules you MUST enforce:
A) Object introduction:
   - If an object is used (ADD/CUT/OPEN/POUR/MIX/etc.) it must not be "introduced" later.
   - A 'GET X' must NOT appear only AFTER we already 'ADD X' or 'OPEN X' or 'CUT X'.
   Example BAD:
     1) Add salt to the bowl
     2) Get salt from the cabinet
   Example GOOD:
     1) Get salt from the cabinet
     2) Add salt to the bowl

B) Container variants are the same object:
   - Treat "mayonnaise" and "mayonnaise jar" as the same identity unless context says otherwise.
   Example: If original had "Get mayonnaise from countertop", then "Open the mayonnaise jar" is allowed only if mayonnaise was already available.

C) Part-words:
   - Treat "tomato stalk" as "tomato", "cucumber blossom region" as "cucumber", etc.
   - If candidate replaced 'Get tomato...' with 'Get cucumber...', then later "Dice tomato" is invalid unless tomato is introduced again.
   Fix by either:
     - inserting 'Get tomato...' before tomato-use, OR
     - cascading the later tomato steps to cucumber (if that keeps procedure sensible), marked as mod='a' with the same eid.

D) Put-back / Get-back ordering:
   - If a step puts an item away and the next step needs it, a realistic correction can be:
     Put back X -> Get X back -> Use X
   (Not: Put back X -> Use X -> Get X back)

E) No nonsense after goal or impossible states:
   - Example: "pour coffee into cup" then "wipe the inside of the cup dry" is not plausible.
   Fix by moving wiping before pouring, or deleting the wipe step if it's an erroneous insertion.

Meta constraints you MUST satisfy:
- Output STRICTLY JSON with keys {"final_steps":[...], "meta":[...]} and NO extra keys.
- meta is timeline order; meta.new indices must cover 0..len(final_steps)-1 exactly once.
- 'u' steps must be verbatim copies of original_steps[old] (ignore whitespace only).
- deletions exist ONLY as meta entries with mod='d', new="".
- transposition must be encoded as exactly one 'ms' and one 'mt' with the same eid.

Optional meta fields (allowed by schema):
- repair_reason: string. Use ONLY for judge-created "physics repair" insertions (mod='i', eid="").
- plan_override: boolean. Set true ONLY when you intentionally deviate from the plan's indices/pairs
  to keep the overall procedure physically feasible (the deviation must still be reasonable).

Meta label semantics (STRICT):
- mod='u' means the step text MUST be a verbatim copy of ORIGINAL[old] (ignore whitespace only).
  If you need to change ANY non-whitespace character, you MUST NOT keep mod='u'.
- mod='e' is the PRIMARY planned error-realization step for its eid (e.g., substitution/wrong_execution/etc.).
- mod='a' is a CASCADE / FEASIBILITY / CONSISTENCY adjustment caused by some error eid.
  Use mod='a' when you minimally adjust a step to keep the whole procedure coherent after an earlier error.
  For mod='a', keep old as int (do not lose old-index coverage), and set eid to the CAUSING error id.

Consistency rule (general):
- If an earlier error changes an object/tool identity OR quantity OR variant (e.g., singular/plural, container naming),
  then any later step that refers to the outdated identity/quantity/variant MUST be adjusted for coherence.
  Those downstream fixes MUST be mod='a' with eid = the causing error id (do not rewrite the whole step).

Meta label semantics (IMPORTANT):
- mod='u': verbatim ORIGINAL[old]. If you need to change ANY token, you MUST NOT keep 'u'.
          Change mod to either 'e' (if this step is the primary planned error step for that eid)
          or 'a' (if this is a cascade/feasibility adjustment caused by some eid).
- mod='e': the primary planned error realization step (substitution/wrong_execution/etc.) for its eid.
- mod='a': cascade/feasibility adjustment. Use this for minimal context fixes caused by an earlier error,
           and ALWAYS set eid to the causing error id. Keep old as int to preserve old coverage.

Wrong-execution policy (etype="wrong_execution"):
- Keep the SAME main action intent and the SAME core arguments (Object and Destination/Target/Location) as in the ORIGINAL step,
  unless the plan explicitly requires changing them.
- Prefer a MANNER/DEGREE/QUALITY mistake that keeps downstream steps meaningful:
  examples of generic manner mistakes: misaligned, not fully seated, slightly loose, uneven, incomplete, partial, off-center.
- If the ORIGINAL semantic representation does not provide a Manner role, you MAY still introduce a small manner/degree mistake
  in the natural language step text (do not invent new objects).
- Do NOT replace a required placement/installation with a different spatial relation that breaks later steps.

Return ONLY the JSON object.
"""

REPAIR_OUTPUT_SCHEMA = {
  "type": "object",
  "properties": {
    "final_steps": {
      "type": "array",
      "items": {"type": "string"}
    },
    "meta": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "old": {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]},
          "new": {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]},
          "mod": {"type": "string"},
          "etype": {"anyOf": [{"type": "string"}, {"type": "null"}]},
          "eid": {"anyOf": [{"type": "string"}, {"type": "null"}]},
          "cid": {"anyOf": [{"type": "string"}, {"type": "null"}]},
          "repair_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
          "plan_override": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
        },
        "required": ["old", "new", "mod", "etype", "eid", "cid"],
        "additionalProperties": False
      }
    }
  },
  "required": ["final_steps", "meta"],
  "additionalProperties": False
}

def format_steps_for_prompt(steps: List[Dict[str, Any]]) -> str:
    out = []
    for s in steps:
        idx = _step_idx(s)
        phase = s.get("phase", "")
        txt = _step_text(s)
        out.append(f"{idx:02d} [{phase}] {txt}")
    return "\n".join(out)

def format_plan_for_prompt(errors: List[Dict[str, Any]], corrections: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    corr_by_eid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in corrections:
        eid = c.get("targets_error_id") or c.get("error_id") or ""
        if eid:
            corr_by_eid[str(eid)].append(c)

    for e in errors:
        eid = str(e.get("event_id") or e.get("error_id") or "").strip()
        etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
        src = e.get("step_index") if isinstance(e.get("step_index"), int) else e.get("src_step_idx")
        extra = []
        if etype == "transposition":
            spec = e.get("spec") if isinstance(e.get("spec"), dict) else {}
            tgt = spec.get("transposition_target")
            extra.append(f"target={tgt}")
        if etype == "insertion":
            extra.append("insert_before=src")
        alt = e.get("alternate_src_indices")
        if alt:
            extra.append(f"alt_del={alt}")

        out.append(f"{eid}: {etype} at src={src}" + (f" ({', '.join(extra)})" if extra else ""))

        for c in corr_by_eid.get(eid, []):
            cid = c.get("correction_id")
            ctype = c.get("correction_type")
            pos = c.get("detect_at_step_index") or c.get("after_step_idx")
            intent = c.get("intent") or ""
            if intent:
                out.append(f"  - {cid}: {ctype} after={pos} | intent: {intent}")
            else:
                out.append(f"  - {cid}: {ctype} after={pos}")

    return "\n".join(out)

def build_repair_user_prompt(
    take: Dict[str, Any],
    original_steps: List[str],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
    schema_issues: List[str],
    plaus_issues: List[Dict[str, Any]],
) -> str:

    scenario = str(take.get("scenario") or "").strip()
    scenario_block = f"SCENARIO:\n{scenario}\n\n" if scenario else ""

    steps_block = format_steps_for_prompt(take.get("steps") or [])
    plan_block = format_plan_for_prompt(take.get("errors") or [], take.get("corrections") or [])
    cand = json.dumps({"final_steps": final_steps, "meta": meta}, ensure_ascii=False, indent=2)

    issues_lines = []
    for s in schema_issues:
        issues_lines.append(f"- SCHEMA: {s}")
    for it in plaus_issues:
        issues_lines.append(f"- PLAUS({it.get('code')} @ {it.get('step_index')}): {it.get('detail')} | step='{it.get('step')}'")
    issues_block = "\n".join(issues_lines) if issues_lines else "(none)"

    # --- EXTRA GUIDANCE for plausibility: seasoning vs prep-actions ---
    afford_guidance = ""
    if any((it.get("code") == "AFFORDANCE_MISMATCH_BASELINE") for it in (plaus_issues or [])):
        afford_guidance = """
AFFORDANCE_MISMATCH_BASELINE guidance (IMPORTANT):
- If you see AFFORDANCE_MISMATCH_BASELINE issues:
  do NOT keep prep-actions (wash/chop/cut/slice/peel) on an entity that is only used as a seasoning in the ORIGINAL steps.
- Prefer changing the SUBSTITUTION target (still a substitution) to a prep-compatible ingredient,
  then re-apply cascades as mod="a" (same eid) to keep downstream steps consistent.
"""

    cascade_guidance = ""
    if any((it.get("code") in {"SUBSTITUTION_TARGET_ALREADY_PRESENT", "SUBSTITUTION_CASCADE_ENTITY"}) for it in (plaus_issues or [])):
        cascade_guidance = """
GET-SUBSTITUTION / CASCADE guidance (IMPORTANT):
- If you see SUBSTITUTION_TARGET_ALREADY_PRESENT:
  do NOT keep a GET-substitution whose target entity already appears earlier in the ORIGINAL steps
  or earlier in the current candidate rewrite. Change the substitution target to a different plausible item.
  (You must still keep it a substitution at the planned src step.)
- If you see SUBSTITUTION_CASCADE_ENTITY:
  downstream steps must refer to the substituted object explicitly (surface form), not to an inferred "content/core" word.
  Example BAD: get tea bag instead of sugar -> pour tea into milk
  Example GOOD: get honey instead of sugar -> pour honey into milk
- Prefer choosing a substitution target in the same functional slot as the original GET:
  ingredient -> ingredient, tool -> tool, so cascades remain noun swaps (no new concepts invented).
"""

    return f"""{scenario_block}ORIGINAL STEPS:

{steps_block}

ERROR+CORRECTION PLAN:
{plan_block}

CANDIDATE:
{cand}

ISSUES FOUND:
{issues_block}

{afford_guidance}
{cascade_guidance}

Repair minimally and output ONLY JSON with keys final_steps/meta.
"""

# -------------------------
# Main per-take judge
# -------------------------
def build_semrep_maps_for_take(
    take: Dict[str, Any],
    vocab_map: Optional[Dict[str, str]],
    semrep_map: Optional[Dict[str, Dict[str, str]]],
    semrep_step_to_id: Optional[Dict[str, str]],
    final_steps: List[str],
    meta: List[Dict[str, Any]],
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Returns:
      semrep_by_old: for original indices
      semrep_by_new: for final indices (aligned by meta old->new when possible)
    """

    semrep_by_old: Dict[int, str] = {}
    for s in (take.get("steps") or []):
        old = _step_idx(s)
        sr0 = s.get("semantic_representation")
        if isinstance(sr0, str) and sr0.strip():
            semrep_by_old[int(old)] = sr0.strip()
            continue
        txt = _step_text(s)
        sid = s.get("step_description_id")
        if (not isinstance(sid, str) or not sid.strip()) and vocab_map is not None:
            sid = resolve_step_id_from_text(txt, vocab_map, extra_map=semrep_step_to_id)
        if isinstance(sid, str) and sid.strip() and semrep_map is not None and sid.strip() in semrep_map:
            sr = semrep_map[sid.strip()].get("semantic_representation") or ""
            if sr:
                semrep_by_old[int(old)] = sr

    semrep_by_new: Dict[int, str] = {}
    # semrep for final steps:
    # - first try lookup by final step text (via semrep_step_to_id + semrep_map)
    # - then override for verbatim u/ms/mt by meta.old -> semrep_by_old
    if semrep_step_to_id is not None and semrep_map is not None:
        for new_i, txt in enumerate(final_steps):
            sid = resolve_step_id_from_text(txt, vocab_map or {}, extra_map=semrep_step_to_id)
            if isinstance(sid, str) and sid in semrep_map:
                sr = semrep_map[sid].get("semantic_representation") or ""
                if sr:
                    semrep_by_new[new_i] = sr
    for m in meta:
        mod = m.get("mod")
        new = m.get("new")
        old = m.get("old")
        if mod in {"u","ms","mt"} and isinstance(new, int) and isinstance(old, int) and old in semrep_by_old:
            semrep_by_new[new] = semrep_by_old[old]

    return semrep_by_old, semrep_by_new

def _score(sch: List[str], plan: List[str], pl: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    Lower is better.
    Priority:
        1) eliminate schema+plan issues first (hard constraints)
        2) then minimize plausibility issues
    """
    hard = len(sch) + len(plan)
    soft = len(pl)
    # (has_any_hard, hard_count, soft_count)
    return (1 if hard > 0 else 0, hard, soft)

def judge_one_take(
    take: Dict[str, Any],
    candidate_rewrite: Dict[str, Any],
    backend: Optional[LLMBackend],
    frame_cache: Optional[Any],
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    disable_llm_fallback: bool,
    vocab_map: Optional[Dict[str, str]],
    semrep_map: Optional[Dict[str, Dict[str, str]]],
    semrep_step_to_id: Optional[Dict[str, str]],
    semrep_extender: Optional[Any],
    take_uid: str,
    take_name: str,
    debug_llm: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    original_steps = [_step_text(s) for s in (take.get("steps") or [])]
    report_rows: List[Dict[str, Any]] = []

    final_steps = candidate_rewrite.get("final_steps")
    meta = candidate_rewrite.get("meta")

    def collect_prompt_images_on_demand(
        take: Dict[str, Any],
        mt: List[Dict[str, Any]],
        issues: List[Dict[str, Any]],
        max_images: int = 2,
    ) -> Tuple[str, List[str]]:
        """
        Returns (caption_text, data_urls).
        We attach frames ONLY for retry attempts (not the first try), and only if cache available.
        """
        if frame_cache is None:
            return "", []
        if EgoVideoFrameCache is None or FrameRequest is None:
            return "",   []

        # choose old indices:
        old_indices = []
        # 1) prioritize wrong_execution/substitution src indices from plan errors
        for e in (take.get("errors") or []):
            etype = str(e.get("type") or e.get("error_type") or "").strip().lower()
            if etype not in {"wrong_execution", "substitution"}:
                continue
            src = e.get("step_index") if isinstance(e.get("step_index"), int) else e.get("src_step_idx")
            if isinstance(src, int):
                old_indices.append(src)

        # 2) also map plausibility issues -> meta.old (if exists)
        for it in issues[:5]:
            ni = it.get("step_index")
            if isinstance(ni, int):
                mm = _find_meta_by_new(mt, ni)
                if isinstance(mm, dict):
                    oi = mm.get("old")
                    if isinstance(oi, int):
                        old_indices.append(oi)

        # uniq preserve order
        seen = set()
        uniq_old = []
        for x in old_indices:
            if x in seen:
                continue
            seen.add(x)
            uniq_old.append (x)

        captions = []
        urls: List[str] = []

        steps = take.get("steps") or []
        for oi in uniq_old:
            if len(urls) >= max_images:
                break
            if not (0 <= oi < len(steps)):
                continue
            st = steps[oi].get("start_time")
            en = steps[oi].get("end_time")
            txt = steps[oi].get("step_description") or ""
            if not (isinstance(st, (int, float)) and isinstance(en, (int, float))):
                continue

            p = frame_cache.get_mid_frame_path(
                FrameRequest(
                    take_name=str(take.get("take_name") or ""),
                    old_index=int(oi),
                    start_time=float(st),
                    end_time=float(en),
                    tag="mid",
                )
            )
            if p is None:
                continue

            captions.append(f"- Image for ORIGINAL old={oi}: '{txt}' (t≈{(float(st)+float(en))/2:.2f}s)")
            urls.append(frame_cache.image_to_data_url(p))

        if not urls:
            return "",   []

        cap = "IMAGES (mid-frames):\n" + "\n".join(captions) + "\n"
        return cap, urls

    def _trace_row(code: str, detail: str, step_index: Any = "") -> None:
        # Writes into the SAME report rows (csv/json). No extra files.
        if not debug_llm:
            return
        report_rows.append({
            "take_uid": take_uid,
            "take_name": take_name,
            "kind": "trace",
            "code": code,
            "step_index": step_index if step_index is not None else "",
            "detail": detail[:2500],  # keep it bounded
        })

    llm_used = False
    def _dbg(msg: str) -> None:
        # Print only when enabled; stderr so it lands in job logs.
        # provenance summary (LLM vs deterministic)
        if not debug_llm:
            return
        print(msg, file=sys.stderr)
        final_score = _score(schema_issues, planned_realization_issues, plaus_issues)
        _trace_row(
            "PROVENANCE",
            f"llm_used={int(llm_used)} base_score={base_score} final_score={final_score} "
            f"final_hard={len(schema_issues)+len(planned_realization_issues)} final_soft={len(plaus_issues)}"
        )

    if not isinstance(final_steps, list) or not all(isinstance(x, str) for x in final_steps):
        final_steps = []
    if not isinstance(meta, list) or not all(isinstance(x, dict) for x in meta):
        meta = []

    meta = normalize_meta(meta)
    dedupe_corrections_in_place(meta)

    # --- normalize/align ---
    canonicalize_meta_new_indices_in_place(meta)
    final_steps = align_final_to_meta_length(final_steps, meta)
    enforce_verbatim_for_u_and_moves(original_steps, final_steps, meta)
    sanitize_nonverbatim_step_texts(final_steps, meta)

    # --- plan-driven insertion enforcement (missing/mismatch) ---
    _ins_changed, _ins_issues = validate_and_repair_insertions_from_plan(
        take=take,
        final_steps=final_steps,
        meta=meta,
        semrep_by_new=None,   # semrep computed later; deterministic insertion doesn't need it
        jaccard_min=0.70,
    )
    if debug_llm and _ins_issues:
        for msg in _ins_issues:
            report_rows.append({
                "take_uid": take_uid,
                "take_name": take_name,
                "kind": "schema",
                "code": "INSERTION_PLAN_ENFORCE",
                "step_index": "",
                "detail": msg,
            })

    # --- force-realize transpositions early ---
    trans_issues = validate_transposition_realized(take, meta)
    bad_eids: Set[str] = set()
    for s in trans_issues:
        m = re.search(r"\beid=([A-Za-z0-9_-]+)\b", str(s))
        if m:
            bad_eids.add(m.group(1))
    for eid, src, tgt in iter_plan_transpositions(take):
        if eid in bad_eids:
            force_realize_transposition(final_steps, meta, eid, src, tgt)

    canonicalize_meta_new_indices_in_place(meta)
    final_steps = align_final_to_meta_length(final_steps, meta)
    enforce_verbatim_for_u_and_moves(original_steps, final_steps, meta)
    sanitize_nonverbatim_step_texts(final_steps, meta)

    def recompute_semrep(fs: List[str], mt: List[Dict[str, Any]]) -> Tuple[Dict[int, str], Dict[int, str], Dict[str, Any]]:
        nonlocal semrep_step_to_id
        if semrep_extender is not None:
            semrep_extender.ensure_for_texts(fs)
            semrep_step_to_id = semrep_extender.step_to_id
        semrep_by_old, semrep_by_new = build_semrep_maps_for_take(
            take, vocab_map, semrep_map, semrep_step_to_id, fs, mt
        )
        baseline = build_original_baseline(original_steps, semrep_by_old=semrep_by_old)
        return semrep_by_old, semrep_by_new, baseline

    def _append_generic_we_manner(text: str, pred: str) -> str:
        base = text.rstrip().rstrip(".")
        p = (pred or "").upper()
        # generic, predicate-class based (no scenario hardcode)
        if p in {"POSITION","PLACE","PUT","INSERT","INSTALL","MOUNT","SET"}:
            return base + ", but do not seat it fully"
        if p in {"TIGHTEN","FASTEN","SECURE","LOCK"}:
            return base + ", but leave it slightly loose"
        if p in {"OPEN"}:
            return base + ", but only partially"
        if p in {"CLOSE","SEAL"}:
            return base + ", but not completely"
        if p in {"MIX","STIR","COMBINE"}:
            return base + ", but only briefly"
        if p in {"CUT","SLICE","CHOP"}:
            return base + ", but unevenly"
        if p in {"ADD","POUR","TRANSFER"}:
            return base + ", but spill a little"
        return base + ", but not quite correctly"

    def coerce_wrong_execution_to_manner(
        take: Dict[str, Any],
        original_steps: List[str],
        final_steps: List[str],
        meta: List[Dict[str, Any]],
        semrep_by_old: Dict[int, str],
        semrep_by_new: Dict[int, str],
    ) -> bool:
        changed = False
        for m in meta:
            if m.get("mod") != "e":
                continue
            if (m.get("etype") or "").strip().lower() != "wrong_execution":
                continue
            old = m.get("old")
            new = m.get("new")
            if not (isinstance(old, int) and isinstance(new, int)):
                continue
            if not (0 <= old < len(original_steps) and 0 <= new < len(final_steps)):
                continue
            sr = semrep_by_old.get(old) or ""
            parsed = parse_semrep_one(sr) if sr else None
            if not parsed:
                continue
            pred, roles = parsed
            dest = _head_entity(roles.get("Destination") or roles.get("Location") or roles.get("Goal") or "")
            orig_instr = _head_entity(roles.get("Instrument") or "")

            cand = final_steps[new]
            orig = original_steps[old]

            cand_sr = semrep_by_new.get(new) or find_semrep_exact(cand) or ""
            cand_parsed = parse_semrep_one(cand_sr) if cand_sr else None
            cand_instr = ""
            if cand_parsed:
                _p2, r2 = cand_parsed
                cand_instr = _head_entity(r2.get("Instrument") or "")
            instr_mismatch = bool(orig_instr) and bool(cand_instr) and (orig_instr != cand_instr)

            # too far from original OR lost core destination mention -> coerce to manner-based WE
            far = jaccard_similarity(normalize_ws(cand).lower(), normalize_ws(orig).lower()) < 0.60
            lost_dest = bool(dest) and (dest not in normalize_step_text(cand))

            if far or lost_dest or instr_mismatch:
                final_steps[new] = _append_generic_we_manner(orig, pred)
                changed = True
        return changed

    def compute_issues(
        fs: List[str],
        mt: List[Dict[str, Any]],
        baseline: Dict[str, Any],
        semrep_by_new: Dict[int, str],
        semrep_by_old: Dict[int, str],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        planned_realization_issues = validate_planned_errors_still_realized(take, original_steps, fs, mt)
        schema_issues = (
            validate_rewrite_schema(original_steps, fs, mt)
            + validate_plan_coverage(take, mt)
            + validate_transposition_realized(take, mt)
            + validate_old_index_coverage(original_steps, mt)
        )
        pl = plausibility_issues(fs, mt, baseline, semrep_by_new=semrep_by_new)
        # Baseline-derived affordance sanity: prevents implausible cascades like WASH/CHOP on seasonings.
        pl += validate_affordance_mismatch_against_baseline_semrep(
            fs, mt, semrep_by_new=semrep_by_new, semrep_by_old=semrep_by_old
        )
        # If we substituted a GET (plan error), but later 'u' steps still use the original
        # entity/substance, create a soft issue so deterministic repair can apply a minimal cascade.
        pl += validate_get_substitution_cascade_semrep(
            fs, mt, original_steps, semrep_by_old=semrep_by_old, semrep_by_new=semrep_by_new
        )
        pl += validate_location_continuity_semrep(fs, mt, semrep_by_new=semrep_by_new)
        # Final de-duplication across plausibility sources.
        seen = set()
        pl_uniq: List[Dict[str, Any]] = []
        for it in pl:
            key = (it.get("code"), it.get("step_index"), it.get("object"))
            if key in seen:
                continue
            seen.add(key)
            pl_uniq.append(it)
        return schema_issues, planned_realization_issues, pl_uniq

    # initial issues
    semrep_by_old, semrep_by_new, baseline = recompute_semrep(final_steps, meta)
    schema_issues, planned_realization_issues, plaus_issues = compute_issues(
        final_steps, meta, baseline, semrep_by_new, semrep_by_old
    )

    # Track best candidate across LLM attempts (even if not perfect).
    base_score = _score(schema_issues, planned_realization_issues, plaus_issues)
    best_score = base_score
    best_candidate: Optional[Tuple[List[str], List[Dict[str, Any]]]] = None
    best_err_summary = ""

    llm_last_err = ""

    # -------------------------
    # 1) LLM-first repair
    # -------------------------
    if (schema_issues or planned_realization_issues or plaus_issues) and (not disable_llm_fallback):
        if backend is None:
            raise RuntimeError("LLM fallback requested but backend is None.")

        llm_used = True
        user_prompt = build_repair_user_prompt(
            take=take,
            original_steps=original_steps,
            final_steps=final_steps,
            meta=meta,
            schema_issues=(schema_issues + planned_realization_issues),
            plaus_issues=plaus_issues,
        )

        for attempt in range(max_retries + 1):
            t = max(0.0, float(temperature) - 0.15 * attempt)

            # --- attach images only on retries ---
            prompt_with_imgs = user_prompt
            img_urls: List[str] = []
            if attempt >= 1 and frame_cache is not None:
                cap, img_urls = collect_prompt_images_on_demand(
                    take=take,
                    mt=meta,
                    issues=plaus_issues,
                    max_images=2,
                )
                if cap:
                    prompt_with_imgs = cap + "\n" + user_prompt

            try:
                if isinstance(backend, OpenAIBackend):
                    obj = openai_repair_strict_json(
                        backend.client,
                        backend.model_id,
                        SYSTEM_REPAIR,
                        prompt_with_imgs,
                        temperature=t,
                        max_output_tokens=max_new_tokens,
                        image_data_urls=img_urls,  
                    )
                else:
                    raw = backend.generate(
                        messages=[{"role": "system", "content": SYSTEM_REPAIR},
                                  {"role": "user", "content": prompt_with_imgs},],
                        temperature=t,
                        max_new_tokens=max_new_tokens,
                        image_data_urls=img_urls,
                    )
                    obj = extract_json_object(raw)

                fs = obj.get("final_steps")
                mt = obj.get("meta")
                if not isinstance(fs, list) or not all(isinstance(x, str) for x in fs):
                    raise ValueError("LLM returned invalid final_steps")
                if not isinstance(mt, list) or not all(isinstance(x, dict) for x in mt):
                    raise ValueError("LLM returned invalid meta")

                mt = normalize_meta(mt)
                dedupe_corrections_in_place(mt)
                canonicalize_meta_new_indices_in_place(mt)
                fs = align_final_to_meta_length(fs, mt)
                enforce_verbatim_for_u_and_moves(original_steps, fs, mt)
                sanitize_nonverbatim_step_texts(fs, mt)

                # force transpositions again if LLM broke them
                tmp = validate_transposition_realized(take, mt)
                bad_llm: Set[str] = set()
                for s in tmp:
                    m = re.search(r"\beid=([A-Za-z0-9_-]+)\b", str(s))
                    if m:
                        bad_llm.add(m.group(1))
                for eid, src, tgt in iter_plan_transpositions(take):
                    if eid in bad_llm:
                        force_realize_transposition(fs, mt, eid, src, tgt)

                canonicalize_meta_new_indices_in_place(mt)
                fs = align_final_to_meta_length(fs, mt)
                enforce_verbatim_for_u_and_moves(original_steps, fs, mt)
                sanitize_nonverbatim_step_texts(fs, mt)

                _, semrep_by_new2, baseline2 = recompute_semrep(fs, mt)
                sch2, plan2, pl2 = compute_issues(fs, mt, baseline2, semrep_by_new2)

                cand_score = _score(sch2, plan2, pl2)

                top = ""
                if sch2 or plan2:
                    top = " ; ".join((sch2 + plan2)[:2])
                elif pl2:
                    top = f"{pl2[0].get('code')}: {pl2[0].get('detail')}"
                _trace_row(
                    "LLM_ATTEMPT",
                    f"attempt={attempt} temp={t:.3f} score={cand_score} hard={len(sch2)+len(plan2)} soft={len(pl2)} top={top}",
                    step_index=attempt
                )

                # Keep the best attempt so far (deepcopy to avoid later mutation).
                if cand_score < best_score:
                    best_score = cand_score
                    best_candidate = (copy.deepcopy(fs), copy.deepcopy(mt))

                    if sch2 or plan2:
                        best_err_summary = " ; ".join((sch2 + plan2)[:3])
                    elif pl2:
                        best_err_summary = f"{pl2[0].get('code')}: {pl2[0].get('detail')}"
                    else:
                        best_err_summary = ""

                err_summary = ""
                if sch2 or plan2:
                    err_summary = " ; ".join((sch2 + plan2)[:3])
                elif pl2:
                    err_summary = f"{pl2[0].get('code')}: {pl2[0].get('detail')}"
                _dbg(
                    f"[LLM_DEBUG] take={take_name} uid={take_uid} attempt={attempt}/{max_retries} "
                    f"temp={t:.3f} score={cand_score} best={best_score} base={base_score} "
                    f"hard={len(sch2)+len(plan2)} soft={len(pl2)} top='{(err_summary or '')[:220]}'"
                )

                if not sch2 and not plan2 and not pl2:
                    final_steps, meta = fs, mt
                    semrep_by_new, baseline = semrep_by_new2, baseline2
                    schema_issues, planned_realization_issues, plaus_issues = [], [], []
                    llm_last_err = ""
                    _trace_row("LLM_ADOPT", f"mode=perfect score={cand_score}", step_index=attempt)
                    break

                llm_last_err = " ; ".join(sch2 + plan2) if (sch2 or plan2) else (pl2[0].get("detail") if pl2 else "unknown")
            except Exception as e:
                llm_last_err = str(e)
                _trace_row("LLM_EXCEPTION", f"attempt={attempt} err={llm_last_err}", step_index=attempt)
                _dbg(
                    f"[LLM_DEBUG] take={take_name} uid={take_uid} attempt={attempt}/{max_retries} "
                    f"EXCEPTION: {llm_last_err[:800]}"
                )

        # If we didn't reach a perfect solution, still adopt the best attempt if it improves score.
        adopted_mode = "base"
        if best_candidate is not None and best_score < base_score:
            final_steps, meta = best_candidate
            adopted_mode = "best"
            _trace_row("LLM_ADOPT", f"mode=best best_score={best_score} base_score={base_score}")
            _dbg(
                f"[LLM_DEBUG] take={take_name} uid={take_uid} adopting best_attempt score={best_score} "
                f"(base={base_score})"
            )
        else:
            _trace_row("LLM_ADOPT", f"mode=base best_score={best_score} base_score={base_score}")
            _dbg(
                f"[LLM_DEBUG] take={take_name} uid={take_uid} keeping base candidate "
                f"(best_score={best_score}, base_score={base_score})"
            )

        # recompute after LLM attempts (whatever candidate we kept)
        semrep_by_old, semrep_by_new, baseline = recompute_semrep(final_steps, meta)
        if coerce_wrong_execution_to_manner(take, original_steps, final_steps, meta, semrep_by_old, semrep_by_new):
            # keep meta/steps consistent after edits
            canonicalize_meta_new_indices_in_place(meta)
            final_steps = align_final_to_meta_length(final_steps, meta)
            enforce_verbatim_for_u_and_moves(original_steps, final_steps, meta)
            sanitize_nonverbatim_step_texts(final_steps, meta)
            semrep_by_old, semrep_by_new, baseline = recompute_semrep(final_steps, meta)
        schema_issues, planned_realization_issues, plaus_issues = compute_issues(
            final_steps, meta, baseline, semrep_by_new, semrep_by_old
        )

    # -------------------------
    # 2) deterministic-last repair
    # -------------------------
    if (schema_issues or planned_realization_issues or plaus_issues):
        # deterministic mostly addresses plausibility, not schema/coverage
        any_change = False
        det_pass = 0
        for _ in range(3):
            if not plaus_issues:
                break
            final_steps, meta, changed = deterministic_repair(
                final_steps, meta, plaus_issues, baseline, semrep_by_new=semrep_by_new, take=take
            )
            if not changed:
                break
            any_change = any_change or changed
            det_pass += 1
            _trace_row("DET_PASS", f"pass={det_pass} changed=1", step_index=det_pass)

            canonicalize_meta_new_indices_in_place(meta)
            final_steps = align_final_to_meta_length(final_steps, meta)
            enforce_verbatim_for_u_and_moves(original_steps, final_steps, meta)
            sanitize_nonverbatim_step_texts(final_steps, meta)

            semrep_by_old, semrep_by_new, baseline = recompute_semrep(final_steps, meta)
            schema_issues, planned_realization_issues, plaus_issues = compute_issues(
                final_steps, meta, baseline, semrep_by_new, semrep_by_old
            )

    # -------------------------
    # FINAL report rows (append final issues)
    # -------------------------
    for s in schema_issues:
        report_rows.append({
            "take_uid": take_uid,
            "take_name": take_name,
            "kind": "schema",
            "code": "SCHEMA",
            "step_index": "",
            "detail": s,
        })
    for s in planned_realization_issues:
        report_rows.append({
            "take_uid": take_uid,
            "take_name": take_name,
            "kind": "schema",
            "code": "PLANNED_ERROR_NOT_REALIZED",
            "step_index": "",
            "detail": s,
        })
    for it in plaus_issues:
        report_rows.append({
            "take_uid": take_uid,
            "take_name": take_name,
            "kind": "plausibility",
            "code": it.get("code"),
            "step_index": it.get("step_index"),
            "detail": it.get("detail"),
        })

    if (schema_issues or planned_realization_issues or plaus_issues) and (not disable_llm_fallback):
        report_rows.append({
            "take_uid": take_uid,
            "take_name": take_name,
            "kind": "llm_first_pass_failed",
            "code": "LLM_FIRST_PASS_FAILED",
            "step_index": "",
            "detail": llm_last_err or "LLM first-pass did not fully fix issues",
        })

    # final normalize
    meta = normalize_meta(meta)
    dedupe_corrections_in_place(meta)
    canonicalize_meta_new_indices_in_place(meta)
    final_steps = align_final_to_meta_length(final_steps, meta)
    enforce_verbatim_for_u_and_moves(original_steps, final_steps, meta)
    sanitize_nonverbatim_step_texts(final_steps, meta)

    return {"final_steps": final_steps, "meta": meta}, report_rows

# -------------------------
# CLI + report writing
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge+repair erroneous procedure rewrites.")
    p.add_argument("--plan", default=str(REPO_ROOT / "data" / "examples" / "split_50_error_plan_with_corrections.json"), help="Path to split_50_error_plan_with_corrections.json")
    p.add_argument("--rewrite", default=str(REPO_ROOT / "data" / "examples" / "split_50_error_instructions_openai.json"), help="Path to writer output JSON")
    p.add_argument("--out", default=str(REPO_ROOT / "local" / "outputs" / "split_50_error_instructions_openai_judged.json"), help="Output judged JSON path")
    p.add_argument("--report_base", default=str(REPO_ROOT / "local" / "outputs" / "openai_judged_report"), help="Base path (without extension) for CSV/JSON reports")

    # MODEL FLAGS (writer-style)
    p.add_argument("--model", required=True, choices=["openai", "qwen"], help="Which backend to use for LLM fallback")

    # filtering (by take_name)
    p.add_argument(
        "--take_name",
        nargs="+",
        default=[],
        help="If set, process only these take_name values (space-separated). Example: --take_name fair_bike_06_15 sfu_cooking026_8",
    )
    p.add_argument("--limit", type=int, default=-1, help="If >0, process only first N (after filtering)")

    # llm behavior
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=2400)
    p.add_argument("--disable_llm_fallback", action="store_true", help="Disable LLM fallback repair (deterministic only)")

    # optional semrep
    p.add_argument("--vocab_csv", default=str(REPO_ROOT / "data" / "resources" / "split_50_vocabulary.csv"), help="Optional vocab CSV for semrep lookup")
    p.add_argument("--semrep_json", default=str(REPO_ROOT / "data" / "resources" / "semantic_representations_split_50.json"), help="Optional semrep JSON for semrep lookup")

    p.add_argument(
        "--debug_llm",
        action="store_true",
        help="Print LLM attempt diagnostics to stderr (no files).",
    )    
    p.add_argument(
        "--use_frames",
        action="store_true",
        help="Attach 1-2 cached mid-frames to OpenAI LLM repair on retry attempts (when issues persist).",
    )
    return p.parse_args()

def write_report_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    cols = ["take_uid", "take_name", "kind", "code", "step_index", "detail"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

def main() -> None:
    args = parse_args()

    plan = json.load(open(args.plan, "r", encoding="utf-8"))
    rew = json.load(open(args.rewrite, "r", encoding="utf-8"))

    takes_plan: Dict[str, Any] = plan.get("takes") or {}
    takes_rew: Dict[str, Any] = rew.get("takes") or {}

    # backend (only needed if llm fallback enabled)
    backend: Optional[LLMBackend] = None
    if not args.disable_llm_fallback:
        backend = make_backend(args.model)
    
    frame_cache = None
    if args.use_frames and (not args.disable_llm_fallback) and EgoVideoFrameCache is not None:
        frame_cache = EgoVideoFrameCache()

    vocab_map: Optional[Dict[str, str]] = None
    semrep_map: Optional[Dict[str, Dict[str, str]]] = None
    semrep_extender: Optional[Any] = None
    semrep_step_to_id: Optional[Dict[str, str]] = None

    if args.vocab_csv and args.semrep_json:
        try:
            vocab_map = load_vocab_csv(args.vocab_csv)
            semrep_map = load_semrep_json(args.semrep_json)
            init_reverse_semrep_map(semrep_map)
            if build_semrep_step_to_id is not None:
                semrep_step_to_id = build_semrep_step_to_id(semrep_map)
        except Exception as e:
            print(f"WARNING: failed to load vocab/semrep: {e}", file=sys.stderr)

    # ALWAYS enable semrep auto-extension when possible (no extra flags)
    if semrep_map is not None and SemRepAutoExtender is not None and args.semrep_json:
        # Keep id prefix consistent with writer (model-based prefix).
        semrep_extender = SemRepAutoExtender(
            semrep_map=semrep_map,
            out_path=args.semrep_json,
            id_prefix=f"{args.model}_ext",
        )
        semrep_step_to_id = semrep_extender.step_to_id

    # build filtered take ids
    take_ids = list(takes_plan.keys())
    take_ids.sort(key=lambda x: str(x))

    # Filter ALWAYS by take_name
    if args.take_name:
        wanted = [str(x).strip() for x in args.take_name if str(x).strip()]
        wanted_set = set(wanted)
        filtered: List[str] = []
        for tid in take_ids:
            t_plan = takes_plan.get(tid) or {}
            t_rew = takes_rew.get(tid) or {}

            take_name = str(t_plan.get("take_name") or t_rew.get("take_name") or "").strip()
            fallback_name = str(t_plan.get("name") or t_rew.get("name") or "").strip()
            key = take_name or fallback_name or str(tid).strip()

            if key in wanted_set:
                filtered.append(tid)
                continue

            # small robustness: substring match (helps if user passes shorter tokens)
            hit = False
            for w in wanted:
                if w and (w == key or w in key or key in w):
                    hit = True
                    break
            if hit:
                filtered.append(tid)

        take_ids = filtered

        if not take_ids:
            print("WARNING: --take_name filter matched 0 takes. Showing a few available take_name values:", file=sys.stderr)
            shown = 0
            for tid0 in list(takes_plan.keys())[:200]:
                t0 = takes_plan.get(tid0) or {}
                tn0 = str(t0.get("take_name") or t0.get("name") or "").strip()
                if tn0:
                    print(f"  tid={str(tid0).strip()} | take_name={tn0}", file=sys.stderr)
                    shown += 1
                if shown >= 30:
                    break

        if semrep_extender is not None:
            semrep_extender.flush()

    if args.limit and int(args.limit) > 0:
        take_ids = take_ids[: int(args.limit)]

    out_obj: Dict[str, Any] = {"takes": {}}
    report_rows: List[Dict[str, Any]] = []

    for tid in take_ids:
        take = takes_plan[tid]
        cand_take = takes_rew.get(tid, {}) or {}

        # --- ensure take_name is filled ---
        take_uid = str(tid)
        take_name = (
            take.get("take_name")
            or cand_take.get("take_name")
            or take.get("name")
            or cand_take.get("name")
            or str(take_uid)
        )

        candidate_rewrite = (cand_take.get("rewrite") or {}) if isinstance(cand_take, dict) else {}
        # If writer produced invalid_schema, try to salvage JSON from raw_output_preview
        if (not candidate_rewrite) and isinstance(cand_take, dict):
            preview = cand_take.get("raw_output_preview")
            if isinstance(preview, str) and preview.strip():
                try:
                    candidate_rewrite = extract_json_object(preview)
                except Exception:
                    pass
        judged_rewrite, rows = judge_one_take(
            take=take,
            candidate_rewrite=candidate_rewrite,
            backend=backend,
            frame_cache=frame_cache,
            temperature=float(args.temperature),
            max_new_tokens=int(args.max_new_tokens),
            max_retries=int(args.max_retries),
            disable_llm_fallback=bool(args.disable_llm_fallback),
            vocab_map=vocab_map,
            semrep_map=semrep_map,
            semrep_step_to_id=semrep_step_to_id,
            semrep_extender=semrep_extender,
            take_uid=take_uid,
            take_name=take_name,
            debug_llm=bool(args.debug_llm),
        )
        report_rows.extend(rows)

        # rows already correspond to this take -> don't try to match by take_uid (often absent in plan)
        status = "ok" if not any(r.get("kind") in {"schema", "plausibility", "llm_first_pass_failed"} for r in rows) else "judged_with_findings"

        out_obj["takes"][tid] = {
            "take_uid": take_uid,
            "take_name": take_name,
            "status": status,
            "rewrite": judged_rewrite,
        }

    # write outputs
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    write_report_csv(args.report_base + ".csv", report_rows)
    with open(args.report_base + ".json", "w", encoding="utf-8") as f:
        json.dump(report_rows, f, ensure_ascii=False, indent=2)

    if semrep_extender is not None:
        semrep_extender.flush()

    print(f"Wrote judged JSON: {args.out}")
    print(f"Wrote report: {args.report_base}.csv / {args.report_base}.json")

if __name__ == "__main__":
    main()
