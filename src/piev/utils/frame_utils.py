#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from piev.config import load_settings

# -------------------------
# Video -> mid-frame cache
# -------------------------

_DEFAULT_SETTINGS = load_settings()
DEFAULT_VIDEO_ROOT = _DEFAULT_SETTINGS.videos_root
DEFAULT_FRAMES_ROOT = _DEFAULT_SETTINGS.frames_root

@dataclass
class FrameRequest:
    take_name: str
    old_index: int
    start_time: float
    end_time: float
    tag: str = "mid"  # you can use "mid", "mid60", etc.


class EgoVideoFrameCache:
    def __init__(
        self,
        video_root: str | Path = DEFAULT_VIDEO_ROOT,
        frames_root: str | Path = DEFAULT_FRAMES_ROOT,
        *,
        mid_alpha: float = 0.55,
        jpg_qv: int = 2,
    ) -> None:
        self.video_root = Path(video_root)
        self.frames_root = Path(frames_root)
        self.frames_root.mkdir(parents=True, exist_ok=True)
        self.mid_alpha = float(mid_alpha)
        self.jpg_qv = int(jpg_qv)

    def _video_path(self, take_name: str) -> Optional[Path]:
        # dataset naming: <take_name>.mp4
        cand = self.video_root / f"{take_name}.mp4"
        if cand.exists():
            return cand
        # tiny robustness
        for ext in (".MP4", ".mkv", ".mov"):
            cand2 = self.video_root / f"{take_name}{ext}"
            if cand2.exists():
                return cand2
        return None

    def _out_path(self, req: FrameRequest) -> Path:
        ms0 = int(round(float(req.start_time) * 1000))
        ms1 = int(round(float(req.end_time) * 1000))
        sub = self.frames_root / req.take_name
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"old{req.old_index:03d}_{ms0}_{ms1}_{req.tag}.jpg"

    def _need_ffmpeg(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH (required for frame extraction).")

    def get_mid_frame_path(self, req: FrameRequest) -> Optional[Path]:
        """
        Returns a cached (or newly extracted) JPG path, or None if video missing / invalid times.
        """
        if not (req.end_time > req.start_time >= 0):
            return None

        video = self._video_path(req.take_name)
        if video is None:
            return None

        out = self._out_path(req)
        if out.exists() and out.stat().st_size > 0:
            return out

        self._need_ffmpeg()

        dur = float(req.end_time) - float(req.start_time)
        t = float(req.start_time) + self.mid_alpha * dur

        # ffmpeg extract
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-q:v", str(self.jpg_qv),
            "-y",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            # if failed, make sure we don't leave a zero-byte file
            try:
                if out.exists() and out.stat().st_size == 0:
                    out.unlink()
            except Exception:
                pass
            return None

        return out if out.exists() and out.stat().st_size > 0 else None

    @staticmethod
    def image_to_data_url(img_path: str | Path) -> str:
        p = Path(img_path)
        mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"


# -------------------------
# Optional: split_50.json lookup (if you ever need it)
# -------------------------

def find_segment_times_in_split50(split50_path: str | Path, take_name: str, old_index: int) -> Optional[Tuple[float, float]]:
    """
    split_50.json structure: {"annotations":[{"take_name":..., "segments":[...]}]}
    Segments are not chronological; we match by segment["step_id"] if possible,
    otherwise by order in segments list is NOT reliable.
    """
    obj = json.loads(Path(split50_path).read_text(encoding="utf-8"))
    anns = obj.get("annotations") or []
    for a in anns:
        if str(a.get("take_name") or "").strip() != take_name:
            continue
        segs = a.get("segments") or []
        # In split_50.json, segment has "step_id" (taxonomy step_id), not "index".
        # If your "old_index" is original list index, prefer plan.json timings instead.
        for s in segs:
            # best-effort: treat "old_index" as position in that take's ORIGINAL ordered steps is unsafe here
            pass
    return None
