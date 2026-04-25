#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replace one window in the video with a Seedance (first & last frame) generated clip,
then splice back with smooth crossfades (ffmpeg xfade/acrossfade),
and shift annotations accordingly (split_50.json-like).

Seedance:
  model default: seedance-1-5-pro-251215
  create task:   POST {base}/contents/generations/tasks
  retrieve:      GET  {base}/contents/generations/tasks/{id}
"""

import os
import time
import json
import argparse
import shutil
import base64
import mimetypes
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------
# Config defaults
# ---------------------------
DEFAULT_VIDEO = str(REPO_ROOT / "local" / "egoexo4d" / "videos_ego" / "georgiatech_covid_18_10.mp4")
DEFAULT_SPLIT = str(REPO_ROOT / "local" / "egoexo4d" / "split_50.json")
DEFAULT_TAKE = "georgiatech_covid_18_10"

SOURCE_STEP_DESC = "Apply a few drops of the test solution to the test indicator"
TARGET_STEP_DESC = "Apply a few drops of the test solution to the wrong side of the test indicator"

DEFAULT_OUT_DIR = str(REPO_ROOT / "local" / "outputs" / "video_generation" / "seedance")
DEFAULT_AUDIO_SOURCE = str(
    REPO_ROOT
    / "local"
    / "egoexo4d"
    / "videos_with_audio"
    / "georgiatech_covid_18_10"
    / "aria_and_cam03_with_audio.mp4"
)

DEFAULT_ERROR_INSTRUCTIONS_JSON = str(REPO_ROOT / "data" / "examples" / "split_50_error_instructions_openai_judged.json")
DEFAULT_NEW_ANNOTATIONS_OUT = str(
    REPO_ROOT
    / "local"
    / "outputs"
    / "video_generation"
    / "new_annotations"
    / "georgiatech_covid_18_10.json"
)

DEFAULT_MODEL_ID = "seedance-1-5-pro-251215"
DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"


# ---------------------------
# Small utils
# ---------------------------
def _run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)

def ffprobe_json(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-of", "json", "-show_format", "-show_streams", video_path]
    return json.loads(subprocess.check_output(cmd).decode("utf-8"))

def video_has_audio(video_path: str) -> bool:
    j = ffprobe_json(video_path)
    return any(s.get("codec_type") == "audio" for s in j.get("streams", []))

def ffprobe_video_specs(video_path: str) -> dict:
    j = ffprobe_json(video_path)
    v = next(s for s in j["streams"] if s.get("codec_type") == "video")
    fmt = j["format"]

    def parse_rate(s: str) -> float:
        if not s:
            return 0.0
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)

    w = int(v["width"])
    h = int(v["height"])
    fps = parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1")
    dur = float(fmt["duration"])
    return {"width": w, "height": h, "fps": fps, "duration": dur}

def extract_segment(video_path: str, t_start: float, t_end: float, out_mp4: str) -> None:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.001, t_end - t_start)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ss", f"{t_start:.6f}",
        "-t",  f"{dur:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        out_mp4
    ]
    _run(cmd)

def extract_audio_segment(video_path: str, t_start: float, dur_sec: float, out_m4a: str) -> None:
    Path(out_m4a).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t_start:.6f}",
        "-t", f"{dur_sec:.6f}",
        "-i", video_path,
        "-vn",
        "-c:a", "aac", "-b:a", "192k",
        out_m4a
    ]
    _run(cmd)

def replace_audio(video_in: str, audio_in: str, out_mp4: str) -> None:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_in,
        "-i", audio_in,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_mp4
    ]
    _run(cmd)

def normalize_clip(in_mp4: str, out_mp4: str, width: int, height: int, fps: float) -> None:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    j = ffprobe_json(in_mp4)
    has_audio = any(s.get("codec_type") == "audio" for s in j["streams"])

    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"fps={fps},setsar=1,format=yuv420p"
    )

    if has_audio:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", in_mp4,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
            out_mp4
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", in_mp4,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-shortest",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
            out_mp4
        ]
    _run(cmd)

def pad_clip_to_min_duration(in_mp4: str, out_mp4: str, min_dur: float) -> float:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    cur = float(ffprobe_video_specs(in_mp4)["duration"])
    if cur >= (min_dur - 0.02):
        if str(Path(in_mp4).resolve()) != str(Path(out_mp4).resolve()):
            shutil.copyfile(in_mp4, out_mp4)
        return cur

    pad = max(0.001, min_dur - cur)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_mp4,
        "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.6f}",
        "-af", f"apad=pad_dur={pad:.6f}",
        "-t", f"{min_dur:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        out_mp4
    ]
    _run(cmd)
    return float(ffprobe_video_specs(out_mp4)["duration"])

def concat_with_crossfade(clips: List[str], out_mp4: str, fade: float = 0.12, fps: float = 30.0) -> None:
    assert len(clips) >= 1
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)

    if len(clips) == 1:
        shutil.copyfile(clips[0], out_mp4)
        return

    durs = [ffprobe_video_specs(c)["duration"] for c in clips]

    inputs = []
    for c in clips:
        inputs += ["-i", c]

    fc = []
    pad = float(fade) + (2.0 / max(1.0, float(fps))) + 0.05
    for i in range(len(clips)):
        fc.append(f"[{i}:v]tpad=stop_mode=clone:stop_duration={pad:.6f},setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[{i}:a]apad=pad_dur={pad:.6f},asetpts=PTS-STARTPTS[a{i}]")

    v_prev = "[v0]"
    a_prev = "[a0]"
    t_acc = float(durs[0])
    eps = 1.0 / max(1.0, float(fps))

    for i in range(1, len(clips)):
        v_in = f"[v{i}]"
        a_in = f"[a{i}]"
        off = max(0.0, t_acc - float(fade) - float(eps))
        fc.append(f"{v_prev}{v_in}xfade=transition=fade:duration={fade:.6f}:offset={off:.6f}[vxf{i}]")
        fc.append(f"{a_prev}{a_in}acrossfade=d={fade:.6f}[axf{i}]")
        v_prev = f"[vxf{i}]"
        a_prev = f"[axf{i}]"
        t_acc = t_acc + float(durs[i]) - float(fade)

    expected = max(0.001, sum(float(x) for x in durs) - float(fade) * float(len(clips) - 1))
    fc.append(f"{v_prev}trim=0:{expected:.6f},setpts=PTS-STARTPTS[vout]")
    fc.append(f"{a_prev}atrim=0:{expected:.6f},asetpts=PTS-STARTPTS[aout]")

    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + [
        "-filter_complex", ";".join(fc),
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        out_mp4
    ]
    _run(cmd)

def download_file(url: str, out_path: str, timeout: int = 600) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def extract_frame_png(video_path: str, t: float, out_png: str) -> None:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    # NOTE: place -ss AFTER -i for decode-accurate seeking (stable boundary frames).
    # PNG avoids JPEG compression artifacts on anchor frames.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ss", f"{t:.6f}",
        "-frames:v", "1",
        out_png
    ]
    _run(cmd)

def file_to_data_url(path: str) -> str:
    p = Path(path)
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/jpeg"
    b = p.read_bytes()
    enc = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{enc}"


# ---------------------------
# split_50.json helpers
# ---------------------------
def load_take(split_json_path: str, take_name: str) -> dict:
    data = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
    take = next(a for a in data["annotations"] if a["take_name"] == take_name)
    return take

def sorted_segments(take: dict) -> List[dict]:
    segs = sorted(take["segments"], key=lambda s: (float(s["start_time"]), float(s["end_time"])))
    for i, s in enumerate(segs):
        s["position"] = i
    return segs

def find_segment_by_desc(segs: List[dict], desc: str) -> dict:
    for s in segs:
        if (s.get("step_description") or "").strip() == desc.strip():
            return s
    raise ValueError(f"Segment not found by description: {desc}")

def print_segments_table(segs: List[dict]) -> None:
    for s in segs:
        print(f"[{s['position']:02d}] {float(s['start_time']):8.3f}–{float(s['end_time']):8.3f} | {s.get('step_description')}")

def load_error_rewrite_for_take(error_json_path: str, take_name: str) -> Optional[dict]:
    p = Path(error_json_path)
    if not p.exists():
        print(f"WARNING: error instructions json not found: {error_json_path}")
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"WARNING: failed to parse error instructions json: {error_json_path} ({e})")
        return None

    takes = d.get("takes") or {}
    for _uid, info in takes.items():
        if isinstance(info, dict) and info.get("take_name") == take_name:
            return info
    print(f"WARNING: take_name='{take_name}' not found in error instructions json: {error_json_path}")
    return None

def _format_time_like(value_float: float, template_value):
    if isinstance(template_value, str):
        return f"{value_float:.3f}"
    return float(value_float)

def apply_rewrite_step_descriptions(
    segs_sorted: List[dict],
    final_steps: Optional[List[str]],
    meta: Optional[List[dict]],
    fallback_source_desc: str,
    fallback_target_desc: str,
) -> List[dict]:
    base = [{k: v for k, v in s.items() if k != "position"} for s in segs_sorted]

    if final_steps and meta:
        n = len(final_steps)
        new_segs: List[Optional[dict]] = [None] * n
        for m in meta:
            try:
                old_i = int(m.get("old"))
                new_i = int(m.get("new"))
            except Exception:
                continue
            mod = (m.get("mod") or "").strip()
            if mod in ("u", "e"):
                if 0 <= old_i < len(base) and 0 <= new_i < n:
                    s = dict(base[old_i])
                    s["step_description"] = final_steps[new_i]
                    new_segs[new_i] = s
        return [s for s in new_segs if s is not None]

    out = []
    for s in base:
        s2 = dict(s)
        if (s2.get("step_description") or "").strip() == fallback_source_desc.strip():
            s2["step_description"] = fallback_target_desc
        out.append(s2)
    return out

def shift_segment_times_after_splice(
    segments: List[dict],
    splice_to: float,
    delta_effective: float,
) -> List[dict]:
    out = []
    for s in segments:
        s2 = dict(s)
        st_raw = s2.get("start_time")
        en_raw = s2.get("end_time")
        st = float(st_raw)
        en = float(en_raw)

        if st >= splice_to:
            st2 = st + delta_effective
            en2 = en + delta_effective
            s2["start_time"] = _format_time_like(st2, st_raw)
            s2["end_time"]   = _format_time_like(en2, en_raw)
        elif st < splice_to < en:
            en2 = en + delta_effective
            s2["end_time"] = _format_time_like(en2, en_raw)

        out.append(s2)
    return out

def write_single_take_split_like_json(
    split_json_path: str,
    take_obj: dict,
    out_path: str,
) -> None:
    src = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
    out = dict(src)
    out["annotations"] = [take_obj]
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved shifted annotations:", str(p))


# ---------------------------
# Seedance API (ModelArk) call
# ---------------------------
def seedance_create_task(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout_sec: int = 60,
) -> str:
    url = base_url.rstrip("/") + "/contents/generations/tasks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    j = r.json()
    # usually: {"id": "..."}
    task_id = j.get("id") or (j.get("data") or {}).get("id")
    if not task_id:
        raise RuntimeError(f"Seedance create_task: unexpected response: {j}")
    return task_id

def seedance_get_task(
    base_url: str,
    api_key: str,
    task_id: str,
    timeout_sec: int = 60,
) -> dict:
    url = base_url.rstrip("/") + f"/contents/generations/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()

def seedance_wait_video_url(
    base_url: str,
    api_key: str,
    task_id: str,
    timeout_sec: int = 1800,
    poll_sec: float = 2.0,
    poll_max_sec: float = 10.0,
) -> Tuple[str, dict]:
    t0 = time.time()
    cur_poll = float(poll_sec)

    while True:
        j = seedance_get_task(base_url, api_key, task_id)
        status = (j.get("status") or "").lower()

        if status in ("succeeded", "success", "completed", "done"):
            content = j.get("content") or {}
            video_url = content.get("video_url") or content.get("url")
            if not video_url:
                # fallback (some wrappers)
                video_url = (j.get("output") or {}).get("video_url")
            if not video_url:
                raise RuntimeError(f"Seedance task succeeded but no video_url. Response: {j}")
            return video_url, j

        if status in ("failed", "error", "cancelled", "canceled"):
            raise RuntimeError(f"Seedance task {task_id} failed, status={status}, response={j}")

        if (time.time() - t0) > float(timeout_sec):
            raise TimeoutError(f"Seedance task timeout after {timeout_sec}s. Last response: {j}")

        time.sleep(cur_poll)
        cur_poll = min(float(poll_max_sec), cur_poll * 1.25)


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--video_path", default=DEFAULT_VIDEO)
    p.add_argument("--split_json", default=DEFAULT_SPLIT)
    p.add_argument("--take_name", default=DEFAULT_TAKE)
    p.add_argument("--source_desc", default=SOURCE_STEP_DESC)
    p.add_argument("--target_desc_text", default=TARGET_STEP_DESC)

    p.add_argument("--audio_source", default=DEFAULT_AUDIO_SOURCE)
    p.add_argument("--audio_offset", type=float, default=0.0)

    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--workdir", default="", help="Optional explicit workdir (else out_dir/take_name/run_id).")

    # splice params
    p.add_argument("--fade", type=float, default=0.12, help="Crossfade seconds at splice boundaries.")
    p.add_argument("--seed", type=int, default=0, help="0 means random seed.")

    p.add_argument("--replace_from", type=float, default=274.0, help="Absolute replacement window start (seconds).")
    p.add_argument("--replace_to", type=float, default=287.0, help="Absolute replacement window end (seconds).")

    # Seedance API params
    p.add_argument("--api_key_env", default="ARK_API_KEY")
    p.add_argument("--base_url_env", default="ARK_BASE_URL")
    p.add_argument("--base_url", default="", help="Override base URL (else env ARK_BASE_URL or default).")

    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--resolution", default="720p", choices=["480p", "720p", "1080p"])
    p.add_argument("--ratio", default="adaptive", choices=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"])
    p.add_argument("--duration", type=int, default=0, help="0 => auto from window, clamped to [2,12].")
    p.add_argument("--generate_audio", action="store_true", help="Ask Seedance to generate audio too.")
    p.add_argument("--camera_fixed", action="store_true", help="Try to keep camera fixed (cf).")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--poll", type=float, default=2.0)

    p.add_argument("--error_instructions_json", default=DEFAULT_ERROR_INSTRUCTIONS_JSON)
    p.add_argument("--new_annotations_out", default=DEFAULT_NEW_ANNOTATIONS_OUT)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {args.api_key_env}. Example: export {args.api_key_env}='...'")

    base_url = (args.base_url.strip()
                or os.environ.get(args.base_url_env, "").strip()
                or DEFAULT_BASE_URL)

    video_path = args.video_path
    split_json = args.split_json
    take_name = args.take_name

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    work = Path(args.workdir) if args.workdir.strip() else (out_dir / take_name / run_id)
    work.mkdir(parents=True, exist_ok=True)

    specs = ffprobe_video_specs(video_path)
    print("Video specs:", specs)

    take = load_take(split_json, take_name)
    segs = sorted_segments(take)
    print("Chronological segments:")
    print_segments_table(segs)

    target = find_segment_by_desc(segs, args.source_desc)
    print(f"\nReplacing step:\n  FROM '{args.source_desc}'\n  TO   '{args.target_desc_text}'")
    cut_from = float(target["start_time"])
    cut_to = float(target["end_time"])
    print(f"Annotated step time: {cut_from:.3f}–{cut_to:.3f}")

    # --- replacement window (absolute or inside step) ---
    eps = 0.00
    if args.replace_from is not None and args.replace_to is not None:
        splice_from = float(args.replace_from) + eps
        splice_to = float(args.replace_to) - eps
        splice_from = max(0.0, splice_from)
        splice_to = min(float(specs["duration"] - 0.001), splice_to)
        if splice_to <= splice_from:
            raise RuntimeError(f"Bad replace window: {splice_from:.3f}..{splice_to:.3f}")
        print(f"Replace window (absolute): {splice_from:.3f}–{splice_to:.3f} (len={splice_to-splice_from:.3f}s)")
    else:
        # fallback: replace the whole annotated step
        splice_from = max(0.0, cut_from)
        splice_to = min(float(specs["duration"] - 0.001), cut_to)
        print(f"Replace window (step): {splice_from:.3f}–{splice_to:.3f} (len={splice_to-splice_from:.3f}s)")

    removed_len = float(splice_to - splice_from)

    # --- extract boundary frames for Seedance ---
    frames_dir = work / "frames"
    fps_src = specs["fps"] if specs["fps"] > 0 else 30.0
    last_safe = max(0.0, float(specs["duration"]) - (1.0 / max(1.0, fps_src)))

    frame_dt = 1.0 / max(1.0, fps_src)
    # first_frame should be the last frame BEFORE the cut (end of part1), not the first frame inside the removed window.
    t_first = min(max(0.0, splice_from - frame_dt), last_safe)
    # last_frame should be the first frame AFTER the cut (start of part2).
    t_last = min(max(0.0, splice_to), last_safe)

    first_png = frames_dir / "first_frame.png"
    last_png = frames_dir / "last_frame.png"
    extract_frame_png(video_path, t_first, str(first_png))
    extract_frame_png(video_path, t_last, str(last_png))
    print("Saved boundary frames:", first_png, last_png)

    first_data = file_to_data_url(str(first_png))
    last_data = file_to_data_url(str(last_png))

    # --- prompt ---
    prompt = (
        "Egocentric head-mounted POV. Keep the scene identical: same light, table, box, paper, hands, COVID test cassette. "
        "No new objects. The right hand with the tube moves left. "
        "Dispense two transparent drops strictly into the left rectangular results window (reading window) of the cassette (near the left hand). "
        "The tube touches the left rectangular reading window of the cassette. Drops fall straight down onto the left window only. "
        "Right sampling well is dry, no drops there. "
        "The test cassette stays on the table all the time. No extra dispensing actions. No cuts, no zoom."
    )

    seed = None if args.seed == 0 else int(args.seed)

    # duration: auto from window if not set; clamp to [2,12]
    dur_req = int(args.duration) if int(args.duration) > 0 else int(round(removed_len))
    dur_req = max(2, min(12, dur_req))

    payload = {
        "model": args.model,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": first_data}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": last_data}, "role": "last_frame"},
        ],
        "resolution": args.resolution,
        "ratio": args.ratio,
        "duration": dur_req,
        "camerafixed": bool(args.camera_fixed),
        "generate_audio": bool(args.generate_audio),
    }
    if seed is not None:
        payload["seed"] = seed

    print("\nCalling Seedance (first & last frame)...")
    task_id = seedance_create_task(base_url, api_key, payload)
    print("Task id:", task_id)

    video_url, task_info = seedance_wait_video_url(
        base_url, api_key, task_id,
        timeout_sec=int(args.timeout),
        poll_sec=float(args.poll),
    )
    print("Seedance video_url:", video_url)

    gen_raw = work / "generated_raw.mp4"
    download_file(video_url, str(gen_raw), timeout=900)
    print("Downloaded generated clip:", gen_raw)

    gen_dur = float(ffprobe_video_specs(str(gen_raw))["duration"])
    if gen_dur < 1.95:
        gen_p = work / "generated_raw_min2s.mp4"
        _ = pad_clip_to_min_duration(str(gen_raw), str(gen_p), 2.0)
        gen_raw = gen_p
        gen_dur = float(ffprobe_video_specs(str(gen_raw))["duration"])

    print(f"Generated duration: {gen_dur:.3f}s | removed window: {removed_len:.3f}s | delta: {gen_dur-removed_len:+.3f}s")

    # --- Overlay original audio for generated clip (aligned to splice_from) ---
    audio_source = args.audio_source
    aoff = float(args.audio_offset)
    has_audio = video_has_audio(audio_source)
    gen_with_audio = gen_raw

    if has_audio:
        audio_dir = work / "audio"
        step_audio = audio_dir / "step_audio.m4a"
        extract_audio_segment(audio_source, splice_from + aoff, float(gen_dur), str(step_audio))
        gen_with_audio = work / "generated_with_audio.mp4"
        replace_audio(str(gen_raw), str(step_audio), str(gen_with_audio))
        print("Generated clip with original audio:", gen_with_audio)
    else:
        print(f"WARNING: audio_source has no audio stream: {audio_source}")

    # --- Cut original into part1 + part2 (no bridge) ---
    parts_dir = work / "parts"
    part1_raw = parts_dir / "part1_raw.mp4"
    extract_segment(video_path, 0.0, splice_from, str(part1_raw))

    has_part2 = splice_to < float(specs["duration"] - 0.05)
    part2_raw = None
    if has_part2:
        part2_raw = parts_dir / "part2_raw.mp4"
        extract_segment(video_path, splice_to, float(specs["duration"]), str(part2_raw))
    else:
        print("WARNING: no part2 (replace window reaches video end).")

    # --- audio mux for part1/part2 to keep continuity in crossfade ---
    if has_audio:
        audio_dir = work / "audio"
        p1a = audio_dir / "part1_audio.m4a"
        extract_audio_segment(audio_source, 0.0 + aoff, float(splice_from), str(p1a))
        p1av = parts_dir / "part1_with_audio.mp4"
        replace_audio(str(part1_raw), str(p1a), str(p1av))
        part1_raw = p1av

        if has_part2 and part2_raw is not None:
            p2a = audio_dir / "part2_audio.m4a"
            extract_audio_segment(audio_source, float(splice_to) + aoff, float(specs["duration"] - splice_to), str(p2a))
            p2av = parts_dir / "part2_with_audio.mp4"
            replace_audio(str(part2_raw), str(p2a), str(p2av))
            part2_raw = p2av

    # --- Normalize all pieces to original geometry/fps ---
    norm_dir = work / "norm"
    fps_out = specs["fps"] if specs["fps"] > 0 else 30.0

    part1 = norm_dir / "part1.mp4"
    normalize_clip(str(part1_raw), str(part1), specs["width"], specs["height"], fps_out)

    gen_norm = norm_dir / "generated.mp4"
    normalize_clip(str(gen_with_audio), str(gen_norm), specs["width"], specs["height"], fps_out)

    part2 = None
    if has_part2 and part2_raw is not None:
        part2 = norm_dir / "part2.mp4"
        normalize_clip(str(part2_raw), str(part2), specs["width"], specs["height"], fps_out)

    # --- Final splice with crossfades (smooth glue, no extra bridge generation) ---
    clips = [str(part1), str(gen_norm)]
    if part2 is not None:
        clips.append(str(part2))

    transitions = max(0, len(clips) - 1)

    final_name = f"{take_name}_seedance_{run_id}.mp4"
    out_final = out_dir / final_name
    concat_with_crossfade(clips, str(out_final), fade=float(args.fade), fps=float(fps_out))
    print("\nSaved final video:", out_final)

    # Save step-only clip too
    step_clip = out_dir / f"{take_name}_seedance_generated_{run_id}.mp4"
    step_clip.write_bytes(Path(gen_norm).read_bytes())
    print("Saved step-only clip:", step_clip)

    # --- Write shifted annotations for THIS take only ---
    # delta_effective
    delta_effective = (float(gen_dur) - float(removed_len)) - float(args.fade) * float(transitions)

    err_take = load_error_rewrite_for_take(args.error_instructions_json, take_name)
    rewrite = (err_take or {}).get("rewrite") or {}
    final_steps = rewrite.get("final_steps")
    meta = rewrite.get("meta")

    segs_for_write = apply_rewrite_step_descriptions(
        segs_sorted=segs,
        final_steps=final_steps if isinstance(final_steps, list) else None,
        meta=meta if isinstance(meta, list) else None,
        fallback_source_desc=args.source_desc,
        fallback_target_desc=args.target_desc_text,
    )

    # Shift everything after the *removed window end* (splice_to)
    segs_for_write = shift_segment_times_after_splice(
        segments=segs_for_write,
        splice_to=float(splice_to),
        delta_effective=float(delta_effective),
    )

    take_out = dict(take)
    take_out["segments"] = segs_for_write
    write_single_take_split_like_json(
        split_json_path=split_json,
        take_obj=take_out,
        out_path=args.new_annotations_out,
    )
