#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seedance pipeline for EgoExo4D take: cmu_bike15_3.

What this script does:
1) Generates two Seedance clips from first/last anchor frames:
   - Window A: 12.0 -> 16.0 (inside "Deflate...")
   - Window B: 58.0 -> end of "Fit the inner tube..."
2) Reassembles final video with reordered clips:
   [0..12] + GenA + [16..58] + GenB + PushTire + ValveStem + InflateTail
3) Applies variable crossfades:
   - short fade on regular boundaries (default 0.12s)
   - long fade (default 3.0s) around moved ValveStem clip
4) Writes updated single-take annotations JSON.

Seedance API:
  POST {base}/contents/generations/tasks
  GET  {base}/contents/generations/tasks/{id}
"""

import argparse
import base64
import json
import mimetypes
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------
# Defaults
# ---------------------------
DEFAULT_VIDEO = str(REPO_ROOT / "local" / "egoexo4d" / "videos_ego" / "cmu_bike15_3.mp4")
DEFAULT_SPLIT = str(REPO_ROOT / "local" / "egoexo4d" / "split_50.json")
DEFAULT_TAKE = "cmu_bike15_3"

DEFAULT_OUT_DIR = str(REPO_ROOT / "local" / "outputs" / "video_generation" / "seedance")
DEFAULT_NEW_ANNOTATIONS_OUT = str(
    REPO_ROOT / "local" / "outputs" / "video_generation" / "new_annotations" / "cmu_bike15_3_seedance.json"
)

DEFAULT_MODEL_ID = "seedance-1-5-pro-251215"
DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"

# Step descriptions in split_50.json
REMOVE_DUSTCAP_DESC = "Remove the dust cap from the valve stem"
DEF_SRC_DESC = "Deflate the wheel using a deflating needle"
FIT_SRC_DESC = "Fit the inner tube into the tire"
VALVE_SRC_DESC = "Insert the valve stem into the hole on the rim"
PUSH_SRC_DESC = "Push the tire back into place on the wheel rim"
INFLATE_SRC_DESC = "Inflate the tube to check if the tire bead is properly seated in the rim"
FINAL_DUSTCAP_DESC = "Cover the valve stem with the dust cap"

# New descriptions
DEF_ERR_DESC = (
    "Deflate the wheel using a deflating needle, but only press it briefly so the wheel barely deflates."
)
DEF_FIX_DESC = (
    "Stop and fix it: press the deflating needle properly until the wheel is fully deflated."
)
FIT_ERR_DESC = (
    "Fit the inner tube into the tire, but leave part of the tube twisted and not fully tucked inside."
)
FIT_FIX_DESC = (
    "Stop and fix it: pull the tube back out, untwist it, and fit the inner tube into the tire evenly."
)
VALVE_LATE_DESC = "Perform the missed step now: insert the valve stem into the hole on the rim."

# Prompt templates
PROMPT_STYLE_LOCK = (
    "Egocentric head-mounted camera, fixed POV, same fisheye lens and same dark circular vignette. "
    "Same bike repair workshop, same bike wheel and tools, same hands and body, same lighting. "
    "No new objects, no swaps, no text overlays, no reframing, no cut."
)

PROMPT_DEF = (
    PROMPT_STYLE_LOCK
    + " Continue naturally from the first frame: the hand holds a screwdriver attached to the valve stem; the hand briefly presses the deflating needle, "
    + "then stops, pulls the screwdriver away; the valve stem stays in plave, the hand checks the wheel, "
    + "then hand position returns to start proper deflation by the last frame."
)

PROMPT_FIT = (
    PROMPT_STYLE_LOCK
    + " Continue naturally from the first frame where the inner tube is not fully tucked and slightly twisted. "
    + "The person inspects and slightly rotates the wheel, realizes the issue, and fits the inner tube "
    + "evenly and correctly so the result matches the last frame."
)


# ---------------------------
# Small utils
# ---------------------------
def _run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def ffprobe_json(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-of", "json", "-show_format", "-show_streams", video_path]
    return json.loads(subprocess.check_output(cmd).decode("utf-8"))


def ffprobe_video_specs(video_path: str) -> dict:
    j = ffprobe_json(video_path)
    v = next(s for s in j["streams"] if s.get("codec_type") == "video")
    fmt = j["format"]

    def parse_rate(value: str) -> float:
        if not value:
            return 0.0
        if "/" in value:
            a, b = value.split("/")
            return float(a) / float(b)
        return float(value)

    w = int(v["width"])
    h = int(v["height"])
    fps = parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1")
    dur = float(fmt["duration"])
    return {"width": w, "height": h, "fps": fps, "duration": dur}


def extract_segment(video_path: str, t_start: float, t_end: float, out_mp4: str) -> None:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.001, t_end - t_start)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-ss",
        f"{t_start:.6f}",
        "-t",
        f"{dur:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-b:a",
        "192k",
        out_mp4,
    ]
    _run(cmd)


def extract_frame_png(video_path: str, t: float, out_png: str) -> None:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-ss",
        f"{t:.6f}",
        "-frames:v",
        "1",
        out_png,
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
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_mp4,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "192k",
            out_mp4,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_mp4,
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-shortest",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "192k",
            out_mp4,
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
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        in_mp4,
        "-vf",
        f"tpad=stop_mode=clone:stop_duration={pad:.6f}",
        "-af",
        f"apad=pad_dur={pad:.6f}",
        "-t",
        f"{min_dur:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-b:a",
        "192k",
        out_mp4,
    ]
    _run(cmd)
    return float(ffprobe_video_specs(out_mp4)["duration"])


def sanitize_fades(durs: List[float], fades: List[float], margin: float = 0.05) -> List[float]:
    assert len(fades) == len(durs) - 1
    out = []
    for i, f in enumerate(fades):
        max_f = max(0.01, min(durs[i], durs[i + 1]) - margin)
        if f > max_f:
            print(
                f"WARNING: fade[{i}]={f:.3f}s too long for clips ({durs[i]:.3f}s, {durs[i+1]:.3f}s). "
                f"Using {max_f:.3f}s."
            )
            out.append(max_f)
        else:
            out.append(float(f))
    return out


def concat_with_variable_crossfade(
    clips: List[str],
    out_mp4: str,
    fades: List[float],
    fps: float = 30.0,
) -> None:
    assert len(clips) >= 1
    assert len(fades) == max(0, len(clips) - 1)
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)

    if len(clips) == 1:
        shutil.copyfile(clips[0], out_mp4)
        return

    durs = [float(ffprobe_video_specs(c)["duration"]) for c in clips]
    fades = sanitize_fades(durs, fades)

    inputs = []
    for c in clips:
        inputs += ["-i", c]

    fc = []
    max_fade = max(float(x) for x in fades) if fades else 0.12
    pad = max_fade + (2.0 / max(1.0, float(fps))) + 0.05
    for i in range(len(clips)):
        fc.append(f"[{i}:v]tpad=stop_mode=clone:stop_duration={pad:.6f},setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[{i}:a]apad=pad_dur={pad:.6f},asetpts=PTS-STARTPTS[a{i}]")

    v_prev = "[v0]"
    a_prev = "[a0]"
    t_acc = float(durs[0])
    eps = 1.0 / max(1.0, float(fps))

    for i in range(1, len(clips)):
        fd = float(fades[i - 1])
        v_in = f"[v{i}]"
        a_in = f"[a{i}]"
        off = max(0.0, t_acc - fd - eps)
        fc.append(f"{v_prev}{v_in}xfade=transition=fade:duration={fd:.6f}:offset={off:.6f}[vxf{i}]")
        fc.append(f"{a_prev}{a_in}acrossfade=d={fd:.6f}[axf{i}]")
        v_prev = f"[vxf{i}]"
        a_prev = f"[axf{i}]"
        t_acc = t_acc + float(durs[i]) - fd

    expected = max(0.001, sum(float(x) for x in durs) - sum(float(x) for x in fades))
    fc.append(f"{v_prev}trim=0:{expected:.6f},setpts=PTS-STARTPTS[vout]")
    fc.append(f"{a_prev}atrim=0:{expected:.6f},asetpts=PTS-STARTPTS[aout]")

    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + [
        "-filter_complex",
        ";".join(fc),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-b:a",
        "192k",
        out_mp4,
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


# ---------------------------
# split_50 helpers
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


def clone_segment(seg: dict, start: float, end: float, desc: Optional[str] = None) -> dict:
    s = {k: v for k, v in seg.items() if k != "position"}
    s["start_time"] = round(float(start), 5)
    s["end_time"] = round(float(end), 5)
    if desc is not None:
        s["step_description"] = desc
    return s


# ---------------------------
# Seedance API
# ---------------------------
def seedance_create_task(base_url: str, api_key: str, payload: dict, timeout_sec: int = 60) -> str:
    url = base_url.rstrip("/") + "/contents/generations/tasks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    j = r.json()
    task_id = j.get("id") or (j.get("data") or {}).get("id")
    if not task_id:
        raise RuntimeError(f"Seedance create_task: unexpected response: {j}")
    return task_id


def seedance_get_task(base_url: str, api_key: str, task_id: str, timeout_sec: int = 60) -> dict:
    url = base_url.rstrip("/") + f"/contents/generations/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
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
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--new_annotations_out", default=DEFAULT_NEW_ANNOTATIONS_OUT)
    p.add_argument("--workdir", default="", help="Optional explicit workdir. Else out_dir/take_name/run_id.")

    # Time anchors from user instructions
    p.add_argument("--step1_keep_until", type=float, default=12.0, help="Keep original until this time before GenA.")
    p.add_argument("--step1_fix_start", type=float, default=16.0, help="Resume original deflation from this time.")
    p.add_argument("--step2_gen_start", type=float, default=58.0, help="Start GenB at this time inside FitTube step.")

    # Crossfades
    p.add_argument("--fade_short", type=float, default=0.12, help="Default fade on regular boundaries.")
    p.add_argument("--fade_valve", type=float, default=3.0, help="Fade before and after moved valve clip.")

    # Seedance API
    p.add_argument("--api_key_env", default="ARK_API_KEY")
    p.add_argument("--base_url_env", default="ARK_BASE_URL")
    p.add_argument("--base_url", default="", help="Override base URL.")
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--resolution", default="720p", choices=["480p", "720p", "1080p"])
    p.add_argument("--ratio", default="adaptive", choices=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"])
    p.add_argument("--duration_a", type=int, default=0, help="Seedance duration for GenA (0 => auto from window).")
    p.add_argument("--duration_b", type=int, default=0, help="Seedance duration for GenB (0 => auto from window).")
    p.add_argument("--seed", type=int, default=0, help="0 => random seed.")
    p.add_argument("--generate_audio", action="store_true", help="Ask Seedance to generate audio.")
    p.add_argument("--camera_fixed", action="store_true", default=True)
    p.add_argument("--no_camera_fixed", action="store_false", dest="camera_fixed")
    p.add_argument(
        "--service_tier",
        default="flex",
        choices=["default", "flex"],
        help="Seedance service tier: default=online inference, flex=offline inference.",
    )
    p.add_argument(
        "--execution_expires_after",
        type=int,
        default=172800,
        help="Task expiry in seconds for flex tier (e.g. 172800 = 48h).",
    )
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--poll", type=float, default=2.0)

    p.add_argument("--reuse_gen_a", default="", help="Reuse existing GenA mp4 path.")
    p.add_argument("--reuse_gen_b", default="", help="Reuse existing GenB mp4 path.")
    return p.parse_args()


def pick_duration(requested: int, window_len: float) -> int:
    if requested and int(requested) > 0:
        return max(2, min(12, int(requested)))
    auto = int(round(window_len))
    return max(2, min(12, auto))


if __name__ == "__main__":
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {args.api_key_env}. Example: export {args.api_key_env}='...'")

    base_url = args.base_url.strip() or os.environ.get(args.base_url_env, "").strip() or DEFAULT_BASE_URL

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    work = Path(args.workdir) if args.workdir.strip() else (out_dir / args.take_name / run_id)
    work.mkdir(parents=True, exist_ok=True)

    specs = ffprobe_video_specs(args.video_path)
    fps_src = specs["fps"] if specs["fps"] > 0 else 30.0
    print("Video specs:", specs)

    take = load_take(args.split_json, args.take_name)
    segs = sorted_segments(take)
    print("Chronological segments:")
    print_segments_table(segs)

    seg_remove_dustcap = find_segment_by_desc(segs, REMOVE_DUSTCAP_DESC)
    seg_def = find_segment_by_desc(segs, DEF_SRC_DESC)
    seg_fit = find_segment_by_desc(segs, FIT_SRC_DESC)
    seg_valve = find_segment_by_desc(segs, VALVE_SRC_DESC)
    seg_push = find_segment_by_desc(segs, PUSH_SRC_DESC)
    seg_inflate = find_segment_by_desc(segs, INFLATE_SRC_DESC)
    seg_final_dustcap = find_segment_by_desc(segs, FINAL_DUSTCAP_DESC)

    seg_get_lever = find_segment_by_desc(segs, "Get a tire lever to remove the tube")
    seg_insert_lever = find_segment_by_desc(segs, "Insert the tire lever between the tire and wheel rim")
    seg_pry = find_segment_by_desc(segs, "Pry out a section of the tire with the tire lever")
    seg_run_lever = find_segment_by_desc(segs, "Run the tire lever around the rim to remove the tire with tube from the wheel rim")
    seg_separate = find_segment_by_desc(segs, "Seperate the inner tube from the tire")

    def_start = float(seg_def["start_time"])
    def_end = float(seg_def["end_time"])
    fit_start = float(seg_fit["start_time"])
    fit_end = float(seg_fit["end_time"])
    valve_start = float(seg_valve["start_time"])
    valve_end = float(seg_valve["end_time"])
    push_start = float(seg_push["start_time"])
    push_end = float(seg_push["end_time"])
    inflate_start = float(seg_inflate["start_time"])
    video_end = float(specs["duration"])

    keep_until = float(args.step1_keep_until)
    fix_start = float(args.step1_fix_start)
    fit_gen_start = float(args.step2_gen_start)

    if not (def_start < keep_until < fix_start < def_end):
        raise RuntimeError(
            f"Bad step1 anchors: need {def_start:.3f} < keep_until < fix_start < {def_end:.3f}, "
            f"got keep_until={keep_until:.3f}, fix_start={fix_start:.3f}"
        )
    if not (fit_start < fit_gen_start < fit_end):
        raise RuntimeError(
            f"Bad step2 anchor: need {fit_start:.3f} < step2_gen_start < {fit_end:.3f}, got {fit_gen_start:.3f}"
        )
    if not (valve_start < valve_end <= push_start < push_end <= inflate_start < video_end):
        raise RuntimeError("Unexpected ordering for valve/push/inflate segments in source annotations.")

    replaced_len_a = float(fix_start - keep_until)
    replaced_len_b = float(fit_end - fit_gen_start)
    moved_valve_len = float(valve_end - valve_start)
    tier_label = "Offline (flex)" if args.service_tier == "flex" else "Online (default)"
    print(f"Seedance service tier: {tier_label}")
    if args.service_tier == "flex":
        print(f"execution_expires_after: {int(args.execution_expires_after)}s")

    print(
        f"\nReplacing step #1:\n"
        f"  FROM '{DEF_SRC_DESC}'\n"
        f"  TO   '{DEF_ERR_DESC}'\n"
        f"  TO   '{DEF_FIX_DESC}'"
    )
    print(f"Annotated step #1 time: {def_start:.3f}–{def_end:.3f}")
    print(f"Replace window #1 (absolute): {keep_until:.3f}–{fix_start:.3f} (len={replaced_len_a:.3f}s)")

    print(
        f"\nReplacing step #2:\n"
        f"  FROM '{FIT_SRC_DESC}'\n"
        f"  TO   '{FIT_ERR_DESC}'\n"
        f"  TO   '{FIT_FIX_DESC}'"
    )
    print(f"Annotated step #2 time: {fit_start:.3f}–{fit_end:.3f}")
    print(f"Replace window #2 (absolute): {fit_gen_start:.3f}–{fit_end:.3f} (len={replaced_len_b:.3f}s)")

    print(
        f"\nMoved correction step:\n"
        f"  FROM '{VALVE_SRC_DESC}' at {valve_start:.3f}–{valve_end:.3f}\n"
        f"  TO   '{VALVE_LATE_DESC}' after '{PUSH_SRC_DESC}' ({push_start:.3f}–{push_end:.3f})"
    )
    print(
        f"Windows: A={keep_until:.3f}-{fix_start:.3f}, "
        f"B={fit_gen_start:.3f}-{fit_end:.3f}, "
        f"moved_valve={valve_start:.3f}-{valve_end:.3f}"
    )

    # --- extract anchor frames ---
    frame_dt = 1.0 / max(1.0, fps_src)
    last_safe = max(0.0, video_end - frame_dt)
    frames_dir = work / "frames"

    a_first_t = min(max(0.0, keep_until - frame_dt), last_safe)
    a_last_t = min(max(0.0, fix_start), last_safe)
    b_first_t = min(max(0.0, fit_gen_start - frame_dt), last_safe)
    b_last_t = min(max(0.0, fit_end), last_safe)

    a_first_png = frames_dir / "a_first.png"
    a_last_png = frames_dir / "a_last.png"
    b_first_png = frames_dir / "b_first.png"
    b_last_png = frames_dir / "b_last.png"

    extract_frame_png(args.video_path, a_first_t, str(a_first_png))
    extract_frame_png(args.video_path, a_last_t, str(a_last_png))
    extract_frame_png(args.video_path, b_first_t, str(b_first_png))
    extract_frame_png(args.video_path, b_last_t, str(b_last_png))
    print("Saved boundary frames:", a_first_png, a_last_png, b_first_png, b_last_png)

    # --- Seedance A ---
    seed = None if args.seed == 0 else int(args.seed)
    dur_a = pick_duration(args.duration_a, fix_start - keep_until)
    dur_b = pick_duration(args.duration_b, fit_end - fit_gen_start)

    gen_a_raw: Path
    if args.reuse_gen_a.strip():
        gen_a_raw = Path(args.reuse_gen_a.strip())
        if not gen_a_raw.exists():
            raise FileNotFoundError(f"--reuse_gen_a not found: {gen_a_raw}")
        print("Reusing generated clip A:", gen_a_raw)
    else:
        payload_a = {
            "model": args.model,
            "content": [
                {"type": "text", "text": PROMPT_DEF},
                {"type": "image_url", "image_url": {"url": file_to_data_url(str(a_first_png))}, "role": "first_frame"},
                {"type": "image_url", "image_url": {"url": file_to_data_url(str(a_last_png))}, "role": "last_frame"},
            ],
            "resolution": args.resolution,
            "ratio": args.ratio,
            "duration": dur_a,
            "camerafixed": bool(args.camera_fixed),
            "generate_audio": bool(args.generate_audio),
            "service_tier": args.service_tier,
        }
        if args.service_tier == "flex":
            payload_a["execution_expires_after"] = int(args.execution_expires_after)
        if seed is not None:
            payload_a["seed"] = seed

        print("\nCalling Seedance A...")
        task_id_a = seedance_create_task(base_url, api_key, payload_a)
        print("Task id A:", task_id_a)
        video_url_a, _ = seedance_wait_video_url(
            base_url,
            api_key,
            task_id_a,
            timeout_sec=int(args.timeout),
            poll_sec=float(args.poll),
        )
        print("Seedance A video_url:", video_url_a)
        gen_a_raw = work / "generated_raw_A.mp4"
        download_file(video_url_a, str(gen_a_raw), timeout=900)
        print("Downloaded generated clip A:", gen_a_raw)

    gen_a_dur = float(ffprobe_video_specs(str(gen_a_raw))["duration"])
    if gen_a_dur < 1.95:
        gen_a_min = work / "generated_raw_A_min2s.mp4"
        _ = pad_clip_to_min_duration(str(gen_a_raw), str(gen_a_min), 2.0)
        gen_a_raw = gen_a_min
        gen_a_dur = float(ffprobe_video_specs(str(gen_a_raw))["duration"])

    # --- Seedance B ---
    gen_b_raw: Path
    if args.reuse_gen_b.strip():
        gen_b_raw = Path(args.reuse_gen_b.strip())
        if not gen_b_raw.exists():
            raise FileNotFoundError(f"--reuse_gen_b not found: {gen_b_raw}")
        print("Reusing generated clip B:", gen_b_raw)
    else:
        payload_b = {
            "model": args.model,
            "content": [
                {"type": "text", "text": PROMPT_FIT},
                {"type": "image_url", "image_url": {"url": file_to_data_url(str(b_first_png))}, "role": "first_frame"},
                {"type": "image_url", "image_url": {"url": file_to_data_url(str(b_last_png))}, "role": "last_frame"},
            ],
            "resolution": args.resolution,
            "ratio": args.ratio,
            "duration": dur_b,
            "camerafixed": bool(args.camera_fixed),
            "generate_audio": bool(args.generate_audio),
            "service_tier": args.service_tier,
        }
        if args.service_tier == "flex":
            payload_b["execution_expires_after"] = int(args.execution_expires_after)
        if seed is not None:
            payload_b["seed"] = seed

        print("\nCalling Seedance B...")
        task_id_b = seedance_create_task(base_url, api_key, payload_b)
        print("Task id B:", task_id_b)
        video_url_b, _ = seedance_wait_video_url(
            base_url,
            api_key,
            task_id_b,
            timeout_sec=int(args.timeout),
            poll_sec=float(args.poll),
        )
        print("Seedance B video_url:", video_url_b)
        gen_b_raw = work / "generated_raw_B.mp4"
        download_file(video_url_b, str(gen_b_raw), timeout=900)
        print("Downloaded generated clip B:", gen_b_raw)

    gen_b_dur = float(ffprobe_video_specs(str(gen_b_raw))["duration"])
    if gen_b_dur < 1.95:
        gen_b_min = work / "generated_raw_B_min2s.mp4"
        _ = pad_clip_to_min_duration(str(gen_b_raw), str(gen_b_min), 2.0)
        gen_b_raw = gen_b_min
        gen_b_dur = float(ffprobe_video_specs(str(gen_b_raw))["duration"])

    print(
        f"\nGenerated A: {gen_a_dur:.3f}s, Generated B: {gen_b_dur:.3f}s | "
        f"replaced windows: {replaced_len_a:.3f}s + {replaced_len_b:.3f}s, "
        f"moved valve window: {moved_valve_len:.3f}s"
    )

    # --- Build raw parts ---
    parts_dir = work / "parts"
    c0_raw = parts_dir / "clip0_0_to_keep.mp4"
    c2_raw = parts_dir / "clip2_fix_to_fitstart.mp4"
    c4_raw = parts_dir / "clip4_push.mp4"
    c5_raw = parts_dir / "clip5_valve_moved.mp4"
    c6_raw = parts_dir / "clip6_inflate_tail.mp4"

    extract_segment(args.video_path, 0.0, keep_until, str(c0_raw))
    extract_segment(args.video_path, fix_start, fit_gen_start, str(c2_raw))
    extract_segment(args.video_path, push_start, push_end, str(c4_raw))
    extract_segment(args.video_path, valve_start, valve_end, str(c5_raw))
    extract_segment(args.video_path, inflate_start, video_end, str(c6_raw))

    # --- Normalize all pieces ---
    norm_dir = work / "norm"
    fps_out = specs["fps"] if specs["fps"] > 0 else 30.0
    w = int(specs["width"])
    h = int(specs["height"])

    c0 = norm_dir / "clip0.mp4"
    c1 = norm_dir / "clip1_genA.mp4"
    c2 = norm_dir / "clip2.mp4"
    c3 = norm_dir / "clip3_genB.mp4"
    c4 = norm_dir / "clip4_push.mp4"
    c5 = norm_dir / "clip5_valve_moved.mp4"
    c6 = norm_dir / "clip6_tail.mp4"

    normalize_clip(str(c0_raw), str(c0), w, h, fps_out)
    normalize_clip(str(gen_a_raw), str(c1), w, h, fps_out)
    normalize_clip(str(c2_raw), str(c2), w, h, fps_out)
    normalize_clip(str(gen_b_raw), str(c3), w, h, fps_out)
    normalize_clip(str(c4_raw), str(c4), w, h, fps_out)
    normalize_clip(str(c5_raw), str(c5), w, h, fps_out)
    normalize_clip(str(c6_raw), str(c6), w, h, fps_out)

    clips = [str(c0), str(c1), str(c2), str(c3), str(c4), str(c5), str(c6)]
    durs = [float(ffprobe_video_specs(c)["duration"]) for c in clips]

    fades = [
        float(args.fade_short),  # c0 -> c1
        float(args.fade_short),  # c1 -> c2
        float(args.fade_short),  # c2 -> c3
        float(args.fade_short),  # c3 -> c4
        float(args.fade_valve),  # c4 -> c5
        float(args.fade_valve),  # c5 -> c6
    ]
    fades = sanitize_fades(durs, fades)

    print("\nClip durations:")
    for i, d in enumerate(durs):
        print(f"  clip{i}: {d:.3f}s")
    print("Fades:", [round(x, 3) for x in fades])

    # --- Final splice ---
    final_name = f"{args.take_name}_seedance_{run_id}.mp4"
    out_final = out_dir / final_name
    concat_with_variable_crossfade(clips, str(out_final), fades=fades, fps=float(fps_out))
    final_dur = float(ffprobe_video_specs(str(out_final))["duration"])
    print("\nSaved final video:", out_final)
    print(f"Final duration: {final_dur:.3f}s")

    # --- Build updated annotations ---
    clip_starts = [0.0]
    for i in range(1, len(durs)):
        clip_starts.append(clip_starts[-1] + durs[i - 1] - fades[i - 1])

    eps = 1e-6

    def map_source_time(src_t: float) -> float:
        t = float(src_t)
        if -eps <= t <= keep_until + eps:
            return clip_starts[0] + (t - 0.0)
        if fix_start - eps <= t <= fit_gen_start + eps:
            return clip_starts[2] + (t - fix_start)
        if push_start - eps <= t <= push_end + eps:
            return clip_starts[4] + (t - push_start)
        if valve_start - eps <= t <= valve_end + eps:
            return clip_starts[5] + (t - valve_start)
        if inflate_start - eps <= t <= video_end + eps:
            return clip_starts[6] + (t - inflate_start)
        raise RuntimeError(f"Source time {t:.6f} not covered by kept source ranges.")

    new_segments: List[dict] = []

    # Keep first step unchanged (mapped)
    new_segments.append(
        clone_segment(
            seg_remove_dustcap,
            map_source_time(float(seg_remove_dustcap["start_time"])),
            map_source_time(float(seg_remove_dustcap["end_time"])),
        )
    )

    # Split deflate into error + fix
    new_segments.append(
        clone_segment(
            seg_def,
            map_source_time(def_start),
            clip_starts[2],
            desc=DEF_ERR_DESC,
        )
    )
    new_segments.append(
        clone_segment(
            seg_def,
            clip_starts[2],
            map_source_time(def_end),
            desc=DEF_FIX_DESC,
        )
    )

    # Middle unchanged steps
    for seg in [seg_get_lever, seg_insert_lever, seg_pry, seg_run_lever, seg_separate]:
        new_segments.append(
            clone_segment(
                seg,
                map_source_time(float(seg["start_time"])),
                map_source_time(float(seg["end_time"])),
            )
        )

    # Split fit step into error + fix
    new_segments.append(
        clone_segment(
            seg_fit,
            map_source_time(fit_start),
            clip_starts[3],
            desc=FIT_ERR_DESC,
        )
    )
    new_segments.append(
        clone_segment(
            seg_fit,
            clip_starts[3],
            clip_starts[4],
            desc=FIT_FIX_DESC,
        )
    )

    # Push step before moved valve
    new_segments.append(
        clone_segment(
            seg_push,
            clip_starts[4],
            clip_starts[5],
        )
    )

    # Moved valve as late correction
    new_segments.append(
        clone_segment(
            seg_valve,
            clip_starts[5],
            clip_starts[6],
            desc=VALVE_LATE_DESC,
        )
    )

    # Inflate + final dust cap mapped from tail clip
    new_segments.append(
        clone_segment(
            seg_inflate,
            clip_starts[6] + (float(seg_inflate["start_time"]) - inflate_start),
            clip_starts[6] + (float(seg_inflate["end_time"]) - inflate_start),
        )
    )
    new_segments.append(
        clone_segment(
            seg_final_dustcap,
            clip_starts[6] + (float(seg_final_dustcap["start_time"]) - inflate_start),
            clip_starts[6] + (float(seg_final_dustcap["end_time"]) - inflate_start),
        )
    )

    new_segments = sorted(new_segments, key=lambda s: (float(s["start_time"]), float(s["end_time"])))

    print("\nNew segments (chronological):")
    for i, s in enumerate(new_segments):
        print(f"[{i:02d}] {float(s['start_time']):8.3f}–{float(s['end_time']):8.3f} | {s.get('step_description')}")

    take_out = dict(take)
    take_out["segments"] = new_segments
    write_single_take_split_like_json(
        split_json_path=args.split_json,
        take_obj=take_out,
        out_path=args.new_annotations_out,
    )

    # manifest = {
    #     "take_name": args.take_name,
    #     "run_id": run_id,
    #     "final_video": str(out_final),
    #     "final_duration": final_dur,
    #     "clips": [
    #         {"name": "clip0_0_to_keep", "path": str(c0), "duration": durs[0]},
    #         {"name": "clip1_genA", "path": str(c1), "duration": durs[1]},
    #         {"name": "clip2_fix_to_fitstart", "path": str(c2), "duration": durs[2]},
    #         {"name": "clip3_genB", "path": str(c3), "duration": durs[3]},
    #         {"name": "clip4_push", "path": str(c4), "duration": durs[4]},
    #         {"name": "clip5_valve_moved", "path": str(c5), "duration": durs[5]},
    #         {"name": "clip6_inflate_tail", "path": str(c6), "duration": durs[6]},
    #     ],
    #     "fades": fades,
    #     "clip_starts": clip_starts,
    #     "anchors": {
    #         "step1_keep_until": keep_until,
    #         "step1_fix_start": fix_start,
    #         "step2_gen_start": fit_gen_start,
    #         "step2_end": fit_end,
    #         "valve_start": valve_start,
    #         "valve_end": valve_end,
    #         "push_start": push_start,
    #         "push_end": push_end,
    #         "inflate_start": inflate_start,
    #     },
    # }
    # manifest_path = work / "run_manifest.json"
    # manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # print("Saved run manifest:", manifest_path)
