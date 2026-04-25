#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
video_generator_sfu_cooking_008_5_kling.py

Replace the last segment of take sfu_cooking_008_5 ("Pour coffee into milk in the cup")
with two AI-generated clips:
  A) wrong_execution: spill some coffee onto countertop
  B) correction: wipe spill + finish pour into the cup

Then splice into the original video with smooth crossfades.

Requires:
  - ffmpeg, ffprobe
  - requests, pyjwt
  - env: KLING_ACCESS_KEY, KLING_SECRET_KEY

Typical workflow:
  1) Create local feature ref clip (8-10s), upload to Google Drive:
     python video_generator_sfu_cooking_008_5.py --make_feature_ref

  2) Run generation + splice:
     python video_generator_sfu_cooking_008_5.py \
       --feature_drive_url "https://drive.google.com/file/d/<ID>/view?usp=sharing"
"""

import os
import time
import json
import base64
import subprocess
import re
import argparse
from pathlib import Path

import requests
import jwt  # pip install pyjwt

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------
# Kling OmniVideo API
# ---------------------------
BASE_URL = "https://api-singapore.klingai.com"

def make_jwt_token(access_key: str, secret_key: str) -> str:
    now = int(time.time())
    payload = {"iss": access_key, "exp": now + 1800, "nbf": now - 5}
    token = jwt.encode(payload, secret_key, algorithm="HS256", headers={"typ": "JWT"})
    return token.decode("utf-8") if isinstance(token, bytes) else token

def auth_headers() -> dict:
    # Fail with a clear message instead of KeyError (saves time when running on new machines).
    ak = os.getenv("KLING_ACCESS_KEY")
    sk = os.getenv("KLING_SECRET_KEY")
    if not ak or not sk:
        raise RuntimeError(
            "Missing KLING_ACCESS_KEY / KLING_SECRET_KEY in environment. "
            "Export them before running this script."
        )
    token = make_jwt_token(ak, sk)
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def create_omni_task(prompt: str, first_frame: str, end_frame: str | None = None,
                     mode: str = "pro", duration: int = 10,
                     external_task_id: str | None = None,
                     feature_video_url: str | None = None) -> str:
    """
    NOTE: duration is expected to be an integer seconds value (typically 3..10).
    """
    url = f"{BASE_URL}/v1/videos/omni-video"
    body = {
        "model_name": "kling-video-o1",
        "prompt": prompt,
        "mode": mode,
        "duration": int(duration),
    }
    # IMPORTANT (Kling Omni/O1 API constraint):
    # If you provide a reference video (video_list), you MUST NOT provide a tail/end frame.
    # Therefore:
    #   - with feature_video_url: send ONLY first_frame
    #   - without feature_video_url: you may send first_frame + end_frame
    image_list = [{"image_url": first_frame, "type": "first_frame"}]
    if (end_frame is not None) and (not feature_video_url):
        image_list.append({"image_url": end_frame, "type": "end_frame"})
    body["image_list"] = image_list
    if feature_video_url:
        body["video_list"] = [{
            "video_url": feature_video_url,
            "refer_type": "feature",
            "keep_original_sound": "no",
        }]
    if external_task_id:
        body["external_task_id"] = external_task_id

    r = requests.post(url, headers=auth_headers(), json=body, timeout=60)
    if r.status_code >= 400:
        print("HTTP", r.status_code)
        print("Response:", r.text)
        print("Body:", json.dumps(body)[:2000], "...")
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"Create task failed: {j}")
    return j["data"]["task_id"]

def get_task(task_id: str) -> dict:
    url = f"{BASE_URL}/v1/videos/omni-video/{task_id}"
    r = requests.get(url, headers=auth_headers(), timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"Get task failed: {j}")
    return j["data"]

def wait_task(task_id: str, poll_sec: int = 5, timeout_sec: int = 1200) -> dict:
    t0 = time.time()
    while True:
        d = get_task(task_id)
        st = d.get("task_status")
        if st in ("succeed", "failed"):
            return d
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Timeout waiting task={task_id}, last_status={st}")
        time.sleep(poll_sec)

def download_file(url: str, out_path: str) -> None:
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


# ---------------------------
# ffmpeg helpers
# ---------------------------
def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)

def ffprobe_json(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-of", "json", "-show_format", "-show_streams", video_path]
    return json.loads(subprocess.check_output(cmd).decode("utf-8"))

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

def extract_frame_png(video_path: str, t_sec: float, out_png: str) -> None:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-update", "1",
        out_png
    ]
    _run(cmd)

def png_to_base64(png_path: str) -> str:
    b = Path(png_path).read_bytes()
    return base64.b64encode(b).decode("utf-8")

def extract_segment(video_path: str, t_start: float, t_end: float, out_mp4: str) -> None:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    # try stream copy, fallback to re-encode
    cmd = ["ffmpeg", "-y", "-ss", f"{t_start:.3f}", "-to", f"{t_end:.3f}", "-i", video_path, "-c", "copy", out_mp4]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{t_start:.3f}", "-to", f"{t_end:.3f}",
            "-i", video_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            out_mp4
        ]
        _run(cmd)

def trim_to_duration(in_mp4: str, out_mp4: str, dur_sec: float) -> None:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_mp4,
        "-t", f"{dur_sec:.3f}",
        "-c", "copy",
        out_mp4
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", in_mp4,
            "-t", f"{dur_sec:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            out_mp4
        ]
        _run(cmd)

def normalize_clip(in_mp4: str, out_mp4: str, width: int, height: int, fps: int = 60) -> None:
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    j = ffprobe_json(in_mp4)
    has_audio = any(s.get("codec_type") == "audio" for s in j["streams"])
    vf = f"scale={width}:{height},fps={fps},format=yuv420p"

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

def concat_with_crossfade(clips: list[str], out_mp4: str, fade: float = 0.25) -> None:
    """
    Smooth concat via xfade + acrossfade.
    Assumes all clips are already normalized: same size/fps/codec/audio layout.
    NOTE: use a slightly longer fade between generated clips to hide minor continuity noise.
    """
    assert len(clips) >= 2
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)

    # Get durations
    durs = [ffprobe_video_specs(c)["duration"] for c in clips]

    # Per-transition fades can be injected by encoding them into the clip filenames conventionally,
    # but we keep it simple and pass a list of fades in via a closure from main (see below).

    # Build filter graph
    # video: [0:v][1:v]xfade=... -> v01; [v01][2:v]xfade=... -> v012 ...
    # audio: [0:a][1:a]acrossfade=... -> a01; ...
    inputs = []
    for c in clips:
        inputs += ["-i", c]

    v_labels = [f"[{i}:v]" for i in range(len(clips))]
    a_labels = [f"[{i}:a]" for i in range(len(clips))]

    fc = []
    # Merge sequentially with xfade/acrossfade
    v_prev = v_labels[0]
    a_prev = a_labels[0]
    t_acc = durs[0]
    for i in range(1, len(clips)):
        v_out = f"[v{i}]"
        a_out = f"[a{i}]"
        # offset = time when xfade starts in the accumulated timeline
        # start fade at (t_acc - fade)
        this_fade = float(getattr(concat_with_crossfade, "_fades", {}).get(i - 1, fade))
        off = max(0.0, t_acc - this_fade)
        fc.append(f"{v_prev}{v_labels[i]}xfade=transition=fade:duration={this_fade}:offset={off}{v_out}")
        fc.append(f"{a_prev}{a_labels[i]}acrossfade=d={this_fade}{a_out}")
        v_prev = v_out
        a_prev = a_out
        t_acc = t_acc + durs[i] - this_fade  # overlap reduces total by fade

    filter_complex = ";".join(fc)

    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", v_prev,
        "-map", a_prev,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        out_mp4
    ]
    _run(cmd)


# ---------------------------
# split_50.json helpers
# ---------------------------
def load_take(split_json_path: str, take_name: str) -> dict:
    data = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
    take = next(a for a in data["annotations"] if a["take_name"] == take_name)
    return take

def sorted_segments(take: dict) -> list[dict]:
    segs = sorted(take["segments"], key=lambda s: (float(s["start_time"]), float(s["end_time"])))
    for i, s in enumerate(segs):
        s["position"] = i
    return segs

def find_segment_by_desc(segs: list[dict], desc: str) -> dict:
    for s in segs:
        if (s.get("step_description") or "").strip() == desc.strip():
            return s
    raise ValueError(f"Segment not found by description: {desc}")

def print_segments_table(segs: list[dict]) -> None:
    for s in segs:
        print(f"[{s['position']:02d}] {float(s['start_time']):8.3f}–{float(s['end_time']):8.3f} | {s.get('step_description')}")


def drive_view_to_direct(url: str) -> str:
    m = re.search(r"/file/d/([^/]+)/", url)
    if not m:
        raise ValueError("Could not parse file_id from drive url")
    file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--video_path",
        default=str(REPO_ROOT / "local" / "egoexo4d" / "videos_ego" / "sfu_cooking_008_5.mp4"),
    )
    p.add_argument(
        "--split_json",
        default=str(REPO_ROOT / "local" / "egoexo4d" / "split_50.json"),
    )
    p.add_argument("--take_name", default="sfu_cooking_008_5")
    p.add_argument(
        "--workdir",
        default=str(REPO_ROOT / "local" / "outputs" / "video_generation" / "kling" / "sfu_cooking_008_5"),
    )
    p.add_argument("--feature_drive_url", default="", help="Google Drive VIEW url for the feature ref video (optional but recommended)")
    p.add_argument("--make_feature_ref", action="store_true", help="Only extract local feature_ref.mp4 and exit (upload it to Drive)")
    p.add_argument("--fade", type=float, default=0.25, help="Crossfade seconds for normal transitions (e.g., part1<->generated<->part2)")
    p.add_argument("--fade_generated", type=float, default=0.35, help="Crossfade seconds specifically between generated clips A->B")
    # Feature-ref timing INSIDE the replaced step (critical for realistic pouring physics).
    # Example: offset=3.5 means start ref at (cut_from + 3.5s) within the original 'pour' segment.
    p.add_argument("--feature_offset", type=float, default=3.5, help="Seconds after cut_from to start feature_ref (inside the replaced segment)")
    p.add_argument("--feature_len", type=float, default=8.0, help="Length of feature_ref in seconds (inside the replaced segment)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    video_path = args.video_path
    split_json = args.split_json
    take_name = args.take_name

    specs = ffprobe_video_specs(video_path)
    print("Video specs:", specs)

    take = load_take(split_json, take_name)
    segs = sorted_segments(take)
    print("Chronological segments:")
    print_segments_table(segs)

    # Original last step segment to replace
    target_desc = "Pour coffee into milk in the cup"
    target = find_segment_by_desc(segs, target_desc)

    cut_from = float(target["start_time"])
    cut_to = float(target["end_time"])
    removed_len = max(0.0, cut_to - cut_from)
    print(f"Replacing segment '{target_desc}' @ {cut_from:.3f}–{cut_to:.3f} (len={removed_len:.3f}s)")

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    # 1) Build a local feature reference clip.
    # IMPORTANT: For realistic pouring/spilling physics, the feature ref must show the ACTUAL pour motion.
    # Therefore we extract feature_ref INSIDE the replaced segment, not from the previous step.
    seg_len = max(0.0, cut_to - cut_from)
    feat_start = cut_from + float(args.feature_offset)
    feat_len = float(args.feature_len)
    # Clamp to stay inside the replaced segment
    feat_start = max(cut_from, min(feat_start, cut_to - 0.10))
    feat_end = min(cut_to - 0.05, feat_start + max(1.0, feat_len))
    # Fallback if the segment is too short (should not happen here, but keep robust)
    if feat_end <= feat_start + 0.25:
        fallback_len = min(9.0, max(2.0, seg_len))
        feat_start = max(0.0, cut_from - fallback_len)
        feat_end = cut_from
        print("WARNING: replaced segment too short for inside-step feature_ref; falling back to pre-step ref.")
    else:
        print("Feature ref will be extracted INSIDE the replaced step for better motion consistency.")

    feature_local = str(work / "feature_ref.mp4")

    extract_segment(video_path, feat_start, feat_end, feature_local)
    print("Local feature ref:", feature_local, f"({feat_start:.3f}–{feat_end:.3f})")

    if args.make_feature_ref:
        print("\nUpload this feature_ref.mp4 to Google Drive (anyone-with-link).")
        print("Then rerun with: --feature_drive_url 'https://drive.google.com/file/d/<ID>/view?usp=sharing'")
        raise SystemExit(0)

    feature_url = ""
    if args.feature_drive_url.strip():
        feature_url = drive_view_to_direct(args.feature_drive_url.strip())
        print("Using feature_url:", feature_url)
    else:
        print("WARNING: no --feature_drive_url provided; generation will run WITHOUT feature reference (less consistent).")

    # 2) Decide durations for two clips (must be <=10 each for Kling O1)
    # We aim to roughly match removed_len (≈18.9s) with 9s + 10s.
    dur_a = 9
    dur_b = 10
    print(f"Planned generated durations: spill={dur_a}s, correction={dur_b}s (total={dur_a+dur_b}s)")

    # 3) Extract boundary frames from original
    # Use frames INSIDE the segment we are replacing:
    # - first frame: just AFTER cut_from (start of the replaced segment)
    # - end frame:   just BEFORE cut_to (end of the replaced segment)
    # This improves visual continuity vs. sampling frames from the neighboring segments.
    eps_in = 0.05
    t_first = min(specs["duration"] - 0.05, max(0.0, cut_from + eps_in))
    t_end   = min(specs["duration"] - 0.05, max(0.0, cut_to   - eps_in))

    frames_dir = work / "frames"
    first_png = str(frames_dir / "first_frame.png")
    end_png = str(frames_dir / "end_frame.png")

    extract_frame_png(video_path, t_first, first_png)
    extract_frame_png(video_path, t_end, end_png)

    first_b64 = png_to_base64(first_png)
    end_b64 = png_to_base64(end_png)

    # 4) Generate Clip A (spill)

    prompt_a = (
        "Egocentric head-mounted POV, wide-angle fisheye with strong vignette. "
        "Same kitchen countertop scene. Continue naturally from the first frame. "
        "The small cup with milk is on the countertop in front of you. "
        "Action: start pouring coffee into the cup, but make a realistic mistake: "
        "tilt the coffee container slightly too fast so the stream briefly misses the cup and "
        "spills a SMALL amount of coffee onto the countertop right next to the cup (make a small puddle or splash). "
        "Some coffee should still go into the cup. Then stop the spill and set the coffee container down. "
        "Keep the spill visible on the countertop. "
        "Realistic liquid behavior, no teleporting, no sudden camera jumps. "
        "Do NOT introduce new objects, do not rearrange the countertop, preserve lighting and existing items. "
        "No tea bags, no sugar, no stirring, no washing, no text overlays."
    )  

    ext_a = f"{take_name}_E01_spill_{int(time.time())}"
    # Clip A uses the feature reference video (if provided) to match motion/camera style.
    # When feature_video_url is present, Kling forbids end_frame (tail image).
    end_a = None if (feature_url and feature_url.strip()) else end_b64
    task_a = create_omni_task(
        prompt=prompt_a,
        first_frame=first_b64,
        end_frame=end_a,
        mode="pro",
        duration=dur_a,
        external_task_id=ext_a,
        feature_video_url=feature_url or None
    )
    print("Kling task A:", task_a)
    res_a = wait_task(task_a)
    if res_a["task_status"] == "failed":
        print("A failed:", res_a.get("task_status_msg"))
        raise SystemExit(1)

    url_a = res_a["task_result"]["videos"][0]["url"]
    clip_a_raw = str(work / f"clipA_raw_{task_a}.mp4")
    download_file(url_a, clip_a_raw)
    print("Downloaded clip A:", clip_a_raw)

    # 5) Use last frame of clip A as first frame for clip B
    mid_png = str(frames_dir / "mid_from_clipA_last.png")
    # take last frame slightly before end
    a_specs = ffprobe_video_specs(clip_a_raw)
    extract_frame_png(clip_a_raw, max(0.0, a_specs["duration"] - 0.02), mid_png)
    mid_b64 = png_to_base64(mid_png)

    # 6) Generate Clip B (wipe + finish pour)

    prompt_b = (
        "Egocentric head-mounted POV, wide-angle fisheye with strong vignette. "
        "Same countertop, continue immediately from the first frame: there is a small coffee spill next to the cup. "
        "Action: put the coffee container down if it is in hand. "
        "Grab the YELLOW cloth/towel that is already on the countertop to the right, "
        "wipe the small coffee puddle with a few realistic strokes until the surface looks mostly clean/dry. "
        "Then pick up the coffee container again and carefully finish pouring coffee into the cup with milk, "
        "without any further spilling. "
        "Maintain consistent objects and placements, realistic motion, no cuts. "
        "Do NOT introduce new objects, do not change the scene, preserve lighting. "
        "No tea bags, no sugar, no stirring, no washing, no text overlays."
    )

    ext_b = f"{take_name}_E01_correction_{int(time.time())}"
    # Clip B MUST be end-anchored back to the original (end_frame=end_b64),
    # so we MUST NOT use a feature reference video here (Kling forbids tail image with video input).
    task_b = create_omni_task(
        prompt=prompt_b,
        first_frame=mid_b64,
        end_frame=end_b64,
        mode="pro",
        duration=dur_b,
        external_task_id=ext_b,
        feature_video_url=None
    )
    print("Kling task B:", task_b)
    res_b = wait_task(task_b)
    if res_b["task_status"] == "failed":
        print("B failed:", res_b.get("task_status_msg"))
        raise SystemExit(1)

    url_b = res_b["task_result"]["videos"][0]["url"]
    clip_b_raw = str(work / f"clipB_raw_{task_b}.mp4")
    download_file(url_b, clip_b_raw)
    print("Downloaded clip B:", clip_b_raw)

    # 7) Cut original part1 (up to cut_from) and (optional) part2 (after cut_to)
    parts_dir = work / "parts"
    part1_raw = str(parts_dir / "part1_raw.mp4")
    extract_segment(video_path, 0.0, cut_from, part1_raw)

    part2_raw = ""
    if cut_to < specs["duration"] - 0.2:
        part2_raw = str(parts_dir / "part2_raw.mp4")
        extract_segment(video_path, cut_to, specs["duration"], part2_raw)

    # 8) Normalize all clips to match for blending/concat
    norm_dir = work / "norm"
    part1 = str(norm_dir / "part1.mp4")
    clipA = str(norm_dir / "clipA.mp4")
    clipB = str(norm_dir / "clipB.mp4")
    normalize_clip(part1_raw, part1, specs["width"], specs["height"], fps=60)
    normalize_clip(clip_a_raw, clipA, specs["width"], specs["height"], fps=60)
    normalize_clip(clip_b_raw, clipB, specs["width"], specs["height"], fps=60)

    # 9) Optional: trim generated clips if you want exact total replacement length
    # Here we keep them as-is (9s + 10s). If you want to match removed_len exactly:
    # trim_to_duration(clipB, clipB_trimmed, max(3.0, removed_len - dur_a))
    # (but note Kling outputs can vary slightly anyway)

    clips = [part1, clipA, clipB]
    if part2_raw:
        part2 = str(norm_dir / "part2.mp4")
        normalize_clip(part2_raw, part2, specs["width"], specs["height"], fps=60)
        clips.append(part2)

    out_final = str(work / f"{take_name}_E01_spill_and_correction_final.mp4")
    # Use a slightly longer crossfade specifically between generated clips A->B
    fades = {}
    # transition indices: 0 is clips[0]->clips[1], 1 is clips[1]->clips[2], etc.
    # Here clips = [part1, clipA, clipB, (optional) part2]
    if len(clips) >= 3:
        fades[1] = float(args.fade_generated)  # clipA -> clipB
    concat_with_crossfade._fades = fades
    concat_with_crossfade(clips, out_final, fade=float(args.fade))
    print("Saved final video:", out_final)

    # Also save step-level clips (for dataset bookkeeping)
    stepA_out = str(work / f"{take_name}_step13_wrong_execution_spill.mp4")
    stepB_out = str(work / f"{take_name}_step14_correction_wipe_and_pour.mp4")
    _run(["cp", clipA, stepA_out])
    _run(["cp", clipB, stepB_out])
    print("Saved step clips:\n ", stepA_out, "\n ", stepB_out)
