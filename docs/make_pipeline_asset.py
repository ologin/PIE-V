#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import textwrap

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("This script needs Pillow: python -m pip install pillow") from exc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "piev-pipeline.png"

W, H = 1800, 1050
BG = "#f7f4ee"
GRID = "#e8dfd1"
INK = "#18212b"
MUTED = "#46515f"
LINE = "#d2cabf"
DARK = "#17212c"
WHITE = "#fffdf8"
YELLOW = "#ffcc3d"
COLORS = ["#00838a", "#6b4dbb", "#c9241c", "#2e7d32", "#9a6a00", "#005bbb"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(62, True)
FONT_SUBTITLE = font(28, True)
FONT_NUM = font(38, True)
FONT_CARD_TITLE = font(25, True)
FONT_CARD_BODY = font(21, True)
FONT_STAT = font(72, True)
FONT_STAT_LABEL = font(27, True)
FONT_TAG = font(24, True)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, fnt, fill: str, line_gap: int = 6) -> None:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)

    x, y = xy
    line_h = fnt.size + line_gap
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h


def arrow(draw: ImageDraw.ImageDraw, x1: int, y: int, x2: int) -> None:
    draw.line((x1, y, x2, y), fill="#9ba4ae", width=5)
    draw.polygon([(x2, y), (x2 - 14, y - 9), (x2 - 14, y + 9)], fill="#9ba4ae")


def draw_grid(draw: ImageDraw.ImageDraw) -> None:
    for x in range(0, W + 1, 90):
        draw.line((x, 0, x, H), fill=GRID, width=1)
    for y in range(0, H + 1, 90):
        draw.line((0, y, W, y), fill=GRID, width=1)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_grid(draw)

    draw.text((92, 84), "PIE-V: Psychologically Inspired Error Injection for Video", font=FONT_TITLE, fill=INK)
    draw.text(
        (96, 166),
        "Structured mistakes, recovery traces, and video edits for egocentric procedural benchmarks",
        font=FONT_SUBTITLE,
        fill=MUTED,
    )

    cards = [
        ("1", "Clean keysteps", "Ego-Exo4D procedures with timings and steps"),
        ("2", "Semantic layer", "Predicate-role SemRep plus role impact priors"),
        ("3", "Error planner", "WE, D, S, I, T sampled by phase and step load"),
        ("4", "Correction planner", "Detection, action, latency, and recovery type"),
        ("5", "LLM writer + judge", "Cascade-consistent rewrites validated and repaired"),
        ("6", "Video synthesis", "Regenerate affected windows and stitch episodes"),
    ]

    x0, y0 = 96, 302
    cw, ch, gap = 244, 276, 34
    for i, (num, title, body) in enumerate(cards):
        x = x0 + i * (cw + gap)
        draw.rounded_rectangle((x, y0, x + cw, y0 + ch), radius=24, fill=WHITE, outline=LINE, width=3)
        draw.rounded_rectangle((x + 22, y0 + 22, x + 82, y0 + 84), radius=20, fill=COLORS[i])
        draw.text((x + 52, y0 + 53), num, font=FONT_NUM, fill=WHITE, anchor="mm")
        draw_wrapped(draw, (x + 23, y0 + 118), title, cw - 46, FONT_CARD_TITLE, INK, line_gap=4)
        draw_wrapped(draw, (x + 23, y0 + 184), body, cw - 42, FONT_CARD_BODY, MUTED, line_gap=6)
        if i < len(cards) - 1:
            arrow(draw, x + cw + 4, y0 + 126, x + cw + gap - 6)

    band = (96, 650, W - 96, 920)
    draw.rounded_rectangle(band, radius=28, fill=DARK)
    stats = [("17", "tasks"), ("50", "Ego-Exo4D scenarios"), ("102", "mistakes"), ("27", "corrections"), ("9", "rubric metrics")]
    stat_x = [180, 520, 850, 1150, 1440]
    for (value, label), x in zip(stats, stat_x):
        draw.text((x, 755), value, font=FONT_STAT, fill=YELLOW, anchor="mm")
        draw.text((x, 824), label, font=FONT_STAT_LABEL, fill=WHITE, anchor="mm")

    tags = [("WE", "#c9241c"), ("D", "#e98200"), ("S", "#6b4dbb"), ("I", "#00838a"), ("T", "#005bbb")]
    tx = 1120
    for label, color in tags:
        draw.rounded_rectangle((tx, 858, tx + 72, 902), radius=16, fill=color)
        draw.text((tx + 36, 880), label, font=FONT_TAG, fill=WHITE, anchor="mm")
        tx += 100

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"Wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
