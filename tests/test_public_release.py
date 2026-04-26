from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from piev.config import REPO_ROOT, load_settings


PUBLIC_PLAN = REPO_ROOT / "data/examples/split_50_error_plan_with_corrections.json"
TAKE_NAMES = REPO_ROOT / "data/manifests/take_names.txt"
ANNOTATION_SUMMARY = REPO_ROOT / "data/annotations/annotation_summary.csv"

ERROR_TYPES = {
    "wrong_execution",
    "deletion",
    "substitution",
    "insertion",
    "transposition",
}

RUBRIC_COLUMNS = {
    "Error Validity",
    "Text Plausibility",
    "Confusability / Difficulty to Notice",
    "Procedure Logic",
    "Procedure Logic (Annotator Confidence)",
    "Sequence Consistency Score",
    "State-Change Coherence",
    "Taxonomy Fit (Error Type)",
    "Video Plausibility",
    "Text–Video Grounding Consistency (episode-level)",
}

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}


def test_default_config_keeps_public_inputs_local_to_repo() -> None:
    settings = load_settings()

    assert settings.openai_model == "gpt-5.2"
    assert settings.qwen_text_model == "Qwen/Qwen2.5-32B-Instruct"
    assert settings.qwen_vl_model == "Qwen/Qwen3-VL-32B-Instruct"

    assert settings.take_names_path == REPO_ROOT / "data/manifests/take_names.txt"
    assert settings.split50_path == REPO_ROOT / "local/egoexo4d/split_50.json"
    assert settings.keystep_train_path == REPO_ROOT / "local/egoexo4d/keystep_train.json"
    assert settings.videos_root == REPO_ROOT / "local/egoexo4d/videos_ego"


def test_public_take_manifest_has_50_unique_take_names() -> None:
    take_names = [line.strip() for line in TAKE_NAMES.read_text().splitlines() if line.strip()]

    assert len(take_names) == 50
    assert len(set(take_names)) == 50
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", take_name) for take_name in take_names)


def test_public_plan_matches_paper_scale_counts() -> None:
    plan = json.loads(PUBLIC_PLAN.read_text())

    takes = plan["takes"]
    errors = [error for take in takes.values() for error in take["errors"]]
    corrections = [
        correction for take in takes.values() for correction in take.get("corrections", [])
    ]
    error_counts = Counter(error["type"] for error in errors)

    assert len(takes) == 50
    assert len(errors) == 102
    assert len(corrections) == 27
    assert set(error_counts) == ERROR_TYPES
    assert plan["meta"]["corrections_meta"]["stats"]["n_corrections_total"] == 27


def test_public_plan_references_valid_steps_and_corrections() -> None:
    plan = json.loads(PUBLIC_PLAN.read_text())

    for take in plan["takes"].values():
        steps = take["steps"]
        errors_by_id = {error["event_id"]: error for error in take["errors"]}

        for error in take["errors"]:
            assert error["type"] in ERROR_TYPES
            assert 0 <= error["step_index"] < len(steps)

        for correction in take.get("corrections", []):
            target_id = correction["targets_error_id"]
            assert target_id in errors_by_id
            assert correction["targets_error_type"] == errors_by_id[target_id]["type"]
            assert 0 <= correction["detect_at_step_index"] < len(steps)


def test_annotation_summary_contains_public_rubric_columns() -> None:
    with ANNOTATION_SUMMARY.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert rows
    assert RUBRIC_COLUMNS.issubset(reader.fieldnames or [])


def test_project_page_references_committed_preview_assets() -> None:
    html = (REPO_ROOT / "docs/index.html").read_text(encoding="utf-8")

    expected_assets = [
        "docs/assets/piev-pipeline.png",
        "docs/assets/georgiatech_correct_vs_generated.gif",
        "docs/assets/sfu_correct_vs_generated.gif",
    ]

    for asset in expected_assets:
        assert (REPO_ROOT / asset).exists()
        assert Path(asset).name in html


def test_git_does_not_track_large_video_media() -> None:
    tracked_files = _git_ls_files()
    forbidden_suffixes = {".avi", ".mkv", ".mov", ".mp4", ".webm"}

    assert not [path for path in tracked_files if path.suffix.lower() in forbidden_suffixes]


def test_public_text_files_do_not_expose_local_paths_or_secrets() -> None:
    tracked_text_files = [path for path in _git_ls_files() if path.suffix.lower() in TEXT_SUFFIXES]
    secret_patterns = [
        re.compile(r"/Users/olga/"),
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(
            r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?key)\s*=\s*['\"](?!\.\.\.)[^'\"]{12,}['\"]"
        ),
    ]

    matches: list[str] = []
    for path in tracked_text_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            if pattern.search(text):
                matches.append(str(path.relative_to(REPO_ROOT)))

    assert matches == []


def _git_ls_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line.strip()]
