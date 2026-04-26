from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "piev").exists():
            return p
    raise RuntimeError("Could not locate PIE-V repository root.")


REPO_ROOT = find_repo_root()


def _resolve_repo_path(value: str) -> Path:
    return (REPO_ROOT / value).resolve()


@dataclass(frozen=True)
class Settings:
    egoexo4d_root: Path
    outputs_root: Path
    resources_root: Path
    annotations_root: Path
    videos_root: Path
    frames_root: Path
    split50_path: Path
    keystep_train_path: Path
    take_names_path: Path
    openai_model: str
    qwen_text_model: str
    qwen_vl_model: str


def load_settings(config_path: str | Path | None = None) -> Settings:
    cfg_path = Path(config_path) if config_path else REPO_ROOT / "configs" / "defaults.toml"
    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)

    paths = raw["paths"]
    models = raw.get("models", {})

    return Settings(
        egoexo4d_root=Path(
            os.getenv("PIEV_EGOEXO4D_ROOT", _resolve_repo_path(paths["egoexo4d_root"]))
        ),
        outputs_root=Path(
            os.getenv("PIEV_OUTPUTS_ROOT", _resolve_repo_path(paths["outputs_root"]))
        ),
        resources_root=Path(
            os.getenv("PIEV_RESOURCES_ROOT", _resolve_repo_path(paths["resources_root"]))
        ),
        annotations_root=Path(
            os.getenv("PIEV_ANNOTATIONS_ROOT", _resolve_repo_path(paths["annotations_root"]))
        ),
        videos_root=Path(os.getenv("PIEV_VIDEO_ROOT", _resolve_repo_path(paths["videos_root"]))),
        frames_root=Path(os.getenv("PIEV_FRAMES_ROOT", _resolve_repo_path(paths["frames_root"]))),
        split50_path=Path(
            os.getenv("PIEV_SPLIT50_PATH", _resolve_repo_path(paths["split50_path"]))
        ),
        keystep_train_path=Path(
            os.getenv("PIEV_KEYSTEP_TRAIN_PATH", _resolve_repo_path(paths["keystep_train_path"]))
        ),
        take_names_path=Path(
            os.getenv("PIEV_TAKE_NAMES_PATH", _resolve_repo_path(paths["take_names_path"]))
        ),
        openai_model=os.getenv("OPENAI_MODEL", models.get("openai_model", "gpt-5.2")),
        qwen_text_model=os.getenv(
            "QWEN_TEXT_MODEL",
            models.get("qwen_text_model", "Qwen/Qwen2.5-32B-Instruct"),
        ),
        qwen_vl_model=os.getenv(
            "QWEN_VL_MODEL",
            models.get("qwen_vl_model", "Qwen/Qwen3-VL-32B-Instruct"),
        ),
    )
