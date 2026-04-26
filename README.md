# PIE-V

[![CI](https://github.com/ologin/PIE-V/actions/workflows/ci.yml/badge.svg)](https://github.com/ologin/PIE-V/actions/workflows/ci.yml)

**PIE-V** (Psychologically Inspired Error injection for Videos) is a framework for
constructing and benchmarking mistake-aware egocentric procedural videos. It augments
clean Ego-Exo4D keystep procedures with structured human-plausible mistakes,
recovery corrections, validated text rewrites, and video edit specifications.

[Paper: How to Correctly Make Mistakes](https://arxiv.org/abs/2604.15134)  
[Project page: PIE-V](https://ologin.github.io/PIE-V/)  
[Release: cvprw26](https://github.com/ologin/PIE-V/releases/tag/cvprw26)

## What Is Included

- A policy-based error planner for five error types: wrong execution, deletion,
  substitution, insertion, and transposition.
- A correction simulator that samples detection, action, latency, and repair type.
- LLM writer and judge stages for cascade-consistent procedure rewrites.
- Semantic-role resources used by the planner, writer, and judge.
- Aggregated human annotation summaries and the public annotation template.
- Example PIE-V outputs for the 50-scenario Ego-Exo4D subset.
- Public-release checks for data integrity, project-page assets, tracked media,
  local paths, and exposed credentials.

Raw Ego-Exo4D videos, original Ego-Exo4D annotation files, and generated MP4 media
remain outside git. The [`cvprw26` GitHub release](https://github.com/ologin/PIE-V/releases/tag/cvprw26) is the media location used by
the paper and project page.

## Repository Layout

```text
src/piev/                 Core PIE-V planner, writer, judge, and utilities
prep/                     Data preparation and semantic representation helpers
analysis/                 Rubric/agreement analysis utilities
baselines/freeform/       Freeform LLM baseline and diagnostics
configs/                  Default local path configuration
data/resources/           Semantic role resources and SRL cache
data/manifests/           Take-name list for the 50-scenario Ego-Exo4D subset
data/examples/            Example PIE-V plans, judged rewrites, and annotations
data/annotations/         Aggregated human annotation summary and template
examples/video_generation Case-study notes for video editing scripts
docs/                     Static project page for GitHub Pages
tests/                    Public-release smoke tests
.github/workflows/        GitHub Actions checks
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional backends:

```bash
python -m pip install -e ".[qwen]"
python -m pip install -e ".[video]"
```

Development checks:

```bash
python -m pip install -e ".[dev]"
python -m ruff format --check .
python -m ruff check .
python -m compileall -q src prep analysis baselines examples/video_generation/scripts docs
python -m pytest -q
```

For OpenAI-backed SRL generation, writing, or judging:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.2"
```

For local Qwen-backed writing or judging, the default model ids are configured in
`configs/defaults.toml` and can be overridden with:

```bash
export QWEN_TEXT_MODEL="Qwen/Qwen2.5-32B-Instruct"
export QWEN_VL_MODEL="Qwen/Qwen3-VL-32B-Instruct"
```

## Local Ego-Exo4D Inputs

The source Ego-Exo4D videos and annotations are distributed under the original
Ego-Exo4D access terms. For local reproduction, download the takes listed in
`data/manifests/take_names.txt` and arrange them as follows:

```text
local/egoexo4d/
  split_50.json
  keystep_train.json
  videos_ego/
    <take_name>.mp4
```

The same paths are recorded in `configs/defaults.toml`. They can also be
overridden with `PIEV_SPLIT50_PATH`, `PIEV_KEYSTEP_TRAIN_PATH`,
`PIEV_VIDEO_ROOT`, and `PIEV_EGOEXO4D_ROOT`.

## Reproduce The Text Pipeline

The example resources in `data/` are enough to inspect outputs and run downstream
stages. To regenerate the full error plan, provide the local Ego-Exo4D split and
keystep taxonomy files.

```bash
piev-error-plan \
  --split50 local/egoexo4d/split_50.json \
  --vocab_csv data/resources/split_50_vocabulary.csv \
  --semrep_json data/resources/semantic_representations_split_50.json \
  --roles_csv data/resources/semantic_roles.csv \
  --roles_by_predicate_csv data/resources/semantic_roles_by_predicate.csv \
  --keystep local/egoexo4d/keystep_train.json \
  --out local/outputs/split_50_error_plan.json \
  --seed 123

piev-corrections \
  --input local/outputs/split_50_error_plan.json \
  --out local/outputs/split_50_error_plan_with_corrections.json \
  --seed 123

piev-write-instructions \
  --model openai \
  --input local/outputs/split_50_error_plan_with_corrections.json \
  --out local/outputs/split_50_error_instructions.json \
  --include_semrep \
  --vocab_csv data/resources/split_50_vocabulary.csv \
  --semrep_json data/resources/semantic_representations_split_50.json

piev-judge-instructions \
  --model openai \
  --plan local/outputs/split_50_error_plan_with_corrections.json \
  --rewrite local/outputs/split_50_error_instructions_openai.json \
  --out local/outputs/split_50_error_instructions_openai_judged.json \
  --report_base local/outputs/openai_judged_report
```

## Data Notes

`data/examples/split_50_error_plan_with_corrections.json` contains the sampled
paper-scale text plan: 50 takes, 102 mistakes, and 27 corrections.
`data/annotations/PIE-V_Annotation_Template.xlsx` is the public template for the
rubric fields used in the human evaluation.

`data/manifests/take_names.txt` contains one Ego-Exo4D `take_name` per line. The
Ego-Exo4D `split_50.json` and `keystep_train.json` files remain local inputs.

## Video Generation

The paper uses provider-specific video-editing case studies for Seedance, Kling,
Runway, and Veo workflows. The examples document the representative
cases used for the project page; the full text pipeline covers all five PIE-V
error types.

## Citation

```bibtex
@article{loginova2026correctly,
  title={How to Correctly Make Mistakes: A Framework for Constructing and Benchmarking Mistake Aware Egocentric Procedural Videos},
  author={Loginova, Olga and Keller, Frank},
  journal={arXiv preprint arXiv:2604.15134},
  year={2026}
}
```
