# DINOv2-LeafCare

Plant disease image classification built on a fine-tuned **DINOv2 ViT-B/14**
backbone. The repository covers the whole path from raw images to a served
model: a dataset integrity audit, a training pipeline, a full evaluation suite
with read-only guarantees, and a FastAPI inference service.

Detailed reference documentation lives in [`docs/`](docs/README.md).

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Key metrics](#key-metrics)
- [Results](#results)
- [Project structure](#project-structure)
- [Core modules](#core-modules)
- [Installation](#installation)
- [Dataset layout](#dataset-layout)
- [Available commands](#available-commands)
- [Configuration and environment variables](#configuration-and-environment-variables)
- [API reference](#api-reference)
- [External services](#external-services)
- [Build and deployment](#build-and-deployment)
- [Quality gates](#quality-gates)
- [Contributing](#contributing)
- [License](#license)
- [Known gaps](#known-gaps)

---

## Overview

The model answers one question: **given a photo of a leaf, which disease is it?**

It is designed to sit behind a crop-identification step rather than replace one.
The LeafCare backend establishes which plant a photo shows before this model is
called, then picks the best-scoring disease class belonging to that crop. That is
why the API ranks the *entire* class vocabulary instead of a truncated top-5 —
a crop whose classes all fall outside the top five would otherwise be answered
with a different plant's disease.

Three principles run through the codebase:

| Principle | How it is enforced |
| --- | --- |
| **Configuration is the single source of truth** | Every tunable value lives in [`configs/config.yaml`](configs/config.yaml). No magic numbers in code, and no environment variables at all. |
| **The class count is never hardcoded** | Training and evaluation discover it from the dataset directory layout; the API reads it from the checkpoint. |
| **Evaluation is provably read-only** | The checkpoint file and the model weights are SHA-256 fingerprinted before and after the pass. A mismatch aborts the run rather than emitting a report. |

## Features

- **Dataset audit as a hard gate** — 13 integrity rules (missing or orphan
  classes, zero-byte files, undecodable images, duplicate filenames, class
  imbalance, and more). Every candidate image is opened twice, `verify()` then
  `load()`, so the verdict comes from a real decode. Any error-severity issue
  aborts training before the backbone is even loaded.
- **Transfer learning on DINOv2** — self-supervised ViT-B/14 backbone from the
  official `torch.hub` repository, plus a linear classification head. The
  backbone can be frozen or fine-tuned end to end from config.
- **Production training loop** — automatic mixed precision (bfloat16 when the
  GPU supports it, float16 otherwise), cosine learning-rate schedule, gradient
  clipping, early stopping, and full resume of model, optimizer, scheduler,
  scaler, epoch and history.
- **Complete evaluation suite** — top-1 and top-k accuracy, macro and weighted
  precision/recall/F1, one-vs-rest ROC-AUC and PR-AUC, expected calibration
  error with a reliability diagram, per-class metrics, a confusion matrix, and a
  latency benchmark with percentiles.
- **Pipeline verification** — a 13-stage end-to-end dry run that exercises the
  whole system on temporary artifacts before a real training run is committed.
- **REST inference service** — FastAPI with single and batch prediction,
  a typed error contract, request-ID correlation, upload validation, and an
  OpenAPI schema. The checkpoint is loaded once at startup, never per request.
- **Reproducibility** — one seed propagated across Python, NumPy, torch, CUDA
  and dataloader workers, with deterministic cuDNN algorithm selection.

## Tech stack

| Layer | Choice | Notes |
| --- | --- | --- |
| Language | Python ≥ 3.11 | Modern typing syntax throughout |
| Package manager | [uv](https://docs.astral.sh/uv/) | `pyproject.toml` + `uv.lock` are the only dependency source of truth |
| Deep learning | PyTorch, torchvision | Resolved from the pinned CUDA 13.0 wheel index |
| Backbone | DINOv2 ViT-B/14 | Loaded via `torch.hub` from `facebookresearch/dinov2` |
| Metrics | scikit-learn | ROC/PR curves, confusion matrix, classification report |
| Numerics | NumPy | Metric computation, seeding |
| Imaging | Pillow | Dataset audit decode, API upload decode |
| Plotting | matplotlib (`Agg`) | Headless figure export |
| API | FastAPI, Uvicorn, python-multipart | ASGI service and multipart upload parsing |
| Config | PyYAML | Single YAML file |
| Progress | tqdm | Training and evaluation progress bars |
| Lint | ruff | Line length 100, target py311, 9 rule families |
| Tests | `unittest` (stdlib) + httpx | 127 tests, no pytest dependency |

## Key metrics

| Dimension | Count |
| --- | --- |
| Python modules in `src/` | 42 (14 root modules, 5 sub-packages, 6 `__init__.py`) |
| Application code | ~8,600 lines |
| Test modules / test methods | 3 / **127** |
| Test code | ~1,600 lines |
| Configuration keys | **94**, in one file |
| CLI entry points | **6** |
| HTTP routes | **4** |
| Pydantic wire schemas | **8** |
| Typed API error classes | **5** (plus 4 exception handlers) |
| Dataset audit rules | **13** |
| Runtime dependencies | **11** (all imported by `src`) |
| Dev dependencies | 2 (`ruff`, `httpx`) |
| Database models | **0** — the service is stateless; there is no database |

## Results

Measured on the held-out `data/test` split. Full report:
[`results/evaluation.json`](results/), interpretation guide:
[docs/METRICS.md](docs/METRICS.md).

| Metric | Value |
| --- | --- |
| Classes | 38 |
| Train / val / test images | 94,814 / 7,608 / 7,671 |
| **Top-1 accuracy** | **0.9840** |
| Top-5 accuracy | 0.9944 |
| Macro F1 | 0.9778 |
| Weighted F1 | 0.9841 |
| Macro ROC-AUC (OvR) | 0.9999 |
| Macro PR-AUC | 0.9964 |
| Expected calibration error | 0.0064 |
| Test loss | 0.0668 |
| Selected checkpoint | `best_model.pt`, epoch 14 |

Throughput, batch size 32, fp32, on the machine the report was produced on:

| Measure | Value |
| --- | --- |
| Forward-pass latency | 4.83 ms / image (p99 batch 154.9 ms) |
| Model-only throughput | 207 images/s |
| End-to-end throughput (incl. decode + load) | 149 images/s |
| Peak GPU memory | 646 MiB |
| Model size | 330 MB, 86,609,702 parameters |

## Project structure

```text
DinoV2-Plant-Disease-Detection/
├── configs/
│   └── config.yaml           # Single source of truth — 94 keys
├── src/
│   ├── paths.py              # PROJECT_ROOT, path resolution, run directories
│   ├── logger.py             # Namespaced logger, UTF-8 console, file handler
│   ├── config.py             # Config object, loading, validation, overrides
│   ├── utils.py              # Timer, parameter counts, JSON/CSV writers, formatters
│   ├── device.py             # Device selection, hardware info, CUDA memory helpers
│   ├── seed.py               # Global reproducibility
│   ├── reporting.py          # Console report primitives (banner, rules, entries)
│   ├── cli.py                # Shared bootstrap + infrastructure entry point
│   ├── model.py              # DINOv2 backbone + classification head
│   ├── verification.py       # Structural model verification
│   ├── audit_dataset.py      # Dataset audit entry point
│   ├── train.py              # Training orchestration
│   ├── evaluate.py           # Evaluation orchestration
│   ├── verify_pipeline.py    # 13-stage end-to-end dry run
│   ├── datasets/             # Audit, transforms, dataloader factories
│   ├── training/             # Engine, optim, checkpoints, AMP, early stopping
│   ├── evaluation/           # Inference pass, metrics, integrity, reporting
│   ├── visualization/        # Training curves, evaluation figures
│   └── api/                  # FastAPI service
├── tests/                    # 127 tests across 3 modules
├── docs/                     # Reference documentation
├── data/                     # Dataset            (git-ignored)
├── checkpoints/              # Model weights      (git-ignored)
├── logs/                     # Per-entry-point logs (git-ignored)
├── results/                  # Reports and figures  (git-ignored)
├── pyproject.toml
└── uv.lock
```

The four git-ignored directories are recreated by `ProjectPaths.create()` on
every run, so a fresh clone needs no `mkdir`.

### Layering

Modules are strictly layered and there are no import cycles:

```text
paths, reporting            →  foundation, no internal imports
logger, config, utils, device, seed  →  infrastructure
model, datasets, training, evaluation, visualization  →  domain
cli, train, evaluate, audit_dataset, verify_pipeline, api  →  orchestration
```

Orchestration modules compose; domain modules never call them.

## Core modules

### `src/datasets/` — data in

| Module | Role |
| --- | --- |
| `validation.py` | The 13-rule audit. Produces a `DatasetAudit` with per-split statistics and severity-tagged issues. Gates everything downstream. |
| `transforms.py` | Train-time augmentation (random resized crop, flip, rotation, colour jitter, random erasing) and the deterministic eval transform (resize + centre crop). |
| `loaders.py` | `ImageFolder`-backed dataloaders with seeded workers, pinned memory and persistent workers. Returns a `DataBundle` carrying the discovered class vocabulary. |

### `src/model.py` — the network

`DinoV2Classifier` = DINOv2 backbone + head. The configured `feature_dim` is
checked against the width the backbone actually reports, so a mismatch fails the
build with an explicit message instead of a shape error deep in training.
Switching to ViT-S/14, ViT-L/14 or ViT-G/14 is a two-key config change.

### `src/training/` — the loop

| Module | Role |
| --- | --- |
| `engine.py` | One train pass and one eval pass over an epoch, sharing a single accumulator so loss and accuracy are computed identically in both. |
| `optim.py` | AdamW builder that excludes frozen parameters and raises if none are trainable; cosine scheduler; gradient clipping. |
| `precision.py` | Resolves the AMP dtype (`auto` → bfloat16 when supported) and builds the matching `GradScaler`. |
| `early_stopping.py` | Monitors a metric with configurable mode, patience and minimum delta. |
| `checkpoints.py` | Save/load of model, optimizer, scheduler, scaler, epoch, history and class vocabulary. Refuses to resume into a different class mapping. |
| `metrics.py` | Per-epoch metric record and the `history.csv` writer. |

### `src/evaluation/` — the verdict

| Module | Role |
| --- | --- |
| `inference.py` | One forward pass over the held-out split in full float32 (deliberately no AMP — calibration and AUC read probability values directly), plus the latency benchmark. |
| `metrics.py` | Turns that single pass into every reported number, including a `MetricLimitations` record stating what the numbers do *not* prove. |
| `integrity.py` | File fingerprinting, parameter digests, and 7 `check_*` functions proving the pass was read-only, finite, in eval mode and covered the whole split. |
| `reporting.py` | Exports the JSON report, per-class CSV, classification report and confusion matrix. |

### `src/api/` — serving

| Module | Role |
| --- | --- |
| `settings.py` | Reads and validates the `api` config section. |
| `inference.py` | `InferenceEngine` — owns the loaded model, decodes uploads, runs the forward pass under a lock, returns ranked predictions. |
| `routes.py` | The 4 endpoints, upload validation and response shaping. |
| `schemas.py` | The 8 Pydantic models forming the wire contract. |
| `errors.py` | 5 typed error classes and 4 handlers producing one consistent error envelope. |
| `dependencies.py` | FastAPI dependency providers for the engine and settings. |
| `main.py` | Assembles the app, loads the checkpoint in a lifespan hook, assigns request IDs. |

## Installation

**Prerequisites**

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU is optional — every stage falls back to CPU

```bash
git clone <repository-url>
cd DinoV2-Plant-Disease-Detection

uv venv
uv sync
```

`uv sync` installs from `uv.lock`. Do not edit the lockfile by hand — change
`pyproject.toml` and run `uv lock`.

Verify the install:

```bash
uv run python -m src.cli
```

This prints an infrastructure report (config loaded, directories created,
device, seed, logging) and exits non-zero if anything is wrong.

## Dataset layout

The dataset is git-ignored. Reconstruct it from your source dataset in
`ImageFolder` layout:

```text
data/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg
```

Every split must contain the **same** class directories — a class present in
`train/` but missing from `val/` is an error-severity audit issue, as is the
reverse. Accepted extensions are `.jpg`, `.jpeg`, `.png`, `.bmp` and `.webp`,
compared case-insensitively.

## Available commands

Six CLI entry points, all of which accept `--config` (default
`configs/config.yaml`):

| # | Command | Produces |
| --- | --- | --- |
| 1 | `uv run python -m src.cli` | Infrastructure report; creates run directories |
| 2 | `uv run python -m src.model` | `results/model_verification.json`, `results/model_summary.txt` |
| 3 | `uv run python -m src.audit_dataset` | `results/dataset_audit.json` |
| 4 | `uv run python -m src.verify_pipeline` | 13-stage dry run; temporary artifacts only |
| 5 | `uv run python -m src.train` | `checkpoints/`, `results/history.csv`, loss + accuracy curves |
| 6 | `uv run python -m src.evaluate` | `results/evaluation.json` + 8 further artifacts |

Plus the server, which takes its settings from the same file:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Extra flags:

| Command | Flag | Purpose |
| --- | --- | --- |
| `src.train` | `--resume <path>` | Continue from a checkpoint, restoring model, optimizer, scheduler, scaler, epoch and history |
| `src.evaluate` | `--checkpoint <path>` | Evaluate a specific checkpoint, overriding `evaluation.checkpoint_filename` |

A typical first run:

```bash
uv run python -m src.cli                 # 1. sanity-check the environment
uv run python -m src.audit_dataset       # 2. prove the dataset is usable
uv run python -m src.verify_pipeline     # 3. dry-run the whole pipeline
uv run python -m src.train               # 4. train
uv run python -m src.evaluate            # 5. score the held-out split
uv run uvicorn src.api.main:app --reload # 6. serve
```

Training runs the dataset audit itself, so step 2 is a fast pre-check rather
than a prerequisite.

## Configuration and environment variables

> **This application reads no environment variables.** There is no `.env` file
> and none is needed. Every setting lives in
> [`configs/config.yaml`](configs/config.yaml).

`.gitignore` still excludes `.env` files as a precaution, so a stray one can
never reach history.

The 94 keys are grouped into these sections:

| Section | Controls |
| --- | --- |
| `project` | Name, version, seed |
| `paths` | Where logs, checkpoints and results are written |
| `device` | `auto`, `cuda` or `cpu` — `cuda` is a preference, not a requirement |
| `logging` | Level, whether to write a log file |
| `reproducibility` | Deterministic algorithms, cuDNN benchmark |
| `model` | Backbone entrypoint, pretrained, freeze, image size, feature width |
| `classifier` | Head type, dropout |
| `dataset` | Root, split names, extensions, augmentation, normalisation, imbalance threshold |
| `dataloader` | Batch size, workers, pinned memory, prefetch |
| `training` | Epochs, gradient clipping, AMP, optimizer, scheduler, early stopping, checkpoint names |
| `visualization` | Figure sizes, DPI, output filenames |
| `evaluation` | Checkpoint, split, top-k, calibration bins, benchmark, output filenames |
| `api` | Title, checkpoint, top-k, upload limits, accepted content types |

Full key-by-key reference: [docs/CONFIG.md](docs/CONFIG.md).

Deployment note: the service reads config at startup only. Changing
`configs/config.yaml` requires a restart.

## API reference

Interactive Swagger UI at `http://localhost:8000/docs` once running.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service readiness, model status, GPU status |
| `GET` | `/metadata` | Backbone, class vocabulary, checkpoint identity and epoch |
| `POST` | `/predict` | Classify one uploaded image |
| `POST` | `/predict/batch` | Classify several images in one forward pass |

**Example**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf.jpg"
```

**Upload validation** — a request breaching any of these is rejected before the
image is decoded:

| Limit | Default | Config key |
| --- | --- | --- |
| Content types | JPEG, PNG, BMP, WebP | `api.allowed_content_types` |
| Max image size | 10 MiB | `api.max_image_bytes` |
| Max batch size | 16 images | `api.max_batch_size` |
| Ranked classes returned | 38 (full vocabulary) | `api.top_k` |

**Error contract** — every non-2xx response returns the same flat envelope:

```json
{
  "request_id": "6b31c854-edf9-4ad7-b291-731d2760bf6a",
  "error": "unsupported_media_type",
  "detail": "'notes.txt' has content type 'text/plain'. Accepted types: ...",
  "status_code": 415
}
```

`request_id` is also echoed in the `X-Request-ID` response header, so a client
log line and a server log line can be joined.

| Status | `error` | Cause |
| --- | --- | --- |
| 400 | `invalid_image` | The upload could not be decoded |
| 413 | `payload_too_large` | Image exceeds `api.max_image_bytes` |
| 415 | `unsupported_media_type` | Content type not in the allow-list |
| 422 | `validation_error` | Malformed request per the Pydantic schema |
| 500 | `inference_failed` | The forward pass raised |
| 503 | `model_not_ready` | The checkpoint has not finished loading |

Full route, schema and middleware reference: [docs/API.md](docs/API.md).

## External services

The project calls exactly one external service, and only when weights are not
already cached:

| Service | Purpose | When |
| --- | --- | --- |
| `torch.hub` → `github.com/facebookresearch/dinov2` | Downloads the official pretrained DINOv2 backbone | First model build; cached in the torch hub directory afterwards |

There is **no** database, no message queue, no third-party API, no telemetry and
no outbound network traffic at inference time. Once the hub cache is warm, the
entire pipeline runs offline.

## Build and deployment

There is no build step — the project is run as a module, not packaged as a
wheel.

**Serving**

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Deployment facts worth knowing before you ship:

| Fact | Implication |
| --- | --- |
| The checkpoint (~1 GB) is read once in the lifespan hook | Startup is slow; readiness is reported by `/health` |
| The forward pass is serialised behind a lock | Use **one** worker per GPU; scale with replicas, not `--workers` |
| The service never reads `data/` | The dataset does not need to ship with the image |
| There is **no authentication** and no rate limiting | Put it behind a gateway that provides both |
| `--reload` is a development flag | Never use it in production |

Full hardening checklist and the gaps to close first:
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Quality gates

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -p "test_*.py"
```

Both gates currently pass: ruff reports no findings, and 127 of 127 tests pass.

`tests/test_milestone3.py` loads the **real** pretrained backbone, so it needs
the `torch.hub` cache populated (or network access on first run).
`tests/test_milestone6.py` substitutes a tiny stand-in backbone inside the
production `InferenceEngine`, so it exercises the genuine preprocessing, softmax
and top-k paths without a trained checkpoint or a GPU.

These gates are run by hand — there is no CI pipeline yet.

## Contributing

1. **Branch** from the default branch.
2. **Put new settings in `configs/config.yaml`**, never as literals in code.
   Add the key to the module's `*_REQUIRED_KEYS` contract so a missing or
   wrongly typed value fails loudly at startup.
3. **Respect the layering.** Domain modules (`model`, `datasets`, `training`,
   `evaluation`, `visualization`) must not import orchestration modules
   (`cli`, `train`, `evaluate`, `api`).
4. **Document behaviour in docstrings**, including a `Raises:` section. The
   codebase documents every public callable; keep it that way.
5. **Run both gates** before opening a PR:
   ```bash
   uv run ruff check .
   uv run python -m unittest discover -s tests -p "test_*.py"
   ```
6. **Add tests** for new behaviour. The suite is stdlib `unittest` — no pytest.
7. **Never commit** dataset files, checkpoints, logs or results. `.gitignore`
   covers all four, plus every common weight-file extension.
8. **Change dependencies via `pyproject.toml`**, then run `uv lock`. Do not hand-edit
   `uv.lock`.
9. **Keep `docs/` truthful.** [docs/CODEMAP.md](docs/CODEMAP.md) tracks file
   responsibilities and technical debt; update it when you change structure.

## License

**No license file is present in this repository**, and `pyproject.toml` declares
no license field. All rights are therefore reserved by default — the code is not
open source as it stands. Add a `LICENSE` file before distributing or accepting
outside contributions.

Note that the DINOv2 backbone weights are downloaded from
`facebookresearch/dinov2` and carry that project's own license terms.

## Known gaps

Tracked in [docs/CODEMAP.md](docs/CODEMAP.md#technical-debt). The ones most
likely to matter to you:

| Gap | Consequence |
| --- | --- |
| No CI, pre-commit hooks or Dockerfile | Quality gates and deployment are manual |
| No authentication or rate limiting on the API | Must not face the internet directly |
| `torch` / `torchvision` have no version floors | Pinning comes only from `uv.lock`; a fresh `uv lock` elsewhere could resolve a different torch |
| `src.evaluate` re-audits every split to score one | Decodes ~110k images to evaluate ~7.7k |
| Some tests assert values from the live `configs/config.yaml` | Editing production config can break the suite |
| `ruff format` is not clean or gated | `ruff check` passes; formatting is not enforced |
| Upload size is checked after buffering | An oversized upload occupies memory before rejection |

## Documentation

| Document | Covers |
| --- | --- |
| [docs/README.md](docs/README.md) | Documentation index and project identity |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layering, module graph, execution flows |
| [docs/API.md](docs/API.md) | Routes, schemas, validation, errors, middleware |
| [docs/MODEL.md](docs/MODEL.md) | Preprocessing, inference, checkpoint format, versioning |
| [docs/METRICS.md](docs/METRICS.md) | Every metric, where it is computed, how to read it |
| [docs/CONFIG.md](docs/CONFIG.md) | All 94 configuration keys |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Dependencies, startup, serving stack, hardening |
| [docs/CODEMAP.md](docs/CODEMAP.md) | File-by-file responsibilities, dead-code audit, technical debt |
