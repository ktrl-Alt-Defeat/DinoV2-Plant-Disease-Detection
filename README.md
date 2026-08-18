# DINOv2-LeafCare

Plant disease image classification built on a fine-tuned **DINOv2 ViT-B/14**
backbone, with a dataset audit, a training pipeline, a full evaluation suite and
a FastAPI inference service.

Detailed documentation lives in [`docs/`](docs/README.md).

## What is implemented

| Capability | Entry point |
| --- | --- |
| Infrastructure bootstrap and environment report | `python -m src.cli` |
| Structural model verification on synthetic tensors | `python -m src.model` |
| Dataset integrity audit | `python -m src.audit_dataset` |
| End-to-end pipeline verification | `python -m src.verify_pipeline` |
| Fine-tuning with AMP, cosine schedule, early stopping, resume | `python -m src.train` |
| Held-out evaluation with integrity guarantees | `python -m src.evaluate` |
| REST inference API | `uvicorn src.api.main:app` |

## Model

| Property | Value |
| --- | --- |
| Backbone | DINOv2 ViT-B/14 (`dinov2_vitb14`), loaded via `torch.hub` |
| Feature dimension | 768 |
| Head | `Linear(768, num_classes)` |
| Input | 224×224 RGB |
| Total parameters | 86,609,702 |

Class count is never hardcoded: training and evaluation derive it from the
dataset directory layout, and the API derives it from the checkpoint.

## Results

Measured on the held-out `data/test` split — see
[`results/evaluation.json`](results/) and [docs/METRICS.md](docs/METRICS.md).

| Metric | Value |
| --- | --- |
| Classes | 38 |
| Test images | 7,671 |
| Top-1 accuracy | 0.9840 |
| Top-5 accuracy | 0.9944 |
| Macro F1 | 0.9778 |
| Expected calibration error | 0.0064 |
| Checkpoint | `best_model.pt`, epoch 14 |

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) as the package manager
- NVIDIA GPU optional; every stage falls back to CPU

`pyproject.toml` and `uv.lock` are the only source of truth for dependencies.
`torch`, `torchvision` and `torchaudio` resolve from the pinned CUDA 13.0 index
declared in `pyproject.toml`. Do not edit `uv.lock` by hand.

## Installation

```bash
uv venv
uv sync
```

## Dataset layout

The dataset is git-ignored and expected at:

```text
data/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg
```

Every split must contain the same class directories. The audit enforces this and
13 other integrity rules before any training starts.

## Usage

```bash
# 1. Verify the environment
uv run python -m src.cli --config configs/config.yaml

# 2. Audit the dataset -> results/dataset_audit.json
uv run python -m src.audit_dataset --config configs/config.yaml

# 3. Verify the pipeline end to end (temporary artifacts only)
uv run python -m src.verify_pipeline --config configs/config.yaml

# 4. Train -> checkpoints/ + results/history.csv + curves
uv run python -m src.train --config configs/config.yaml
uv run python -m src.train --config configs/config.yaml --resume checkpoints/last_model.pt

# 5. Evaluate -> results/evaluation.json + 8 further artifacts
uv run python -m src.evaluate --config configs/config.yaml

# 6. Serve
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Training runs the dataset audit itself and aborts before the backbone loads if
the dataset carries any error-severity issue.

## API

Swagger UI at `http://localhost:8000/docs` once the service is running.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service, model and GPU status |
| GET | `/metadata` | Model, class vocabulary, checkpoint identity |
| POST | `/predict` | Classify one image |
| POST | `/predict/batch` | Classify several images in one forward pass |

The checkpoint is read once at startup and never reloaded per request. The
service does not require `data/` to be present. There is **no authentication** —
see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the hardening gaps.

## Repository structure

```text
├── configs/config.yaml   # Single source of truth, 94 keys
├── src/
│   ├── cli.py            # Shared bootstrap + infrastructure CLI
│   ├── config.py         # Config loading, validation, overrides
│   ├── model.py          # DINOv2 backbone + classification head
│   ├── train.py          # Training orchestration
│   ├── evaluate.py       # Evaluation orchestration
│   ├── audit_dataset.py  # Dataset audit CLI
│   ├── verify_pipeline.py
│   ├── datasets/         # Audit, transforms, dataloaders
│   ├── training/         # Engine, optim, checkpoints, AMP, early stopping
│   ├── evaluation/       # Inference, metrics, integrity, reporting
│   ├── visualization/    # Training curves, evaluation figures
│   └── api/              # FastAPI service
├── tests/                # 127 tests across 3 modules
├── docs/                 # Full documentation
├── data/                 # Dataset (git-ignored)
├── checkpoints/          # Model weights (git-ignored)
├── logs/                 # Run logs (git-ignored)
└── results/              # Reports and figures (git-ignored)
```

## Configuration

All behaviour is driven by [`configs/config.yaml`](configs/config.yaml). The
application reads **no environment variables**. Every entry point accepts
`--config`; `src.train` also accepts `--resume` and `src.evaluate` accepts
`--checkpoint`.

Switching backbone is a configuration change: set `model.name` to
`dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14` or `dinov2_vitg14` and set
`model.feature_dim` to that backbone's embedding width. The width is checked
against what the backbone reports, so a mismatch fails the build with an explicit
message. Note that `tests/test_milestone1.py` and `tests/test_milestone3.py`
assert against the values currently in `configs/config.yaml`.

Full key reference: [docs/CONFIG.md](docs/CONFIG.md).

## Quality gates

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -p "test_*.py"
```

`tests/test_milestone3.py` loads the real pretrained backbone and needs the
`torch.hub` cache populated.

## Documentation

| Document | Covers |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layering, module graph, execution flows |
| [docs/API.md](docs/API.md) | Routes, schemas, validation, errors, middleware |
| [docs/MODEL.md](docs/MODEL.md) | Preprocessing, inference, checkpoint format, versioning |
| [docs/METRICS.md](docs/METRICS.md) | Every metric, where computed, interpretation |
| [docs/CONFIG.md](docs/CONFIG.md) | All 94 configuration keys |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Dependencies, startup, serving stack |
| [docs/CODEMAP.md](docs/CODEMAP.md) | File-by-file responsibilities, technical debt |

## Known gaps

Tracked in [docs/CODEMAP.md](docs/CODEMAP.md#technical-debt). The main ones:
`torchaudio` and `huggingface-hub` are declared but unimported; the torch stack
has no version floors; `pyproject.toml` version (`0.1.0`) and
`project.version` (`1.0.0`) disagree; there is no CI, Dockerfile or
authentication.
