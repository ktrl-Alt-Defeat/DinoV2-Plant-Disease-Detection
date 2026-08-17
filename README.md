# DINOv2-LeafCare

Fine-tuning a pretrained **DINOv2-S** (`dinov2_vits14`) Vision Transformer for multi-class
plant disease classification.

This repository is built milestone by milestone. It currently delivers the **engineering
foundation** (Milestone 1), the **DINOv2 backbone** (Milestone 2) and the **configurable
classification head with structural verification** (Milestone 3). No dataset access, no
training and no evaluation code exists yet; `model.num_classes` is a placeholder until the
dataset is integrated.

## Repository structure

```text
DinoV2-LeafCare/
├── configs/
│   └── config.yaml          # Single source of truth for every tunable value
├── src/
│   ├── cli.py               # Infrastructure entry point (python -m src.cli)
│   ├── config.py            # YAML loading, dotted-key access, validation
│   ├── device.py            # CUDA detection, CPU fallback, hardware introspection
│   ├── logger.py            # Singleton logger, console + optional UTF-8 file output
│   ├── model.py             # DINOv2 backbone + classifier (python -m src.model)
│   ├── paths.py             # Project root resolution and directory management
│   ├── reporting.py         # Shared console report formatting
│   ├── seed.py              # Reproducibility across Python, NumPy, PyTorch and cuDNN
│   ├── utils.py             # Generic helpers (JSON/CSV/text I/O, timing, formatting)
│   ├── verification.py      # Synthetic structural verification of the model
│   ├── models/              # Reserved — model components beyond the backbone
│   ├── datasets/            # Reserved — Milestone 4
│   ├── training/            # Reserved — Milestone 5
│   ├── evaluation/          # Reserved — Milestone 6
│   └── visualization/       # Reserved — Milestone 7
├── tests/
│   ├── test_milestone1.py   # Synthetic unit tests for the foundation
│   └── test_milestone3.py   # Synthetic unit tests for the integrated model
├── checkpoints/             # Model weights (git-ignored)
├── logs/                    # Run logs (git-ignored)
├── results/                 # Metrics, summaries and reports (git-ignored)
├── pyproject.toml           # Dependency specification
└── uv.lock                  # Fully resolved, reproducible dependency lock
```

Directory creation lives in `src/paths.py`; no other module creates directories, and every
path is built with `pathlib` relative to the repository root, so the project behaves
identically on Windows, macOS and Linux.

`src/model.py` defines the model; `src/verification.py` holds the verification harness that
`python -m src.model` runs. Keeping the two apart means importing the model never pulls in
the test harness.

`tests/test_milestone3.py` supersedes the Milestone 2 suite: the model contract changed
from "feature extractor" to "backbone plus classification head", and every check the older
suite performed is re-expressed there against the integrated model.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) as the package manager

`pyproject.toml` and `uv.lock` are the only source of truth for dependencies. Do not edit
`uv.lock` by hand.

## Installation

```bash
git clone <repository-url>
cd DinoV2-LeafCare

uv venv
uv sync
```

`uv sync` reproduces the exact locked environment, including the `dev` dependency group.

Adding dependencies later:

```bash
uv add <package>              # runtime dependency
uv add --dev <package>        # development dependency
```

## Environment setup check

Run the infrastructure bootstrap to verify the environment:

```bash
uv run python -m src.cli
# or with an explicit configuration
uv run python -m src.cli --config configs/config.yaml
```

The command loads the configuration, initialises the logger, seeds every random number
generator, creates `logs/`, `checkpoints/` and `results/`, detects the compute device and
prints a summary ending in `STATUS: PASS`.

## Model verification

```bash
uv run python -m src.model
# or with an explicit configuration
uv run python -m src.model --config configs/config.yaml
```

This builds the configured DINOv2 backbone with the official pretrained weights, attaches
the classification head, runs a forward pass on synthetic `torch.randn(2, 3, 224, 224)`
tensors under `torch.inference_mode()`, then checks the embedding shape, the logit shape,
CPU and CUDA execution and the absence of NaN/Inf values before writing two artifacts into
`results/`:

- `model_summary.txt` — model facts and the full module tree
- `model_verification.json` — machine-readable check results

The first run downloads the official weights (~85 MB) from `facebookresearch/dinov2` via
`torch.hub`; later runs reuse the local cache.

## Verification

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
uv run ruff check .
```

## Configuration

All behaviour is driven by `configs/config.yaml`. The `project`, `paths`, `device`,
`logging`, `reproducibility`, `model` and `classifier` sections are active today;
`dataset`, `training` and `evaluation` are reserved for the milestones below.

Switching backbone is a configuration change only — set `model.name` to `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14` or `dinov2_vitg14`, and `model.feature_dim` to that
backbone's embedding width. The width is checked against what the backbone actually
reports, so a mismatch fails the build with an explicit message instead of being trusted.

The head is described by `classifier.type` (`linear`) and `classifier.dropout`; a dropout
probability above `0` prepends a `Dropout` layer to the linear projection.

> `model.num_classes: 10` is a **placeholder for architectural verification only**. It is
> replaced with the real class count derived from the dataset metadata in Milestone 4.

## Milestone roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| 1 | Project foundation and engineering infrastructure | Complete |
| 2 | DINOv2-S backbone and structural verification | Complete |
| 3 | Classification head and model integration | Complete |
| 4 | Dataset pipeline, dataloaders and real class count | Planned |
| 5 | Fine-tuning loop, schedulers and checkpointing | Planned |
| 6 | Evaluation, metrics and reporting | Planned |
| 7 | Visualisation, inference and benchmarking | Planned |
