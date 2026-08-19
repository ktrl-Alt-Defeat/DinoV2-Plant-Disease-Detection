# Configuration

## Environment variables

**Not Found.** A scan of `src/` and `tests/` for `os.environ`, `os.getenv`,
`dotenv` and `ENV[` returns zero matches. The application reads **no** environment
variables. All runtime behaviour comes from
[configs/config.yaml](../configs/config.yaml) and CLI flags.

`.env` file support: **Not Found** (`python-dotenv` is present only as a
transitive dependency of `uvicorn[standard]`, never imported by this code).

## Configuration files

| File | Role | Source |
| --- | --- | --- |
| `configs/config.yaml` | Single source of truth, 94 keys | [configs/config.yaml](../configs/config.yaml) |
| `pyproject.toml` | Dependencies, ruff config, uv index pinning | [pyproject.toml](../pyproject.toml) |
| `uv.lock` | Resolved lockfile | [uv.lock](../uv.lock) |
| `.gitignore` | Ignores `logs/`, `results/`, `checkpoints/`, `data/`, `.venv/`, caches | [.gitignore](../.gitignore) |

Default config path: `DEFAULT_CONFIG_PATH = "configs/config.yaml"`
([src/cli.py:28](../src/cli.py)). Every entry point accepts `--config`
([src/cli.py:58](../src/cli.py)).

## Loading and validation

| Stage | Behaviour | Source |
| --- | --- | --- |
| Resolve | Relative paths anchored at repo root | [src/paths.py:27](../src/paths.py) |
| Parse | `yaml.safe_load`; empty or non-mapping root rejected | [src/config.py:148](../src/config.py) |
| Validate | `REQUIRED_KEYS` (11 infra keys) checked at load | [src/config.py:24](../src/config.py) |
| Access | Dotted keys; missing key without default raises `ConfigError` | [src/config.py:64](../src/config.py) |
| Immutability | Constructor deep-copies; every `get` returns a deep copy | [src/config.py:62](../src/config.py) |
| Override | `with_overrides` returns a **new** `Config` | [src/config.py:167](../src/config.py) |

`validate_keys` collects **all** problems and raises once
([src/config.py:189](../src/config.py)). `bool` is treated separately from `int`
so flags and numbers are not interchangeable ([src/config.py:224](../src/config.py)).

### Load-time required keys

`project.name`, `project.version`, `project.seed`, `paths.logs`,
`paths.checkpoints`, `paths.results`, `device.preferred`, `logging.level`,
`logging.save_file`, `reproducibility.deterministic`, `reproducibility.benchmark`
([src/config.py:24](../src/config.py)).

All other sections are validated lazily by their consuming specification.

## Configuration reference

### `project`

| Key | Default | Consumer |
| --- | --- | --- |
| `project.name` | `dinov2_plant_disease` | Default log filename |
| `project.version` | `1.0.0` | API `model_version`, `/health`, `/metadata` |
| `project.seed` | `42` | `set_seed`, dataloader generator |

### `paths`

| Key | Default | Notes |
| --- | --- | --- |
| `paths.logs` | `logs` | Created by `ProjectPaths.create` |
| `paths.checkpoints` | `checkpoints` | Checkpoint root |
| `paths.results` | `results` | All reports and figures |

Absolute values are honoured unchanged ([src/paths.py:27](../src/paths.py)).

### `device`, `logging`, `reproducibility`

| Key | Default | Accepted / effect |
| --- | --- | --- |
| `device.preferred` | `auto` | `auto`, `cuda`, `cpu` ([src/device.py:12](../src/device.py)). `cuda` warns and falls back to CPU when unavailable |
| `logging.level` | `INFO` | Any `logging` level name, case-insensitive |
| `logging.save_file` | `true` | `false` disables the file handler |
| `reproducibility.deterministic` | `true` | `torch.backends.cudnn.deterministic` |
| `reproducibility.benchmark` | `false` | `torch.backends.cudnn.benchmark` |

### `model`, `classifier`

| Key | Default | Notes |
| --- | --- | --- |
| `model.name` | `dinov2_vitb14` | `torch.hub` entrypoint |
| `model.pretrained` | `true` | Downloads official weights on first use |
| `model.freeze_backbone` | `false` | `true` leaves only the head trainable |
| `model.image_size` | `224` | Must be a multiple of the backbone patch size (14) |
| `model.feature_dim` | `768` | Checked against the backbone's advertised width |
| `model.num_classes` | `10` | **Fallback only.** Overridden by dataset (train/eval) or checkpoint (API) |
| `classifier.type` | `linear` | Only `linear` supported |
| `classifier.dropout` | `0.0` | Must satisfy `0 <= d < 1`; `>0` prepends `Dropout` |

### `dataset`

| Key | Default |
| --- | --- |
| `dataset.root` | `data` |
| `dataset.splits.train` / `.val` / `.test` | `train` / `val` / `test` |
| `dataset.extensions` | `[.jpg, .jpeg, .png, .bmp, .webp]` |
| `dataset.resize_size` | `256` |
| `dataset.imbalance_ratio_threshold` | `10.0` |
| `dataset.audit_filename` | `dataset_audit.json` |
| `dataset.normalization.mean` | `[0.485, 0.456, 0.406]` |
| `dataset.normalization.std` | `[0.229, 0.224, 0.225]` |

Extensions are matched case-insensitively and must be keys of
`EXTENSION_FORMATS` ([src/datasets/validation.py:25](../src/datasets/validation.py)),
which maps `.jpg`/`.jpeg` to `{JPEG, MPO}` and each other suffix to a single format.

### `dataset.augmentation` (training only)

| Key | Default |
| --- | --- |
| `random_resized_crop_scale` | `[0.7, 1.0]` |
| `random_resized_crop_ratio` | `[0.75, 1.3333]` |
| `horizontal_flip_probability` | `0.5` |
| `rotation_degrees` | `15.0` |
| `color_jitter_brightness` / `_contrast` / `_saturation` | `0.2` |
| `color_jitter_hue` | `0.05` (max `0.5`) |
| `random_erasing_probability` | `0.25` |

### `dataloader`

| Key | Default | Notes |
| --- | --- | --- |
| `batch_size` | `32` | Must be positive |
| `num_workers` | `4` | `0` disables the two worker options below, logged explicitly |
| `pin_memory` | `true` | Silently downgraded to `false` on CPU, logged |
| `persistent_workers` | `true` | Applied only when `num_workers > 0` |
| `prefetch_factor` | `4` | Applied only when `num_workers > 0` |
| `drop_last` | `false` | Applied to the **training split only** ([loaders.py:274](../src/datasets/loaders.py)) |

### `training`

| Key | Default | Notes |
| --- | --- | --- |
| `epochs` | `30` | Must be a positive int |
| `log_filename` | `train.log` | |
| `history_filename` | `history.csv` | |
| `gradient_clip_norm` | `1.0` | `<= 0` disables clipping |
| `amp.enabled` | `true` | |
| `amp.dtype` | `auto` | `auto` \| `bf16` \| `fp16`; bf16 downgrades to fp16 with a warning when unsupported |
| `optimizer.name` | `adamw` | Only `adamw` |
| `optimizer.learning_rate` | `0.0001` | |
| `optimizer.weight_decay` | `0.05` | |
| `optimizer.betas` | `[0.9, 0.999]` | Both in `[0,1)` |
| `scheduler.name` | `cosine` | Only `cosine` (`CosineAnnealingLR`) |
| `scheduler.min_learning_rate` | `1e-06` | `eta_min` |
| `early_stopping.enabled` | `true` | |
| `early_stopping.monitor` | `val_accuracy` | One of `MONITORABLE_METRICS` |
| `early_stopping.mode` | `max` | `min` \| `max` |
| `early_stopping.patience` | `5` | |
| `early_stopping.min_delta` | `0.001` | |
| `checkpoints.best_filename` | `best_model.pt` | |
| `checkpoints.last_filename` | `last_model.pt` | |

AMP is applied only on CUDA; on CPU it is disabled and logged
([src/training/precision.py](../src/training/precision.py)). `GradScaler` is
enabled only for fp16 (`uses_grad_scaler`).

### `visualization`

| Key | Default |
| --- | --- |
| `loss_curve_filename` | `loss_curve.png` |
| `accuracy_curve_filename` | `accuracy_curve.png` |
| `figure_width` / `figure_height` | `10.0` / `6.0` |
| `dpi` | `150` |

### `evaluation`

| Key | Default | Notes |
| --- | --- | --- |
| `checkpoint_filename` | `best_model.pt` | Overridable by `--checkpoint` |
| `split` | `test` | Must be `train`, `val` or `test` |
| `log_filename` | `evaluate.log` | |
| `top_k` | `5` | Clamped to class count |
| `calibration_bins` | `15` | |
| `probability_tolerance` | `1e-05` | Row-sum tolerance |
| `benchmark.warmup_batches` | `5` | May be `0` |
| `benchmark.measured_batches` | `50` | Must be positive |
| `confusion_matrix_figure_size` | `16.0` | Square edge, inches |
| `filenames.*` | 9 artifact names | See [METRICS.md](METRICS.md#exported-artifacts) |

### `api`

| Key | Default | Notes |
| --- | --- | --- |
| `title` | `DINOv2-LeafCare Inference API` | OpenAPI title |
| `description` | see config | OpenAPI description |
| `checkpoint_filename` | `best_model.pt` | Loaded once at startup |
| `log_filename` | `api.log` | |
| `top_k` | `5` | Clamped to class count |
| `max_batch_size` | `16` | Batch route only |
| `max_image_bytes` | `10485760` (10 MiB) | Per file |
| `allowed_content_types` | `[image/jpeg, image/png, image/bmp, image/webp]` | Exact match after lowercase + `;` strip |

`ApiSettings.version` is taken from `project.version`, not from the `api` section
([src/api/settings.py:69](../src/api/settings.py)).

## CLI flags

| Flag | Entry points | Default |
| --- | --- | --- |
| `--config` | all six CLI modules | `configs/config.yaml` |
| `--resume` | `src.train` | `None` |
| `--checkpoint` | `src.evaluate` | `None` (falls back to config) |

Sources: [src/cli.py:58](../src/cli.py),
[src/train.py](../src/train.py), [src/evaluate.py:340](../src/evaluate.py).

## Feature flags

Boolean switches that change behaviour:

| Flag | Effect when `false` |
| --- | --- |
| `logging.save_file` | No file handler; console only |
| `model.pretrained` | Random backbone init; `weights_source` reports it |
| `model.freeze_backbone` | (when `true`) only the head is trainable |
| `training.amp.enabled` | fp32 training |
| `training.early_stopping.enabled` | Never stops early; best tracking continues |
| `dataloader.pin_memory` / `persistent_workers` | Worker options unapplied |
| `reproducibility.deterministic` / `benchmark` | cuDNN algorithm selection |

## Version fields

| Field | Value | Read by |
| --- | --- | --- |
| `pyproject.toml` `version` | `1.0.0` | Packaging only — **never read at runtime** |
| `configs/config.yaml` `project.version` | `1.0.0` | `src/cli.py`, `src/api/settings.py` |

The two agree, but nothing enforces that: they are separate fields read by
separate code paths. Bump both together.

## Runtime-derived values

Not configurable; computed at run time:

| Value | Derived from | Source |
| --- | --- | --- |
| `num_classes` | `ImageFolder` directories, or checkpoint | [loaders.py:239](../src/datasets/loaders.py), [api/inference.py:145](../src/api/inference.py) |
| Device | `device.preferred` + CUDA availability | [src/device.py:36](../src/device.py) |
| AMP dtype | `training.amp.dtype` + `torch.cuda.is_bf16_supported()` | [precision.py](../src/training/precision.py) |
| Effective `top_k` | `min(top_k, num_classes)` | [metrics.py:292](../src/evaluation/metrics.py) |
| Scheduler `T_max` | `training.epochs` | [optim.py](../src/training/optim.py) |
