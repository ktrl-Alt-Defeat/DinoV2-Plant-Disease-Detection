# Code map

## Directories

| Path | Tracked? | Purpose |
| --- | --- | --- |
| `configs/` | yes | `config.yaml`, the single source of truth |
| `src/` | yes | All application code: 14 root modules + 5 packages, 42 `.py` files (6 are `__init__.py`) |
| `tests/` | yes | 3 unittest modules, 127 test methods |
| `docs/` | yes | This documentation |
| `data/` | git-ignored | `train/`, `val/`, `test/` `ImageFolder` layout |
| `checkpoints/` | git-ignored | `best_model.pt`, `last_model.pt` |
| `logs/` | git-ignored | Per-entry-point log files |
| `results/` | git-ignored | Reports and figures (15 files present) |

## Root modules — `src/`

| File | Lines | Responsibility |
| --- | --- | --- |
| [paths.py](../src/paths.py) | 87 | `PROJECT_ROOT`, `resolve`, `ensure_directory`, `ProjectPaths` |
| [logger.py](../src/logger.py) | 123 | Root logger `dinov2_leafcare`, idempotent config, UTF-8 console |
| [config.py](../src/config.py) | 241 | `Config`, `load_config`, `validate_keys`, `with_overrides` |
| [utils.py](../src/utils.py) | 199 | `Timer`, `count_parameters`, JSON/CSV/text writers, formatters |
| [device.py](../src/device.py) | 122 | `get_device`, `get_device_info`, `DeviceInfo`, `synchronize`, `reset_peak_memory`, `peak_memory_mib` |
| [seed.py](../src/seed.py) | 56 | `set_seed` across Python, NumPy, torch, CUDA, cuDNN |
| [reporting.py](../src/reporting.py) | 48 | Console banner/rule/entry primitives |
| [cli.py](../src/cli.py) | 203 | `build_parser`, `bootstrap`, `BootstrapReport`, infra CLI |
| [model.py](../src/model.py) | 437 | `DinoV2Classifier`, specs, `build_model`, `build_classifier` |
| [verification.py](../src/verification.py) | 286 | Structural model verification, artifact writers |
| [audit_dataset.py](../src/audit_dataset.py) | 114 | Dataset audit CLI |
| [train.py](../src/train.py) | 559 | Training orchestration CLI |
| [evaluate.py](../src/evaluate.py) | 560 | Evaluation orchestration CLI |
| [verify_pipeline.py](../src/verify_pipeline.py) | 407 | 13-stage pipeline verification CLI |

## `src/datasets/`

| File | Lines | Key definitions |
| --- | --- | --- |
| [validation.py](../src/datasets/validation.py) | 541 | `DatasetSpecification`, `DatasetAudit`, `DatasetIssue`, `SplitStatistics`, `audit_dataset`, 13 issue categories |
| [transforms.py](../src/datasets/transforms.py) | 259 | `TransformSpecification`, `build_train_transform`, `build_eval_transform` |
| [loaders.py](../src/datasets/loaders.py) | 321 | `DataLoaderSpecification`, `DataBundle`, `build_dataloaders`, `_seed_worker` |

### Audit issue categories

`missing_root`, `missing_split`, `no_classes`, `missing_class`, `orphan_class`,
`empty_class`, `zero_byte_file`, `invalid_extension`, `unsupported_format`,
`unreadable_image`, `corrupt_image`, `duplicate_filename`, `class_imbalance`
([validation.py:41–66](../src/datasets/validation.py)).

Severity: all are `error` except `duplicate_filename` and `class_imbalance`,
which are `warning` ([validation.py:452](../src/datasets/validation.py),
[:517](../src/datasets/validation.py)).

Each candidate file is opened twice — `verify()` then `load()` — so decoding is
authoritative ([validation.py:391](../src/datasets/validation.py)).

## `src/training/`

| File | Lines | Key definitions |
| --- | --- | --- |
| [checkpoints.py](../src/training/checkpoints.py) | 333 | `save_checkpoint`, `load_checkpoint`, `load_model_checkpoint`, `read_checkpoint`, `CheckpointContents`, `CheckpointMetadata`, `ResumeState`, `CheckpointError` |
| [optim.py](../src/training/optim.py) | 207 | `build_optimizer` (AdamW), `build_scheduler` (Cosine), `clip_gradients`, `current_learning_rate` |
| [engine.py](../src/training/engine.py) | 157 | `train_one_epoch`, `evaluate`, `_OutcomeAccumulator`, `log_epoch_start` |
| [early_stopping.py](../src/training/early_stopping.py) | 160 | `EarlyStopping`, `EarlyStoppingSpecification` |
| [precision.py](../src/training/precision.py) | 138 | `PrecisionSpecification.resolve`, `build_grad_scaler` |
| [metrics.py](../src/training/metrics.py) | 94 | `EpochMetrics`, `EpochOutcome`, `write_history` |

`build_optimizer` excludes frozen parameters and raises if none are trainable
([optim.py](../src/training/optim.py)).

## `src/evaluation/`

| File | Lines | Key definitions |
| --- | --- | --- |
| [metrics.py](../src/evaluation/metrics.py) | 555 | `compute_metrics`, `OverallMetrics`, `ClassMetrics`, `CalibrationSummary`, `MetricLimitations`, `EvaluationMetrics` |
| [inference.py](../src/evaluation/inference.py) | 296 | `run_inference`, `benchmark_inference`, `softmax_probabilities`, `BenchmarkResult` |
| [integrity.py](../src/evaluation/integrity.py) | 211 | `fingerprint_file`, `parameter_digest`, 7 `check_*` functions, `enforce` |
| [reporting.py](../src/evaluation/reporting.py) | 132 | `write_reports`, `ReportFilenames` |

## `src/visualization/`

| File | Lines | Key definitions |
| --- | --- | --- |
| [plots.py](../src/visualization/plots.py) | 157 | `PlotSpecification`, `write_curves`, `_line_plot` |
| [evaluation_plots.py](../src/visualization/evaluation_plots.py) | 364 | `EvaluationPlotSpecification`, `write_evaluation_figures`, 4 private plotters |

Both select the `Agg` backend before importing `pyplot`.

## `src/api/`

| File | Lines | Key definitions |
| --- | --- | --- |
| [main.py](../src/api/main.py) | 161 | `create_app`, `_prepare`, `_lifespan_factory`, `_load_engine`, `_assign_request_id`, `app` |
| [inference.py](../src/api/inference.py) | 337 | `InferenceEngine`, `decode_image`, `_rebuild_model`, `ScoredClass`, `ImagePrediction`, `BatchPrediction`, `CheckpointIdentity` |
| [routes.py](../src/api/routes.py) | 243 | `health`, `metadata`, `predict`, `predict_batch`, `_decode_uploads`, `_to_response` |
| [errors.py](../src/api/errors.py) | 149 | `ApiError` + 5 subclasses, 4 handlers, `error_response` |
| [schemas.py](../src/api/schemas.py) | 116 | 8 Pydantic models |
| [dependencies.py](../src/api/dependencies.py) | 54 | `get_engine`, `get_settings`, `EngineDependency`, `SettingsDependency` |
| [settings.py](../src/api/settings.py) | 92 | `ApiSettings` |

## Tests

| File | Tests | Scope |
| --- | --- | --- |
| [test_milestone1.py](../tests/test_milestone1.py) | 44 | Config, paths, logging, seed, device, utils, CLI |
| [test_milestone3.py](../tests/test_milestone3.py) | 48 | Model construction, weights, forward, freeze, artifacts |
| [test_milestone6.py](../tests/test_milestone6.py) | 35 | API endpoints, validation, errors, engine, decoding |

`test_milestone3.py` loads the **real** pretrained backbone; `test_milestone6.py`
substitutes a stand-in backbone inside the production `InferenceEngine`.

## Dead code

An AST + reference scan over all 172 public module-level definitions in `src/`
found **one** name with zero textual references outside its definition:
`predict_batch` ([routes.py:145](../src/api/routes.py)). It is **not dead** — it is
registered by the `@router.post("/predict/batch")` decorator and exercised by
`BatchEndpointTests`.

**Conclusion: no dead public code.** A companion scan of every module-level
constant and private name found no unreferenced definitions either, and
`ruff check` (rule `F`) reports no unused imports or locals.

The empty `src/models/` placeholder package was removed; `src/model.py` is the
only model module, so the near-identical-name hazard is gone.

## TODO / FIXME comments

A scan of `src/`, `tests/`, `configs/`, `README.md` and `pyproject.toml` for
`TODO`, `FIXME`, `XXX`, `HACK` and `NotImplementedError` returns **zero** matches.

Three occurrences of the word "placeholder" exist, all documentation prose:

| Location | Text |
| --- | --- |
| [src/api/errors.py:79](../src/api/errors.py) | docstring: "or a placeholder" |
| [tests/test_milestone3.py:66](../tests/test_milestone3.py) | comment on the test class count |
| [README.md:9,138](../README.md) | describes `model.num_classes` |

## Technical debt

| # | Issue | Evidence | Impact |
| --- | --- | --- | --- |
| 1 | ~~Root `README.md` stale~~ **Resolved** | Rewritten to describe the implemented system and index `docs/`. Verified: no `vits14`/384-dim/"no evaluation code" claims remain outside the backbone-switching list | — |
| 2 | ~~`torchaudio` unused~~ **Resolved** | Removed from [pyproject.toml](../pyproject.toml) and `uv.lock` | — |
| 3 | ~~`huggingface-hub` unused~~ **Resolved** | Removed from [pyproject.toml](../pyproject.toml) and `uv.lock` (with its `hf-xet` transitive) | — |
| 4 | **No version floors on torch stack** | `"torch"`, `"torchvision"` unpinned | A fresh `uv lock` could resolve an incompatible torch |
| 5 | ~~Version fields unsynchronised~~ **Resolved** | `pyproject.toml` now `1.0.0`, matching `project.version`. Still two hand-maintained fields | — |
| 6 | ~~`pyproject.toml` description stale~~ **Resolved** | Now names the ViT-B/14 backbone | — |
| 7 | ~~`src/model.py` vs `src/models/`~~ **Resolved** | Empty `src/models/` package deleted | — |
| 8 | ~~`with_overrides` duplicated in tests~~ **Resolved** | `override_config` [test_milestone3.py](../tests/test_milestone3.py) now delegates to `src.config.with_overrides` | — |
| 9 | **Tests coupled to the live config file** | `test_milestone1.py` and `test_milestone3.py` assert against `configs/config.yaml` values (`dinov2_vitb14`, `768`, `dinov2_plant_disease`) | Editing production config breaks tests. `SettingsTests` in `test_milestone6.py` was decoupled; the rest were not |
| 10 | **Evaluation re-audits the whole dataset** | `_build_bundle` audits all splits to evaluate one ([evaluate.py:385](../src/evaluate.py)) | Decodes ~110k images to score ~7.7k |
| 11 | **`Config.get` deep-copies on every access** | [config.py:85](../src/config.py) | Per-call cost if used in a hot loop |
| 12 | **`is_backbone_frozen` rescans all parameters** | [model.py:243](../src/model.py); used inside `describe()` | O(params) per property read |
| 13 | **`forward_on` mutates model device** | [verification.py:106](../src/verification.py) | Side effect not implied by the name |
| 14 | **API size limit applied post-buffering** | [routes.py:213](../src/api/routes.py) | Oversized upload occupies memory before rejection |
| 15 | **No CI, no pre-commit, no Dockerfile** | Absent from repo | Gates run manually |
| 16 | **`ruff format` not clean** | `ruff check` passes; `ruff format --check` reports 14 of 55 files would be reformatted | Formatting not gated |

## Duplicated logic

| Pair | Files | Notes |
| --- | --- | --- |
| ~~Config override~~ | [config.py](../src/config.py) / [test_milestone3.py](../tests/test_milestone3.py) | **Resolved.** The test helper now calls `with_overrides` |
| ~~GPU memory helpers~~ | [training/engine.py](../src/training/engine.py) / [evaluation/inference.py](../src/evaluation/inference.py) | **Resolved.** `synchronize`, `reset_peak_memory` and `peak_memory_mib` now live once in [device.py](../src/device.py) |
| Console report rendering | [verification.py:171](../src/verification.py) / [verify_pipeline.py](../src/verify_pipeline.py) / [evaluate.py:267](../src/evaluate.py) / [audit_dataset.py:37](../src/audit_dataset.py) | All build on `src/reporting.py` primitives but each re-implements layout. Left as is: the four reports are genuinely different documents, and a shared abstraction would have to be re-parameterised per caller |
| Figure specification | [plots.py](../src/visualization/plots.py) / [evaluation_plots.py](../src/visualization/evaluation_plots.py) | `EvaluationPlotSpecification` reuses `PlotSpecification` for size/dpi, so duplication is partial |

## Undocumented components

Components previously absent from the root `README.md`. The rewritten README now
summarises each and links to the detailed page:

| Component | Documented in |
| --- | --- |
| Dataset audit and its 13 issue categories | [ARCHITECTURE.md](ARCHITECTURE.md), this file |
| Training pipeline, AMP, early stopping, checkpoints | [MODEL.md](MODEL.md), [CONFIG.md](CONFIG.md) |
| Evaluation metric suite and integrity checks | [METRICS.md](METRICS.md) |
| FastAPI service | [API.md](API.md) |
| `dataloader`, `training`, `evaluation`, `visualization`, `api` config sections | [CONFIG.md](CONFIG.md) |

## Test coverage gaps

| Area | Status |
| --- | --- |
| `src/datasets/validation.py` | No dedicated unit tests |
| `src/datasets/loaders.py`, `transforms.py` | Exercised only via `verify_pipeline`, not unit tests |
| `src/training/*` | No unit tests; covered by `verify_pipeline` at runtime |
| `src/evaluation/*` | No unit tests |
| `src/visualization/*` | No unit tests |
| `src/api/*` | 35 unit tests |
| Real checkpoint load path (`_rebuild_model`) | Not unit-tested — API tests use a stand-in backbone |
