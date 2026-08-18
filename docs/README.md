# DINOv2-LeafCare — Documentation

Reverse-engineered documentation for this repository. Every statement below is
derived from source files in this repository and cites them. Anything that could
not be verified in code is recorded as **Not Found**.

The root [`README.md`](../README.md) is the project overview; this directory holds
the detailed reference.

## Index

| Document | Covers |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Package layout, module dependency graph, execution flows |
| [API.md](API.md) | FastAPI routes, schemas, validation, middleware, exceptions, examples |
| [MODEL.md](MODEL.md) | Backbone, head, preprocessing, inference, checkpoint format, versioning |
| [METRICS.md](METRICS.md) | Every implemented metric, where computed, interpretation |
| [CONFIG.md](CONFIG.md) | All 94 configuration keys, defaults, consumers, environment variables |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Dependencies, build, startup, serving stack, commands |
| [CODEMAP.md](CODEMAP.md) | Directory / file / class / function responsibilities, technical debt |

## Project identity

| Field | Value | Source |
| --- | --- | --- |
| Distribution name | `dinov2-leafcare` | [pyproject.toml:2](../pyproject.toml) |
| Distribution version | `0.1.0` | [pyproject.toml:3](../pyproject.toml) |
| Runtime `project.version` | `1.0.0` | [configs/config.yaml:9](../configs/config.yaml) |
| Python requirement | `>=3.11` | [pyproject.toml:6](../pyproject.toml) |
| Lint config | ruff, line-length 100, target py311 | [pyproject.toml](../pyproject.toml) |

> The two version fields differ and are read by different code paths. The API and
> all reports surface `project.version` (`1.0.0`), never the distribution version.
> See [CONFIG.md](CONFIG.md#version-fields).

## Executable entry points

Six modules define `if __name__ == "__main__"`, plus one ASGI application object.

| Command | Module | Purpose |
| --- | --- | --- |
| `python -m src.cli` | [src/cli.py:202](../src/cli.py) | Infrastructure bootstrap and environment report |
| `python -m src.model` | [src/model.py:432](../src/model.py) | Structural model verification on synthetic tensors |
| `python -m src.audit_dataset` | [src/audit_dataset.py:113](../src/audit_dataset.py) | Dataset integrity audit |
| `python -m src.verify_pipeline` | [src/verify_pipeline.py:406](../src/verify_pipeline.py) | End-to-end training pipeline verification |
| `python -m src.train` | [src/train.py:563](../src/train.py) | Fine-tuning run |
| `python -m src.evaluate` | [src/evaluate.py:559](../src/evaluate.py) | Held-out split evaluation |
| `uvicorn src.api.main:app` | [src/api/main.py:161](../src/api/main.py) | Inference REST API |

## Repository scope

| Metric | Value | Source |
| --- | --- | --- |
| Python files (`src` + `tests`) | 46 (43 + 3) | `find` excluding `__pycache__` |
| Total lines | 10,339 | `wc -l` over the same set |
| Test methods | 127 (44 + 48 + 35) | [tests/](../tests/) |
| Public module-level definitions in `src` | 172 | AST scan |
| Unreferenced public definitions | 0 (see [CODEMAP.md](CODEMAP.md#dead-code)) | Reference scan |

## Documentation conventions

- Line references point at the definition site at the time of writing.
- Tables state observed defaults from [configs/config.yaml](../configs/config.yaml),
  not recommended values.
- Only implemented behaviour is documented. Reserved or empty modules are listed
  as such in [CODEMAP.md](CODEMAP.md).
