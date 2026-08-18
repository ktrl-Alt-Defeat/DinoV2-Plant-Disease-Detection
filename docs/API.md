# API

FastAPI inference service. All facts below are from [src/api/](../src/api/).

## Application

| Property | Value | Source |
| --- | --- | --- |
| ASGI object | `app` | [src/api/main.py:161](../src/api/main.py) |
| Factory | `create_app(*, config_path, engine)` | [src/api/main.py:49](../src/api/main.py) |
| Title | `api.title` config value | [src/api/settings.py](../src/api/settings.py) |
| Version | `project.version` (`1.0.0`) | [src/api/settings.py:69](../src/api/settings.py) |
| Swagger UI | `/docs` | [src/api/main.py:41](../src/api/main.py) |
| OpenAPI schema | `/openapi.json` | [src/api/main.py:44](../src/api/main.py) |
| ReDoc | **Not Found** — not configured | — |

`create_app` accepts an optional pre-built `engine`; when supplied, the lifespan
handler skips checkpoint loading ([src/api/main.py:118](../src/api/main.py)).

## Startup and shutdown

| Phase | Action | Source |
| --- | --- | --- |
| Import | `app = create_app()` runs `_prepare()`: load config, configure logging, create dirs, `set_seed` | [src/api/main.py:82](../src/api/main.py) |
| Startup (lifespan) | Set `app.state.settings`, then load engine into `app.state.engine` | [src/api/main.py:115](../src/api/main.py) |
| Shutdown | `app.state.engine = None` | [src/api/main.py:123](../src/api/main.py) |

The checkpoint is read **once** at startup by `_load_engine`
([src/api/main.py:129](../src/api/main.py)) from
`paths.checkpoints / api.checkpoint_filename`. Importing the module does not load
weights.

## Routes

| Method | Path | Handler | Response model | Tag |
| --- | --- | --- | --- | --- |
| GET | `/health` | `health` | `HealthResponse` | operations |
| GET | `/metadata` | `metadata` | `MetadataResponse` | operations |
| POST | `/predict` | `predict` | `PredictionResponse` | inference |
| POST | `/predict/batch` | `predict_batch` | `BatchPredictionResponse` | inference |

Source: [src/api/routes.py:66–149](../src/api/routes.py). All four handlers are
**synchronous**, so FastAPI dispatches them onto its worker thread pool
([src/api/routes.py:1](../src/api/routes.py)).

## Response models

Defined in [src/api/schemas.py](../src/api/schemas.py). Models carrying a
`model_version` field set `protected_namespaces=()` because Pydantic reserves the
`model_` prefix ([src/api/schemas.py:19](../src/api/schemas.py)).

### `PredictionResponse` ([:27](../src/api/schemas.py))

| Field | Type | Constraint |
| --- | --- | --- |
| `request_id` | str | — |
| `filename` | str | — |
| `predicted_class` | str | — |
| `predicted_index` | int | `ge=0` |
| `confidence` | float | `ge=0.0, le=1.0` |
| `top_k` | list[`ClassPrediction`] | — |
| `inference_time_ms` | float | `ge=0.0` |
| `model_version` | str | — |

`ClassPrediction` ([:19](../src/api/schemas.py)): `label` (str), `index` (int, `ge=0`),
`confidence` (float, `ge=0.0, le=1.0`).

### `BatchPredictionResponse` ([:42](../src/api/schemas.py))

`request_id`, `count` (`ge=0`), `predictions` (list[`PredictionResponse`]),
`inference_time_ms` (`ge=0.0`), `model_version`.

`inference_time_ms` is the duration of the **single batched forward pass** and is
copied onto every nested `PredictionResponse`
([src/api/routes.py:177](../src/api/routes.py)).

### `HealthResponse` ([:66](../src/api/schemas.py))

| Field | Type | Notes |
| --- | --- | --- |
| `status` | str | Literal `"ok"` from `STATUS_OK` ([routes.py:38](../src/api/routes.py)) |
| `model_loaded` | bool | Hardcoded `True` — the route is unreachable unless the engine resolved |
| `device` | str | `str(engine.device)` |
| `cuda_available` | bool | `engine.device.type == "cuda"` |
| `gpu` | `GpuStatus \| None` | `None` on CPU |
| `uptime_seconds` | float | Since engine construction, `time.monotonic()` |
| `version` | str | `project.version` |

`GpuStatus` ([:56](../src/api/schemas.py)): `name`, `capability` (`sm_XY`),
`total_memory_mib`, `allocated_memory_mib`, `reserved_memory_mib`. Populated by
`InferenceEngine.gpu_status()` ([src/api/inference.py:241](../src/api/inference.py)).

### `MetadataResponse` ([:89](../src/api/schemas.py))

`model_version`, `backbone`, `backbone_display`, `feature_dim`, `image_size`,
`num_classes`, `classes`, `class_to_idx`, `total_parameters`, `device`,
`precision`, `top_k`, `max_batch_size`, `checkpoint` (`CheckpointInfo`).

`CheckpointInfo` ([:80](../src/api/schemas.py)): `filename`, `sha256`, `epoch`,
`best_value`.

### `ErrorResponse` ([:110](../src/api/schemas.py))

`request_id`, `error` (code), `detail`, `status_code`. Returned for **every**
non-2xx response.

## Validation

Applied in `_decode_uploads` ([src/api/routes.py:186](../src/api/routes.py)),
in this order, before any decoding work:

| Order | Check | Failure | Status |
| --- | --- | --- | --- |
| 1 | Batch size `> api.max_batch_size` (batch route only) | `PayloadTooLargeError` | 413 |
| 2 | Content type in `api.allowed_content_types` | `UnsupportedMediaTypeError` | 415 |
| 3 | Byte length `> api.max_image_bytes` | `PayloadTooLargeError` | 413 |
| 4 | Non-empty + PIL-decodable | `InvalidImageError` | 400 |

Content type is lowercased and split on `;` to strip parameters
([routes.py:203](../src/api/routes.py)). Decoded images are converted to RGB via
`decode_image` ([src/api/inference.py:285](../src/api/inference.py)).

Empty `files` list on the batch route raises `InvalidImageError`
([routes.py:157](../src/api/routes.py)).

> Size is enforced **after** the body is buffered by Starlette, not during
> transfer. See [DEPLOYMENT.md](DEPLOYMENT.md#limits).

## Exceptions

| Class | Status | `error` code | Source |
| --- | --- | --- | --- |
| `ApiError` (base) | 500 | `internal_error` | [errors.py:32](../src/api/errors.py) |
| `ModelNotReadyError` | 503 | `model_not_ready` | [errors.py:43](../src/api/errors.py) |
| `UnsupportedMediaTypeError` | 415 | `unsupported_media_type` | [errors.py:50](../src/api/errors.py) |
| `PayloadTooLargeError` | 413 | `payload_too_large` | [errors.py:57](../src/api/errors.py) |
| `InvalidImageError` | 400 | `invalid_image` | [errors.py:64](../src/api/errors.py) |
| `InferenceFailedError` | 500 | `inference_failed` | [errors.py:71](../src/api/errors.py) |

### Registered handlers

| Exception | Handler | Produces |
| --- | --- | --- |
| `ApiError` | `handle_api_error` | Its own status + code, logged at WARNING |
| `StarletteHTTPException` | `handle_http_exception` | `http_error` (e.g. 404) |
| `RequestValidationError` | `handle_validation_error` | 422 `validation_error`, joined field errors |
| `Exception` | `handle_unexpected_error` | 500 `internal_error`, logged with traceback, internals not leaked |

Source: [src/api/main.py:74–77](../src/api/main.py),
[src/api/errors.py:100–149](../src/api/errors.py).

`InferenceFailedError` is raised when the forward pass yields non-finite
probabilities ([src/api/inference.py:199](../src/api/inference.py)).

## Middleware

Exactly one HTTP middleware is registered.

| Middleware | Behaviour | Source |
| --- | --- | --- |
| `_assign_request_id` | Reads `X-Request-ID` from the request, else generates `uuid.uuid4()`. Stores on `request.state.request_id` and echoes on the response header. | [src/api/main.py:143](../src/api/main.py) |

**CORS middleware: Not Found.** **Compression middleware: Not Found.**
**Rate limiting: Not Found.**

## Authentication and authorization

**Not Found.** No authentication, API key, token, OAuth or dependency-based
security scheme exists anywhere in `src/`. All endpoints are unauthenticated.

## Dependency injection

| Provider | Returns | Failure | Source |
| --- | --- | --- | --- |
| `get_engine` | `app.state.engine` | `ModelNotReadyError` (503) if `None` | [dependencies.py:26](../src/api/dependencies.py) |
| `get_settings` | `app.state.settings` | `ModelNotReadyError` (503) if `None` | [dependencies.py:38](../src/api/dependencies.py) |

Exposed as `Annotated` aliases `EngineDependency` and `SettingsDependency`
([dependencies.py:47](../src/api/dependencies.py)). Tests override them through
`app.dependency_overrides` ([tests/test_milestone6.py](../tests/test_milestone6.py)).

## Concurrency

`InferenceEngine.predict` serialises the forward pass behind a
`threading.Lock` ([src/api/inference.py:124](../src/api/inference.py),
[:188](../src/api/inference.py)). Preprocessing (`self._transform`) runs
**outside** the lock; device transfer, forward, softmax and synchronisation run
inside it.

## Examples

Request and response shapes are taken from the declared schemas.

```bash
# Health
curl http://localhost:8000/health

# Single image
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf.jpg;type=image/jpeg"

# Batch (repeat the 'files' field)
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@leaf1.jpg;type=image/jpeg" \
  -F "files=@leaf2.jpg;type=image/jpeg"

# Supply your own trace id
curl -X POST http://localhost:8000/predict \
  -H "X-Request-ID: trace-123" \
  -F "file=@leaf.jpg;type=image/jpeg"
```

`POST /predict` response shape:

```json
{
  "request_id": "trace-123",
  "filename": "leaf.jpg",
  "predicted_class": "corn_maize___healthy",
  "predicted_index": 9,
  "confidence": 1.0,
  "top_k": [{ "label": "corn_maize___healthy", "index": 9, "confidence": 1.0 }],
  "inference_time_ms": 4.68,
  "model_version": "1.0.0"
}
```

Error shape (all non-2xx):

```json
{
  "request_id": "1f0c...",
  "error": "unsupported_media_type",
  "detail": "'notes.txt' has content type 'text/plain'. Accepted types: image/bmp, image/jpeg, image/png, image/webp.",
  "status_code": 415
}
```

Field names, types and the `error` code vocabulary are verified against
[src/api/schemas.py](../src/api/schemas.py) and
[src/api/errors.py](../src/api/errors.py). The literal values above are
illustrative.

## Documented response codes in OpenAPI

Both prediction routes declare 400, 413, 415, 422 and 503 with `ErrorResponse`
via `_PREDICT_RESPONSES` ([src/api/routes.py:41](../src/api/routes.py)). The
`operations` routes declare no extra responses.

## Test coverage

35 test methods in [tests/test_milestone6.py](../tests/test_milestone6.py) across
9 classes: `SettingsTests`, `HealthEndpointTests`, `MetadataEndpointTests`,
`PredictEndpointTests`, `BatchEndpointTests`, `DocumentationTests`,
`ErrorContractTests`, `EngineTests`, `DecodeTests`. Tests inject a stand-in
backbone into the production `InferenceEngine` rather than mocking the engine.
