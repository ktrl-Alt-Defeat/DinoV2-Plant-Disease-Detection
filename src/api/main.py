"""Application factory and ASGI entry point.

``uv run uvicorn src.api.main:app`` serves this module's ``app``. The model is
loaded once inside the lifespan handler, before the first request is accepted,
and released when the process shuts down. Importing this module does not load
weights, which keeps test collection and tooling fast.

Swagger UI is served at ``/docs`` and the raw schema at ``/openapi.json``.
"""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import ENGINE_ATTRIBUTE, SETTINGS_ATTRIBUTE
from src.api.errors import (
    REQUEST_ID_ATTRIBUTE,
    REQUEST_ID_HEADER,
    ApiError,
    handle_api_error,
    handle_http_exception,
    handle_unexpected_error,
    handle_validation_error,
)
from src.api.inference import InferenceEngine
from src.api.routes import router
from src.api.settings import ApiSettings
from src.cli import DEFAULT_CONFIG_PATH
from src.config import load_config
from src.logger import configure_logging, get_logger
from src.paths import ProjectPaths, resolve
from src.seed import set_seed

#: Route prefix documenting the interactive schema.
DOCS_URL: Final[str] = "/docs"

#: Route serving the raw OpenAPI document.
OPENAPI_URL: Final[str] = "/openapi.json"

_LOGGER: Final = get_logger("api.main")


def create_app(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    engine: InferenceEngine | None = None,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        config_path: Configuration driving the service.
        engine: Pre-built engine. When supplied the lifespan handler skips
            loading a checkpoint, which is how tests exercise the endpoints
            without a trained model on disk.
    """
    settings = _prepare(config_path, load_model=engine is None)

    application = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.version,
        docs_url=DOCS_URL,
        openapi_url=OPENAPI_URL,
        lifespan=_lifespan_factory(config_path, settings, engine),
    )

    application.middleware("http")(_assign_request_id)
    application.add_exception_handler(ApiError, handle_api_error)
    application.add_exception_handler(StarletteHTTPException, handle_http_exception)
    application.add_exception_handler(RequestValidationError, handle_validation_error)
    application.add_exception_handler(Exception, handle_unexpected_error)
    application.include_router(router)
    return application


def _prepare(config_path: str | Path, *, load_model: bool) -> ApiSettings:
    """Resolve settings and initialise logging, directories and determinism."""
    config = load_config(config_path)
    settings = ApiSettings.from_config(config)

    log_dir = resolve(config.get("paths.logs")) if config.get("logging.save_file") else None
    configure_logging(
        level=config.get("logging.level"),
        log_dir=log_dir,
        filename=settings.log_filename,
    )
    ProjectPaths.from_mapping(config.section("paths")).create()
    set_seed(
        config.get("project.seed"),
        deterministic=config.get("reproducibility.deterministic"),
        benchmark=config.get("reproducibility.benchmark"),
    )
    _LOGGER.info(
        "API configured: %s v%s, model loading %s.",
        settings.title,
        settings.version,
        "enabled" if load_model else "skipped (engine supplied)",
    )
    return settings


def _lifespan_factory(
    config_path: str | Path,
    settings: ApiSettings,
    engine: InferenceEngine | None,
) -> Callable[[FastAPI], object]:
    """Build the lifespan handler that owns the model for the process lifetime."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        setattr(application.state, SETTINGS_ATTRIBUTE, settings)
        loaded = engine if engine is not None else _load_engine(config_path, settings)
        setattr(application.state, ENGINE_ATTRIBUTE, loaded)
        try:
            yield
        finally:
            setattr(application.state, ENGINE_ATTRIBUTE, None)
            _LOGGER.info("API shutting down; model released.")

    return lifespan


def _load_engine(config_path: str | Path, settings: ApiSettings) -> InferenceEngine:
    """Load the configured checkpoint exactly once."""
    config = load_config(config_path)
    checkpoint = resolve(config.get("paths.checkpoints")) / settings.checkpoint_filename

    _LOGGER.info("Loading checkpoint %s.", checkpoint)
    return InferenceEngine.load(
        config,
        checkpoint,
        top_k=settings.top_k,
        version=settings.version,
    )


async def _assign_request_id(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach an identifier to every request and echo it on the response.

    A client-supplied ``X-Request-ID`` is honoured so a call can be traced
    across services; otherwise a fresh identifier is generated.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    setattr(request.state, REQUEST_ID_ATTRIBUTE, request_id)

    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


#: ASGI application served by uvicorn.
app = create_app()
