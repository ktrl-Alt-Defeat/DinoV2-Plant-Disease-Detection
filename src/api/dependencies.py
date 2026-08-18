"""Dependency providers.

Endpoints declare what they need and FastAPI supplies it, which keeps the route
handlers free of construction logic and lets a test replace the engine through
``app.dependency_overrides`` without touching a checkpoint.

Both the engine and the settings live on ``app.state``, populated once during
startup, so a request never triggers construction.
"""

from typing import Annotated, Final

from fastapi import Depends, Request

from src.api.errors import ModelNotReadyError
from src.api.inference import InferenceEngine
from src.api.settings import ApiSettings

#: ``app.state`` attribute holding the loaded engine.
ENGINE_ATTRIBUTE: Final[str] = "engine"

#: ``app.state`` attribute holding the resolved settings.
SETTINGS_ATTRIBUTE: Final[str] = "settings"


def get_engine(request: Request) -> InferenceEngine:
    """Return the engine loaded at startup.

    Raises:
        ModelNotReadyError: If startup has not finished loading the model.
    """
    engine = getattr(request.app.state, ENGINE_ATTRIBUTE, None)
    if engine is None:
        raise ModelNotReadyError("The model is still loading; retry shortly.")
    return engine


def get_settings(request: Request) -> ApiSettings:
    """Return the settings resolved at startup.

    Raises:
        ModelNotReadyError: If startup has not finished resolving configuration.
    """
    settings = getattr(request.app.state, SETTINGS_ATTRIBUTE, None)
    if settings is None:
        raise ModelNotReadyError("The service is still starting; retry shortly.")
    return settings


#: Injected engine, ready to score a batch.
EngineDependency = Annotated[InferenceEngine, Depends(get_engine)]

#: Injected settings, carrying the upload limits.
SettingsDependency = Annotated[ApiSettings, Depends(get_settings)]
