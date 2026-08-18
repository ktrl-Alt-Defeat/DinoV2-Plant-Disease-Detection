"""Production inference API.

:mod:`~src.api.inference` owns the loaded model, :mod:`~src.api.routes` exposes
it over HTTP, :mod:`~src.api.schemas` is the wire contract and
:mod:`~src.api.main` assembles them into an ASGI application.

The application object is imported lazily through :func:`create_app` so that
importing this package never loads a checkpoint.
"""

from src.api.inference import InferenceEngine
from src.api.settings import ApiSettings

__all__ = ["ApiSettings", "InferenceEngine"]
