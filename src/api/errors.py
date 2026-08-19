"""Structured error handling.

Every failure the service can produce is one of these exceptions, each carrying
the HTTP status and the machine-readable code it should surface. Handlers
registered in :mod:`src.api.main` render them all through the same
:class:`~src.api.schemas.ErrorResponse` body, so a client parses one shape
whether a request was malformed, too large or hit an unexpected fault.
"""

from typing import Final

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.schemas import ErrorResponse
from src.logger import get_logger

#: Header carrying the per-request identifier.
REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

#: Attribute the middleware stores the identifier under.
REQUEST_ID_ATTRIBUTE: Final[str] = "request_id"

#: Used when a failure happens before the middleware assigned an identifier.
UNKNOWN_REQUEST_ID: Final[str] = "unknown"

_LOGGER: Final = get_logger("api.errors")


class ApiError(Exception):
    """Base class for failures the service reports deliberately."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ModelNotReadyError(ApiError):
    """Raised when a request arrives before the model finished loading."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_ready"


class UnsupportedMediaTypeError(ApiError):
    """Raised when an upload declares a content type the service does not accept."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"


class PayloadTooLargeError(ApiError):
    """Raised when an upload exceeds the configured size or batch limit."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"


class InvalidImageError(ApiError):
    """Raised when an upload is empty or cannot be decoded as an image."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "invalid_image"


class InferenceFailedError(ApiError):
    """Raised when the forward pass could not produce a usable result."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "inference_failed"


def request_id_of(request: Request) -> str:
    """Return the identifier assigned to ``request``, or a placeholder."""
    return getattr(request.state, REQUEST_ID_ATTRIBUTE, UNKNOWN_REQUEST_ID)


def error_response(request: Request, *, status_code: int, code: str, detail: str) -> JSONResponse:
    """Render one error body, echoing the request identifier in the header."""
    request_id = request_id_of(request)
    payload = ErrorResponse(
        request_id=request_id, error=code, detail=detail, status_code=status_code
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(),
        headers={REQUEST_ID_HEADER: request_id},
    )


async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    """Render a deliberate service failure."""
    _LOGGER.warning(
        "%s %s -> %d %s: %s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.code,
        exc.detail,
    )
    return error_response(
        request, status_code=exc.status_code, code=exc.code, detail=exc.detail
    )


async def handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Render an HTTP error raised by the framework, such as an unknown route."""
    return error_response(
        request,
        status_code=exc.status_code,
        code="http_error",
        detail=str(exc.detail),
    )


async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render a request that did not satisfy the endpoint signature."""
    detail = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )
    return error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="validation_error",
        detail=detail or "Request failed validation.",
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Render an unanticipated fault without leaking internals to the client."""
    _LOGGER.exception(
        "Unhandled error on %s %s: %s", request.method, request.url.path, exc
    )
    return error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        detail="The service encountered an unexpected error.",
    )
