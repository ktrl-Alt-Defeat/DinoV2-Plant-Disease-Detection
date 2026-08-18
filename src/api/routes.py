"""HTTP endpoints.

Handlers are deliberately thin: they validate the upload, hand decoded images to
the engine and shape the result. Every rejection raises a typed
:class:`~src.api.errors.ApiError` so the response body has one structure.

The handlers are synchronous, so FastAPI runs them on its worker thread pool and
the event loop is never blocked by the forward pass.
"""

from collections.abc import Sequence
from typing import Annotated, Final

from fastapi import APIRouter, File, Request, UploadFile, status

from src.api.dependencies import EngineDependency, SettingsDependency
from src.api.errors import (
    InvalidImageError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    request_id_of,
)
from src.api.inference import ImagePrediction, decode_image
from src.api.schemas import (
    BatchPredictionResponse,
    CheckpointInfo,
    ClassPrediction,
    ErrorResponse,
    GpuStatus,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
)
from src.api.settings import ApiSettings
from src.logger import get_logger

#: Status reported by ``GET /health`` when the model is resident and ready.
STATUS_OK: Final[str] = "ok"

#: Responses documented for the prediction endpoints in the OpenAPI schema.
_PREDICT_RESPONSES: Final[dict[int | str, dict[str, object]]] = {
    status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Undecodable image."},
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "Upload exceeds the size or batch limit.",
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "model": ErrorResponse,
        "description": "Content type is not an accepted image type.",
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": ErrorResponse,
        "description": "Request did not satisfy the endpoint signature.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "Model is still loading.",
    },
}

_LOGGER: Final = get_logger("api.routes")

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service, model and GPU status",
    tags=["operations"],
)
def health(engine: EngineDependency) -> HealthResponse:
    """Report whether the service is ready to serve predictions."""
    gpu = engine.gpu_status()
    return HealthResponse(
        status=STATUS_OK,
        model_loaded=True,
        device=str(engine.device),
        cuda_available=engine.device.type == "cuda",
        gpu=GpuStatus(**gpu) if gpu is not None else None,
        uptime_seconds=engine.uptime_seconds,
        version=engine.version,
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Model, class vocabulary and version",
    tags=["operations"],
)
def metadata(engine: EngineDependency, settings: SettingsDependency) -> MetadataResponse:
    """Describe the served model and the classes it can predict."""
    checkpoint = engine.checkpoint
    return MetadataResponse(
        **engine.describe(),
        max_batch_size=settings.max_batch_size,
        checkpoint=CheckpointInfo(
            filename=checkpoint.filename,
            sha256=checkpoint.sha256,
            epoch=checkpoint.epoch,
            best_value=checkpoint.best_value,
        ),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses=_PREDICT_RESPONSES,
    summary="Classify a single image",
    tags=["inference"],
)
def predict(
    request: Request,
    engine: EngineDependency,
    settings: SettingsDependency,
    file: Annotated[UploadFile, File(description="Image to classify.")],
) -> PredictionResponse:
    """Classify one uploaded image."""
    images = _decode_uploads([file], settings)
    result = engine.predict(images)
    request_id = request_id_of(request)

    _LOGGER.info(
        "request_id=%s POST /predict file=%s -> %s (%.4f) in %.2f ms",
        request_id,
        file.filename,
        result.predictions[0].top.label,
        result.predictions[0].top.confidence,
        result.inference_time_ms,
    )
    return _to_response(
        result.predictions[0], request_id, result.inference_time_ms, engine.version
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses=_PREDICT_RESPONSES,
    summary="Classify several images in one forward pass",
    tags=["inference"],
)
def predict_batch(
    request: Request,
    engine: EngineDependency,
    settings: SettingsDependency,
    files: Annotated[list[UploadFile], File(description="Images to classify.")],
) -> BatchPredictionResponse:
    """Classify a batch of uploaded images.

    Raises:
        PayloadTooLargeError: If the batch exceeds ``api.max_batch_size``.
        InvalidImageError: If no file was supplied.
    """
    if not files:
        raise InvalidImageError("No file was supplied.")
    if len(files) > settings.max_batch_size:
        raise PayloadTooLargeError(
            f"Batch of {len(files)} images exceeds the limit of {settings.max_batch_size}."
        )

    images = _decode_uploads(files, settings)
    result = engine.predict(images)
    request_id = request_id_of(request)

    _LOGGER.info(
        "request_id=%s POST /predict/batch count=%d in %.2f ms",
        request_id,
        len(result.predictions),
        result.inference_time_ms,
    )
    return BatchPredictionResponse(
        request_id=request_id,
        count=len(result.predictions),
        predictions=[
            _to_response(prediction, request_id, result.inference_time_ms, engine.version)
            for prediction in result.predictions
        ],
        inference_time_ms=result.inference_time_ms,
        model_version=engine.version,
    )


def _decode_uploads(
    files: Sequence[UploadFile],
    settings: ApiSettings,
) -> list[tuple[str, object]]:
    """Validate and decode every upload.

    Content type and size are checked before decoding, so an oversized or
    mistyped upload is rejected without spending time on it.

    Raises:
        UnsupportedMediaTypeError: If a content type is not accepted.
        PayloadTooLargeError: If a file exceeds ``api.max_image_bytes``.
        InvalidImageError: If a file is empty or cannot be decoded.
    """
    decoded: list[tuple[str, object]] = []
    for upload in files:
        filename = upload.filename or "upload"
        content_type = (upload.content_type or "").split(";")[0].strip().lower()

        if content_type not in settings.allowed_content_types:
            accepted = ", ".join(sorted(settings.allowed_content_types))
            raise UnsupportedMediaTypeError(
                f"'{filename}' has content type '{content_type or 'unset'}'. "
                f"Accepted types: {accepted}."
            )

        payload = upload.file.read()
        if len(payload) > settings.max_image_bytes:
            raise PayloadTooLargeError(
                f"'{filename}' is {len(payload):,} bytes, above the "
                f"{settings.max_image_bytes:,} byte limit."
            )

        decoded.append((filename, decode_image(filename, payload)))
    return decoded


def _to_response(
    prediction: ImagePrediction,
    request_id: str,
    inference_time_ms: float,
    version: str,
) -> PredictionResponse:
    """Shape one image prediction into the response contract."""
    top = prediction.top
    return PredictionResponse(
        request_id=request_id,
        filename=prediction.filename,
        predicted_class=top.label,
        predicted_index=top.index,
        confidence=top.confidence,
        top_k=[
            ClassPrediction(label=scored.label, index=scored.index, confidence=scored.confidence)
            for scored in prediction.ranked
        ],
        inference_time_ms=inference_time_ms,
        model_version=version,
    )
