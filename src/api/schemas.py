"""Request and response models.

These types are the API contract: FastAPI derives the OpenAPI schema shown at
``/docs`` from them, so anything documented here is what a client actually
receives.

Pydantic reserves the ``model_`` prefix for its own attributes, so every model
carrying a ``model_version`` field clears ``protected_namespaces``.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

#: Frees the ``model_`` field prefix, which the API contract uses.
_ALLOW_MODEL_PREFIX: Final[ConfigDict] = ConfigDict(protected_namespaces=())


class ClassPrediction(BaseModel):
    """One ranked class and the probability assigned to it."""

    label: str = Field(description="Class name as discovered from the dataset layout.")
    index: int = Field(ge=0, description="Class index the model emits.")
    confidence: float = Field(ge=0.0, le=1.0, description="Softmax probability.")


class PredictionResponse(BaseModel):
    """Prediction for a single image."""

    model_config = _ALLOW_MODEL_PREFIX

    request_id: str = Field(description="Identifier echoed in the X-Request-ID header.")
    filename: str = Field(description="Name of the uploaded file.")
    predicted_class: str = Field(description="Highest scoring class.")
    predicted_index: int = Field(ge=0, description="Index of the highest scoring class.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the top class.")
    top_k: list[ClassPrediction] = Field(description="Ranked predictions, most likely first.")
    inference_time_ms: float = Field(ge=0.0, description="Forward pass duration, milliseconds.")
    model_version: str = Field(description="Version of the served model.")


class BatchPredictionResponse(BaseModel):
    """Predictions for a batch of images, scored in one forward pass."""

    model_config = _ALLOW_MODEL_PREFIX

    request_id: str = Field(description="Identifier echoed in the X-Request-ID header.")
    count: int = Field(ge=0, description="Number of images scored.")
    predictions: list[PredictionResponse] = Field(description="One entry per uploaded image.")
    inference_time_ms: float = Field(
        ge=0.0, description="Duration of the single batched forward pass, milliseconds."
    )
    model_version: str = Field(description="Version of the served model.")


class GpuStatus(BaseModel):
    """State of the CUDA device backing the service, when there is one."""

    name: str = Field(description="GPU model name.")
    capability: str = Field(description="CUDA compute capability, e.g. sm_120.")
    total_memory_mib: float = Field(ge=0.0, description="Total device memory.")
    allocated_memory_mib: float = Field(ge=0.0, description="Currently allocated by this process.")
    reserved_memory_mib: float = Field(ge=0.0, description="Reserved by the caching allocator.")


class HealthResponse(BaseModel):
    """Liveness and readiness of the service."""

    model_config = _ALLOW_MODEL_PREFIX

    status: str = Field(description="'ok' when the model is loaded and ready.")
    model_loaded: bool = Field(description="Whether weights are resident in memory.")
    device: str = Field(description="Device the model runs on.")
    cuda_available: bool = Field(description="Whether CUDA is visible to the process.")
    gpu: GpuStatus | None = Field(default=None, description="GPU detail, absent on CPU.")
    uptime_seconds: float = Field(ge=0.0, description="Seconds since the model finished loading.")
    version: str = Field(description="Service version.")


class CheckpointInfo(BaseModel):
    """Identity of the checkpoint being served."""

    filename: str = Field(description="File name of the served checkpoint.")
    sha256: str = Field(description="SHA-256 of the checkpoint file.")
    epoch: int = Field(ge=0, description="Training epoch the checkpoint was written at.")
    best_value: float = Field(description="Monitored metric value that selected it.")


class MetadataResponse(BaseModel):
    """Description of the served model and its class vocabulary."""

    model_config = _ALLOW_MODEL_PREFIX

    model_version: str = Field(description="Version of the served model.")
    backbone: str = Field(description="Backbone entrypoint, e.g. dinov2_vitb14.")
    backbone_display: str = Field(description="Readable backbone name.")
    feature_dim: int = Field(gt=0, description="Embedding width of the backbone.")
    image_size: int = Field(gt=0, description="Square input resolution.")
    num_classes: int = Field(gt=0, description="Number of classes the head predicts.")
    classes: list[str] = Field(description="Class names ordered by index.")
    class_to_idx: dict[str, int] = Field(description="Class name to index mapping.")
    total_parameters: int = Field(gt=0, description="Parameter count of the served model.")
    device: str = Field(description="Device the model runs on.")
    precision: str = Field(description="Numeric precision inference runs in.")
    top_k: int = Field(gt=0, description="Number of ranked predictions returned.")
    max_batch_size: int = Field(gt=0, description="Largest accepted batch.")
    checkpoint: CheckpointInfo = Field(description="Identity of the served checkpoint.")


class ErrorResponse(BaseModel):
    """Body returned for every non-2xx response."""

    request_id: str = Field(description="Identifier echoed in the X-Request-ID header.")
    error: str = Field(description="Machine-readable error code.")
    detail: str = Field(description="Human-readable explanation.")
    status_code: int = Field(description="HTTP status code of the response.")
