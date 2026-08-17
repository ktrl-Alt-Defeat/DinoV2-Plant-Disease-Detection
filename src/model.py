"""DINOv2 backbone with a configurable classification head.

The backbone encodes an image batch into embeddings and the head turns those
embeddings into class logits. Both halves are described by the configuration,
so a different backbone or a different head is a configuration change rather
than a code change.

Running ``python -m src.model`` executes the structural verification defined in
:mod:`src.verification`.
"""

import re
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import nn

from src import utils
from src.config import Config, TypeSpec, validate_keys
from src.device import get_device
from src.logger import get_logger

#: ``torch.hub`` repository publishing the official DINOv2 weights.
DINOV2_HUB_REPOSITORY: Final[str] = "facebookresearch/dinov2"

#: Canonical location the official checkpoints are downloaded from.
DINOV2_WEIGHTS_URL_TEMPLATE: Final[str] = (
    "https://dl.fbaipublicfiles.com/dinov2/{name}/{name}_pretrain.pth"
)

#: Official backbone entrypoints. Any other entrypoint published by the
#: repository is accepted as well; this tuple documents the supported variants
#: and is quoted when a backbone fails to load.
KNOWN_BACKBONES: Final[tuple[str, ...]] = (
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
)

#: Classification head architectures accepted by ``classifier.type``.
SUPPORTED_CLASSIFIER_TYPES: Final[frozenset[str]] = frozenset({"linear"})

#: Configuration contract of the ``model`` section.
MODEL_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("model.name", str),
    ("model.pretrained", bool),
    ("model.freeze_backbone", bool),
    ("model.image_size", int),
    ("model.feature_dim", int),
    ("model.num_classes", int),
)

#: Configuration contract of the ``classifier`` section.
CLASSIFIER_REQUIRED_KEYS: Final[tuple[tuple[str, TypeSpec], ...]] = (
    ("classifier.type", str),
    ("classifier.dropout", (int, float)),
)

#: Attributes a backbone may use to advertise its embedding width.
_FEATURE_DIM_ATTRIBUTES: Final[tuple[str, ...]] = ("embed_dim", "num_features")

#: Matches ``dinov2_vits14`` and friends so a display name can be derived.
_BACKBONE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^dinov2_vit(?P<size>[a-z])(?P<patch>\d+)$"
)

_LOGGER: Final = get_logger("model")


class ModelBuildError(RuntimeError):
    """Raised when the model cannot be assembled from the configuration."""


@dataclass(frozen=True)
class ModelSpecification:
    """Backbone settings resolved from the ``model`` configuration section."""

    name: str
    pretrained: bool
    freeze_backbone: bool
    image_size: int
    feature_dim: int
    num_classes: int

    @classmethod
    def from_config(cls, config: Config) -> "ModelSpecification":
        """Read and validate the ``model`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If ``image_size``, ``feature_dim`` or ``num_classes``
                is not a positive integer.
        """
        validate_keys(config, MODEL_REQUIRED_KEYS, context="model section")

        specification = cls(
            name=config.get("model.name"),
            pretrained=config.get("model.pretrained"),
            freeze_backbone=config.get("model.freeze_backbone"),
            image_size=config.get("model.image_size"),
            feature_dim=config.get("model.feature_dim"),
            num_classes=config.get("model.num_classes"),
        )

        for key, value in (
            ("model.image_size", specification.image_size),
            ("model.feature_dim", specification.feature_dim),
            ("model.num_classes", specification.num_classes),
        ):
            if value <= 0:
                raise ValueError(f"{key} must be positive, got {value}.")

        return specification

    @property
    def display_name(self) -> str:
        """Readable backbone name, e.g. ``"DINOv2 ViT-S/14"``."""
        match = _BACKBONE_NAME_PATTERN.match(self.name)
        if match is None:
            return self.name
        return f"DINOv2 ViT-{match['size'].upper()}/{match['patch']}"


@dataclass(frozen=True)
class ClassifierSpecification:
    """Classification head settings resolved from the ``classifier`` section."""

    type: str
    dropout: float

    @classmethod
    def from_config(cls, config: Config) -> "ClassifierSpecification":
        """Read and validate the ``classifier`` section of ``config``.

        Raises:
            ConfigError: If a required key is missing or has the wrong type.
            ValueError: If the head type is unsupported or the dropout
                probability is outside ``[0, 1)``.
        """
        validate_keys(config, CLASSIFIER_REQUIRED_KEYS, context="classifier section")

        head_type = config.get("classifier.type").strip().lower()
        if head_type not in SUPPORTED_CLASSIFIER_TYPES:
            supported = ", ".join(sorted(SUPPORTED_CLASSIFIER_TYPES))
            raise ValueError(
                f"Unsupported classifier.type '{head_type}'. Supported types: {supported}."
            )

        dropout = float(config.get("classifier.dropout"))
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"classifier.dropout must be within [0, 1), got {dropout}.")

        return cls(type=head_type, dropout=dropout)

    @property
    def display_name(self) -> str:
        """Readable head description, e.g. ``"Linear"`` or ``"Dropout(p=0.20) → Linear"``."""
        head = self.type.capitalize()
        if self.dropout > 0.0:
            return f"Dropout(p={self.dropout:.2f}) → {head}"
        return head


class DinoV2Classifier(nn.Module):
    """A DINOv2 Vision Transformer with a classification head on top.

    :meth:`forward_features` exposes the raw ``[batch, feature_dim]`` embeddings
    and :meth:`forward` turns them into ``[batch, num_classes]`` logits.
    """

    def __init__(
        self,
        backbone: nn.Module,
        specification: ModelSpecification,
        classifier_specification: ClassifierSpecification,
    ) -> None:
        super().__init__()
        self.specification = specification
        self.classifier_specification = classifier_specification

        feature_dim = _resolve_feature_dim(backbone)
        _validate_feature_dim(specification.feature_dim, feature_dim, specification.name)

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.patch_size = _resolve_patch_size(backbone)
        _validate_image_size(specification.image_size, self.patch_size)

        self.classifier = build_classifier(
            classifier_specification,
            feature_dim=feature_dim,
            num_classes=specification.num_classes,
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into embeddings of shape ``[batch, feature_dim]``."""
        return self.backbone(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``[batch, num_classes]``."""
        return self.classifier(self.forward_features(inputs))

    def freeze_backbone(self) -> None:
        """Disable gradients for every backbone parameter, leaving the head trainable."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def count_parameters(self) -> int:
        """Return the total number of parameters, head included."""
        return utils.count_parameters(self).total

    def count_trainable_parameters(self) -> int:
        """Return the number of parameters that would receive gradients, head included."""
        return utils.count_parameters(self).trainable

    def model_size_mb(self) -> float:
        """Return the in-memory size of parameters and buffers, in MiB."""
        return utils.model_size_mb(self)

    @property
    def name(self) -> str:
        """Backbone entrypoint name, as configured."""
        return self.specification.name

    @property
    def image_size(self) -> int:
        """Square input resolution the model is configured for."""
        return self.specification.image_size

    @property
    def num_classes(self) -> int:
        """Width of the logit vector produced by :meth:`forward`."""
        return self.specification.num_classes

    @property
    def is_pretrained(self) -> bool:
        """Whether official pretrained weights were requested and loaded."""
        return self.specification.pretrained

    @property
    def is_backbone_frozen(self) -> bool:
        """Whether every backbone parameter currently has gradients disabled."""
        return not any(parameter.requires_grad for parameter in self.backbone.parameters())

    @property
    def weights_source(self) -> str:
        """Human readable origin of the backbone weights."""
        if not self.specification.pretrained:
            return "Random initialisation (model.pretrained is false)"
        return DINOV2_WEIGHTS_URL_TEMPLATE.format(name=self.specification.name)

    @property
    def device(self) -> torch.device:
        """Device the parameters currently live on."""
        return next(self.parameters()).device

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the architecture and its parameter counts."""
        counts = utils.count_parameters(self)
        return {
            "backbone": self.name,
            "backbone_display": self.specification.display_name,
            "hub_repository": DINOV2_HUB_REPOSITORY,
            "pretrained": self.is_pretrained,
            "weights_source": self.weights_source,
            "feature_dim": self.feature_dim,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "classifier": self.classifier_specification.display_name,
            "classifier_type": self.classifier_specification.type,
            "dropout": self.classifier_specification.dropout,
            "num_classes": self.num_classes,
            "frozen_backbone": self.is_backbone_frozen,
            "total_parameters": counts.total,
            "trainable_parameters": counts.trainable,
            "frozen_parameters": counts.frozen,
            "model_size_mb": round(self.model_size_mb(), 2),
            "device": str(self.device),
        }


def build_model(config: Config) -> DinoV2Classifier:
    """Build the configured DINOv2 classifier, ready for inference.

    The backbone is loaded from the official repository, the head is attached,
    the backbone is optionally frozen, and the model is moved to the device
    resolved from ``device.preferred`` and switched to evaluation mode. No
    optimizer, scheduler or training state is created.

    Raises:
        ConfigError: If the ``model`` or ``classifier`` section is incomplete.
        ValueError: If a configured value is unsupported.
        ModelBuildError: If the backbone cannot be loaded or is incompatible.
    """
    specification = ModelSpecification.from_config(config)
    classifier_specification = ClassifierSpecification.from_config(config)

    backbone = load_backbone(specification.name, pretrained=specification.pretrained)
    model = DinoV2Classifier(backbone, specification, classifier_specification)

    if specification.freeze_backbone:
        model.freeze_backbone()

    device = get_device(config.get("device.preferred"))
    model.to(device)
    model.eval()

    _LOGGER.info(
        "Built %s + %s head (feature_dim=%d, num_classes=%d, pretrained=%s, frozen=%s) on %s.",
        specification.name,
        classifier_specification.display_name,
        model.feature_dim,
        specification.num_classes,
        specification.pretrained,
        model.is_backbone_frozen,
        device,
    )
    return model


def load_backbone(name: str, *, pretrained: bool) -> nn.Module:
    """Load a DINOv2 backbone from the official ``torch.hub`` repository.

    Args:
        name: Repository entrypoint, e.g. ``"dinov2_vits14"``.
        pretrained: Whether to download the official pretrained weights.

    Raises:
        ModelBuildError: If the entrypoint is unknown or the download fails.
    """
    try:
        backbone = torch.hub.load(
            DINOV2_HUB_REPOSITORY,
            name,
            pretrained=pretrained,
            trust_repo=True,
        )
    except Exception as error:
        known = ", ".join(KNOWN_BACKBONES)
        raise ModelBuildError(
            f"Unable to load backbone '{name}' from {DINOV2_HUB_REPOSITORY}: {error}. "
            f"Known entrypoints: {known}."
        ) from error

    if not isinstance(backbone, nn.Module):
        raise ModelBuildError(
            f"Entrypoint '{name}' returned {type(backbone).__name__}, expected an nn.Module."
        )
    return backbone


def build_classifier(
    specification: ClassifierSpecification,
    *,
    feature_dim: int,
    num_classes: int,
) -> nn.Module:
    """Build the classification head described by ``specification``.

    A positive dropout probability prepends a ``Dropout`` layer to the linear
    projection; otherwise the head is the projection alone.

    Raises:
        ValueError: If the head type is not supported.
    """
    if specification.type not in SUPPORTED_CLASSIFIER_TYPES:
        supported = ", ".join(sorted(SUPPORTED_CLASSIFIER_TYPES))
        raise ValueError(
            f"Unsupported classifier type '{specification.type}'. Supported types: {supported}."
        )

    projection = nn.Linear(feature_dim, num_classes)
    if specification.dropout > 0.0:
        return nn.Sequential(nn.Dropout(p=specification.dropout), projection)
    return projection


def _resolve_feature_dim(backbone: nn.Module) -> int:
    """Read the embedding width advertised by ``backbone``.

    Raises:
        ModelBuildError: If no attribute exposes a usable feature dimension.
    """
    for attribute in _FEATURE_DIM_ATTRIBUTES:
        value = getattr(backbone, attribute, None)
        if isinstance(value, int) and value > 0:
            return value

    expected = ", ".join(_FEATURE_DIM_ATTRIBUTES)
    raise ModelBuildError(
        f"Backbone {type(backbone).__name__} exposes no feature dimension; "
        f"one of these attributes is required: {expected}."
    )


def _validate_feature_dim(configured: int, advertised: int, backbone_name: str) -> None:
    """Check the configured feature width against the one the backbone reports.

    Raises:
        ModelBuildError: If the two disagree.
    """
    if configured == advertised:
        return
    raise ModelBuildError(
        f"model.feature_dim ({configured}) does not match the embedding width of "
        f"backbone '{backbone_name}' ({advertised}). Set model.feature_dim to {advertised}."
    )


def _resolve_patch_size(backbone: nn.Module) -> int | None:
    """Read the patch size of ``backbone``, or ``None`` when it is not advertised."""
    value = getattr(backbone, "patch_size", None)
    return value if isinstance(value, int) and value > 0 else None


def _validate_image_size(image_size: int, patch_size: int | None) -> None:
    """Ensure the input resolution tiles exactly into patches.

    Raises:
        ModelBuildError: If ``image_size`` is not a multiple of ``patch_size``.
    """
    if patch_size is None or image_size % patch_size == 0:
        return
    raise ModelBuildError(
        f"model.image_size ({image_size}) must be a multiple of the backbone "
        f"patch size ({patch_size})."
    )


if __name__ == "__main__":
    # Imported here so that ``import src.model`` stays free of the verification
    # harness, which depends on this module.
    from src.verification import main

    raise SystemExit(main())
