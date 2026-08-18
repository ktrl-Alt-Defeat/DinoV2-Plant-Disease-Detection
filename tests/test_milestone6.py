"""Unit tests for the Milestone 6 inference API.

Every test drives the real application through ``TestClient``: real routing,
real Pydantic validation, real error handlers and the real upload pipeline. Only
the backbone is replaced, by a tiny stand-in module wrapped in the production
:class:`~src.api.inference.InferenceEngine`, so the tests exercise the genuine
preprocessing, softmax and top-k paths without a trained checkpoint on disk and
without a GPU.
"""

import io
import threading
import unittest
from typing import Final

import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from src.api.dependencies import get_settings
from src.api.errors import REQUEST_ID_HEADER
from src.api.inference import (
    CheckpointIdentity,
    InferenceEngine,
    ScoredClass,
    decode_image,
)
from src.api.main import create_app
from src.api.settings import ApiSettings
from src.config import load_config
from src.datasets.transforms import TransformSpecification, build_eval_transform
from src.logger import shutdown_logging

#: Classes the stand-in model predicts.
CLASS_NAMES: Final[tuple[str, ...]] = (
    "apple___healthy",
    "apple___scab",
    "corn___blight",
    "grape___healthy",
    "tomato___late_blight",
    "tomato___healthy",
)

TOP_K: Final[int] = 5
IMAGE_SIZE: Final[int] = 224
JPEG_TYPE: Final[str] = "image/jpeg"


class StubBackbone(nn.Module):
    """DINOv2-shaped stand-in exposing the attributes the model contract needs."""

    def __init__(self, embed_dim: int = 768, patch_size: int = 14) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one embedding per image."""
        return self.projection(inputs).flatten(2).mean(-1)


def build_stub_engine(top_k: int = TOP_K) -> InferenceEngine:
    """Build a real engine around a stand-in model, on CPU."""
    from src.model import ClassifierSpecification, DinoV2Classifier, ModelSpecification

    specification = ModelSpecification(
        name="dinov2_vitb14",
        pretrained=False,
        freeze_backbone=False,
        image_size=IMAGE_SIZE,
        feature_dim=768,
        num_classes=len(CLASS_NAMES),
    )
    model = DinoV2Classifier(
        StubBackbone(), specification, ClassifierSpecification(type="linear", dropout=0.0)
    )
    transform = build_eval_transform(
        TransformSpecification.from_config(load_config("configs/config.yaml"))
    )
    return InferenceEngine(
        model=model,
        class_names=CLASS_NAMES,
        device=torch.device("cpu"),
        transform=transform,
        top_k=top_k,
        version="1.0.0",
        checkpoint=CheckpointIdentity(
            filename="best_model.pt", sha256="a" * 64, epoch=14, best_value=0.9873
        ),
    )


def encode_image(
    *,
    size: tuple[int, int] = (256, 256),
    image_format: str = "JPEG",
    color: tuple[int, int, int] = (30, 120, 60),
) -> bytes:
    """Return an encoded synthetic image."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def image_upload(name: str = "leaf.jpg", content_type: str = JPEG_TYPE) -> tuple:
    """Return a multipart tuple describing one uploaded image."""
    return (name, encode_image(), content_type)


class ApiTestCase(unittest.TestCase):
    """Base class serving the app with a stand-in engine."""

    client: TestClient
    engine: InferenceEngine

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = build_stub_engine()
        cls.application = create_app(engine=cls.engine)
        cls.client = TestClient(cls.application, raise_server_exceptions=False)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        shutdown_logging()


class SettingsTests(unittest.TestCase):
    """1. Configuration of the service."""

    def test_settings_are_read_from_the_repository_configuration(self) -> None:
        settings = ApiSettings.from_config(load_config("configs/config.yaml"))

        self.assertEqual(settings.checkpoint_filename, "best_model.pt")
        self.assertEqual(settings.top_k, 5)
        self.assertGreater(settings.max_batch_size, 0)
        self.assertGreater(settings.max_image_bytes, 0)
        self.assertIn(JPEG_TYPE, settings.allowed_content_types)

    def test_invalid_limits_are_rejected(self) -> None:
        from src.config import Config

        for key in ("top_k", "max_batch_size", "max_image_bytes"):
            payload = load_config("configs/config.yaml").as_dict()
            payload["api"][key] = 0
            with self.subTest(key=key), self.assertRaises(ValueError):
                ApiSettings.from_config(Config(payload))

    def test_empty_content_type_list_is_rejected(self) -> None:
        from src.config import Config

        payload = load_config("configs/config.yaml").as_dict()
        payload["api"]["allowed_content_types"] = []
        with self.assertRaises(ValueError):
            ApiSettings.from_config(Config(payload))


class HealthEndpointTests(ApiTestCase):
    """2. GET /health."""

    def test_health_reports_a_ready_service(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["device"], "cpu")
        self.assertFalse(body["cuda_available"])
        self.assertIsNone(body["gpu"])
        self.assertGreaterEqual(body["uptime_seconds"], 0.0)
        self.assertEqual(body["version"], "1.0.0")

    def test_health_echoes_a_request_id(self) -> None:
        response = self.client.get("/health")

        self.assertIn(REQUEST_ID_HEADER, response.headers)
        self.assertTrue(response.headers[REQUEST_ID_HEADER])


class MetadataEndpointTests(ApiTestCase):
    """3. GET /metadata."""

    def test_metadata_describes_the_model_and_classes(self) -> None:
        response = self.client.get("/metadata")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_version"], "1.0.0")
        self.assertEqual(body["backbone"], "dinov2_vitb14")
        self.assertEqual(body["backbone_display"], "DINOv2 ViT-B/14")
        self.assertEqual(body["feature_dim"], 768)
        self.assertEqual(body["image_size"], IMAGE_SIZE)
        self.assertEqual(body["num_classes"], len(CLASS_NAMES))
        self.assertEqual(body["classes"], list(CLASS_NAMES))
        self.assertEqual(body["precision"], "fp32")
        self.assertEqual(body["top_k"], TOP_K)

    def test_metadata_reports_the_served_checkpoint(self) -> None:
        body = self.client.get("/metadata").json()

        self.assertEqual(body["checkpoint"]["filename"], "best_model.pt")
        self.assertEqual(body["checkpoint"]["epoch"], 14)
        self.assertEqual(len(body["checkpoint"]["sha256"]), 64)

    def test_class_to_idx_is_consistent_with_the_class_list(self) -> None:
        body = self.client.get("/metadata").json()

        self.assertEqual(
            body["class_to_idx"], {name: index for index, name in enumerate(CLASS_NAMES)}
        )


class PredictEndpointTests(ApiTestCase):
    """4. POST /predict."""

    def test_predict_returns_the_documented_contract(self) -> None:
        response = self.client.post("/predict", files={"file": image_upload()})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            set(body),
            {
                "request_id",
                "filename",
                "predicted_class",
                "predicted_index",
                "confidence",
                "top_k",
                "inference_time_ms",
                "model_version",
            },
        )
        self.assertEqual(body["filename"], "leaf.jpg")
        self.assertIn(body["predicted_class"], CLASS_NAMES)
        self.assertEqual(body["model_version"], "1.0.0")
        self.assertGreater(body["inference_time_ms"], 0.0)

    def test_top_k_is_ranked_and_normalised(self) -> None:
        body = self.client.post("/predict", files={"file": image_upload()}).json()
        ranked = body["top_k"]

        self.assertEqual(len(ranked), TOP_K)
        confidences = [entry["confidence"] for entry in ranked]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in confidences))
        self.assertAlmostEqual(body["confidence"], confidences[0])
        self.assertEqual(body["predicted_class"], ranked[0]["label"])
        self.assertEqual(body["predicted_index"], ranked[0]["index"])

    def test_probabilities_sum_to_one_across_all_classes(self) -> None:
        engine = build_stub_engine(top_k=len(CLASS_NAMES))
        with TestClient(create_app(engine=engine)) as client:
            body = client.post("/predict", files={"file": image_upload()}).json()

        total = sum(entry["confidence"] for entry in body["top_k"])
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_png_upload_is_accepted(self) -> None:
        payload = ("leaf.png", encode_image(image_format="PNG"), "image/png")
        response = self.client.post("/predict", files={"file": payload})

        self.assertEqual(response.status_code, 200)

    def test_request_id_is_echoed_into_body_and_header(self) -> None:
        response = self.client.post(
            "/predict", files={"file": image_upload()}, headers={REQUEST_ID_HEADER: "trace-123"}
        )

        self.assertEqual(response.headers[REQUEST_ID_HEADER], "trace-123")
        self.assertEqual(response.json()["request_id"], "trace-123")

    def test_missing_file_is_rejected(self) -> None:
        response = self.client.post("/predict")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "validation_error")

    def test_unsupported_content_type_is_rejected(self) -> None:
        payload = ("notes.txt", b"not an image", "text/plain")
        response = self.client.post("/predict", files={"file": payload})

        self.assertEqual(response.status_code, 415)
        body = response.json()
        self.assertEqual(body["error"], "unsupported_media_type")
        self.assertIn("text/plain", body["detail"])

    def test_undecodable_image_is_rejected(self) -> None:
        payload = ("broken.jpg", b"\xff\xd8\xff\xe0 not really a jpeg", JPEG_TYPE)
        response = self.client.post("/predict", files={"file": payload})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_image")

    def test_empty_upload_is_rejected(self) -> None:
        response = self.client.post("/predict", files={"file": ("empty.jpg", b"", JPEG_TYPE)})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_image")

    def test_oversized_upload_is_rejected(self) -> None:
        application = create_app(engine=build_stub_engine())
        application.dependency_overrides[get_settings] = lambda: _shrunk_settings(
            max_image_bytes=128
        )

        with TestClient(application) as client:
            response = client.post("/predict", files={"file": image_upload()})

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"], "payload_too_large")


class BatchEndpointTests(ApiTestCase):
    """5. POST /predict/batch."""

    def test_batch_scores_every_image(self) -> None:
        files = [("files", image_upload(f"leaf_{index}.jpg")) for index in range(3)]
        response = self.client.post("/predict/batch", files=files)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 3)
        self.assertEqual(len(body["predictions"]), 3)
        self.assertEqual(
            [entry["filename"] for entry in body["predictions"]],
            ["leaf_0.jpg", "leaf_1.jpg", "leaf_2.jpg"],
        )
        self.assertGreater(body["inference_time_ms"], 0.0)

    def test_every_batch_entry_matches_the_single_contract(self) -> None:
        files = [("files", image_upload(f"leaf_{index}.jpg")) for index in range(2)]
        body = self.client.post("/predict/batch", files=files).json()

        for entry in body["predictions"]:
            self.assertEqual(len(entry["top_k"]), TOP_K)
            self.assertIn(entry["predicted_class"], CLASS_NAMES)
            self.assertEqual(entry["request_id"], body["request_id"])
            self.assertEqual(entry["model_version"], "1.0.0")

    def test_batch_beyond_the_limit_is_rejected(self) -> None:
        application = create_app(engine=build_stub_engine())
        application.dependency_overrides[get_settings] = lambda: _shrunk_settings(
            max_batch_size=2
        )

        files = [("files", image_upload(f"leaf_{index}.jpg")) for index in range(3)]
        with TestClient(application) as client:
            response = client.post("/predict/batch", files=files)

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"], "payload_too_large")

    def test_one_bad_image_rejects_the_whole_batch(self) -> None:
        files = [
            ("files", image_upload("good.jpg")),
            ("files", ("bad.jpg", b"garbage", JPEG_TYPE)),
        ]
        response = self.client.post("/predict/batch", files=files)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_image")

    def test_batch_of_one_matches_the_single_endpoint_shape(self) -> None:
        response = self.client.post("/predict/batch", files=[("files", image_upload())])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)


class DocumentationTests(ApiTestCase):
    """6. GET /docs and the OpenAPI schema."""

    def test_swagger_ui_is_served(self) -> None:
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_openapi_schema_documents_every_endpoint(self) -> None:
        schema = self.client.get("/openapi.json").json()

        for path in ("/health", "/metadata", "/predict", "/predict/batch"):
            self.assertIn(path, schema["paths"])
        self.assertIn("PredictionResponse", schema["components"]["schemas"])
        self.assertIn("ErrorResponse", schema["components"]["schemas"])


class ErrorContractTests(ApiTestCase):
    """7. Error handling."""

    def test_unknown_route_uses_the_error_contract(self) -> None:
        response = self.client.get("/does-not-exist")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(set(response.json()), {"request_id", "error", "detail", "status_code"})

    def test_service_unavailable_before_the_model_loads(self) -> None:
        application = create_app(engine=build_stub_engine())
        application.state.engine = None

        client = TestClient(application, raise_server_exceptions=False)
        response = client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "model_not_ready")


class EngineTests(unittest.TestCase):
    """8. The engine itself."""

    @classmethod
    def tearDownClass(cls) -> None:
        shutdown_logging()

    def test_model_is_in_evaluation_mode_and_produces_no_gradient(self) -> None:
        engine = build_stub_engine()
        image = decode_image("leaf.jpg", encode_image())

        result = engine.predict([("leaf.jpg", image)])

        self.assertEqual(len(result.predictions), 1)
        ranked = result.predictions[0].ranked
        self.assertTrue(all(0.0 <= entry.confidence <= 1.0 for entry in ranked))

    def test_repeated_calls_reuse_one_model_instance(self) -> None:
        engine = build_stub_engine()
        image = decode_image("leaf.jpg", encode_image())

        first = engine.predict([("a.jpg", image)])
        second = engine.predict([("a.jpg", image)])

        self.assertEqual(
            [entry.label for entry in first.predictions[0].ranked],
            [entry.label for entry in second.predictions[0].ranked],
        )

    def test_concurrent_predictions_are_serialised_and_consistent(self) -> None:
        engine = build_stub_engine()
        image = decode_image("leaf.jpg", encode_image())
        expected = engine.predict([("a.jpg", image)]).predictions[0].top
        results: list[ScoredClass] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                results.append(engine.predict([("a.jpg", image)]).predictions[0].top)
            except Exception as error:  # recorded, then asserted empty on the main thread
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        for scored in results:
            self.assertEqual(scored.label, expected.label)
            self.assertAlmostEqual(scored.confidence, expected.confidence, places=5)

    def test_top_k_is_clamped_to_the_class_count(self) -> None:
        engine = build_stub_engine(top_k=99)
        image = decode_image("leaf.jpg", encode_image())

        result = engine.predict([("a.jpg", image)])

        self.assertEqual(len(result.predictions[0].ranked), len(CLASS_NAMES))

    def test_empty_batch_is_rejected(self) -> None:
        engine = build_stub_engine()

        with self.assertRaises(ValueError):
            engine.predict([])

    def test_invalid_engine_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_stub_engine(top_k=0)


class DecodeTests(unittest.TestCase):
    """9. Image decoding."""

    def test_rgba_png_is_converted_to_rgb(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(buffer, format="PNG")

        image = decode_image("leaf.png", buffer.getvalue())

        self.assertEqual(image.mode, "RGB")

    def test_garbage_raises_invalid_image(self) -> None:
        from src.api.errors import InvalidImageError

        with self.assertRaises(InvalidImageError):
            decode_image("broken.jpg", b"definitely not an image")


def _shrunk_settings(**overrides: int) -> ApiSettings:
    """Return the repository settings with tighter limits, for limit tests."""
    from dataclasses import replace

    return replace(ApiSettings.from_config(load_config("configs/config.yaml")), **overrides)


if __name__ == "__main__":
    unittest.main()
