"""Unit tests for the Milestone 3 DINOv2 classifier integration.

Every test is synthetic: random tensors only, no dataset, no optimizer, no
scheduler, no gradient computation and no training loop. The backbone is built
once per test class because loading the official weights dominates the runtime.

This suite supersedes ``test_milestone2.py``: the model contract changed from
"feature extractor" to "backbone plus classification head", so the checks that
covered the head-less model are re-expressed here against the integrated one.
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml
from torch import nn

from src import verification
from src.config import Config, ConfigError, load_config
from src.logger import shutdown_logging
from src.model import (
    CLASSIFIER_REQUIRED_KEYS,
    DINOV2_HUB_REPOSITORY,
    KNOWN_BACKBONES,
    MODEL_REQUIRED_KEYS,
    SUPPORTED_CLASSIFIER_TYPES,
    ClassifierSpecification,
    DinoV2Classifier,
    ModelBuildError,
    ModelSpecification,
    build_classifier,
    build_model,
)
from src.utils import read_json
from src.verification import (
    FAILED,
    MODEL_SUMMARY_FILENAME,
    MODEL_VERIFICATION_FILENAME,
    PASSED,
    SKIPPED,
    VerificationCheck,
    VerificationReport,
    forward_on,
    render_report,
    synthetic_batch,
    verify_model,
    write_artifacts,
)

REPOSITORY_CONFIG: Path = Path("configs") / "config.yaml"

#: Feature width of the official DINOv2 ViT-S/14 backbone.
EXPECTED_FEATURE_DIM: int = 384

#: Patch size shared by every official DINOv2 backbone.
EXPECTED_PATCH_SIZE: int = 14

#: Placeholder class count used for architectural verification.
EXPECTED_NUM_CLASSES: int = 10

#: Parameters of the default ``Linear(384 -> 10)`` head: weight matrix plus bias.
EXPECTED_HEAD_PARAMETERS: int = EXPECTED_FEATURE_DIM * EXPECTED_NUM_CLASSES + EXPECTED_NUM_CLASSES

CPU: torch.device = torch.device("cpu")


def override_config(**sections: dict[str, object]) -> Config:
    """Return the repository configuration with the named sections updated."""
    payload = load_config(REPOSITORY_CONFIG).as_dict()
    for name, values in sections.items():
        payload[name].update(values)
    return Config(payload)


def build_test_model(**model_overrides: object) -> DinoV2Classifier:
    """Build the model on CPU so tests behave identically on every machine."""
    return build_model(override_config(model=model_overrides, device={"preferred": "cpu"}))


class SharedModelTestCase(unittest.TestCase):
    """Base class building the pretrained model once for the whole class."""

    model: DinoV2Classifier

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_test_model()

    @classmethod
    def tearDownClass(cls) -> None:
        shutdown_logging()


class ConfigurationTests(unittest.TestCase):
    """1. Configuration loading."""

    def test_model_and_classifier_sections_are_complete(self) -> None:
        config = load_config(REPOSITORY_CONFIG)

        self.assertEqual(config.get("model.name"), "dinov2_vits14")
        self.assertTrue(config.get("model.pretrained"))
        self.assertFalse(config.get("model.freeze_backbone"))
        self.assertEqual(config.get("model.image_size"), 224)
        self.assertEqual(config.get("model.feature_dim"), EXPECTED_FEATURE_DIM)
        self.assertEqual(config.get("model.num_classes"), EXPECTED_NUM_CLASSES)
        self.assertEqual(config.get("classifier.type"), "linear")
        self.assertEqual(config.get("classifier.dropout"), 0.0)
        self.assertEqual(config.get("device.preferred"), "auto")

    def test_specifications_are_read_from_configuration(self) -> None:
        config = load_config(REPOSITORY_CONFIG)

        model_specification = ModelSpecification.from_config(config)
        classifier_specification = ClassifierSpecification.from_config(config)

        self.assertEqual(model_specification.name, "dinov2_vits14")
        self.assertEqual(model_specification.feature_dim, EXPECTED_FEATURE_DIM)
        self.assertEqual(model_specification.num_classes, EXPECTED_NUM_CLASSES)
        self.assertEqual(model_specification.display_name, "DINOv2 ViT-S/14")
        self.assertEqual(classifier_specification.type, "linear")
        self.assertEqual(classifier_specification.dropout, 0.0)
        self.assertEqual(classifier_specification.display_name, "Linear")

    def test_required_keys_cover_the_documented_contract(self) -> None:
        self.assertEqual(
            {key for key, _ in MODEL_REQUIRED_KEYS},
            {
                "model.name",
                "model.pretrained",
                "model.freeze_backbone",
                "model.image_size",
                "model.feature_dim",
                "model.num_classes",
            },
        )
        self.assertEqual(
            {key for key, _ in CLASSIFIER_REQUIRED_KEYS},
            {"classifier.type", "classifier.dropout"},
        )

    def test_missing_key_is_reported(self) -> None:
        payload = load_config(REPOSITORY_CONFIG).as_dict()
        del payload["model"]["num_classes"]

        with self.assertRaises(ConfigError) as raised:
            ModelSpecification.from_config(Config(payload))

        self.assertIn("model.num_classes", str(raised.exception))

    def test_wrong_key_type_is_reported(self) -> None:
        with self.assertRaises(ConfigError) as raised:
            ModelSpecification.from_config(override_config(model={"num_classes": "ten"}))

        self.assertIn("model.num_classes", str(raised.exception))

    def test_integer_dropout_is_accepted(self) -> None:
        specification = ClassifierSpecification.from_config(
            override_config(classifier={"dropout": 0})
        )

        self.assertEqual(specification.dropout, 0.0)

    def test_non_positive_values_are_rejected(self) -> None:
        for key in ("image_size", "feature_dim", "num_classes"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ModelSpecification.from_config(override_config(model={key: 0}))

    def test_unsupported_classifier_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as raised:
            ClassifierSpecification.from_config(override_config(classifier={"type": "mlp"}))

        self.assertIn("linear", str(raised.exception))
        self.assertEqual(SUPPORTED_CLASSIFIER_TYPES, frozenset({"linear"}))

    def test_dropout_outside_the_unit_interval_is_rejected(self) -> None:
        for dropout in (-0.1, 1.0, 1.5):
            config = override_config(classifier={"dropout": dropout})
            with self.subTest(dropout=dropout), self.assertRaises(ValueError):
                ClassifierSpecification.from_config(config)


class BackboneTests(SharedModelTestCase):
    """2. Backbone initialization and 3. pretrained weights."""

    def test_backbone_is_built_from_configuration(self) -> None:
        self.assertIsInstance(self.model, DinoV2Classifier)
        self.assertEqual(self.model.name, "dinov2_vits14")
        self.assertIn(self.model.name, KNOWN_BACKBONES)
        self.assertEqual(self.model.patch_size, EXPECTED_PATCH_SIZE)
        self.assertEqual(self.model.device.type, "cpu")
        self.assertFalse(self.model.training)

    def test_unknown_backbone_is_reported(self) -> None:
        with self.assertRaises(ModelBuildError) as raised:
            build_test_model(name="dinov2_not_a_backbone")

        self.assertIn("dinov2_vitb14", str(raised.exception))

    def test_image_size_must_tile_into_patches(self) -> None:
        with self.assertRaises(ModelBuildError) as raised:
            build_test_model(image_size=225)

        self.assertIn("patch size", str(raised.exception))

    def test_pretrained_weights_differ_from_random_initialisation(self) -> None:
        pretrained = build_test_model(pretrained=True).backbone.state_dict()
        randomly_initialised = build_test_model(pretrained=False).backbone.state_dict()

        self.assertEqual(set(pretrained), set(randomly_initialised))
        differing = [
            key
            for key, tensor in pretrained.items()
            if not torch.equal(tensor, randomly_initialised[key])
        ]
        self.assertTrue(differing, "Pretrained weights are identical to a random init.")

    def test_weight_source_points_at_the_official_repository(self) -> None:
        self.assertTrue(self.model.is_pretrained)
        self.assertIn("dinov2_vits14_pretrain.pth", self.model.weights_source)
        self.assertEqual(self.model.describe()["hub_repository"], DINOV2_HUB_REPOSITORY)

    def test_random_initialisation_is_reported_as_such(self) -> None:
        model = build_test_model(pretrained=False)

        self.assertFalse(model.is_pretrained)
        self.assertIn("Random initialisation", model.weights_source)


class FeatureDimensionTests(SharedModelTestCase):
    """4. Feature dimension validation."""

    def test_feature_dim_comes_from_the_backbone(self) -> None:
        self.assertEqual(self.model.feature_dim, EXPECTED_FEATURE_DIM)
        self.assertEqual(self.model.backbone.embed_dim, self.model.feature_dim)

    def test_mismatched_feature_dim_is_rejected(self) -> None:
        with self.assertRaises(ModelBuildError) as raised:
            build_test_model(feature_dim=768)

        message = str(raised.exception)
        self.assertIn("768", message)
        self.assertIn(str(EXPECTED_FEATURE_DIM), message)


class ClassifierConstructionTests(SharedModelTestCase):
    """5. Classifier construction and 6. dropout configuration."""

    def test_default_head_is_a_bare_linear_layer(self) -> None:
        head = self.model.classifier

        self.assertIsInstance(head, nn.Linear)
        self.assertEqual(head.in_features, EXPECTED_FEATURE_DIM)
        self.assertEqual(head.out_features, EXPECTED_NUM_CLASSES)

    def test_positive_dropout_prepends_a_dropout_layer(self) -> None:
        specification = ClassifierSpecification(type="linear", dropout=0.25)

        head = build_classifier(
            specification, feature_dim=EXPECTED_FEATURE_DIM, num_classes=EXPECTED_NUM_CLASSES
        )

        self.assertIsInstance(head, nn.Sequential)
        self.assertIsInstance(head[0], nn.Dropout)
        self.assertIsInstance(head[1], nn.Linear)
        self.assertEqual(head[0].p, 0.25)
        self.assertEqual(specification.display_name, "Dropout(p=0.25) → Linear")

    def test_zero_dropout_produces_no_dropout_layer(self) -> None:
        head = build_classifier(
            ClassifierSpecification(type="linear", dropout=0.0),
            feature_dim=EXPECTED_FEATURE_DIM,
            num_classes=EXPECTED_NUM_CLASSES,
        )

        self.assertIsInstance(head, nn.Linear)

    def test_head_width_follows_the_configured_class_count(self) -> None:
        model = build_test_model(num_classes=7)

        self.assertEqual(model.num_classes, 7)
        self.assertEqual(model.classifier.out_features, 7)

    def test_unsupported_head_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_classifier(
                ClassifierSpecification(type="transformer", dropout=0.0),
                feature_dim=EXPECTED_FEATURE_DIM,
                num_classes=EXPECTED_NUM_CLASSES,
            )


class ForwardPassTests(SharedModelTestCase):
    """7. Forward feature shape, 8. output shape, 9. batch dimension preservation."""

    def test_forward_features_returns_embeddings(self) -> None:
        outputs = forward_on(self.model, CPU)

        self.assertEqual(tuple(outputs.features.shape), (2, EXPECTED_FEATURE_DIM))
        self.assertEqual(outputs.features.dtype, torch.float32)

    def test_forward_returns_logits(self) -> None:
        outputs = forward_on(self.model, CPU)

        self.assertEqual(tuple(outputs.logits.shape), (2, EXPECTED_NUM_CLASSES))
        self.assertEqual(outputs.logits.dtype, torch.float32)

    def test_forward_is_the_head_applied_to_the_features(self) -> None:
        inputs = synthetic_batch(2, self.model.image_size, CPU)

        with torch.inference_mode():
            features = self.model.forward_features(inputs)
            expected = self.model.classifier(features)
            logits = self.model(inputs)

        self.assertTrue(torch.equal(logits, expected))

    def test_batch_dimension_is_preserved(self) -> None:
        for batch_size in (1, 2, 3):
            with self.subTest(batch_size=batch_size):
                outputs = forward_on(self.model, CPU, batch_size)

                self.assertEqual(outputs.features.shape[0], batch_size)
                self.assertEqual(outputs.logits.shape[0], batch_size)
                self.assertEqual(outputs.logits.shape[1], EXPECTED_NUM_CLASSES)

    def test_inference_produces_no_gradients(self) -> None:
        outputs = forward_on(self.model, CPU)

        self.assertFalse(outputs.logits.requires_grad)
        self.assertIsNone(outputs.logits.grad_fn)


class FreezeTests(unittest.TestCase):
    """10. Backbone freezing."""

    @classmethod
    def tearDownClass(cls) -> None:
        shutdown_logging()

    def test_freezing_leaves_only_the_classifier_trainable(self) -> None:
        model = build_test_model(freeze_backbone=True)

        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        self.assertTrue(model.is_backbone_frozen)
        self.assertEqual(model.count_trainable_parameters(), EXPECTED_HEAD_PARAMETERS)

    def test_unfrozen_backbone_keeps_every_parameter_trainable(self) -> None:
        model = build_test_model(freeze_backbone=False)

        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.is_backbone_frozen)
        self.assertEqual(model.count_trainable_parameters(), model.count_parameters())

    def test_frozen_model_still_produces_logits(self) -> None:
        model = build_test_model(freeze_backbone=True)

        outputs = forward_on(model, CPU)

        self.assertEqual(tuple(outputs.logits.shape), (2, EXPECTED_NUM_CLASSES))


class DeviceExecutionTests(SharedModelTestCase):
    """11. CPU inference and 12. CUDA inference (if available)."""

    def test_cpu_inference_succeeds(self) -> None:
        outputs = forward_on(self.model, CPU)

        self.assertEqual(outputs.logits.device.type, "cpu")
        self.assertEqual(tuple(outputs.logits.shape), (2, EXPECTED_NUM_CLASSES))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this machine.")
    def test_cuda_inference_succeeds(self) -> None:
        self.addCleanup(self.model.to, CPU)

        outputs = forward_on(self.model, torch.device("cuda"))

        self.assertEqual(outputs.logits.device.type, "cuda")
        self.assertEqual(tuple(outputs.logits.shape), (2, EXPECTED_NUM_CLASSES))

    def test_cuda_check_reflects_gpu_availability(self) -> None:
        report = verify_model(self.model)
        cuda_check = next(check for check in report.checks if check.name == "CUDA Forward")

        expected = PASSED if torch.cuda.is_available() else SKIPPED
        self.assertEqual(cuda_check.status, expected)


class FiniteValueTests(SharedModelTestCase):
    """13. NaN protection and 14. Inf protection."""

    def test_logits_and_features_are_finite(self) -> None:
        outputs = forward_on(self.model, CPU)

        for name, tensor in (("features", outputs.features), ("logits", outputs.logits)):
            with self.subTest(tensor=name):
                self.assertFalse(torch.isnan(tensor).any())
                self.assertFalse(torch.isinf(tensor).any())
                self.assertTrue(torch.isfinite(tensor).all())

    def test_verification_reports_nan_and_inf_checks_as_passing(self) -> None:
        report = verify_model(self.model)
        statuses = {check.name: check.status for check in report.checks}

        self.assertEqual(statuses["NaN Check"], PASSED)
        self.assertEqual(statuses["Inf Check"], PASSED)

    def test_non_finite_values_are_detected(self) -> None:
        polluted = torch.tensor([[0.0, float("nan"), float("inf")]])

        self.assertTrue(torch.isnan(polluted).any())
        self.assertTrue(torch.isinf(polluted).any())


class ParameterCountingTests(SharedModelTestCase):
    """15. Parameter counting and 16. trainable parameter counting."""

    def test_counts_include_the_classifier(self) -> None:
        head_parameters = sum(p.numel() for p in self.model.classifier.parameters())
        backbone_parameters = sum(p.numel() for p in self.model.backbone.parameters())

        self.assertEqual(self.model.count_parameters(), backbone_parameters + head_parameters)
        self.assertEqual(head_parameters, EXPECTED_HEAD_PARAMETERS)

    def test_counts_match_the_underlying_parameters(self) -> None:
        expected_total = sum(p.numel() for p in self.model.parameters())
        expected_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.assertEqual(self.model.count_parameters(), expected_total)
        self.assertEqual(self.model.count_trainable_parameters(), expected_trainable)

    def test_vit_small_has_the_expected_scale(self) -> None:
        total = self.model.count_parameters()

        self.assertGreater(total, 20_000_000)
        self.assertLess(total, 25_000_000)

    def test_model_size_is_consistent_with_parameter_count(self) -> None:
        size_mb = self.model.model_size_mb()

        self.assertGreater(size_mb, 0.0)
        self.assertAlmostEqual(size_mb, self.model.count_parameters() * 4 / 1024**2, delta=1.0)

    def test_describe_reports_the_full_architecture(self) -> None:
        summary = self.model.describe()

        self.assertEqual(summary["backbone_display"], "DINOv2 ViT-S/14")
        self.assertEqual(summary["feature_dim"], EXPECTED_FEATURE_DIM)
        self.assertEqual(summary["classifier"], "Linear")
        self.assertEqual(summary["num_classes"], EXPECTED_NUM_CLASSES)
        self.assertFalse(summary["frozen_backbone"])
        self.assertEqual(
            summary["total_parameters"],
            summary["trainable_parameters"] + summary["frozen_parameters"],
        )
        self.assertEqual(summary["device"], "cpu")


class ArtifactTests(SharedModelTestCase):
    """17. Model summary generation and 18. verification artifact generation."""

    def setUp(self) -> None:
        self._temp_dir = TemporaryDirectory()
        self.results_dir = Path(self._temp_dir.name)
        self.addCleanup(self._temp_dir.cleanup)

    def test_report_status_is_pass(self) -> None:
        report = verify_model(self.model)

        self.assertTrue(report.passed)
        self.assertEqual(report.status, PASSED)
        self.assertEqual(len(report.checks), 6)
        self.assertEqual(
            [check.name for check in report.checks],
            [
                "Feature Extraction",
                "Classifier Output",
                "CPU Forward",
                "CUDA Forward",
                "NaN Check",
                "Inf Check",
            ],
        )

    def test_console_report_matches_the_expected_layout(self) -> None:
        rendered = render_report(verify_model(self.model))

        self.assertIn("MILESTONE 3 — MODEL INTEGRATION VERIFICATION", rendered)
        self.assertIn(f"MODEL STATUS : {PASSED}", rendered)
        for label in (
            "Backbone",
            "Pretrained",
            "Feature Dimension",
            "Classifier",
            "Classes",
            "Frozen Backbone",
            "Total Parameters",
            "Trainable Parameters",
            "Approximate Model Size",
        ):
            self.assertIn(f"{label}:", rendered)
        for index in range(1, 7):
            self.assertIn(f"Verification {index}:", rendered)

    def test_summary_artifact_reports_the_classifier(self) -> None:
        report = verify_model(self.model)

        artifacts = write_artifacts(self.model, report, self.results_dir)
        contents = artifacts["model_summary"].read_text(encoding="utf-8")

        self.assertEqual(artifacts["model_summary"].name, MODEL_SUMMARY_FILENAME)
        self.assertIn("DINOv2 ViT-S/14", contents)
        self.assertIn("Classifier:", contents)
        self.assertIn(f"in_features={EXPECTED_FEATURE_DIM}", contents)
        self.assertIn(f"out_features={EXPECTED_NUM_CLASSES}", contents)
        self.assertIn("Module tree", contents)

    def test_verification_artifact_is_written_with_the_expected_schema(self) -> None:
        report = verify_model(self.model)

        artifacts = write_artifacts(self.model, report, self.results_dir)
        payload = read_json(artifacts["model_verification"])

        self.assertEqual(artifacts["model_verification"].name, MODEL_VERIFICATION_FILENAME)
        self.assertEqual(payload["status"], PASSED)
        self.assertEqual(payload["model"]["backbone"], "dinov2_vits14")
        self.assertEqual(payload["model"]["feature_dim"], EXPECTED_FEATURE_DIM)
        self.assertEqual(payload["model"]["num_classes"], EXPECTED_NUM_CLASSES)
        self.assertEqual(payload["model"]["classifier_type"], "linear")
        self.assertEqual(payload["model"]["dropout"], 0.0)
        self.assertEqual(len(payload["checks"]), 6)
        for check in payload["checks"]:
            self.assertEqual(set(check), {"name", "status", "details"})
            self.assertIn(check["status"], {PASSED, FAILED, SKIPPED})

    def test_failed_checks_drive_the_overall_status(self) -> None:
        report = VerificationReport(
            model_summary=self.model.describe(),
            checks=(VerificationCheck("Classifier Output", FAILED, "shape mismatch"),),
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.status, FAILED)
        self.assertIn(f"MODEL STATUS : {FAILED}", render_report(report))


class VerificationCliTests(unittest.TestCase):
    """End-to-end run of ``python -m src.model`` against a temporary results directory."""

    def setUp(self) -> None:
        self._temp_dir = TemporaryDirectory()
        self.temp_path = Path(self._temp_dir.name)
        self.addCleanup(self._temp_dir.cleanup)
        self.addCleanup(shutdown_logging)

    def _temporary_config(self) -> Path:
        payload = load_config(REPOSITORY_CONFIG).as_dict()
        payload["paths"] = {
            name: str(self.temp_path / name) for name in ("logs", "checkpoints", "results")
        }
        payload["device"]["preferred"] = "cpu"

        config_path = self.temp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return config_path

    def test_cli_run_passes_and_writes_both_artifacts(self) -> None:
        config_path = self._temporary_config()
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            exit_code = verification.main(["--config", str(config_path)])

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("MILESTONE 3 — MODEL INTEGRATION VERIFICATION", output)
        self.assertIn(f"MODEL STATUS : {PASSED}", output)

        results_dir = self.temp_path / "results"
        self.assertTrue((results_dir / MODEL_SUMMARY_FILENAME).is_file())
        self.assertTrue((results_dir / MODEL_VERIFICATION_FILENAME).is_file())

    def test_cli_reports_failure_for_an_unusable_configuration(self) -> None:
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            exit_code = verification.main(["--config", str(self.temp_path / "absent.yaml")])

        self.assertEqual(exit_code, 1)
        self.assertIn(f"MODEL STATUS : {FAILED}", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
