"""Unit tests for the Milestone 1 engineering foundation.

The suite is fully synthetic: it uses temporary directories and tiny in-memory
tensors, and never touches a dataset or a pretrained backbone.
"""

import compileall
import io
import json
import logging
import random
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from src import cli, paths, utils
from src.config import Config, ConfigError, load_config
from src.device import SUPPORTED_PREFERENCES, get_device, get_device_info
from src.logger import LOGGER_NAME, configure_logging, get_logger, shutdown_logging
from src.paths import PROJECT_ROOT, ProjectPaths, ensure_directory, project_root, resolve
from src.seed import MAX_SEED, set_seed

MINIMAL_CONFIG: dict[str, object] = {
    "project": {"name": "test_project", "version": "0.0.1", "seed": 7},
    "paths": {"logs": "logs", "checkpoints": "checkpoints", "results": "results"},
    "device": {"preferred": "cpu"},
    "logging": {"level": "INFO", "save_file": False},
    "reproducibility": {"deterministic": True, "benchmark": False},
}


def write_config(directory: Path, overrides: dict[str, object] | None = None) -> Path:
    """Write a synthetic configuration file into ``directory`` and return its path."""
    payload: dict[str, object] = {
        section: dict(values) for section, values in MINIMAL_CONFIG.items()
    }
    payload.update(overrides or {})

    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


class TempDirTestCase(unittest.TestCase):
    """Base class providing an isolated temporary directory per test."""

    def setUp(self) -> None:
        self._temp_dir = TemporaryDirectory()
        self.temp_path = Path(self._temp_dir.name)
        self.addCleanup(self._temp_dir.cleanup)


class ConfigLoadingTests(TempDirTestCase):
    """1. Configuration loading."""

    def test_loads_repository_configuration(self) -> None:
        config = load_config(Path("configs") / "config.yaml")

        self.assertIsInstance(config, Config)
        self.assertEqual(config.get("project.name"), "dinov2_s_plant_disease")
        self.assertEqual(config.get("project.version"), "1.0.0")
        self.assertEqual(config.get("project.seed"), 42)
        self.assertEqual(config.get("model.name"), "dinov2_vits14")

    def test_dotted_lookup_default_and_membership(self) -> None:
        config = load_config(write_config(self.temp_path))

        self.assertEqual(config.get("project.seed"), 7)
        self.assertEqual(config.get("training.epochs", 25), 25)
        self.assertIn("logging.level", config)
        self.assertNotIn("logging.absent", config)

    def test_section_and_as_dict_return_copies(self) -> None:
        config = load_config(write_config(self.temp_path))

        section = config.section("paths")
        section["logs"] = "mutated"
        snapshot = config.as_dict()
        snapshot["project"]["name"] = "mutated"

        self.assertEqual(config.get("paths.logs"), "logs")
        self.assertEqual(config.get("project.name"), "test_project")


class ConfigValidationTests(TempDirTestCase):
    """2. Configuration validation."""

    def test_missing_file_reports_path(self) -> None:
        missing = self.temp_path / "absent.yaml"

        with self.assertRaises(ConfigError) as raised:
            load_config(missing)

        self.assertIn(str(missing), str(raised.exception))

    def test_missing_required_key_is_reported(self) -> None:
        payload = {section: dict(values) for section, values in MINIMAL_CONFIG.items()}
        del payload["device"]
        config_path = self.temp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

        with self.assertRaises(ConfigError) as raised:
            load_config(config_path)

        self.assertIn("device.preferred", str(raised.exception))

    def test_wrong_value_type_is_reported(self) -> None:
        config_path = write_config(
            self.temp_path,
            overrides={"project": {"name": "x", "version": "1", "seed": "forty-two"}},
        )

        with self.assertRaises(ConfigError) as raised:
            load_config(config_path)

        self.assertIn("project.seed", str(raised.exception))
        self.assertIn("int", str(raised.exception))

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        config_path = write_config(
            self.temp_path,
            overrides={"project": {"name": "x", "version": "1", "seed": True}},
        )

        with self.assertRaises(ConfigError):
            load_config(config_path)

    def test_malformed_and_empty_documents_are_rejected(self) -> None:
        broken = self.temp_path / "broken.yaml"
        broken.write_text("project: [unclosed\n", encoding="utf-8")
        empty = self.temp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        scalar = self.temp_path / "scalar.yaml"
        scalar.write_text("just-a-string\n", encoding="utf-8")

        for candidate in (broken, empty, scalar):
            with self.subTest(candidate=candidate.name), self.assertRaises(ConfigError):
                load_config(candidate)

    def test_missing_key_message_lists_available_keys(self) -> None:
        config = load_config(write_config(self.temp_path))

        with self.assertRaises(ConfigError) as raised:
            config.get("project.unknown")

        self.assertIn("name", str(raised.exception))


class DirectoryCreationTests(TempDirTestCase):
    """3. Directory creation."""

    def test_project_paths_are_created(self) -> None:
        mapping = {
            "logs": str(self.temp_path / "logs"),
            "checkpoints": str(self.temp_path / "ckpt"),
            "results": str(self.temp_path / "out"),
        }

        created = ProjectPaths.from_mapping(mapping).create()

        for directory in created.as_dict().values():
            self.assertTrue(directory.is_dir())

    def test_creation_is_idempotent(self) -> None:
        target = self.temp_path / "nested" / "deep"

        first = ensure_directory(target)
        second = ensure_directory(target)

        self.assertEqual(first, second)
        self.assertTrue(second.is_dir())

    def test_missing_path_entries_are_reported(self) -> None:
        with self.assertRaises(KeyError) as raised:
            ProjectPaths.from_mapping({"logs": "logs"})

        message = str(raised.exception)
        self.assertIn("checkpoints", message)
        self.assertIn("results", message)


class LoggerTests(TempDirTestCase):
    """4. Logger initialisation."""

    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(shutdown_logging)

    def test_console_only_logger_has_single_handler(self) -> None:
        logger = configure_logging(level="DEBUG")

        self.assertEqual(logger.name, LOGGER_NAME)
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertEqual(len(logger.handlers), 1)
        self.assertFalse(logger.propagate)

    def test_repeated_configuration_does_not_duplicate_handlers(self) -> None:
        first = configure_logging(level="INFO", log_dir=self.temp_path)
        handler_count = len(first.handlers)
        second = configure_logging(level="INFO", log_dir=self.temp_path)

        self.assertIs(first, second)
        self.assertEqual(len(second.handlers), handler_count)

    def test_file_logging_writes_utf8_records(self) -> None:
        logger = configure_logging(level="INFO", log_dir=self.temp_path, filename="run.log")
        logger.info("Café — 温度")
        shutdown_logging()

        log_file = self.temp_path / "run.log"
        contents = log_file.read_text(encoding="utf-8")
        self.assertIn("Café — 温度", contents)
        self.assertIn("INFO", contents)

    def test_child_loggers_are_namespaced_and_shared(self) -> None:
        child = get_logger("device")

        self.assertEqual(child.name, f"{LOGGER_NAME}.device")
        self.assertIs(child, get_logger("device"))
        self.assertIs(get_logger(), logging.getLogger(LOGGER_NAME))

    def test_unknown_level_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            configure_logging(level="LOUD")


class SeedTests(unittest.TestCase):
    """5. Seed reproducibility."""

    def _draw(self) -> tuple[float, float, float]:
        return (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()),
        )

    def test_same_seed_produces_same_sequences(self) -> None:
        set_seed(123)
        first = self._draw()
        set_seed(123)
        second = self._draw()

        self.assertEqual(first, second)

    def test_different_seeds_produce_different_sequences(self) -> None:
        set_seed(1)
        first = self._draw()
        set_seed(2)
        second = self._draw()

        self.assertNotEqual(first, second)

    def test_cudnn_flags_follow_arguments(self) -> None:
        set_seed(0, deterministic=True, benchmark=False)

        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

        set_seed(0, deterministic=False, benchmark=True)

        self.assertFalse(torch.backends.cudnn.deterministic)
        self.assertTrue(torch.backends.cudnn.benchmark)

    def test_invalid_seeds_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            set_seed("42")
        with self.assertRaises(TypeError):
            set_seed(True)
        with self.assertRaises(ValueError):
            set_seed(-1)
        with self.assertRaises(ValueError):
            set_seed(MAX_SEED + 1)

    def tearDown(self) -> None:
        set_seed(MINIMAL_CONFIG["project"]["seed"])


class DeviceTests(unittest.TestCase):
    """6. Device detection."""

    def test_cpu_preference_returns_cpu(self) -> None:
        device = get_device("cpu")

        self.assertIsInstance(device, torch.device)
        self.assertEqual(device.type, "cpu")

    def test_auto_matches_cuda_availability(self) -> None:
        device = get_device("auto")

        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(device.type, expected)

    def test_cuda_preference_falls_back_when_unavailable(self) -> None:
        device = get_device("CUDA")

        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
        else:
            self.assertEqual(device.type, "cpu")

    def test_unsupported_preference_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as raised:
            get_device("tpu")

        self.assertIn("tpu", str(raised.exception))
        self.assertTrue(SUPPORTED_PREFERENCES.issubset({"auto", "cuda", "cpu"}))

    def test_device_info_reports_hardware(self) -> None:
        device = get_device("cpu")
        info = get_device_info(device)

        self.assertIs(info.device, device)
        self.assertTrue(info.name)
        self.assertIsNone(info.total_memory_bytes)
        self.assertTrue(info.cuda_version is None or isinstance(info.cuda_version, str))
        self.assertTrue(info.cudnn_version is None or isinstance(info.cudnn_version, int))


class UtilsTests(TempDirTestCase):
    """7. Utility functions."""

    def test_parameter_counting_and_model_size(self) -> None:
        module = torch.nn.Linear(4, 2, bias=True)
        module.bias.requires_grad_(False)

        counts = utils.count_parameters(module)

        self.assertEqual(counts.total, 10)
        self.assertEqual(counts.trainable, 8)
        self.assertEqual(counts.frozen, 2)
        self.assertGreater(utils.model_size_mb(module), 0.0)

    def test_json_roundtrip_creates_parent_directories(self) -> None:
        target = self.temp_path / "nested" / "payload.json"
        payload = {"accuracy": 0.91, "labels": ["healthy", "rust"], "note": "café"}

        written = utils.write_json(target, payload)

        self.assertEqual(written, target)
        self.assertEqual(utils.read_json(target), payload)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), payload)

    def test_write_csv_uses_header_and_row_order(self) -> None:
        target = self.temp_path / "metrics.csv"
        rows = [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.25}]

        utils.write_csv(target, rows)

        lines = target.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["epoch,loss", "1,0.5", "2,0.25"])

    def test_write_csv_requires_fieldnames_for_empty_rows(self) -> None:
        with self.assertRaises(ValueError):
            utils.write_csv(self.temp_path / "empty.csv", [])

        target = utils.write_csv(self.temp_path / "empty.csv", [], fieldnames=["epoch"])
        self.assertEqual(target.read_text(encoding="utf-8").splitlines(), ["epoch"])

    def test_timer_measures_elapsed_time(self) -> None:
        with utils.Timer() as timer:
            sum(range(10_000))

        frozen = timer.elapsed
        self.assertGreater(frozen, 0.0)
        self.assertEqual(frozen, timer.elapsed)

        with self.assertRaises(RuntimeError):
            _ = utils.Timer().elapsed

    def test_formatting_helpers(self) -> None:
        self.assertEqual(utils.format_bytes(0), "0.00 B")
        self.assertEqual(utils.format_bytes(1536), "1.50 KiB")
        self.assertEqual(utils.format_bytes(8 * 1024**3), "8.00 GiB")
        self.assertEqual(utils.format_duration(0.25), "250 ms")
        self.assertEqual(utils.format_duration(12.5), "12.50 s")
        self.assertEqual(utils.format_duration(3903), "1h 05m 03s")
        self.assertEqual(utils.format_duration(75), "1m 15s")

        with self.assertRaises(ValueError):
            utils.format_bytes(-1)
        with self.assertRaises(ValueError):
            utils.format_duration(-1)


class PathResolutionTests(TempDirTestCase):
    """8. Path resolution."""

    def test_project_root_hosts_the_package(self) -> None:
        root = project_root()

        self.assertEqual(root, PROJECT_ROOT)
        self.assertTrue((root / "pyproject.toml").is_file())
        self.assertTrue((root / "src" / "cli.py").is_file())

    def test_relative_paths_anchor_at_project_root(self) -> None:
        self.assertEqual(resolve("logs"), PROJECT_ROOT / "logs")
        self.assertEqual(resolve(Path("a") / "b"), PROJECT_ROOT / "a" / "b")

    def test_absolute_paths_are_preserved(self) -> None:
        absolute = self.temp_path / "elsewhere"

        self.assertEqual(resolve(absolute), absolute)

    def test_required_path_keys_match_configuration(self) -> None:
        config = load_config(Path("configs") / "config.yaml")

        self.assertEqual(set(paths.REQUIRED_PATH_KEYS), set(config.section("paths")))


class CliTests(TempDirTestCase):
    """9. CLI execution."""

    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(shutdown_logging)

    def _synthetic_config(self, **logging_overrides: object) -> Path:
        logging_section = {"level": "INFO", "save_file": True}
        logging_section.update(logging_overrides)
        return write_config(
            self.temp_path,
            overrides={
                "paths": {
                    "logs": str(self.temp_path / "logs"),
                    "checkpoints": str(self.temp_path / "checkpoints"),
                    "results": str(self.temp_path / "results"),
                },
                "logging": logging_section,
            },
        )

    def test_default_config_path_is_the_repository_configuration(self) -> None:
        arguments = cli.build_parser().parse_args([])

        self.assertEqual(arguments.config, Path(cli.DEFAULT_CONFIG_PATH))
        self.assertTrue(resolve(arguments.config).is_file())

    def test_successful_run_reports_pass(self) -> None:
        config_path = self._synthetic_config()
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            exit_code = cli.main(["--config", str(config_path)])

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("MILESTONE 1", output)
        self.assertIn("STATUS: PASS", output)
        self.assertIn("Infrastructure:\n    READY", output)
        for step in ("Directories", "Configuration", "Logger", "Seed", "Device"):
            self.assertIn(f"{step}:\n    PASS", output)

    def test_successful_run_creates_directories_and_log_file(self) -> None:
        config_path = self._synthetic_config()

        with redirect_stdout(io.StringIO()):
            cli.main(["--config", str(config_path)])
        shutdown_logging()

        for name in ("logs", "checkpoints", "results"):
            self.assertTrue((self.temp_path / name).is_dir())
        self.assertTrue((self.temp_path / "logs" / "test_project.log").is_file())

    def test_file_logging_can_be_disabled(self) -> None:
        config_path = self._synthetic_config(save_file=False)

        with redirect_stdout(io.StringIO()):
            cli.main(["--config", str(config_path)])

        self.assertFalse((self.temp_path / "logs" / "test_project.log").exists())

    def test_missing_configuration_reports_failure(self) -> None:
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            exit_code = cli.main(["--config", str(self.temp_path / "absent.yaml")])

        self.assertEqual(exit_code, 1)
        self.assertIn("STATUS: FAIL", buffer.getvalue())

    def test_bootstrap_report_describes_completed_steps(self) -> None:
        report = cli.bootstrap(self._synthetic_config())

        self.assertEqual(set(report.completed_steps), set(cli._SUMMARY_STEPS))
        self.assertEqual(report.device_info.device.type, "cpu")
        self.assertGreaterEqual(report.duration_seconds, 0.0)
        self.assertIn("STATUS: PASS", cli.render_summary(report))


class SyntaxTests(unittest.TestCase):
    """10. Syntax verification."""

    def test_every_source_file_compiles(self) -> None:
        source_root = PROJECT_ROOT / "src"

        compiled = compileall.compile_dir(
            str(source_root),
            quiet=2,
            force=True,
            legacy=False,
        )

        self.assertTrue(compiled, f"Syntax errors detected under {source_root}.")

    def test_expected_package_layout_exists(self) -> None:
        expected_modules = (
            "__init__.py",
            "cli.py",
            "config.py",
            "device.py",
            "logger.py",
            "paths.py",
            "seed.py",
            "utils.py",
        )
        expected_packages = ("models", "datasets", "training", "evaluation", "visualization")

        for module in expected_modules:
            with self.subTest(module=module):
                self.assertTrue((PROJECT_ROOT / "src" / module).is_file())

        for package in expected_packages:
            with self.subTest(package=package):
                self.assertTrue((PROJECT_ROOT / "src" / package / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
