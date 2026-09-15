from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CONFIG_RELATIVE_PATH = Path("config") / "config.json"

REQUIRED_CONFIG_SECTIONS = {
    "project",
    "artifacts",
    "inference",
    "logging",
    "service",
}

REQUIRED_ARTIFACT_KEYS = {
    "preprocessor",
    "feature_names",
    "feature_config",
    "model_bundle",
}


class ConfigurationError(RuntimeError):
    """Raised when the project configuration is missing or invalid."""


def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the project root.

    The project root is identified by the presence of:
    - config/config.json
    - compose.yaml

    The search starts from this module by default and walks upward.
    """

    start_path = (start or Path(__file__)).resolve()

    if start_path.is_file():
        start_path = start_path.parent

    candidates = [
        start_path,
        *start_path.parents,
    ]

    for candidate in candidates:
        config_file = candidate / CONFIG_RELATIVE_PATH
        compose_file = candidate / "compose.yaml"

        if config_file.is_file() and compose_file.is_file():
            return candidate

    raise ConfigurationError(
        "Could not locate the Olist-MLOps project root. "
        "Expected to find config/config.json and compose.yaml."
    )


def get_config_path() -> Path:
    """Return the absolute path to the main configuration file."""

    return find_project_root() / CONFIG_RELATIVE_PATH


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """
    Load and validate config/config.json.

    The configuration is cached so the file is not repeatedly
    read from disk during the lifetime of the application.
    """

    config_path = get_config_path()

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON in configuration file: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigurationError(
            "The root configuration object must be a JSON object."
        )

    missing_sections = REQUIRED_CONFIG_SECTIONS - set(config.keys())

    if missing_sections:
        raise ConfigurationError(
            "Missing required configuration sections: "
            + ", ".join(sorted(missing_sections))
        )

    artifacts = config.get("artifacts")

    if not isinstance(artifacts, dict):
        raise ConfigurationError(
            "The 'artifacts' configuration section must be a JSON object."
        )

    missing_artifacts = REQUIRED_ARTIFACT_KEYS - set(artifacts.keys())

    if missing_artifacts:
        raise ConfigurationError(
            "Missing required artifact configuration keys: "
            + ", ".join(sorted(missing_artifacts))
        )

    return config


def resolve_project_path(path_value: str | Path) -> Path:
    """
    Convert a project-relative path into an absolute path.

    Absolute paths are also accepted, although the project configuration
    should normally use relative paths for portability.
    """

    path = Path(path_value)

    if path.is_absolute():
        return path.resolve()

    return (find_project_root() / path).resolve()


def get_artifact_paths() -> dict[str, Path]:
    """
    Return the configured artifact paths as absolute Path objects.
    """

    config = load_config()
    artifact_config = config["artifacts"]

    return {
        name: resolve_project_path(path_value)
        for name, path_value in artifact_config.items()
    }


def validate_artifact_paths() -> dict[str, Path]:
    """
    Verify that all required inference artifacts exist.

    Returns
    -------
    dict[str, Path]
        Validated absolute artifact paths.

    Raises
    ------
    FileNotFoundError
        If one or more required artifact files are missing.
    """

    artifact_paths = get_artifact_paths()

    missing = {
        name: path
        for name, path in artifact_paths.items()
        if not path.is_file()
    }

    if missing:
        missing_details = "\n".join(
            f"- {name}: {path}"
            for name, path in missing.items()
        )

        raise FileNotFoundError(
            "Required inference artifact files are missing:\n"
            f"{missing_details}"
        )

    return artifact_paths
