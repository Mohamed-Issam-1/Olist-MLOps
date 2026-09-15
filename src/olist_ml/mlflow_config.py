from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow

from olist_ml.config import (
    find_project_root,
    load_config,
)


class MlflowConfigurationError(RuntimeError):
    """Raised when MLflow configuration is invalid."""


@dataclass(frozen=True)
class MlflowSettings:
    backend_database: Path
    artifact_directory: Path
    experiment_name: str
    registered_model_name: str
    model_name: str
    production_alias: str

    @property
    def tracking_uri(self) -> str:
        """
        Return a portable absolute SQLite tracking URI.
        """

        path = (
            self.backend_database
            .resolve()
            .as_posix()
        )

        return (
            f"sqlite:///{path}"
        )

    @property
    def artifact_uri(self) -> str:
        """
        Return the local artifact directory as a file URI.
        """

        return (
            self.artifact_directory
            .resolve()
            .as_uri()
        )


def _require_non_empty_string(
    config: dict,
    key: str,
) -> str:
    value = config.get(
        key
    )

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise MlflowConfigurationError(
            f"MLflow configuration '{key}' "
            "must be a non-empty string."
        )

    return value.strip()


def _resolve_local_path(
    project_root: Path,
    value: str,
    *,
    key: str,
) -> Path:
    configured_path = Path(
        value
    )

    if configured_path.is_absolute():
        raise MlflowConfigurationError(
            f"MLflow configuration '{key}' "
            "must use a project-relative path."
        )

    return (
        project_root
        / configured_path
    ).resolve()


def load_mlflow_settings() -> MlflowSettings:
    """
    Load validated MLflow settings from project configuration.
    """

    config = load_config()

    mlflow_config = config.get(
        "mlflow"
    )

    if not isinstance(
        mlflow_config,
        dict,
    ):
        raise MlflowConfigurationError(
            "Missing 'mlflow' section in project configuration."
        )

    project_root = (
        find_project_root()
    )

    backend_database_value = (
        _require_non_empty_string(
            mlflow_config,
            "backend_database",
        )
    )

    artifact_directory_value = (
        _require_non_empty_string(
            mlflow_config,
            "artifact_directory",
        )
    )

    return MlflowSettings(
        backend_database=(
            _resolve_local_path(
                project_root,
                backend_database_value,
                key="backend_database",
            )
        ),
        artifact_directory=(
            _resolve_local_path(
                project_root,
                artifact_directory_value,
                key="artifact_directory",
            )
        ),
        experiment_name=(
            _require_non_empty_string(
                mlflow_config,
                "experiment_name",
            )
        ),
        registered_model_name=(
            _require_non_empty_string(
                mlflow_config,
                "registered_model_name",
            )
        ),
        model_name=(
            _require_non_empty_string(
                mlflow_config,
                "model_name",
            )
        ),
        production_alias=(
            _require_non_empty_string(
                mlflow_config,
                "production_alias",
            )
        ),
    )


def configure_mlflow() -> MlflowSettings:
    """
    Configure MLflow tracking and registry for this project.

    This function does not train or register a model.
    """

    settings = (
        load_mlflow_settings()
    )

    settings.artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        settings.tracking_uri
    )

    mlflow.set_registry_uri(
        settings.tracking_uri
    )

    return settings
