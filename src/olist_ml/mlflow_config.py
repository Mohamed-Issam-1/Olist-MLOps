from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import mlflow

from olist_ml.config import (
    find_project_root,
    load_config,
)


class MlflowConfigurationError(
    RuntimeError
):
    """Raised when MLflow configuration is invalid."""


TRACKING_URI_ENV = (
    "MLFLOW_TRACKING_URI"
)

REGISTRY_URI_ENV = (
    "MLFLOW_REGISTRY_URI"
)

ARTIFACT_URI_ENV = (
    "OLIST_MLFLOW_ARTIFACT_URI"
)


@dataclass(
    frozen=True
)
class MlflowSettings:
    backend_database: Path
    artifact_directory: Path

    experiment_name: str
    registered_model_name: str
    model_name: str
    production_alias: str

    tracking_uri_override: (
        str | None
    ) = None

    registry_uri_override: (
        str | None
    ) = None

    artifact_uri_override: (
        str | None
    ) = None

    @property
    def tracking_uri(
        self,
    ) -> str:
        """
        Return the MLflow tracking URI.

        Local development uses SQLite. A deployment
        can override it through MLFLOW_TRACKING_URI.
        """

        if (
            self.tracking_uri_override
            is not None
        ):
            return (
                self
                .tracking_uri_override
            )

        path = (
            self.backend_database
            .resolve()
            .as_posix()
        )

        return (
            f"sqlite:///{path}"
        )

    @property
    def registry_uri(
        self,
    ) -> str:
        """
        Return the MLflow model registry URI.

        By default the registry shares the tracking
        backend. Deployments may configure a separate
        registry URI when needed.
        """

        if (
            self.registry_uri_override
            is not None
        ):
            return (
                self
                .registry_uri_override
            )

        return self.tracking_uri

    @property
    def artifact_uri(
        self,
    ) -> str:
        """
        Return the experiment artifact location.

        Local development uses the project
        mlartifacts directory. Containers may use
        an artifact location exposed by MLflow.
        """

        if (
            self.artifact_uri_override
            is not None
        ):
            return (
                self
                .artifact_uri_override
            )

        return (
            self.artifact_directory
            .resolve()
            .as_uri()
        )

    @property
    def uses_local_artifact_directory(
        self,
    ) -> bool:
        return (
            self.artifact_uri_override
            is None
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


def _read_optional_uri(
    environment_variable: str,
) -> str | None:
    value = os.getenv(
        environment_variable
    )

    if value is None:
        return None

    value = value.strip()

    if not value:
        raise MlflowConfigurationError(
            "MLflow environment variable "
            f"'{environment_variable}' "
            "must not be empty."
        )

    return value


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


def load_mlflow_settings(
) -> MlflowSettings:
    """
    Load validated MLflow settings.

    Project configuration supplies portable local
    defaults. Deployment environment variables can
    override service-facing MLflow URIs.
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
            "Missing 'mlflow' section "
            "in project configuration."
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
        tracking_uri_override=(
            _read_optional_uri(
                TRACKING_URI_ENV
            )
        ),
        registry_uri_override=(
            _read_optional_uri(
                REGISTRY_URI_ENV
            )
        ),
        artifact_uri_override=(
            _read_optional_uri(
                ARTIFACT_URI_ENV
            )
        ),
    )


def configure_mlflow(
) -> MlflowSettings:
    """
    Configure MLflow tracking and registry.

    Local development creates the project artifact
    directory. Remote/container deployments leave
    artifact storage management to the MLflow stack.

    This function never trains or registers a model.
    """

    settings = (
        load_mlflow_settings()
    )

    if (
        settings
        .uses_local_artifact_directory
    ):
        settings.artifact_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    mlflow.set_tracking_uri(
        settings.tracking_uri
    )

    mlflow.set_registry_uri(
        settings.registry_uri
    )

    return settings
