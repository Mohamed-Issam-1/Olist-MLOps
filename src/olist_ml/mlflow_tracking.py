from __future__ import annotations

from typing import Any

from mlflow.tracking import MlflowClient

from olist_ml.mlflow_config import (
    MlflowSettings,
    configure_mlflow,
)


class MlflowTrackingError(RuntimeError):
    """Raised when MLflow tracking state is invalid."""


def _normalize_uri(
    uri: str,
) -> str:
    return uri.rstrip("/")


def get_mlflow_client(
    settings: MlflowSettings,
) -> MlflowClient:
    """
    Build an MLflow client using the configured
    tracking and registry backend.
    """

    return MlflowClient(
        tracking_uri=settings.tracking_uri,
        registry_uri=settings.registry_uri,
    )


def _validate_experiment(
    experiment: Any,
    settings: MlflowSettings,
) -> None:
    """
    Validate that the stored experiment matches
    the project configuration.
    """

    if experiment is None:
        raise MlflowTrackingError(
            "MLflow experiment could not be loaded."
        )

    if (
        experiment.name
        != settings.experiment_name
    ):
        raise MlflowTrackingError(
            "MLflow experiment name does not "
            "match project configuration."
        )

    lifecycle_stage = getattr(
        experiment,
        "lifecycle_stage",
        None,
    )

    if lifecycle_stage != "active":
        raise MlflowTrackingError(
            "MLflow experiment is not active: "
            f"{lifecycle_stage!r}."
        )

    stored_artifact_location = (
        _normalize_uri(
            experiment.artifact_location
        )
    )

    configured_artifact_location = (
        _normalize_uri(
            settings.artifact_uri
        )
    )

    if (
        stored_artifact_location
        != configured_artifact_location
    ):
        raise MlflowTrackingError(
            "MLflow experiment artifact location "
            "does not match project configuration. "
            f"Stored: {experiment.artifact_location}; "
            f"Configured: {settings.artifact_uri}"
        )


def ensure_experiment():
    """
    Configure MLflow and create or retrieve
    the project's persistent experiment.

    This function does not train or register models.
    """

    settings = (
        configure_mlflow()
    )

    client = get_mlflow_client(
        settings
    )

    experiment = (
        client.get_experiment_by_name(
            settings.experiment_name
        )
    )

    if experiment is None:
        experiment_id = (
            client.create_experiment(
                name=(
                    settings
                    .experiment_name
                ),
                artifact_location=(
                    settings
                    .artifact_uri
                ),
            )
        )

        experiment = (
            client.get_experiment(
                experiment_id
            )
        )

    _validate_experiment(
        experiment,
        settings,
    )

    return (
        settings,
        client,
        experiment,
    )
