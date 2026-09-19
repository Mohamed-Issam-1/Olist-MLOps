from __future__ import annotations

import logging
from dataclasses import dataclass

from mlflow.exceptions import (
    MlflowException,
)

from olist_ml.mlflow_config import (
    MlflowSettings,
    configure_mlflow,
)
from olist_ml.mlflow_registry import (
    register_task2_model,
)
from olist_ml.mlflow_tracking import (
    get_mlflow_client,
)

LOGGER = logging.getLogger(__name__)


class MlflowBootstrapError(RuntimeError):
    """Raised when production model bootstrap fails."""


@dataclass(frozen=True)
class ModelBootstrapResult:
    registered_model_name: str
    version: str
    alias: str
    model_uri: str
    created: bool


def _is_missing_resource_error(
    exc: MlflowException,
) -> bool:
    return (
        getattr(
            exc,
            "error_code",
            None,
        )
        == "RESOURCE_DOES_NOT_EXIST"
    )


def _get_existing_production_model(
    client,
    settings: MlflowSettings,
) -> ModelBootstrapResult | None:
    """
    Resolve the configured production alias.

    Return None only when the registered model itself
    does not exist. If the model exists but its
    production alias is missing, fail explicitly.
    """

    try:
        client.get_registered_model(settings.registered_model_name)
    except MlflowException as exc:
        if _is_missing_resource_error(exc):
            return None

        raise MlflowBootstrapError("Could not inspect the registered model.") from exc

    try:
        model_version = client.get_model_version_by_alias(
            settings.registered_model_name,
            settings.production_alias,
        )
    except MlflowException as exc:
        if _is_missing_resource_error(exc):
            raise MlflowBootstrapError(
                "Registered model exists, but the "
                "configured production alias is missing."
            ) from exc

        raise MlflowBootstrapError(
            "Could not resolve the production model alias."
        ) from exc

    model_uri = f"models:/{settings.registered_model_name}@{settings.production_alias}"

    return ModelBootstrapResult(
        registered_model_name=(settings.registered_model_name),
        version=str(model_version.version),
        alias=(settings.production_alias),
        model_uri=model_uri,
        created=False,
    )


def ensure_production_model() -> ModelBootstrapResult:
    """
    Ensure that the configured production model exists.

    The existing model is reused when its production
    alias is already available. A new registration is
    performed only for an empty registry.

    No training or refitting occurs here.
    """

    settings = configure_mlflow()

    client = get_mlflow_client(settings)

    existing = _get_existing_production_model(
        client,
        settings,
    )

    if existing is not None:
        LOGGER.info(
            "Using existing MLflow model %s@%s version %s.",
            existing.registered_model_name,
            existing.alias,
            existing.version,
        )

        return existing

    registered = register_task2_model()

    result = ModelBootstrapResult(
        registered_model_name=(registered.registered_model_name),
        version=str(registered.version),
        alias=registered.alias,
        model_uri=registered.model_uri,
        created=True,
    )

    LOGGER.info(
        "Registered MLflow model %s@%s version %s.",
        result.registered_model_name,
        result.alias,
        result.version,
    )

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )

    result = ensure_production_model()

    LOGGER.info(
        "MLflow production model ready: name=%s alias=%s version=%s created=%s uri=%s",
        result.registered_model_name,
        result.alias,
        result.version,
        result.created,
        result.model_uri,
    )


if __name__ == "__main__":
    main()
