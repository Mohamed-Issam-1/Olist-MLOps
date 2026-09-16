from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlflow
import pandas as pd

from olist_ml.artifacts import (
    InferenceArtifacts,
)
from olist_ml.mlflow_model import (
    OlistLateDeliveryPythonModel,
)
from olist_ml.mlflow_tracking import (
    ensure_experiment,
)


class MlflowModelLoadError(RuntimeError):
    """Raised when the production model cannot be loaded from MLflow."""


@dataclass(frozen=True)
class RegisteredInferenceModel:
    """
    Loaded production model and its resolved MLflow metadata.
    """

    pyfunc_model: Any
    artifacts: InferenceArtifacts
    registered_model_name: str
    version: str
    alias: str
    run_id: str
    model_uri: str

    @property
    def source(
        self,
    ) -> str:
        return "mlflow-registry"

    @property
    def model_type(
        self,
    ) -> str:
        return type(
            self.artifacts.model
        ).__name__

    @property
    def classification_threshold(
        self,
    ) -> float:
        return (
            self.artifacts
            .classification_threshold
        )

    def predict(
        self,
        raw_orders: pd.DataFrame,
    ) -> pd.DataFrame:
        return self.pyfunc_model.predict(
            raw_orders
        )


def _load_registered_inference_model() -> RegisteredInferenceModel:
    """
    Resolve the configured production alias and load
    the corresponding MLflow PyFunc model.

    The returned version is the actual registry version,
    not a hardcoded configuration value.
    """

    (
        settings,
        client,
        _experiment,
    ) = ensure_experiment()

    try:
        model_version = (
            client
            .get_model_version_by_alias(
                settings.registered_model_name,
                settings.production_alias,
            )
        )
    except Exception as exc:
        raise MlflowModelLoadError(
            "Could not resolve the configured "
            "MLflow production model alias."
        ) from exc

    version = str(
        model_version.version
    )

    run_id = str(
        model_version.run_id
    )

    status = str(
        getattr(
            model_version,
            "status",
            "",
        )
    ).upper()

    if status != "READY":
        raise MlflowModelLoadError(
            "Resolved MLflow model version is not READY. "
            f"Version={version}, status={status!r}."
        )

    model_uri = (
        "models:/"
        f"{settings.registered_model_name}"
        f"@{settings.production_alias}"
    )

    try:
        pyfunc_model = (
            mlflow.pyfunc.load_model(
                model_uri
            )
        )
    except Exception as exc:
        raise MlflowModelLoadError(
            "Could not load the production model "
            f"from MLflow: {model_uri}"
        ) from exc

    try:
        python_model = (
            pyfunc_model
            .unwrap_python_model()
        )
    except Exception as exc:
        raise MlflowModelLoadError(
            "Could not unwrap the registered "
            "MLflow Python model."
        ) from exc

    if not isinstance(
        python_model,
        OlistLateDeliveryPythonModel,
    ):
        raise MlflowModelLoadError(
            "Registered MLflow model has an unexpected "
            "Python model implementation: "
            f"{type(python_model).__name__}."
        )

    try:
        artifacts = (
            python_model.artifacts
        )
    except Exception as exc:
        raise MlflowModelLoadError(
            "Registered MLflow model did not load "
            "its fitted inference artifacts."
        ) from exc

    return RegisteredInferenceModel(
        pyfunc_model=pyfunc_model,
        artifacts=artifacts,
        registered_model_name=(
            settings.registered_model_name
        ),
        version=version,
        alias=settings.production_alias,
        run_id=run_id,
        model_uri=model_uri,
    )


@lru_cache(maxsize=1)
def _get_cached_registered_inference_model(
) -> RegisteredInferenceModel:
    """
    Load the registry model once per application process.
    """

    return (
        _load_registered_inference_model()
    )


def load_registered_inference_model(
    *,
    refresh: bool = False,
) -> RegisteredInferenceModel:
    """
    Return the production model resolved from MLflow Registry.

    Set refresh=True to re-resolve the alias and reload
    the model, for example after an alias promotion.
    """

    if refresh:
        (
            _get_cached_registered_inference_model
            .cache_clear()
        )

    return (
        _get_cached_registered_inference_model()
    )


def clear_registered_inference_model_cache(
) -> None:
    """
    Clear the process-local production model cache.
    """

    (
        _get_cached_registered_inference_model
        .cache_clear()
    )
