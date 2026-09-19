from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd

from olist_ml.artifacts import (
    InferenceArtifacts,
    validate_artifact_compatibility,
)
from olist_ml.inference import (
    predict_orders,
)

REQUIRED_MLFLOW_ARTIFACTS = {
    "preprocessor",
    "model_bundle",
    "feature_names",
    "feature_config",
}


class MlflowModelError(RuntimeError):
    """Raised when the MLflow inference model cannot operate safely."""


def _load_json(
    path: str | Path,
) -> Any:
    path = Path(path)

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise MlflowModelError(f"Could not load MLflow JSON artifact: {path}") from exc


class OlistLateDeliveryPythonModel(mlflow.pyfunc.PythonModel):
    """
    End-to-end Olist late-delivery inference model.

    The model uses only fitted Task 2 artifacts:

    raw order data
        -> deterministic feature engineering
        -> fitted preprocessor.transform()
        -> fitted LogisticRegression.predict_proba()
        -> saved classification threshold

    No fit or refit operation is performed.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self._artifacts: InferenceArtifacts | None = None

    @property
    def artifacts(
        self,
    ) -> InferenceArtifacts:
        if self._artifacts is None:
            raise MlflowModelError(
                "MLflow model artifacts have not been loaded. "
                "load_context() must run before prediction."
            )

        return self._artifacts

    def load_context(
        self,
        context,
    ) -> None:
        """
        Load fitted inference artifacts packaged inside
        the MLflow model.
        """

        artifact_paths = getattr(
            context,
            "artifacts",
            None,
        )

        if not isinstance(
            artifact_paths,
            dict,
        ):
            raise MlflowModelError(
                "MLflow model context does not contain an artifact mapping."
            )

        missing = REQUIRED_MLFLOW_ARTIFACTS - set(artifact_paths)

        if missing:
            raise MlflowModelError(
                "MLflow model is missing required artifacts: "
                + ", ".join(sorted(missing))
            )

        try:
            preprocessor = joblib.load(artifact_paths["preprocessor"])

            model_bundle = joblib.load(artifact_paths["model_bundle"])

        except Exception as exc:
            raise MlflowModelError(
                "Could not load fitted MLflow inference artifacts."
            ) from exc

        feature_names = _load_json(artifact_paths["feature_names"])

        feature_config = _load_json(artifact_paths["feature_config"])

        artifacts = InferenceArtifacts(
            preprocessor=preprocessor,
            feature_names=feature_names,
            feature_config=feature_config,
            model_bundle=model_bundle,
        )

        try:
            self._artifacts = validate_artifact_compatibility(artifacts)
        except Exception as exc:
            raise MlflowModelError(
                "Packaged MLflow inference artifacts are incompatible."
            ) from exc

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Run the existing production inference pipeline.
        """

        return predict_orders(
            model_input,
            artifacts=self.artifacts,
        )
