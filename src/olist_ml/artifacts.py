from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any

import joblib

from olist_ml.config import validate_artifact_paths


REQUIRED_FEATURE_CONFIG_KEYS = {
    "numeric_features",
    "categorical_features",
    "raw_feature_count",
    "transformed_feature_count",
}

REQUIRED_MODEL_BUNDLE_KEYS = {
    "model",
    "classification_threshold",
    "model_type",
    "feature_count",
}


class ArtifactValidationError(RuntimeError):
    """Raised when inference artifacts are missing or incompatible."""


@dataclass(frozen=True)
class InferenceArtifacts:
    """
    Container for all fitted objects and metadata required
    by the inference pipeline.
    """

    preprocessor: Any
    feature_names: list[str]
    feature_config: dict[str, Any]
    model_bundle: dict[str, Any]

    @property
    def model(self) -> Any:
        """Return the fitted prediction model."""

        return self.model_bundle["model"]

    @property
    def classification_threshold(self) -> float:
        """Return the saved classification threshold."""

        return float(
            self.model_bundle[
                "classification_threshold"
            ]
        )


def _load_json(path: Path) -> Any:
    """Load a JSON artifact from disk."""

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(
            f"Invalid JSON artifact: {path}"
        ) from exc


def validate_artifact_compatibility(
    artifacts: InferenceArtifacts,
) -> InferenceArtifacts:
    """
    Validate that the saved preprocessing and model artifacts
    describe the same feature pipeline.
    """

    feature_config = artifacts.feature_config
    model_bundle = artifacts.model_bundle
    feature_names = artifacts.feature_names
    preprocessor = artifacts.preprocessor
    model = artifacts.model

    # -------------------------
    # Basic artifact structure
    # -------------------------

    if not isinstance(
        feature_config,
        dict,
    ):
        raise ArtifactValidationError(
            "feature_config must be a dictionary."
        )

    if not isinstance(
        model_bundle,
        dict,
    ):
        raise ArtifactValidationError(
            "model_bundle must be a dictionary."
        )

    if not (
        isinstance(feature_names, list)
        and all(
            isinstance(name, str)
            for name in feature_names
        )
    ):
        raise ArtifactValidationError(
            "feature_names must be a list of strings."
        )

    missing_feature_config_keys = (
        REQUIRED_FEATURE_CONFIG_KEYS
        - set(feature_config.keys())
    )

    if missing_feature_config_keys:
        raise ArtifactValidationError(
            "Missing feature configuration keys: "
            + ", ".join(
                sorted(
                    missing_feature_config_keys
                )
            )
        )

    missing_model_bundle_keys = (
        REQUIRED_MODEL_BUNDLE_KEYS
        - set(model_bundle.keys())
    )

    if missing_model_bundle_keys:
        raise ArtifactValidationError(
            "Missing model bundle keys: "
            + ", ".join(
                sorted(
                    missing_model_bundle_keys
                )
            )
        )

    # -------------------------
    # Raw feature compatibility
    # -------------------------

    numeric_features = feature_config[
        "numeric_features"
    ]

    categorical_features = feature_config[
        "categorical_features"
    ]

    if not isinstance(
        numeric_features,
        list,
    ) or not isinstance(
        categorical_features,
        list,
    ):
        raise ArtifactValidationError(
            "Numeric and categorical feature "
            "definitions must be lists."
        )

    expected_raw_features = [
        *numeric_features,
        *categorical_features,
    ]

    configured_raw_count = feature_config[
        "raw_feature_count"
    ]

    if (
        configured_raw_count
        != len(expected_raw_features)
    ):
        raise ArtifactValidationError(
            "Raw feature count does not match "
            "the configured feature lists."
        )

    if not hasattr(
        preprocessor,
        "feature_names_in_",
    ):
        raise ArtifactValidationError(
            "Preprocessor does not contain "
            "feature_names_in_."
        )

    preprocessor_input_features = [
        str(name)
        for name
        in preprocessor.feature_names_in_
    ]

    if (
        preprocessor_input_features
        != expected_raw_features
    ):
        raise ArtifactValidationError(
            "Preprocessor input feature order does "
            "not match feature_config."
        )

    # -------------------------
    # Transformed feature compatibility
    # -------------------------

    if not hasattr(
        preprocessor,
        "get_feature_names_out",
    ):
        raise ArtifactValidationError(
            "Preprocessor cannot provide "
            "transformed feature names."
        )

    preprocessor_output_features = [
        str(name)
        for name
        in preprocessor.get_feature_names_out()
    ]

    if (
        preprocessor_output_features
        != feature_names
    ):
        raise ArtifactValidationError(
            "Saved feature_names do not match "
            "the fitted preprocessor output."
        )

    if not hasattr(
        model,
        "n_features_in_",
    ):
        raise ArtifactValidationError(
            "Model does not contain n_features_in_."
        )

    transformed_counts = {
        int(
            feature_config[
                "transformed_feature_count"
            ]
        ),
        len(feature_names),
        len(preprocessor_output_features),
        int(
            model_bundle[
                "feature_count"
            ]
        ),
        int(
            model.n_features_in_
        ),
    }

    if len(transformed_counts) != 1:
        raise ArtifactValidationError(
            "Transformed feature counts are "
            "inconsistent across artifacts."
        )

    # -------------------------
    # Model metadata
    # -------------------------

    actual_model_type = (
        type(model).__name__
    )

    configured_model_type = (
        model_bundle[
            "model_type"
        ]
    )

    if (
        actual_model_type
        != configured_model_type
    ):
        raise ArtifactValidationError(
            "Model type metadata does not match "
            "the loaded model object."
        )

    if not callable(
        getattr(
            model,
            "predict_proba",
            None,
        )
    ):
        raise ArtifactValidationError(
            "Loaded model does not support "
            "predict_proba()."
        )

    threshold = model_bundle[
        "classification_threshold"
    ]

    if (
        isinstance(threshold, bool)
        or not isinstance(
            threshold,
            Real,
        )
        or not 0 <= threshold <= 1
    ):
        raise ArtifactValidationError(
            "Classification threshold must be "
            "a number between 0 and 1."
        )

    return artifacts


@lru_cache(maxsize=1)
def load_inference_artifacts() -> InferenceArtifacts:
    """
    Load and validate all artifacts required for inference.

    Artifacts are cached so fitted preprocessing and model objects
    are loaded only once during the application process.
    """

    paths = validate_artifact_paths()

    preprocessor = joblib.load(
        paths[
            "preprocessor"
        ]
    )

    model_bundle = joblib.load(
        paths[
            "model_bundle"
        ]
    )

    feature_names = _load_json(
        paths[
            "feature_names"
        ]
    )

    feature_config = _load_json(
        paths[
            "feature_config"
        ]
    )

    artifacts = InferenceArtifacts(
        preprocessor=preprocessor,
        feature_names=feature_names,
        feature_config=feature_config,
        model_bundle=model_bundle,
    )

    return validate_artifact_compatibility(
        artifacts
    )
