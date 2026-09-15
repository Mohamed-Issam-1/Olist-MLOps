from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from olist_ml.artifacts import (
    InferenceArtifacts,
    load_inference_artifacts,
)
from olist_ml.config import load_config
from olist_ml.features import engineer_features
from olist_ml.preprocessing import transform_features


class InferenceError(RuntimeError):
    """Raised when model inference cannot be completed safely."""


def _resolve_positive_class(
    positive_class: Any | None = None,
) -> Any:
    """
    Return the configured positive target class.

    If an explicit value is not supplied, the value is loaded
    from config/config.json.
    """

    if positive_class is not None:
        return positive_class

    config = load_config()

    try:
        return config[
            "inference"
        ][
            "positive_class"
        ]

    except (
        KeyError,
        TypeError,
    ) as exc:
        raise InferenceError(
            "The positive inference class is not "
            "configured correctly."
        ) from exc


def _get_positive_class_index(
    model: Any,
    positive_class: Any,
) -> int:
    """
    Find the probability-column index belonging to the
    configured positive class.
    """

    if not hasattr(
        model,
        "classes_",
    ):
        raise InferenceError(
            "Loaded model does not expose classes_."
        )

    classes = np.asarray(
        model.classes_
    )

    matches = np.flatnonzero(
        classes == positive_class
    )

    if len(matches) != 1:
        raise InferenceError(
            "Configured positive class "
            f"{positive_class!r} was not found "
            "exactly once in model.classes_. "
            f"Available classes: {classes.tolist()}."
        )

    return int(
        matches[0]
    )


def predict_late_probabilities(
    transformed_features,
    artifacts: InferenceArtifacts | None = None,
    positive_class: Any | None = None,
) -> np.ndarray:
    """
    Predict the probability of the configured late-delivery class.
    """

    artifacts = (
        artifacts
        or load_inference_artifacts()
    )

    model = artifacts.model

    positive_class = (
        _resolve_positive_class(
            positive_class
        )
    )

    positive_class_index = (
        _get_positive_class_index(
            model,
            positive_class,
        )
    )

    probability_matrix = np.asarray(
        model.predict_proba(
            transformed_features
        ),
        dtype=float,
    )

    if probability_matrix.ndim != 2:
        raise InferenceError(
            "predict_proba() must return "
            "a two-dimensional array."
        )

    expected_rows = (
        transformed_features.shape[0]
    )

    if (
        probability_matrix.shape[0]
        != expected_rows
    ):
        raise InferenceError(
            "Prediction row count does not match "
            "the transformed input row count."
        )

    classes = np.asarray(
        model.classes_
    )

    if (
        probability_matrix.shape[1]
        != len(classes)
    ):
        raise InferenceError(
            "Probability-column count does not match "
            "the number of model classes."
        )

    if not np.all(
        np.isfinite(
            probability_matrix
        )
    ):
        raise InferenceError(
            "Model produced NaN or infinite probabilities."
        )

    if (
        np.any(
            probability_matrix < 0
        )
        or np.any(
            probability_matrix > 1
        )
    ):
        raise InferenceError(
            "Model produced probabilities outside "
            "the valid range [0, 1]."
        )

    return probability_matrix[
        :,
        positive_class_index,
    ]


def apply_classification_threshold(
    probabilities,
    threshold: float,
) -> np.ndarray:
    """
    Convert positive-class probabilities to binary predictions.

    The comparison exactly reproduces Notebook 06:
    probability >= threshold.
    """

    if (
        isinstance(
            threshold,
            bool,
        )
        or not isinstance(
            threshold,
            Real,
        )
        or not 0 <= threshold <= 1
    ):
        raise InferenceError(
            "Classification threshold must be "
            "a number between 0 and 1."
        )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise InferenceError(
            "Probability input must be "
            "one-dimensional."
        )

    if not np.all(
        np.isfinite(
            probabilities
        )
    ):
        raise InferenceError(
            "Probabilities contain NaN or infinite values."
        )

    if (
        np.any(
            probabilities < 0
        )
        or np.any(
            probabilities > 1
        )
    ):
        raise InferenceError(
            "Probabilities must be between 0 and 1."
        )

    return (
        probabilities
        >= float(
            threshold
        )
    ).astype(int)


def predict_orders(
    raw_orders: pd.DataFrame,
    artifacts: InferenceArtifacts | None = None,
    positive_class: Any | None = None,
) -> pd.DataFrame:
    """
    Run the complete production inference pipeline.

    Raw orders
        -> deterministic feature engineering
        -> saved fitted preprocessing
        -> saved fitted model
        -> late probability
        -> binary prediction
    """

    if not isinstance(
        raw_orders,
        pd.DataFrame,
    ):
        raise InferenceError(
            "raw_orders must be a pandas DataFrame."
        )

    if raw_orders.empty:
        raise InferenceError(
            "Cannot run inference on an empty DataFrame."
        )

    artifacts = (
        artifacts
        or load_inference_artifacts()
    )

    engineered = engineer_features(
        raw_orders
    )

    transformed = transform_features(
        engineered,
        artifacts,
    )

    probabilities = (
        predict_late_probabilities(
            transformed,
            artifacts,
            positive_class,
        )
    )

    predictions = (
        apply_classification_threshold(
            probabilities,
            artifacts.classification_threshold,
        )
    )

    result = pd.DataFrame(
        {
            "late_probability":
                probabilities,
            "predicted_is_late":
                predictions,
        },
        index=raw_orders.index,
    )

    if (
        "order_id"
        in raw_orders.columns
    ):
        result.insert(
            0,
            "order_id",
            raw_orders[
                "order_id"
            ].values,
        )

    return result.reset_index(
        drop=True
    )
