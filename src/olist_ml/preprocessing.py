from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from olist_ml.artifacts import (
    InferenceArtifacts,
    load_inference_artifacts,
)
from olist_ml.features import (
    select_model_features,
)


class PreprocessingError(RuntimeError):
    """Raised when inference preprocessing produces invalid output."""


def prepare_raw_model_features(
    engineered_data: pd.DataFrame,
    artifacts: InferenceArtifacts | None = None,
) -> pd.DataFrame:
    """
    Select the 31 raw model features in the exact order used
    when the fitted preprocessor was trained.
    """

    artifacts = artifacts or load_inference_artifacts()

    feature_config = artifacts.feature_config

    numeric_features = feature_config["numeric_features"]

    categorical_features = feature_config["categorical_features"]

    return select_model_features(
        engineered_data,
        numeric_features,
        categorical_features,
    )


def transform_features(
    engineered_data: pd.DataFrame,
    artifacts: InferenceArtifacts | None = None,
):
    """
    Transform engineered order data using the saved fitted
    preprocessor.

    This function performs transform() only. It never fits or
    refits preprocessing objects during inference.
    """

    artifacts = artifacts or load_inference_artifacts()

    raw_features = prepare_raw_model_features(
        engineered_data,
        artifacts,
    )

    transformed = artifacts.preprocessor.transform(raw_features)

    expected_feature_count = len(artifacts.feature_names)

    if transformed.shape[1] != expected_feature_count:
        raise PreprocessingError(
            "Transformed feature count does not "
            "match the saved feature metadata. "
            f"Expected {expected_feature_count}, "
            f"got {transformed.shape[1]}."
        )

    if sparse.issparse(transformed):
        values = transformed.data
    else:
        values = np.asarray(transformed)

    if not np.all(np.isfinite(values)):
        raise PreprocessingError("Preprocessing produced NaN or infinite values.")

    return transformed
