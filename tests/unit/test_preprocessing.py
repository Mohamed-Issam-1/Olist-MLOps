import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from olist_ml.artifacts import (
    InferenceArtifacts,
)
from olist_ml.preprocessing import (
    PreprocessingError,
    prepare_raw_model_features,
    transform_features,
)


class DummyPreprocessor:
    def __init__(
        self,
        output=None,
    ):
        self.output = (
            output
            if output is not None
            else sparse.csr_matrix(
                [
                    [
                        1.0,
                        0.0,
                        2.0,
                    ]
                ]
            )
        )

        self.transform_calls = 0
        self.last_input = None

    def transform(
        self,
        values,
    ):
        self.transform_calls += 1
        self.last_input = values.copy()

        return self.output


class DummyModel:
    n_features_in_ = 3

    def predict_proba(
        self,
        values,
    ):
        return values


def make_artifacts(
    preprocessor=None,
):
    if preprocessor is None:
        preprocessor = (
            DummyPreprocessor()
        )

    return InferenceArtifacts(
        preprocessor=preprocessor,
        feature_names=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        feature_config={
            "numeric_features": [
                "number_b",
                "number_a",
            ],
            "categorical_features": [
                "category",
            ],
            "raw_feature_count": 3,
            "transformed_feature_count": 3,
        },
        model_bundle={
            "model": DummyModel(),
            "classification_threshold": 0.5,
            "model_type": "DummyModel",
            "feature_count": 3,
        },
    )


def make_engineered_data():
    return pd.DataFrame(
        {
            "category": [
                "A",
            ],
            "number_a": [
                1.0,
            ],
            "number_b": [
                2.0,
            ],
            "unused_column": [
                999,
            ],
        }
    )


def test_prepare_raw_model_features_uses_saved_order():
    artifacts = (
        make_artifacts()
    )

    result = (
        prepare_raw_model_features(
            make_engineered_data(),
            artifacts,
        )
    )

    assert list(
        result.columns
    ) == [
        "number_b",
        "number_a",
        "category",
    ]


def test_transform_features_uses_transform_once():
    preprocessor = (
        DummyPreprocessor()
    )

    artifacts = (
        make_artifacts(
            preprocessor
        )
    )

    transform_features(
        make_engineered_data(),
        artifacts,
    )

    assert (
        preprocessor.transform_calls
        == 1
    )


def test_transform_features_passes_correct_column_order():
    preprocessor = (
        DummyPreprocessor()
    )

    artifacts = (
        make_artifacts(
            preprocessor
        )
    )

    transform_features(
        make_engineered_data(),
        artifacts,
    )

    assert list(
        preprocessor.last_input.columns
    ) == [
        "number_b",
        "number_a",
        "category",
    ]


def test_transform_features_returns_expected_shape():
    artifacts = (
        make_artifacts()
    )

    result = transform_features(
        make_engineered_data(),
        artifacts,
    )

    assert result.shape == (
        1,
        3,
    )


def test_transform_features_rejects_wrong_feature_count():
    preprocessor = (
        DummyPreprocessor(
            output=sparse.csr_matrix(
                [
                    [
                        1.0,
                        2.0,
                    ]
                ]
            )
        )
    )

    artifacts = (
        make_artifacts(
            preprocessor
        )
    )

    with pytest.raises(
        PreprocessingError,
        match="feature count",
    ):
        transform_features(
            make_engineered_data(),
            artifacts,
        )


def test_transform_features_rejects_nan_output():
    preprocessor = (
        DummyPreprocessor(
            output=np.array(
                [
                    [
                        1.0,
                        np.nan,
                        2.0,
                    ]
                ]
            )
        )
    )

    artifacts = (
        make_artifacts(
            preprocessor
        )
    )

    with pytest.raises(
        PreprocessingError,
        match="NaN or infinite",
    ):
        transform_features(
            make_engineered_data(),
            artifacts,
        )


def test_transform_features_rejects_infinite_sparse_output():
    preprocessor = (
        DummyPreprocessor(
            output=sparse.csr_matrix(
                [
                    [
                        1.0,
                        np.inf,
                        2.0,
                    ]
                ]
            )
        )
    )

    artifacts = (
        make_artifacts(
            preprocessor
        )
    )

    with pytest.raises(
        PreprocessingError,
        match="NaN or infinite",
    ):
        transform_features(
            make_engineered_data(),
            artifacts,
        )
