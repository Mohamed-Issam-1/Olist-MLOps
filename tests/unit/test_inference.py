import numpy as np
import pandas as pd
import pytest

import olist_ml.inference as inference_module
from olist_ml.artifacts import InferenceArtifacts
from olist_ml.inference import (
    InferenceError,
    apply_classification_threshold,
    predict_late_probabilities,
    predict_orders,
)


class DummyModel:
    def __init__(
        self,
        classes=None,
        probabilities=None,
    ):
        self.classes_ = np.asarray(classes if classes is not None else [0, 1])

        self.probabilities = np.asarray(
            probabilities
            if probabilities is not None
            else [
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=float,
        )

        self.n_features_in_ = 3

    def predict_proba(
        self,
        values,
    ):
        return self.probabilities


def make_artifacts(
    model=None,
    threshold=0.4,
):
    if model is None:
        model = DummyModel()

    return InferenceArtifacts(
        preprocessor=object(),
        feature_names=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        feature_config={
            "numeric_features": [],
            "categorical_features": [],
            "raw_feature_count": 0,
            "transformed_feature_count": 3,
        },
        model_bundle={
            "model": model,
            "classification_threshold": threshold,
            "model_type": type(model).__name__,
            "feature_count": 3,
        },
    )


def test_predict_late_probabilities_selects_positive_class():
    artifacts = make_artifacts()

    transformed = np.zeros(
        (
            2,
            3,
        )
    )

    result = predict_late_probabilities(
        transformed,
        artifacts,
        positive_class=1,
    )

    np.testing.assert_allclose(
        result,
        [
            0.2,
            0.9,
        ],
    )


def test_predict_late_probabilities_handles_reversed_class_order():
    model = DummyModel(
        classes=[
            1,
            0,
        ],
        probabilities=[
            [
                0.2,
                0.8,
            ],
            [
                0.9,
                0.1,
            ],
        ],
    )

    artifacts = make_artifacts(model=model)

    transformed = np.zeros(
        (
            2,
            3,
        )
    )

    result = predict_late_probabilities(
        transformed,
        artifacts,
        positive_class=1,
    )

    np.testing.assert_allclose(
        result,
        [
            0.2,
            0.9,
        ],
    )


def test_predict_late_probabilities_rejects_missing_positive_class():
    model = DummyModel(
        classes=[
            0,
            2,
        ]
    )

    artifacts = make_artifacts(model=model)

    with pytest.raises(
        InferenceError,
        match="positive class",
    ):
        predict_late_probabilities(
            np.zeros(
                (
                    2,
                    3,
                )
            ),
            artifacts,
            positive_class=1,
        )


def test_predict_late_probabilities_rejects_invalid_values():
    model = DummyModel(
        probabilities=[
            [
                0.8,
                1.2,
            ],
            [
                0.1,
                0.9,
            ],
        ]
    )

    artifacts = make_artifacts(model=model)

    with pytest.raises(
        InferenceError,
        match="valid range",
    ):
        predict_late_probabilities(
            np.zeros(
                (
                    2,
                    3,
                )
            ),
            artifacts,
            positive_class=1,
        )


def test_threshold_comparison_is_inclusive():
    result = apply_classification_threshold(
        [
            0.2,
            0.4,
            0.6,
        ],
        threshold=0.4,
    )

    np.testing.assert_array_equal(
        result,
        [
            0,
            1,
            1,
        ],
    )


def test_threshold_rejects_invalid_value():
    with pytest.raises(
        InferenceError,
        match="threshold",
    ):
        apply_classification_threshold(
            [
                0.5,
            ],
            threshold=1.5,
        )


def test_predict_orders_returns_expected_output(
    monkeypatch,
):
    raw_orders = pd.DataFrame(
        {
            "order_id": [
                "order-1",
                "order-2",
            ],
            "dummy": [
                1,
                2,
            ],
        }
    )

    artifacts = make_artifacts(threshold=0.4)

    monkeypatch.setattr(
        inference_module,
        "engineer_features",
        lambda data: data.copy(),
    )

    monkeypatch.setattr(
        inference_module,
        "transform_features",
        lambda data, artifacts: np.zeros(
            (
                len(data),
                3,
            )
        ),
    )

    monkeypatch.setattr(
        inference_module,
        "predict_late_probabilities",
        lambda *args, **kwargs: np.asarray(
            [
                0.25,
                0.75,
            ]
        ),
    )

    result = predict_orders(
        raw_orders,
        artifacts,
        positive_class=1,
    )

    assert list(result.columns) == [
        "order_id",
        "late_probability",
        "predicted_is_late",
    ]

    assert list(result["order_id"]) == [
        "order-1",
        "order-2",
    ]

    np.testing.assert_allclose(
        result["late_probability"],
        [
            0.25,
            0.75,
        ],
    )

    np.testing.assert_array_equal(
        result["predicted_is_late"],
        [
            0,
            1,
        ],
    )


def test_predict_orders_rejects_empty_input():
    with pytest.raises(
        InferenceError,
        match="empty",
    ):
        predict_orders(
            pd.DataFrame(),
            make_artifacts(),
            positive_class=1,
        )
