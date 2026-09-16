import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

import olist_ml.mlflow_model as mlflow_model
from olist_ml.mlflow_model import (
    MlflowModelError,
    OlistLateDeliveryPythonModel,
)


class DummyPreprocessor:
    def __init__(
        self,
    ):
        self.feature_names_in_ = np.asarray(
            [
                "number_a",
                "category_a",
            ]
        )

    def get_feature_names_out(
        self,
    ):
        return np.asarray(
            [
                "numeric__number_a",
                "categorical__category_a_A",
            ]
        )

    def transform(
        self,
        values,
    ):
        return np.zeros(
            (
                len(values),
                2,
            ),
            dtype=float,
        )


class DummyModel:
    def __init__(
        self,
    ):
        self.n_features_in_ = 2

        self.classes_ = np.asarray(
            [
                0,
                1,
            ]
        )

    def predict_proba(
        self,
        values,
    ):
        count = values.shape[0]

        return np.tile(
            [
                [
                    0.8,
                    0.2,
                ]
            ],
            (
                count,
                1,
            ),
        )


def make_context(
    tmp_path,
):
    preprocessor_path = (
        tmp_path
        / "preprocessor.joblib"
    )

    model_bundle_path = (
        tmp_path
        / "model_bundle.joblib"
    )

    feature_names_path = (
        tmp_path
        / "feature_names.json"
    )

    feature_config_path = (
        tmp_path
        / "feature_config.json"
    )

    joblib.dump(
        DummyPreprocessor(),
        preprocessor_path,
    )

    joblib.dump(
        {
            "model":
                DummyModel(),
            "classification_threshold":
                0.4,
            "model_type":
                "DummyModel",
            "feature_count":
                2,
        },
        model_bundle_path,
    )

    feature_names_path.write_text(
        json.dumps(
            [
                "numeric__number_a",
                "categorical__category_a_A",
            ]
        ),
        encoding="utf-8",
    )

    feature_config_path.write_text(
        json.dumps(
            {
                "numeric_features": [
                    "number_a",
                ],
                "categorical_features": [
                    "category_a",
                ],
                "raw_feature_count": 2,
                "transformed_feature_count": 2,
            }
        ),
        encoding="utf-8",
    )

    return SimpleNamespace(
        artifacts={
            "preprocessor":
                str(
                    preprocessor_path
                ),
            "model_bundle":
                str(
                    model_bundle_path
                ),
            "feature_names":
                str(
                    feature_names_path
                ),
            "feature_config":
                str(
                    feature_config_path
                ),
        }
    )


def test_load_context_loads_validated_artifacts(
    tmp_path,
):
    model = (
        OlistLateDeliveryPythonModel()
    )

    model.load_context(
        make_context(
            tmp_path
        )
    )

    assert (
        type(
            model.artifacts.model
        ).__name__
        == "DummyModel"
    )

    assert (
        model
        .artifacts
        .classification_threshold
        == pytest.approx(
            0.4
        )
    )

    assert len(
        model.artifacts.feature_names
    ) == 2


def test_load_context_rejects_missing_artifact(
    tmp_path,
):
    context = make_context(
        tmp_path
    )

    del context.artifacts[
        "model_bundle"
    ]

    model = (
        OlistLateDeliveryPythonModel()
    )

    with pytest.raises(
        MlflowModelError,
        match="missing required artifacts",
    ):
        model.load_context(
            context
        )


def test_predict_requires_loaded_context():
    model = (
        OlistLateDeliveryPythonModel()
    )

    with pytest.raises(
        MlflowModelError,
        match="have not been loaded",
    ):
        model.predict(
            None,
            pd.DataFrame(
                {
                    "value": [
                        1,
                    ]
                }
            ),
        )


def test_predict_reuses_production_inference(
    monkeypatch,
    tmp_path,
):
    model = (
        OlistLateDeliveryPythonModel()
    )

    model.load_context(
        make_context(
            tmp_path
        )
    )

    model_input = pd.DataFrame(
        {
            "order_id": [
                "order-1",
            ]
        }
    )

    expected = pd.DataFrame(
        {
            "order_id": [
                "order-1",
            ],
            "late_probability": [
                0.2,
            ],
            "predicted_is_late": [
                0,
            ],
        }
    )

    captured = {}

    def fake_predict_orders(
        raw_orders,
        artifacts=None,
        positive_class=None,
    ):
        captured[
            "raw_orders"
        ] = raw_orders

        captured[
            "artifacts"
        ] = artifacts

        return expected.copy()

    monkeypatch.setattr(
        mlflow_model,
        "predict_orders",
        fake_predict_orders,
    )

    actual = model.predict(
        None,
        model_input,
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
    )

    assert (
        captured[
            "raw_orders"
        ]
        is model_input
    )

    assert (
        captured[
            "artifacts"
        ]
        is model.artifacts
    )
