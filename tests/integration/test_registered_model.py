from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist_ml.mlflow_loader import (
    clear_registered_inference_model_cache,
    load_registered_inference_model,
)
from olist_ml.prediction_service import (
    get_prediction_input_columns,
)

EXPECTED_OUTPUT_COLUMNS = [
    "order_id",
    "late_probability",
    "predicted_is_late",
]

PROBABILITY_TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def runtime_model():
    """
    Load the real production model from the configured
    MLflow Registry alias once for this test module.
    """

    clear_registered_inference_model_cache()

    model = load_registered_inference_model(refresh=True)

    yield model

    clear_registered_inference_model_cache()


@pytest.fixture(scope="module")
def parity_data(
    runtime_model,
):
    """
    Load the real Task 2 final test split and its
    saved Notebook 6 predictions.
    """

    test_data = pd.read_parquet("artifacts/03_splits/test.parquet")

    expected = pd.read_csv("artifacts/06_model/test_predictions.csv")

    input_columns = get_prediction_input_columns(
        test_data,
        runtime_model.artifacts,
    )

    model_input = test_data[input_columns].copy()

    return (
        test_data,
        expected,
        input_columns,
        model_input,
    )


def test_registered_model_loads_with_expected_metadata(
    runtime_model,
):
    """
    The production alias must resolve to a usable
    registered inference model.
    """

    assert runtime_model.source == "mlflow-registry"

    assert runtime_model.alias == "champion"

    assert runtime_model.registered_model_name == "olist-late-delivery"

    assert runtime_model.version.isdigit()

    assert int(runtime_model.version) >= 1

    assert runtime_model.model_type == "LogisticRegression"

    assert 0.0 < runtime_model.classification_threshold < 1.0

    assert runtime_model.model_uri == ("models:/olist-late-delivery@champion")


def test_registered_model_predicts_known_order(
    runtime_model,
    parity_data,
):
    """
    A known Task 2 test order must reproduce the
    saved Notebook 6 prediction.
    """

    (
        _test_data,
        expected,
        _input_columns,
        model_input,
    ) = parity_data

    actual = runtime_model.predict(model_input.head(1))

    assert list(actual.columns) == EXPECTED_OUTPUT_COLUMNS

    assert len(actual) == 1

    assert str(actual.iloc[0]["order_id"]) == str(expected.iloc[0]["order_id"])

    assert float(actual.iloc[0]["late_probability"]) == pytest.approx(
        float(expected.iloc[0]["late_probability"]),
        abs=PROBABILITY_TOLERANCE,
    )

    assert int(actual.iloc[0]["predicted_is_late"]) == int(
        expected.iloc[0]["predicted_is_late"]
    )


def test_registered_model_reproduces_full_task2_test_output(
    runtime_model,
    parity_data,
):
    """
    The registry-backed production model must reproduce
    Notebook 6 across the complete final test set.
    """

    (
        test_data,
        expected,
        input_columns,
        model_input,
    ) = parity_data

    actual = runtime_model.predict(model_input)

    assert len(test_data) == 14471

    assert len(expected) == 14471

    assert len(actual) == 14471

    assert len(input_columns) == 31

    assert list(actual.columns) == EXPECTED_OUTPUT_COLUMNS

    actual_order_ids = actual["order_id"].astype(str).to_numpy()

    expected_order_ids = expected["order_id"].astype(str).to_numpy()

    np.testing.assert_array_equal(
        actual_order_ids,
        expected_order_ids,
    )

    actual_probabilities = actual["late_probability"].to_numpy(dtype=float)

    expected_probabilities = expected["late_probability"].to_numpy(dtype=float)

    np.testing.assert_allclose(
        actual_probabilities,
        expected_probabilities,
        rtol=0.0,
        atol=PROBABILITY_TOLERANCE,
    )

    assert np.all(actual_probabilities >= 0.0)

    assert np.all(actual_probabilities <= 1.0)

    actual_predictions = actual["predicted_is_late"].to_numpy(dtype=int)

    expected_predictions = expected["predicted_is_late"].to_numpy(dtype=int)

    np.testing.assert_array_equal(
        actual_predictions,
        expected_predictions,
    )

    assert set(np.unique(actual_predictions)).issubset(
        {
            0,
            1,
        }
    )

    assert int(actual_predictions.sum()) == 2964
