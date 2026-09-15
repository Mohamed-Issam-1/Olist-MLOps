from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import olist_ml.prediction_service as service


def make_artifacts():
    return SimpleNamespace(
        model=SimpleNamespace(),
        classification_threshold=0.4,
    )


def test_get_model_runtime_metadata():
    original = service.load_config

    service.load_config = lambda: {
        "model": {
            "source":
                "local-artifact",
            "version":
                "version-1",
        }
    }

    try:
        source, version = (
            service
            .get_model_runtime_metadata()
        )

        assert source == (
            "local-artifact"
        )

        assert version == (
            "version-1"
        )

    finally:
        service.load_config = original


def test_predict_orders_with_logging_returns_predictions(
    monkeypatch,
):
    logger = Mock()

    predictions = pd.DataFrame(
        {
            "order_id": [
                "order-1",
            ],
            "late_probability": [
                0.7,
            ],
            "predicted_is_late": [
                1,
            ],
        }
    )

    monkeypatch.setattr(
        service,
        "get_logger",
        lambda name:
            logger,
    )

    monkeypatch.setattr(
        service,
        "get_model_runtime_metadata",
        lambda: (
            "local-artifact",
            "version-1",
        ),
    )

    monkeypatch.setattr(
        service,
        "predict_orders",
        lambda raw_orders, artifacts:
            predictions.copy(),
    )

    prediction_log_mock = Mock(
        return_value="request-1"
    )

    monkeypatch.setattr(
        service,
        "get_prediction_input_columns",
        lambda raw_orders, artifacts: [
            "order_id",
        ],
    )

    monkeypatch.setattr(
        service,
        "log_prediction_batch",
        prediction_log_mock,
    )

    raw_orders = pd.DataFrame(
        {
            "order_id": [
                "order-1",
            ]
        }
    )

    result = (
        service
        .predict_orders_with_logging(
            raw_orders,
            make_artifacts(),
        )
    )

    pd.testing.assert_frame_equal(
        result,
        predictions,
    )

    prediction_log_mock.assert_called_once()

    logger.info.assert_called_once()


def test_predict_orders_with_logging_logs_failure(
    monkeypatch,
):
    logger = Mock()

    monkeypatch.setattr(
        service,
        "get_logger",
        lambda name:
            logger,
    )

    monkeypatch.setattr(
        service,
        "get_model_runtime_metadata",
        lambda: (
            "local-artifact",
            "version-1",
        ),
    )

    def fail_prediction(
        raw_orders,
        artifacts,
    ):
        raise ValueError(
            "bad input"
        )

    monkeypatch.setattr(
        service,
        "predict_orders",
        fail_prediction,
    )

    with pytest.raises(
        ValueError,
        match="bad input",
    ):
        service.predict_orders_with_logging(
            pd.DataFrame(
                {
                    "order_id": [
                        "order-1",
                    ]
                }
            ),
            make_artifacts(),
        )

    logger.exception.assert_called_once()
