from unittest.mock import Mock

import pandas as pd
import pytest

import olist_ml.prediction_service as service


class DummyArtifacts:
    pass


class FakeRuntimeModel:
    def __init__(
        self,
        predictions=None,
    ):
        self.source = (
            "mlflow-registry"
        )

        self.version = "1"

        self.alias = "champion"

        self.model_uri = (
            "models:/"
            "olist-late-delivery"
            "@champion"
        )

        self.model_type = (
            "LogisticRegression"
        )

        self.classification_threshold = (
            0.0822110764307922
        )

        self.artifacts = (
            DummyArtifacts()
        )

        self.predictions = (
            predictions
            if predictions is not None
            else pd.DataFrame(
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
        )

        self.predict_calls = []

    def predict(
        self,
        raw_orders,
    ):
        self.predict_calls.append(
            raw_orders
        )

        return (
            self.predictions
            .copy()
        )


def configure_logging_mocks(
    monkeypatch,
):
    logger = Mock()

    monkeypatch.setattr(
        service,
        "get_logger",
        lambda name:
            logger,
    )

    prediction_log = Mock(
        return_value="request-1"
    )

    monkeypatch.setattr(
        service,
        "log_prediction_batch",
        prediction_log,
    )

    monkeypatch.setattr(
        service,
        "get_prediction_input_columns",
        lambda raw_orders, artifacts: [
            "order_id",
        ],
    )

    return (
        logger,
        prediction_log,
    )


def test_prediction_service_loads_registry_model(
    monkeypatch,
):
    (
        logger,
        prediction_log,
    ) = configure_logging_mocks(
        monkeypatch
    )

    runtime_model = (
        FakeRuntimeModel()
    )

    load_mock = Mock(
        return_value=runtime_model
    )

    monkeypatch.setattr(
        service,
        "load_registered_inference_model",
        load_mock,
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
            raw_orders
        )
    )

    pd.testing.assert_frame_equal(
        result,
        runtime_model.predictions,
    )

    load_mock.assert_called_once_with()

    assert (
        runtime_model.predict_calls
        == [
            raw_orders
        ]
    )

    prediction_log.assert_called_once()

    call_kwargs = (
        prediction_log
        .call_args
        .kwargs
    )

    assert (
        call_kwargs[
            "model_type"
        ]
        == "LogisticRegression"
    )

    assert (
        call_kwargs[
            "model_version"
        ]
        == "1"
    )

    assert (
        call_kwargs[
            "threshold"
        ]
        == pytest.approx(
            0.0822110764307922
        )
    )

    assert (
        call_kwargs[
            "input_columns"
        ]
        == [
            "order_id",
        ]
    )

    logger.info.assert_called_once()


def test_prediction_service_accepts_injected_registry_model(
    monkeypatch,
):
    (
        _logger,
        _prediction_log,
    ) = configure_logging_mocks(
        monkeypatch
    )

    runtime_model = (
        FakeRuntimeModel()
    )

    load_mock = Mock()

    monkeypatch.setattr(
        service,
        "load_registered_inference_model",
        load_mock,
    )

    result = (
        service
        .predict_orders_with_logging(
            pd.DataFrame(
                {
                    "order_id": [
                        "order-1",
                    ]
                }
            ),
            runtime_model=(
                runtime_model
            ),
        )
    )

    assert len(
        result
    ) == 1

    load_mock.assert_not_called()


def test_prediction_service_logs_registry_failure(
    monkeypatch,
):
    logger = Mock()

    monkeypatch.setattr(
        service,
        "get_logger",
        lambda name:
            logger,
    )

    def fail_load():
        raise RuntimeError(
            "registry unavailable"
        )

    monkeypatch.setattr(
        service,
        "load_registered_inference_model",
        fail_load,
    )

    with pytest.raises(
        RuntimeError,
        match="registry unavailable",
    ):
        (
            service
            .predict_orders_with_logging(
                pd.DataFrame(
                    {
                        "order_id": [
                            "order-1",
                        ]
                    }
                )
            )
        )

    logger.exception.assert_called_once()

    log_args = (
        logger
        .exception
        .call_args
        .args
    )

    assert (
        "mlflow-registry"
        in log_args
    )

    assert (
        "unresolved"
        in log_args
    )


def test_prediction_service_logs_resolved_version_on_prediction_failure(
    monkeypatch,
):
    logger = Mock()

    monkeypatch.setattr(
        service,
        "get_logger",
        lambda name:
            logger,
    )

    runtime_model = (
        FakeRuntimeModel()
    )

    def fail_prediction(
        raw_orders,
    ):
        raise ValueError(
            "bad input"
        )

    runtime_model.predict = (
        fail_prediction
    )

    with pytest.raises(
        ValueError,
        match="bad input",
    ):
        (
            service
            .predict_orders_with_logging(
                pd.DataFrame(
                    {
                        "order_id": [
                            "order-1",
                        ]
                    }
                ),
                runtime_model=(
                    runtime_model
                ),
            )
        )

    logger.exception.assert_called_once()

    log_args = (
        logger
        .exception
        .call_args
        .args
    )

    assert (
        "mlflow-registry"
        in log_args
    )

    assert "1" in log_args
