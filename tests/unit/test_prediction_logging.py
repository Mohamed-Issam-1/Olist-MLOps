import json
import logging

import pandas as pd
import pytest

import olist_ml.prediction_logging as prediction_logging


def reset_prediction_logger():
    logger = logging.getLogger(
        prediction_logging.PREDICTION_LOGGER_NAME
    )

    for handler in list(
        logger.handlers
    ):
        handler.close()

        logger.removeHandler(
            handler
        )


def configure_test_environment(
    tmp_path,
    monkeypatch,
):
    reset_prediction_logger()

    monkeypatch.setattr(
        prediction_logging,
        "load_config",
        lambda: {
            "logging": {
                "directory": "logs",
                "prediction_log":
                    "predictions.jsonl",
            }
        },
    )

    monkeypatch.setattr(
        prediction_logging,
        "resolve_project_path",
        lambda value:
            tmp_path / value,
    )


def test_configure_prediction_logger_creates_jsonl_file(
    tmp_path,
    monkeypatch,
):
    configure_test_environment(
        tmp_path,
        monkeypatch,
    )

    prediction_logging.configure_prediction_logger()

    assert (
        tmp_path
        / "logs"
        / "predictions.jsonl"
    ).exists()

    reset_prediction_logger()


def test_log_prediction_batch_writes_request_and_predictions(
    tmp_path,
    monkeypatch,
):
    configure_test_environment(
        tmp_path,
        monkeypatch,
    )

    raw_orders = pd.DataFrame(
        {
            "order_id": [
                "order-1",
                "order-2",
            ],
            "feature_a": [
                1,
                2,
            ],
        }
    )

    predictions = pd.DataFrame(
        {
            "order_id": [
                "order-1",
                "order-2",
            ],
            "late_probability": [
                0.2,
                0.8,
            ],
            "predicted_is_late": [
                0,
                1,
            ],
        }
    )

    request_id = (
        prediction_logging
        .log_prediction_batch(
            raw_orders,
            predictions,
            input_columns=[
                "order_id",
                "feature_a",
            ],
            latency_ms=12.5,
            model_type=
                "LogisticRegression",
            model_version=
                "test-version",
            threshold=0.4,
        )
    )

    reset_prediction_logger()

    log_path = (
        tmp_path
        / "logs"
        / "predictions.jsonl"
    )

    lines = (
        log_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert len(lines) == 3

    records = [
        json.loads(
            line
        )
        for line in lines
    ]

    assert (
        records[0][
            "event"
        ]
        == "prediction_request"
    )

    assert (
        records[0][
            "request_id"
        ]
        == request_id
    )

    assert (
        records[0][
            "request_size"
        ]
        == 2
    )

    assert (
        records[0][
            "model_version"
        ]
        == "test-version"
    )

    assert (
        records[1][
            "input"
        ][
            "order_id"
        ]
        == "order-1"
    )

    assert set(
        records[1][
            "input"
        ].keys()
    ) == {
        "order_id",
        "feature_a",
    }

    assert (
        records[1][
            "output"
        ][
            "late_probability"
        ]
        == pytest.approx(
            0.2
        )
    )

    assert (
        records[2][
            "output"
        ][
            "predicted_is_late"
        ]
        == 1
    )


def test_log_prediction_batch_rejects_row_count_mismatch(
    tmp_path,
    monkeypatch,
):
    configure_test_environment(
        tmp_path,
        monkeypatch,
    )

    raw_orders = pd.DataFrame(
        {
            "order_id": [
                "order-1",
            ]
        }
    )

    predictions = pd.DataFrame(
        {
            "late_probability": [
                0.1,
                0.2,
            ],
            "predicted_is_late": [
                0,
                0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="count",
    ):
        prediction_logging.log_prediction_batch(
            raw_orders,
            predictions,
            input_columns=[
                "order_id",
            ],
            latency_ms=1.0,
            model_type="Model",
            model_version="v1",
            threshold=0.5,
        )

    reset_prediction_logger()


def test_log_prediction_batch_rejects_missing_output_column(
    tmp_path,
    monkeypatch,
):
    configure_test_environment(
        tmp_path,
        monkeypatch,
    )

    raw_orders = pd.DataFrame(
        {
            "order_id": [
                "order-1",
            ]
        }
    )

    predictions = pd.DataFrame(
        {
            "late_probability": [
                0.1,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="predicted_is_late",
    ):
        prediction_logging.log_prediction_batch(
            raw_orders,
            predictions,
            input_columns=[
                "order_id",
            ],
            latency_ms=1.0,
            model_type="Model",
            model_version="v1",
            threshold=0.5,
        )

    reset_prediction_logger()
