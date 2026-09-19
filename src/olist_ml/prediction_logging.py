from __future__ import annotations

import json
import logging
from datetime import (
    date,
    datetime,
    timezone,
)
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from olist_ml.config import (
    load_config,
    resolve_project_path,
)

PREDICTION_LOGGER_NAME = "olist_ml.predictions"

REQUIRED_PREDICTION_COLUMNS = {
    "late_probability",
    "predicted_is_late",
}


def _json_default(value):
    """
    Convert common pandas/numpy values into JSON-safe values.
    """

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if pd.isna(value):
        return None

    return str(value)


def configure_prediction_logger() -> logging.Logger:
    """
    Configure a dedicated JSONL prediction logger.

    Prediction records are written separately from the
    general application log.
    """

    config = load_config()
    logging_config = config["logging"]

    logger = logging.getLogger(PREDICTION_LOGGER_NAME)

    if logger.handlers:
        return logger

    log_directory = resolve_project_path(logging_config["directory"])

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_log_path = log_directory / logging_config["prediction_log"]

    handler = logging.FileHandler(
        prediction_log_path,
        encoding="utf-8",
    )

    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.setLevel(logging.INFO)

    logger.addHandler(handler)

    logger.propagate = False

    return logger


def log_prediction_batch(
    raw_orders: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    input_columns: list[str],
    latency_ms: float,
    model_type: str,
    model_version: str,
    threshold: float,
) -> str:
    """
    Write one request summary and one JSON record per prediction.

    Returns the generated request ID.
    """

    if len(raw_orders) != len(predictions):
        raise ValueError("Raw order count and prediction count must match.")

    missing_columns = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            "Prediction log data is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    missing_input_columns = [
        column for column in input_columns if column not in raw_orders.columns
    ]

    if missing_input_columns:
        raise ValueError(
            "Prediction input log columns are missing: "
            + ", ".join(missing_input_columns)
        )

    logger = configure_prediction_logger()

    request_id = str(uuid4())

    timestamp = datetime.now(timezone.utc).isoformat()

    batch_record = {
        "event": "prediction_request",
        "timestamp": timestamp,
        "request_id": request_id,
        "request_size": len(raw_orders),
        "input_columns": list(input_columns),
        "latency_ms": float(latency_ms),
        "model_type": str(model_type),
        "model_version": str(model_version),
        "threshold": float(threshold),
    }

    logger.info(
        json.dumps(
            batch_record,
            ensure_ascii=False,
            default=_json_default,
        )
    )

    raw_orders_reset = raw_orders.reset_index(drop=True)

    predictions_reset = predictions.reset_index(drop=True)

    for index in range(len(predictions_reset)):
        prediction_row = predictions_reset.iloc[index]

        raw_row = raw_orders_reset.iloc[index]

        input_record = {column: raw_row[column] for column in input_columns}

        prediction_record = {
            "event": "prediction",
            "timestamp": timestamp,
            "request_id": request_id,
            "input": input_record,
            "output": {
                "late_probability": float(prediction_row["late_probability"]),
                "predicted_is_late": int(prediction_row["predicted_is_late"]),
            },
            "model_type": str(model_type),
            "model_version": str(model_version),
            "threshold": float(threshold),
        }

        logger.info(
            json.dumps(
                prediction_record,
                ensure_ascii=False,
                default=_json_default,
            )
        )

    return request_id
