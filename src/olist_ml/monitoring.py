from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from olist_ml.config import load_config, resolve_project_path

MONITORING_LOGGER_NAME = "olist_ml.monitoring"

PREDICTION_PATHS = {
    "/predict",
    "/predict/batch",
}


def configure_monitoring_logger() -> logging.Logger:
    """Configure the structured API monitoring JSONL logger."""

    config = load_config()

    logging_config = config["logging"]
    monitoring_config = config["monitoring"]

    logger = logging.getLogger(MONITORING_LOGGER_NAME)

    if logger.handlers:
        return logger

    log_directory = resolve_project_path(logging_config["directory"])

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_log_path = log_directory / monitoring_config["request_log"]

    handler = logging.FileHandler(
        request_log_path,
        encoding="utf-8",
    )

    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def log_api_request(
    *,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """
    Write one structured monitoring event for
    a prediction API request.
    """

    if path not in PREDICTION_PATHS:
        return

    logger = configure_monitoring_logger()

    record = {
        "event": "api_request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": str(method),
        "path": str(path),
        "status_code": int(status_code),
        "latency_ms": float(latency_ms),
    }

    logger.info(
        json.dumps(
            record,
            ensure_ascii=False,
        )
    )


def _read_jsonl(
    path: Path,
) -> tuple[list[dict], int]:
    """
    Read JSONL records while ignoring malformed lines.
    """

    if not path.exists():
        return [], 0

    records: list[dict] = []
    invalid_count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            text = line.strip()

            if not text:
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                invalid_count += 1
                continue

            if isinstance(record, dict):
                records.append(record)
            else:
                invalid_count += 1

    return records, invalid_count


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def _is_error_status(value) -> bool:
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return True

    return status_code >= 400


def build_monitoring_summary() -> dict:
    """
    Aggregate API and prediction monitoring metrics.

    Monitoring never fits, retrains, or modifies
    the production model.
    """

    config = load_config()

    logging_config = config["logging"]
    monitoring_config = config["monitoring"]

    log_directory = resolve_project_path(logging_config["directory"])

    request_log_path = log_directory / monitoring_config["request_log"]

    prediction_log_path = log_directory / logging_config["prediction_log"]

    request_records, invalid_request_records = _read_jsonl(request_log_path)

    prediction_records, invalid_prediction_records = _read_jsonl(prediction_log_path)

    request_events = [
        record
        for record in request_records
        if record.get("event") == "api_request"
        and record.get("path") in PREDICTION_PATHS
    ]

    latencies = [
        latency
        for latency in (
            _safe_float(record.get("latency_ms")) for record in request_events
        )
        if latency is not None
    ]

    request_count = len(request_events)

    error_count = sum(
        1 for record in request_events if _is_error_status(record.get("status_code"))
    )

    error_rate = error_count / request_count if request_count else 0.0

    average_latency_ms = float(np.mean(latencies)) if latencies else None

    p95_latency_ms = (
        float(
            np.percentile(
                latencies,
                95,
            )
        )
        if latencies
        else None
    )

    max_latency_ms = float(max(latencies)) if latencies else None

    prediction_events = [
        record for record in prediction_records if record.get("event") == "prediction"
    ]

    probabilities: list[float] = []
    predicted_labels: list[int] = []

    for record in prediction_events:
        output = record.get("output")

        if not isinstance(output, dict):
            continue

        probability = _safe_float(output.get("late_probability"))

        if probability is None:
            continue

        try:
            label = int(output.get("predicted_is_late"))
        except (TypeError, ValueError):
            continue

        if label not in {0, 1}:
            continue

        probabilities.append(probability)
        predicted_labels.append(label)

    prediction_count = len(predicted_labels)

    predicted_late_count = sum(predicted_labels)

    predicted_on_time_count = prediction_count - predicted_late_count

    predicted_late_rate = (
        predicted_late_count / prediction_count if prediction_count else None
    )

    mean_late_probability = float(np.mean(probabilities)) if probabilities else None

    baseline = monitoring_config["baseline"]

    drift_thresholds = monitoring_config["drift_thresholds"]

    minimum_predictions = int(monitoring_config["minimum_predictions_for_drift"])

    late_rate_delta = None
    mean_probability_delta = None
    drift_detected = False

    if prediction_count < minimum_predictions:
        drift_status = "insufficient_data"

    else:
        late_rate_delta = abs(
            predicted_late_rate - float(baseline["predicted_late_rate"])
        )

        mean_probability_delta = abs(
            mean_late_probability - float(baseline["mean_late_probability"])
        )

        drift_detected = late_rate_delta > float(
            drift_thresholds["predicted_late_rate_absolute_delta"]
        ) or mean_probability_delta > float(
            drift_thresholds["mean_late_probability_absolute_delta"]
        )

        drift_status = "drift_detected" if drift_detected else "stable"

    alert_thresholds = monitoring_config["alert_thresholds"]

    alerts: list[str] = []

    if request_count and error_rate > float(alert_thresholds["error_rate"]):
        alerts.append("high_error_rate")

    if average_latency_ms is not None and average_latency_ms > float(
        alert_thresholds["average_latency_ms"]
    ):
        alerts.append("high_average_latency")

    if drift_detected:
        alerts.append("prediction_drift")

    return {
        "status": ("alert" if alerts else "ok"),
        "requests": {
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": float(error_rate),
            "average_latency_ms": (average_latency_ms),
            "p95_latency_ms": (p95_latency_ms),
            "max_latency_ms": (max_latency_ms),
        },
        "predictions": {
            "prediction_count": (prediction_count),
            "predicted_late_count": (predicted_late_count),
            "predicted_on_time_count": (predicted_on_time_count),
            "predicted_late_rate": (predicted_late_rate),
            "mean_late_probability": (mean_late_probability),
        },
        "drift": {
            "status": drift_status,
            "minimum_predictions": (minimum_predictions),
            "baseline_prediction_count": int(baseline["prediction_count"]),
            "baseline_predicted_late_rate": float(baseline["predicted_late_rate"]),
            "baseline_mean_late_probability": float(baseline["mean_late_probability"]),
            "predicted_late_rate_absolute_delta": (late_rate_delta),
            "mean_late_probability_absolute_delta": (mean_probability_delta),
            "predicted_late_rate_threshold": float(
                drift_thresholds["predicted_late_rate_absolute_delta"]
            ),
            "mean_late_probability_threshold": float(
                drift_thresholds["mean_late_probability_absolute_delta"]
            ),
        },
        "alerts": alerts,
        "invalid_log_records": {
            "requests": invalid_request_records,
            "predictions": (invalid_prediction_records),
        },
    }
