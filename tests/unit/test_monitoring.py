import json
import logging

import pytest

import olist_ml.monitoring as monitoring


def reset_monitoring_logger():
    logger = logging.getLogger(monitoring.MONITORING_LOGGER_NAME)

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def configure_test_environment(
    tmp_path,
    monkeypatch,
    *,
    minimum_predictions=2,
):
    reset_monitoring_logger()

    config = {
        "logging": {
            "directory": "logs",
            "prediction_log": "predictions.jsonl",
        },
        "monitoring": {
            "request_log": "requests.jsonl",
            "minimum_predictions_for_drift": (minimum_predictions),
            "baseline": {
                "prediction_count": 100,
                "predicted_late_rate": 0.20,
                "mean_late_probability": 0.10,
                "std_late_probability": 0.05,
            },
            "drift_thresholds": {
                "predicted_late_rate_absolute_delta": 0.10,
                "mean_late_probability_absolute_delta": 0.05,
            },
            "alert_thresholds": {
                "error_rate": 0.05,
                "average_latency_ms": 150.0,
            },
        },
    }

    monkeypatch.setattr(
        monitoring,
        "load_config",
        lambda: config,
    )

    monkeypatch.setattr(
        monitoring,
        "resolve_project_path",
        lambda value: tmp_path / value,
    )

    return config


def write_jsonl(path, records):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_log_api_request_records_only_prediction_paths(
    tmp_path,
    monkeypatch,
):
    configure_test_environment(
        tmp_path,
        monkeypatch,
    )

    monitoring.log_api_request(
        method="GET",
        path="/health",
        status_code=200,
        latency_ms=1.0,
    )

    monitoring.log_api_request(
        method="POST",
        path="/predict",
        status_code=200,
        latency_ms=12.5,
    )

    reset_monitoring_logger()

    request_log = tmp_path / "logs" / "requests.jsonl"

    lines = request_log.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["event"] == "api_request"
    assert record["path"] == "/predict"
    assert record["status_code"] == 200

    assert record["latency_ms"] == pytest.approx(12.5)


def test_build_monitoring_summary_aggregates_metrics_and_alerts(
    tmp_path,
    monkeypatch,
):
    configure_test_environment(
        tmp_path,
        monkeypatch,
    )

    request_log = tmp_path / "logs" / "requests.jsonl"

    prediction_log = tmp_path / "logs" / "predictions.jsonl"

    write_jsonl(
        request_log,
        [
            {
                "event": "api_request",
                "path": "/predict",
                "status_code": 200,
                "latency_ms": 100.0,
            },
            {
                "event": "api_request",
                "path": "/predict",
                "status_code": 422,
                "latency_ms": 200.0,
            },
            {
                "event": "api_request",
                "path": "/predict/batch",
                "status_code": 200,
                "latency_ms": 300.0,
            },
        ],
    )

    probabilities = [
        0.10,
        0.20,
        0.30,
        0.40,
    ]

    labels = [
        0,
        1,
        1,
        0,
    ]

    write_jsonl(
        prediction_log,
        [
            {
                "event": "prediction",
                "output": {
                    "late_probability": probability,
                    "predicted_is_late": label,
                },
            }
            for probability, label in zip(
                probabilities,
                labels,
                strict=True,
            )
        ],
    )

    summary = monitoring.build_monitoring_summary()

    assert summary["status"] == "alert"

    assert summary["requests"]["request_count"] == 3

    assert summary["requests"]["error_count"] == 1

    assert summary["requests"]["error_rate"] == pytest.approx(1 / 3)

    assert summary["requests"]["average_latency_ms"] == pytest.approx(200.0)

    assert summary["requests"]["max_latency_ms"] == pytest.approx(300.0)

    assert summary["predictions"]["prediction_count"] == 4

    assert summary["predictions"]["predicted_late_rate"] == pytest.approx(0.5)

    assert summary["predictions"]["mean_late_probability"] == pytest.approx(0.25)

    assert summary["drift"]["status"] == "drift_detected"

    assert "high_error_rate" in (summary["alerts"])

    assert "high_average_latency" in (summary["alerts"])

    assert "prediction_drift" in (summary["alerts"])


def test_build_monitoring_summary_reports_insufficient_drift_data(
    tmp_path,
    monkeypatch,
):
    configure_test_environment(
        tmp_path,
        monkeypatch,
        minimum_predictions=30,
    )

    prediction_log = tmp_path / "logs" / "predictions.jsonl"

    write_jsonl(
        prediction_log,
        [
            {
                "event": "prediction",
                "output": {
                    "late_probability": 0.05,
                    "predicted_is_late": 0,
                },
            }
        ],
    )

    summary = monitoring.build_monitoring_summary()

    assert summary["status"] == "ok"

    assert summary["drift"]["status"] == "insufficient_data"

    assert summary["drift"]["predicted_late_rate_absolute_delta"] is None

    assert summary["alerts"] == []


def test_build_monitoring_summary_ignores_malformed_jsonl(
    tmp_path,
    monkeypatch,
):
    configure_test_environment(
        tmp_path,
        monkeypatch,
    )

    request_log = tmp_path / "logs" / "requests.jsonl"

    request_log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_log.write_text(
        '{"event":"api_request","path":"/predict",'
        '"status_code":200,"latency_ms":10}\n'
        "not-json\n",
        encoding="utf-8",
    )

    summary = monitoring.build_monitoring_summary()

    assert summary["requests"]["request_count"] == 1

    assert summary["invalid_log_records"]["requests"] == 1
