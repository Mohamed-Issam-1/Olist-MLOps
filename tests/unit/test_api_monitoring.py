from fastapi.testclient import TestClient

import app.main as api_module


def build_test_summary():
    return {
        "status": "ok",
        "requests": {
            "request_count": 2,
            "error_count": 0,
            "error_rate": 0.0,
            "average_latency_ms": 25.0,
            "p95_latency_ms": 29.5,
            "max_latency_ms": 30.0,
        },
        "predictions": {
            "prediction_count": 2,
            "predicted_late_count": 1,
            "predicted_on_time_count": 1,
            "predicted_late_rate": 0.5,
            "mean_late_probability": 0.4,
        },
        "drift": {
            "status": "insufficient_data",
            "minimum_predictions": 30,
            "baseline_prediction_count": 14471,
            "baseline_predicted_late_rate": (0.2048234399834151),
            "baseline_mean_late_probability": (0.05929370779772134),
            "predicted_late_rate_absolute_delta": None,
            "mean_late_probability_absolute_delta": None,
            "predicted_late_rate_threshold": 0.10,
            "mean_late_probability_threshold": 0.03,
        },
        "alerts": [],
        "invalid_log_records": {
            "requests": 0,
            "predictions": 0,
        },
    }


def test_monitoring_endpoint_returns_summary(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "build_monitoring_summary",
        build_test_summary,
    )

    client = TestClient(api_module.app)

    response = client.get("/monitoring")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"

    assert body["requests"]["request_count"] == 2

    assert body["predictions"]["prediction_count"] == 2

    assert body["drift"]["status"] == "insufficient_data"


def test_prediction_validation_error_is_recorded_by_middleware(
    monkeypatch,
):
    recorded = []

    def fake_log_api_request(
        *,
        method,
        path,
        status_code,
        latency_ms,
    ):
        recorded.append(
            {
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency_ms": latency_ms,
            }
        )

    monkeypatch.setattr(
        api_module,
        "log_api_request",
        fake_log_api_request,
    )

    client = TestClient(api_module.app)

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422
    assert len(recorded) == 1

    event = recorded[0]

    assert event["method"] == "POST"
    assert event["path"] == "/predict"
    assert event["status_code"] == 422
    assert event["latency_ms"] >= 0.0
