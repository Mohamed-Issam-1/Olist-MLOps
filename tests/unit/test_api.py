from types import SimpleNamespace

from fastapi.testclient import (
    TestClient,
)

import app.main as api
from olist_ml.mlflow_loader import (
    MlflowModelLoadError,
)


client = TestClient(
    api.app
)


def make_runtime_model():
    return SimpleNamespace(
        source="mlflow-registry",
        registered_model_name=(
            "olist-late-delivery"
        ),
        alias="champion",
        version="1",
        run_id="run-123",
        model_uri=(
            "models:/"
            "olist-late-delivery"
            "@champion"
        ),
        model_type="LogisticRegression",
        classification_threshold=(
            0.0822110764307922
        ),
    )


def test_health_route():
    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert (
        body[
            "status"
        ]
        == "ok"
    )

    assert (
        body[
            "service"
        ]
        == (
            "Olist Late Delivery "
            "Inference API"
        )
    )

    assert (
        body[
            "api_version"
        ]
        == "1.0.0"
    )


def test_model_info_route(
    monkeypatch,
):
    runtime_model = (
        make_runtime_model()
    )

    monkeypatch.setattr(
        api,
        "load_registered_inference_model",
        lambda:
            runtime_model,
    )

    response = client.get(
        "/model-info"
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert (
        body[
            "source"
        ]
        == "mlflow-registry"
    )

    assert (
        body[
            "registered_model_name"
        ]
        == "olist-late-delivery"
    )

    assert (
        body[
            "alias"
        ]
        == "champion"
    )

    assert (
        body[
            "version"
        ]
        == "1"
    )

    assert (
        body[
            "model_type"
        ]
        == "LogisticRegression"
    )

    assert (
        body[
            "classification_threshold"
        ]
        == 0.0822110764307922
    )


def test_model_info_returns_503_when_registry_unavailable(
    monkeypatch,
):
    def fail_load():
        raise MlflowModelLoadError(
            "registry unavailable"
        )

    monkeypatch.setattr(
        api,
        "load_registered_inference_model",
        fail_load,
    )

    response = client.get(
        "/model-info"
    )

    assert (
        response.status_code
        == 503
    )

    assert response.json() == {
        "detail":
            "Production model is unavailable."
    }


def test_openapi_contains_foundation_routes():
    response = client.get(
        "/openapi.json"
    )

    assert (
        response.status_code
        == 200
    )

    schema = response.json()

    assert (
        schema[
            "info"
        ][
            "title"
        ]
        == (
            "Olist Late Delivery "
            "Inference API"
        )
    )

    assert (
        schema[
            "info"
        ][
            "version"
        ]
        == "1.0.0"
    )

    assert (
        "/health"
        in schema[
            "paths"
        ]
    )

    assert (
        "/model-info"
        in schema[
            "paths"
        ]
    )
