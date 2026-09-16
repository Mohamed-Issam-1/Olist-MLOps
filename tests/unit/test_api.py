from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

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


def make_valid_order_payload():
    return {
        "order_id":
            "b3b54427f53d13f6063ef7007bf7d371",

        "item_count": 1.0,
        "unique_products": 1.0,
        "unique_sellers": 1.0,

        "total_item_price": 55.0,
        "avg_item_price": 55.0,

        "total_freight_value": 7.65,
        "avg_freight_value": 7.65,

        "unique_product_categories": 1.0,

        "avg_product_weight_g": 200.0,
        "max_product_weight_g": 200.0,

        "avg_product_length_cm": 16.0,
        "avg_product_height_cm": 2.0,
        "avg_product_width_cm": 20.0,

        "avg_product_photos_qty": 5.0,

        "unique_seller_states": 1.0,

        "payment_records": 1.0,
        "payment_types_count": 1.0,
        "payment_total": 62.65,
        "payment_installments_max": 1.0,

        "customer_state": "SP",

        "primary_product_category":
            "watches_gifts",

        "primary_seller_state": "SP",

        "primary_payment_type":
            "boleto",

        "avg_seller_lat":
            -23.652366177840182,

        "avg_seller_lng":
            -46.75575337195744,

        "customer_lat":
            -23.609430024757696,

        "customer_lng":
            -46.66050227039207,

        "order_approved_at":
            "2018-06-22T02:59:29",

        "order_estimated_delivery_date":
            "2018-07-04T00:00:00",

        "order_purchase_timestamp":
            "2018-06-21T08:41:07",
    }


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


def test_predict_single_order(
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

    prediction_mock = Mock(
        return_value=pd.DataFrame(
            {
                "order_id": [
                    (
                        "b3b54427f53d13f6063ef700"
                        "7bf7d371"
                    )
                ],
                "late_probability": [
                    0.02952043625959755
                ],
                "predicted_is_late": [
                    0
                ],
            }
        )
    )

    monkeypatch.setattr(
        api,
        "predict_orders_with_logging",
        prediction_mock,
    )

    response = client.post(
        "/predict",
        json=(
            make_valid_order_payload()
        ),
    )

    assert (
        response.status_code
        == 200
    )

    assert response.json() == {
        "order_id":
            "b3b54427f53d13f6063ef7007bf7d371",

        "predicted_is_late":
            0,

        "late_probability":
            0.02952043625959755,

        "model_version":
            "1",
    }

    prediction_mock.assert_called_once()

    call_args = (
        prediction_mock
        .call_args
    )

    raw_order = (
        call_args
        .args[
            0
        ]
    )

    assert isinstance(
        raw_order,
        pd.DataFrame,
    )

    assert (
        raw_order.shape
        == (
            1,
            31,
        )
    )

    assert (
        raw_order.iloc[
            0
        ][
            "order_id"
        ]
        == (
            "b3b54427f53d13f6063ef700"
            "7bf7d371"
        )
    )

    assert (
        call_args
        .kwargs[
            "runtime_model"
        ]
        is runtime_model
    )


def test_predict_rejects_missing_required_field():
    payload = (
        make_valid_order_payload()
    )

    del payload[
        "payment_total"
    ]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )


def test_predict_rejects_extra_field():
    payload = (
        make_valid_order_payload()
    )

    payload[
        "future_delivery_result"
    ] = "leakage"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )


def test_predict_rejects_invalid_timestamp():
    payload = (
        make_valid_order_payload()
    )

    payload[
        "order_purchase_timestamp"
    ] = "not-a-timestamp"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )


def test_predict_returns_503_when_registry_unavailable(
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

    response = client.post(
        "/predict",
        json=(
            make_valid_order_payload()
        ),
    )

    assert (
        response.status_code
        == 503
    )

    assert response.json() == {
        "detail":
            "Production model is unavailable."
    }


def test_openapi_contains_prediction_route():
    response = client.get(
        "/openapi.json"
    )

    assert (
        response.status_code
        == 200
    )

    schema = response.json()

    paths = schema[
        "paths"
    ]

    assert (
        "/health"
        in paths
    )

    assert (
        "/model-info"
        in paths
    )

    assert (
        "/predict"
        in paths
    )

    assert (
        "post"
        in paths[
            "/predict"
        ]
    )

    assert (
        "OrderPredictionRequest"
        in schema[
            "components"
        ][
            "schemas"
        ]
    )

    assert (
        "PredictionResponse"
        in schema[
            "components"
        ][
            "schemas"
        ]
    )
