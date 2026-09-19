from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(description=("Current service health status."))

    service: str = Field(description=("Configured API service name."))

    api_version: str = Field(description=("API contract version."))


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str

    registered_model_name: str

    alias: str

    version: str

    run_id: str

    model_uri: str

    model_type: str

    classification_threshold: float = Field(
        gt=0.0,
        lt=1.0,
    )


class OrderPredictionRequest(BaseModel):
    """
    One order at the production prediction point.

    Every source column required by the fitted Task 2
    inference pipeline must be present. Nullable feature
    values are allowed because the fitted preprocessor
    handles missing values without refitting.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "order_id": "b3b54427f53d13f6063ef7007bf7d371",
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
                "primary_product_category": "watches_gifts",
                "primary_seller_state": "SP",
                "primary_payment_type": "boleto",
                "avg_seller_lat": -23.652366177840182,
                "avg_seller_lng": -46.75575337195744,
                "customer_lat": -23.609430024757696,
                "customer_lng": -46.66050227039207,
                "order_approved_at": "2018-06-22T02:59:29",
                "order_estimated_delivery_date": "2018-07-04T00:00:00",
                "order_purchase_timestamp": "2018-06-21T08:41:07",
            }
        },
    )

    order_id: str = Field(min_length=1)

    item_count: float | None

    unique_products: float | None

    unique_sellers: float | None

    total_item_price: float | None

    avg_item_price: float | None

    total_freight_value: float | None

    avg_freight_value: float | None

    unique_product_categories: float | None

    avg_product_weight_g: float | None

    max_product_weight_g: float | None

    avg_product_length_cm: float | None

    avg_product_height_cm: float | None

    avg_product_width_cm: float | None

    avg_product_photos_qty: float | None

    unique_seller_states: float | None

    payment_records: float | None

    payment_types_count: float | None

    payment_total: float | None

    payment_installments_max: float | None

    customer_state: str | None

    primary_product_category: str | None

    primary_seller_state: str | None

    primary_payment_type: str | None

    avg_seller_lat: float | None

    avg_seller_lng: float | None

    customer_lat: float | None

    customer_lng: float | None

    order_approved_at: datetime | None

    order_estimated_delivery_date: datetime | None

    order_purchase_timestamp: datetime | None


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(description=("Order identifier supplied in the request."))

    predicted_is_late: Literal[
        0,
        1,
    ] = Field(description=("1 means predicted late; 0 means predicted on time."))

    late_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=("Probability assigned to the late class."),
    )

    model_version: str = Field(description=("Resolved MLflow Registry model version."))


class BatchPredictionRequest(BaseModel):
    """
    One or more orders submitted for batch inference.

    The maximum accepted batch size is controlled by
    service.max_batch_size in the project configuration.
    """

    model_config = ConfigDict(extra="forbid")

    orders: list[OrderPredictionRequest] = Field(
        min_length=1,
        description=("Orders to score in one inference request."),
    )


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(
        ge=1,
        description=("Number of predictions returned."),
    )

    model_version: str = Field(description=("Resolved MLflow Registry model version."))

    predictions: list[PredictionResponse] = Field(min_length=1)


class MonitoringRequestMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)

    average_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
    )

    p95_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
    )

    max_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
    )


class MonitoringPredictionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_count: int = Field(ge=0)
    predicted_late_count: int = Field(ge=0)
    predicted_on_time_count: int = Field(ge=0)

    predicted_late_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    mean_late_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class MonitoringDriftMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "insufficient_data",
        "stable",
        "drift_detected",
    ]

    minimum_predictions: int = Field(ge=1)
    baseline_prediction_count: int = Field(ge=1)

    baseline_predicted_late_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    baseline_mean_late_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    predicted_late_rate_absolute_delta: float | None = Field(
        default=None,
        ge=0.0,
    )

    mean_late_probability_absolute_delta: float | None = Field(
        default=None,
        ge=0.0,
    )

    predicted_late_rate_threshold: float = Field(
        ge=0.0,
    )

    mean_late_probability_threshold: float = Field(
        ge=0.0,
    )


class MonitoringInvalidLogRecords(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: int = Field(ge=0)
    predictions: int = Field(ge=0)


class MonitoringResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "ok",
        "alert",
    ]

    requests: MonitoringRequestMetrics
    predictions: MonitoringPredictionMetrics
    drift: MonitoringDriftMetrics
    alerts: list[str]
    invalid_log_records: MonitoringInvalidLogRecords
