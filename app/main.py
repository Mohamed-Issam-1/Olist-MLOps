from __future__ import annotations

import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    status,
)

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    OrderPredictionRequest,
    PredictionResponse,
)
from olist_ml.config import (
    load_config,
)
from olist_ml.mlflow_loader import (
    MlflowModelLoadError,
    load_registered_inference_model,
)
from olist_ml.prediction_service import (
    PredictionServiceError,
    predict_orders_with_logging,
)


def _load_service_config() -> dict:
    config = load_config()

    service_config = config.get(
        "service"
    )

    if not isinstance(
        service_config,
        dict,
    ):
        raise RuntimeError(
            "Missing service configuration."
        )

    return service_config


SERVICE_CONFIG = (
    _load_service_config()
)


app = FastAPI(
    title=str(
        SERVICE_CONFIG[
            "name"
        ]
    ),
    version=str(
        SERVICE_CONFIG[
            "api_version"
        ]
    ),
    description=(
        "Production inference API for predicting "
        "late Olist deliveries."
    ),
)


def _load_production_model():
    """
    Load the currently configured MLflow production model
    and translate registry availability failures into HTTP 503.
    """

    try:
        return (
            load_registered_inference_model()
        )

    except MlflowModelLoadError as exc:
        raise HTTPException(
            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Production model is unavailable."
            ),
        ) from exc


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=[
        "Service",
    ],
)
def health_check() -> HealthResponse:
    """
    Lightweight liveness endpoint.

    It intentionally does not load the ML model.
    """

    return HealthResponse(
        status="ok",
        service=str(
            SERVICE_CONFIG[
                "name"
            ]
        ),
        api_version=str(
            SERVICE_CONFIG[
                "api_version"
            ]
        ),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=[
        "Model",
    ],
)
def model_info() -> ModelInfoResponse:
    """
    Return metadata for the currently resolved
    MLflow production model.
    """

    runtime_model = (
        _load_production_model()
    )

    return ModelInfoResponse(
        source=runtime_model.source,
        registered_model_name=(
            runtime_model
            .registered_model_name
        ),
        alias=runtime_model.alias,
        version=runtime_model.version,
        run_id=runtime_model.run_id,
        model_uri=runtime_model.model_uri,
        model_type=runtime_model.model_type,
        classification_threshold=(
            runtime_model
            .classification_threshold
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=[
        "Prediction",
    ],
    summary=(
        "Predict whether one order will be delivered late"
    ),
)
def predict_order(
    request: OrderPredictionRequest,
) -> PredictionResponse:
    """
    Predict late-delivery probability for one order.

    The fitted production model is loaded from the configured
    MLflow Registry alias. No fitting or retraining occurs.
    """

    runtime_model = (
        _load_production_model()
    )

    raw_order = pd.DataFrame(
        [
            request.model_dump()
        ]
    )

    try:
        predictions = (
            predict_orders_with_logging(
                raw_order,
                runtime_model=(
                    runtime_model
                ),
            )
        )

    except (
        PredictionServiceError,
        ValueError,
        TypeError,
    ) as exc:
        raise HTTPException(
            status_code=(
                status
                .HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail=(
                "Prediction input could not be processed."
            ),
        ) from exc

    if len(
        predictions
    ) != 1:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Prediction service returned an "
                "unexpected number of results."
            ),
        )

    prediction = (
        predictions.iloc[
            0
        ]
    )

    return PredictionResponse(
        order_id=str(
            prediction[
                "order_id"
            ]
        ),
        predicted_is_late=int(
            prediction[
                "predicted_is_late"
            ]
        ),
        late_probability=float(
            prediction[
                "late_probability"
            ]
        ),
        model_version=(
            runtime_model.version
        ),
    )
