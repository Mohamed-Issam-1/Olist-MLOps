from __future__ import annotations

from fastapi import (
    FastAPI,
    HTTPException,
    status,
)

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
)
from olist_ml.config import (
    load_config,
)
from olist_ml.mlflow_loader import (
    MlflowModelLoadError,
    load_registered_inference_model,
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

    try:
        runtime_model = (
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
