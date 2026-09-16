from __future__ import annotations

from time import perf_counter

import pandas as pd

from olist_ml.artifacts import (
    InferenceArtifacts,
)
from olist_ml.features import (
    ENGINEERED_FEATURES,
    REQUIRED_ENGINEERING_COLUMNS,
)
from olist_ml.logging_config import (
    get_logger,
)
from olist_ml.mlflow_loader import (
    RegisteredInferenceModel,
    load_registered_inference_model,
)
from olist_ml.prediction_logging import (
    log_prediction_batch,
)


class PredictionServiceError(RuntimeError):
    """Raised when prediction service input is invalid."""


def get_prediction_input_columns(
    raw_orders: pd.DataFrame,
    artifacts: InferenceArtifacts,
) -> list[str]:
    """
    Return only source columns that are valid at the
    configured prediction point.

    Leakage, target, review, and post-delivery columns
    are intentionally excluded.
    """

    feature_config = (
        artifacts.feature_config
    )

    configured_features = [
        *feature_config[
            "numeric_features"
        ],
        *feature_config[
            "categorical_features"
        ],
    ]

    direct_features = [
        feature
        for feature in configured_features
        if feature
        not in ENGINEERED_FEATURES
    ]

    required_columns = list(
        direct_features
    )

    for column in sorted(
        REQUIRED_ENGINEERING_COLUMNS
    ):
        if (
            column
            not in required_columns
        ):
            required_columns.append(
                column
            )

    missing_columns = [
        column
        for column in required_columns
        if column
        not in raw_orders.columns
    ]

    if missing_columns:
        raise PredictionServiceError(
            "Prediction input is missing required "
            "source columns: "
            + ", ".join(
                missing_columns
            )
        )

    if (
        "order_id"
        in raw_orders.columns
    ):
        return [
            "order_id",
            *required_columns,
        ]

    return required_columns


def predict_orders_with_logging(
    raw_orders: pd.DataFrame,
    runtime_model: RegisteredInferenceModel | None = None,
) -> pd.DataFrame:
    """
    Execute production inference through MLflow Registry
    with application and prediction logging.

    The production model is resolved from the configured
    MLflow registry alias unless explicitly injected for
    testing.
    """

    logger = get_logger(
        "prediction_service"
    )

    model_source = (
        "mlflow-registry"
    )

    model_version = (
        "unresolved"
    )

    start_time = perf_counter()

    try:
        runtime_model = (
            runtime_model
            or load_registered_inference_model()
        )

        model_source = (
            runtime_model.source
        )

        model_version = (
            runtime_model.version
        )

        predictions = (
            runtime_model.predict(
                raw_orders
            )
        )

        latency_ms = (
            perf_counter()
            - start_time
        ) * 1000

        input_columns = (
            get_prediction_input_columns(
                raw_orders,
                runtime_model.artifacts,
            )
        )

        request_id = (
            log_prediction_batch(
                raw_orders,
                predictions,
                input_columns=input_columns,
                latency_ms=latency_ms,
                model_type=(
                    runtime_model
                    .model_type
                ),
                model_version=(
                    runtime_model
                    .version
                ),
                threshold=(
                    runtime_model
                    .classification_threshold
                ),
            )
        )

        late_prediction_count = int(
            predictions[
                "predicted_is_late"
            ].sum()
        )

        logger.info(
            (
                "prediction_request_completed "
                "request_id=%s "
                "request_size=%d "
                "late_predictions=%d "
                "latency_ms=%.3f "
                "model_type=%s "
                "model_source=%s "
                "model_version=%s "
                "model_alias=%s "
                "model_uri=%s"
            ),
            request_id,
            len(
                predictions
            ),
            late_prediction_count,
            latency_ms,
            runtime_model.model_type,
            model_source,
            model_version,
            runtime_model.alias,
            runtime_model.model_uri,
        )

        return predictions

    except Exception:
        latency_ms = (
            perf_counter()
            - start_time
        ) * 1000

        request_size = (
            len(
                raw_orders
            )
            if hasattr(
                raw_orders,
                "__len__",
            )
            else -1
        )

        logger.exception(
            (
                "prediction_request_failed "
                "request_size=%d "
                "latency_ms=%.3f "
                "model_source=%s "
                "model_version=%s"
            ),
            request_size,
            latency_ms,
            model_source,
            model_version,
        )

        raise
