from __future__ import annotations

from time import perf_counter

import pandas as pd

from olist_ml.artifacts import (
    InferenceArtifacts,
    load_inference_artifacts,
)
from olist_ml.config import load_config
from olist_ml.features import (
    ENGINEERED_FEATURES,
    REQUIRED_ENGINEERING_COLUMNS,
)
from olist_ml.inference import predict_orders
from olist_ml.logging_config import get_logger
from olist_ml.prediction_logging import (
    log_prediction_batch,
)


class PredictionServiceError(RuntimeError):
    """Raised when prediction service configuration is invalid."""


def get_model_runtime_metadata() -> tuple[str, str]:
    """
    Return configured model source and model version.
    """

    config = load_config()

    model_config = config.get(
        "model"
    )

    if not isinstance(
        model_config,
        dict,
    ):
        raise PredictionServiceError(
            "Missing model configuration."
        )

    source = model_config.get(
        "source"
    )

    version = model_config.get(
        "version"
    )

    if not source:
        raise PredictionServiceError(
            "Model source is not configured."
        )

    if not version:
        raise PredictionServiceError(
            "Model version is not configured."
        )

    return (
        str(source),
        str(version),
    )


def get_prediction_input_columns(
    raw_orders: pd.DataFrame,
    artifacts: InferenceArtifacts,
) -> list[str]:
    """
    Return only the source columns that are valid at the
    configured prediction point.

    Leakage, target, review, and post-delivery columns are
    intentionally excluded.
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
    artifacts: InferenceArtifacts | None = None,
) -> pd.DataFrame:
    """
    Execute production inference with application and
    prediction logging.
    """

    logger = get_logger(
        "prediction_service"
    )

    model_source, model_version = (
        get_model_runtime_metadata()
    )

    artifacts = (
        artifacts
        or load_inference_artifacts()
    )

    start_time = perf_counter()

    try:
        predictions = predict_orders(
            raw_orders,
            artifacts,
        )

        latency_ms = (
            perf_counter()
            - start_time
        ) * 1000

        input_columns = (
            get_prediction_input_columns(
                raw_orders,
                artifacts,
            )
        )

        request_id = (
            log_prediction_batch(
                raw_orders,
                predictions,
                input_columns=input_columns,
                latency_ms=latency_ms,
                model_type=type(
                    artifacts.model
                ).__name__,
                model_version=model_version,
                threshold=(
                    artifacts
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
                "model_version=%s"
            ),
            request_id,
            len(
                predictions
            ),
            late_prediction_count,
            latency_ms,
            type(
                artifacts.model
            ).__name__,
            model_source,
            model_version,
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
