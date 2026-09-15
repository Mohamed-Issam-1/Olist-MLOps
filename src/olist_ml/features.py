from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


FIXED_HOLIDAY_MONTH_DAYS = {
    (1, 1),
    (4, 21),
    (5, 1),
    (9, 7),
    (10, 12),
    (11, 2),
    (11, 15),
    (12, 25),
}


REQUIRED_ENGINEERING_COLUMNS = {
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "customer_state",
    "primary_seller_state",
    "customer_lat",
    "customer_lng",
    "avg_seller_lat",
    "avg_seller_lng",
}


ENGINEERED_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "estimated_delivery_days",
    "approval_hours",
    "is_fixed_national_holiday",
    "same_customer_seller_state",
    "customer_seller_distance_km",
]


def _validate_engineering_columns(df: pd.DataFrame) -> None:
    """
    Check that all columns required for deterministic feature
    engineering are available.
    """

    missing_columns = (
        REQUIRED_ENGINEERING_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns required for feature engineering: "
            + ", ".join(sorted(missing_columns))
        )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the deterministic feature-engineering logic from
    Notebook 05.

    The input DataFrame is copied before modification, so the
    caller's original data is not changed.
    """

    _validate_engineering_columns(df)

    data = df.copy()

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        data[column] = pd.to_datetime(
            data[column],
            errors="coerce",
        )

    purchase_time = data[
        "order_purchase_timestamp"
    ]

    # -------------------------
    # Time features
    # -------------------------

    data[
        "purchase_month"
    ] = (
        purchase_time
        .dt.month
        .astype("Int64")
        .astype("string")
    )

    data[
        "purchase_weekday"
    ] = (
        purchase_time
        .dt.day_name()
        .astype("string")
    )

    data[
        "purchase_hour"
    ] = purchase_time.dt.hour

    data[
        "estimated_delivery_days"
    ] = (
        data[
            "order_estimated_delivery_date"
        ]
        - purchase_time
    ).dt.total_seconds() / 86400

    data[
        "approval_hours"
    ] = (
        data[
            "order_approved_at"
        ]
        - purchase_time
    ).dt.total_seconds() / 3600

    # -------------------------
    # Fixed national holiday
    # -------------------------

    month_day = list(
        zip(
            purchase_time.dt.month,
            purchase_time.dt.day,
        )
    )

    data[
        "is_fixed_national_holiday"
    ] = [
        int(
            value
            in FIXED_HOLIDAY_MONTH_DAYS
        )
        if not (
            pd.isna(value[0])
            or pd.isna(value[1])
        )
        else np.nan
        for value in month_day
    ]

    # -------------------------
    # Same customer/seller state
    # -------------------------

    valid_states = (
        data[
            "customer_state"
        ].notna()
        &
        data[
            "primary_seller_state"
        ].notna()
    )

    data[
        "same_customer_seller_state"
    ] = np.nan

    data.loc[
        valid_states,
        "same_customer_seller_state",
    ] = (
        data.loc[
            valid_states,
            "customer_state",
        ]
        ==
        data.loc[
            valid_states,
            "primary_seller_state",
        ]
    ).astype(int)

    # -------------------------
    # Customer-seller distance
    # -------------------------

    customer_lat = np.radians(
        data[
            "customer_lat"
        ]
    )

    customer_lng = np.radians(
        data[
            "customer_lng"
        ]
    )

    seller_lat = np.radians(
        data[
            "avg_seller_lat"
        ]
    )

    seller_lng = np.radians(
        data[
            "avg_seller_lng"
        ]
    )

    delta_lat = (
        seller_lat
        - customer_lat
    )

    delta_lng = (
        seller_lng
        - customer_lng
    )

    a = (
        np.sin(
            delta_lat / 2
        ) ** 2
        +
        np.cos(
            customer_lat
        )
        *
        np.cos(
            seller_lat
        )
        *
        np.sin(
            delta_lng / 2
        ) ** 2
    )

    c = (
        2
        * np.arcsin(
            np.sqrt(a)
        )
    )

    data[
        "customer_seller_distance_km"
    ] = (
        6371.0 * c
    )

    return data


def select_model_features(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """
    Select model input columns in the exact order defined by
    the saved feature configuration.
    """

    final_features = [
        *numeric_features,
        *categorical_features,
    ]

    missing_columns = [
        column
        for column in final_features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot select model features. "
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    return df.loc[
        :,
        final_features,
    ].copy()
