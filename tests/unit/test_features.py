import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from olist_ml.features import (
    ENGINEERED_FEATURES,
    engineer_features,
    select_model_features,
)


def make_order_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_purchase_timestamp": ["2017-09-07 10:30:00"],
            "order_approved_at": ["2017-09-07 12:30:00"],
            "order_estimated_delivery_date": ["2017-09-17 10:30:00"],
            "customer_state": ["SP"],
            "primary_seller_state": ["SP"],
            "customer_lat": [0.0],
            "customer_lng": [0.0],
            "avg_seller_lat": [0.0],
            "avg_seller_lng": [1.0],
        }
    )


def test_engineer_features_creates_expected_features():
    data = make_order_dataframe()

    result = engineer_features(data)

    for feature in ENGINEERED_FEATURES:
        assert feature in result.columns


def test_engineer_features_calculates_time_features():
    data = make_order_dataframe()

    result = engineer_features(data)

    assert (
        result.loc[
            0,
            "purchase_month",
        ]
        == "9"
    )

    assert (
        result.loc[
            0,
            "purchase_weekday",
        ]
        == "Thursday"
    )

    assert (
        result.loc[
            0,
            "purchase_hour",
        ]
        == 10
    )

    assert result.loc[
        0,
        "estimated_delivery_days",
    ] == pytest.approx(10.0)

    assert result.loc[
        0,
        "approval_hours",
    ] == pytest.approx(2.0)


def test_engineer_features_detects_fixed_holiday():
    data = make_order_dataframe()

    result = engineer_features(data)

    assert (
        result.loc[
            0,
            "is_fixed_national_holiday",
        ]
        == 1
    )


def test_engineer_features_detects_same_state():
    data = make_order_dataframe()

    result = engineer_features(data)

    assert (
        result.loc[
            0,
            "same_customer_seller_state",
        ]
        == 1
    )


def test_engineer_features_calculates_haversine_distance():
    data = make_order_dataframe()

    result = engineer_features(data)

    distance = result.loc[
        0,
        "customer_seller_distance_km",
    ]

    assert distance == pytest.approx(
        111.1949,
        rel=1e-4,
    )


def test_engineer_features_preserves_input_dataframe():
    data = make_order_dataframe()
    original = data.copy(deep=True)

    engineer_features(data)

    assert_frame_equal(
        data,
        original,
    )


def test_engineer_features_preserves_missing_state():
    data = make_order_dataframe()

    data.loc[
        0,
        "primary_seller_state",
    ] = np.nan

    result = engineer_features(data)

    assert pd.isna(
        result.loc[
            0,
            "same_customer_seller_state",
        ]
    )


def test_engineer_features_preserves_missing_coordinates():
    data = make_order_dataframe()

    data.loc[
        0,
        "customer_lat",
    ] = np.nan

    result = engineer_features(data)

    assert pd.isna(
        result.loc[
            0,
            "customer_seller_distance_km",
        ]
    )


def test_engineer_features_rejects_missing_required_column():
    data = make_order_dataframe()

    data = data.drop(columns=["avg_seller_lng"])

    with pytest.raises(
        ValueError,
        match="avg_seller_lng",
    ):
        engineer_features(data)


def test_select_model_features_preserves_requested_order():
    data = pd.DataFrame(
        {
            "category": ["A"],
            "number_b": [2],
            "number_a": [1],
        }
    )

    result = select_model_features(
        data,
        numeric_features=[
            "number_a",
            "number_b",
        ],
        categorical_features=[
            "category",
        ],
    )

    assert list(result.columns) == [
        "number_a",
        "number_b",
        "category",
    ]


def test_select_model_features_rejects_missing_feature():
    data = pd.DataFrame(
        {
            "feature_a": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="feature_b",
    ):
        select_model_features(
            data,
            numeric_features=[
                "feature_a",
                "feature_b",
            ],
            categorical_features=[],
        )
