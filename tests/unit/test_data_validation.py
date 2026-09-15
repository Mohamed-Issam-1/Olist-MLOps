import pandas as pd
import pytest

import olist_ml.data_validation as validation


def test_load_data_quality_config():
    config = (
        validation
        .load_data_quality_config()
    )

    assert (
        config[
            "suite_name"
        ]
        == "olist_labeled_data_quality"
    )

    assert len(
        config[
            "columns"
        ]
    ) == 46

    assert (
        config[
            "allowed_is_late"
        ]
        == [
            0,
            1,
        ]
    )


def test_target_consistency_accepts_valid_data():
    dataframe = pd.DataFrame(
        {
            "delay_days": [
                -5,
                0,
                1,
                10,
            ],
            "is_late": [
                0,
                0,
                1,
                1,
            ],
        }
    )

    validation.validate_target_consistency(
        dataframe
    )


def test_target_consistency_rejects_mismatch():
    dataframe = pd.DataFrame(
        {
            "delay_days": [
                -5,
                3,
            ],
            "is_late": [
                0,
                0,
            ],
        }
    )

    with pytest.raises(
        validation.DataValidationError,
        match="1 rows disagree",
    ):
        validation.validate_target_consistency(
            dataframe
        )


def test_target_consistency_rejects_missing_columns():
    dataframe = pd.DataFrame(
        {
            "is_late": [
                0,
            ]
        }
    )

    with pytest.raises(
        validation.DataValidationError,
        match="delay_days",
    ):
        validation.validate_target_consistency(
            dataframe
        )


def test_validate_labeled_dataframe_rejects_empty():
    with pytest.raises(
        validation.DataValidationError,
        match="must not be empty",
    ):
        validation.validate_labeled_dataframe(
            pd.DataFrame()
        )
