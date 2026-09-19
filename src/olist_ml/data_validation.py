from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

from olist_ml.config import (
    find_project_root,
    resolve_project_path,
)

DATA_QUALITY_CONFIG_PATH = "config/data_quality.json"


class DataValidationError(RuntimeError):
    """Raised when production data validation fails."""


def load_data_quality_config() -> dict[str, Any]:
    """
    Load and validate the data-quality configuration.
    """

    path = resolve_project_path(DATA_QUALITY_CONFIG_PATH)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    dataset_config = config.get("labeled_dataset")

    if not isinstance(
        dataset_config,
        dict,
    ):
        raise DataValidationError("Missing labeled_dataset data-quality configuration.")

    required_keys = {
        "path",
        "data_source_name",
        "data_asset_name",
        "batch_definition_name",
        "suite_name",
        "validation_definition_name",
        "row_count",
        "columns",
        "strict_not_null_columns",
        "mostly_not_null",
        "allowed_order_status",
        "allowed_is_late",
        "allowed_customer_states",
        "numeric_ranges",
    }

    missing = required_keys - set(dataset_config)

    if missing:
        raise DataValidationError(
            "Data-quality configuration is missing: " + ", ".join(sorted(missing))
        )

    return dataset_config


def get_gx_context():
    """
    Return the project's persistent Great Expectations context.
    """

    project_root = find_project_root()

    return gx.get_context(
        mode="file",
        project_root_dir=str(project_root),
    )


def get_or_create_batch_definition(
    context,
    config: dict[str, Any],
):
    """
    Create or retrieve the pandas runtime data components.
    """

    data_source_name = config["data_source_name"]

    try:
        data_source = context.data_sources.get(data_source_name)
    except (
        KeyError,
        LookupError,
    ):
        data_source = context.data_sources.add_pandas(name=data_source_name)

    asset_name = config["data_asset_name"]

    try:
        data_asset = data_source.get_asset(asset_name)
    except (
        KeyError,
        LookupError,
    ):
        data_asset = data_source.add_dataframe_asset(name=asset_name)

    batch_definition_name = config["batch_definition_name"]

    try:
        batch_definition = data_asset.get_batch_definition(batch_definition_name)
    except (
        KeyError,
        LookupError,
    ):
        batch_definition = data_asset.add_batch_definition_whole_dataframe(
            batch_definition_name
        )

    return batch_definition


def build_expectation_suite(
    context,
    config: dict[str, Any],
):
    """
    Build and persist the labeled-order Expectation Suite.
    """

    suite = gx.ExpectationSuite(name=config["suite_name"])

    row_count = config["row_count"]

    suite.add_expectation(
        gxe.ExpectTableRowCountToBeBetween(
            min_value=row_count["min"],
            max_value=row_count["max"],
        )
    )

    suite.add_expectation(
        gxe.ExpectTableColumnCountToEqual(value=len(config["columns"]))
    )

    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(
            column_set=config["columns"],
            exact_match=True,
        )
    )

    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="order_id"))

    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="order_id"))

    for column in config["strict_not_null_columns"]:
        if column == "order_id":
            continue

        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column))

    for (
        column,
        mostly,
    ) in config["mostly_not_null"].items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(
                column=column,
                mostly=float(mostly),
            )
        )

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="order_status",
            value_set=config["allowed_order_status"],
        )
    )

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="is_late",
            value_set=config["allowed_is_late"],
        )
    )

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="customer_state",
            value_set=config["allowed_customer_states"],
        )
    )

    for (
        column,
        limits,
    ) in config["numeric_ranges"].items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=limits["min"],
                max_value=limits["max"],
            )
        )

    return context.suites.add_or_update(suite)


def validate_target_consistency(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verify that is_late agrees exactly with delay_days.

    A positive delay means late; zero or negative means on-time.
    """

    required = {
        "delay_days",
        "is_late",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise DataValidationError(
            "Target consistency check is missing columns: " + ", ".join(sorted(missing))
        )

    expected_target = dataframe["delay_days"].gt(0).astype(int)

    actual_target = dataframe["is_late"].astype(int)

    mismatch_count = int((expected_target != actual_target).sum())

    if mismatch_count:
        raise DataValidationError(
            "Target consistency validation failed: "
            f"{mismatch_count} rows disagree between "
            "delay_days and is_late."
        )


def configure_labeled_validation():
    """
    Persist GX suite and validation definition.
    """

    config = load_data_quality_config()

    context = get_gx_context()

    batch_definition = get_or_create_batch_definition(
        context,
        config,
    )

    suite = build_expectation_suite(
        context,
        config,
    )

    validation_definition = gx.ValidationDefinition(
        name=config["validation_definition_name"],
        data=batch_definition,
        suite=suite,
    )

    validation_definition = context.validation_definitions.add_or_update(
        validation_definition
    )

    return (
        context,
        validation_definition,
    )


def validate_labeled_dataframe(
    dataframe: pd.DataFrame,
    *,
    raise_on_failure: bool = True,
):
    """
    Validate a labeled Olist dataframe with GX and semantic checks.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise DataValidationError("Labeled data must be a pandas DataFrame.")

    if dataframe.empty:
        raise DataValidationError("Labeled data must not be empty.")

    validate_target_consistency(dataframe)

    (
        _context,
        validation_definition,
    ) = configure_labeled_validation()

    result = validation_definition.run(batch_parameters={"dataframe": dataframe})

    if not result.success and raise_on_failure:
        statistics = result.statistics or {}

        raise DataValidationError(
            f"Great Expectations validation failed. Statistics: {statistics}"
        )

    return result


def validate_labeled_dataset(
    *,
    raise_on_failure: bool = True,
):
    """
    Load and validate the configured labeled dataset.
    """

    config = load_data_quality_config()

    dataset_path = resolve_project_path(config["path"])

    if not Path(dataset_path).exists():
        raise DataValidationError(f"Labeled dataset does not exist: {dataset_path}")

    dataframe = pd.read_parquet(dataset_path)

    return validate_labeled_dataframe(
        dataframe,
        raise_on_failure=raise_on_failure,
    )
