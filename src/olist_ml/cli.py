from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from typing import Sequence

import pandas as pd

from olist_ml.logging_config import (
    get_logger,
)
from olist_ml.mlflow_loader import (
    load_registered_inference_model,
)
from olist_ml.prediction_service import (
    predict_orders_with_logging,
)


LOGGER = get_logger(
    "cli"
)


class CliError(RuntimeError):
    """Raised when CLI input cannot be processed."""


def load_orders(
    input_path: Path,
) -> pd.DataFrame:
    """
    Load one or more raw orders from JSON or CSV.

    JSON accepts either:
    - one object
    - a list of objects
    """

    if not input_path.is_file():
        raise CliError(
            f"Input file does not exist: {input_path}"
        )

    suffix = input_path.suffix.lower()

    try:
        if suffix == ".csv":
            orders = pd.read_csv(
                input_path
            )

        elif suffix == ".json":
            payload = json.loads(
                input_path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(
                payload,
                dict,
            ):
                payload = [
                    payload
                ]

            if (
                not isinstance(
                    payload,
                    list,
                )
                or not payload
                or not all(
                    isinstance(
                        item,
                        dict,
                    )
                    for item in payload
                )
            ):
                raise CliError(
                    "JSON input must contain "
                    "one object or a non-empty "
                    "list of objects."
                )

            orders = pd.DataFrame(
                payload
            )

        else:
            raise CliError(
                "Unsupported input format. "
                "Use .json or .csv."
            )

    except CliError:
        raise

    except Exception as exc:
        raise CliError(
            f"Could not read input file: {input_path}"
        ) from exc

    if orders.empty:
        raise CliError(
            "Input file contains no orders."
        )

    return orders


def predict_file(
    input_path: Path,
) -> pd.DataFrame:
    """
    Run production inference for an input file.

    The model is resolved from MLflow Registry.
    No model fitting or retraining occurs.
    """

    raw_orders = load_orders(
        input_path
    )

    runtime_model = (
        load_registered_inference_model()
    )

    predictions = (
        predict_orders_with_logging(
            raw_orders,
            runtime_model=runtime_model,
        )
    )

    result = predictions.copy()

    result[
        "model_version"
    ] = runtime_model.version

    return result


def serialize_predictions(
    predictions: pd.DataFrame,
) -> str:
    """Serialize prediction rows as formatted JSON."""

    records = predictions.to_dict(
        orient="records"
    )

    return json.dumps(
        records,
        ensure_ascii=False,
        indent=2,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="olist-predict",
        description=(
            "Predict late Olist deliveries "
            "using the registered production model."
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Path to a .json or .csv "
            "file containing raw order data."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path for JSON prediction output. "
            "If omitted, results are written to stdout."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the production prediction CLI."""

    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    try:
        predictions = predict_file(
            args.input
        )

        payload = serialize_predictions(
            predictions
        )

        if args.output is None:
            sys.stdout.write(
                payload + "\n"
            )

        else:
            args.output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            args.output.write_text(
                payload + "\n",
                encoding="utf-8",
            )

            LOGGER.info(
                "cli_prediction_output_written "
                "path=%s rows=%d",
                args.output,
                len(
                    predictions
                ),
            )

        return 0

    except Exception as exc:
        LOGGER.exception(
            "cli_prediction_failed input=%s",
            args.input,
        )

        sys.stderr.write(
            f"Prediction failed: {exc}\n"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
