from __future__ import annotations

import json

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import olist_ml.cli as cli


def test_load_orders_reads_single_json(
    tmp_path,
):
    input_path = (
        tmp_path
        / "order.json"
    )

    input_path.write_text(
        json.dumps(
            {
                "order_id": "order-1",
                "price": 100.0,
            }
        ),
        encoding="utf-8",
    )

    orders = cli.load_orders(
        input_path
    )

    assert len(
        orders
    ) == 1

    assert (
        orders.loc[
            0,
            "order_id",
        ]
        == "order-1"
    )


def test_load_orders_reads_csv(
    tmp_path,
):
    input_path = (
        tmp_path
        / "orders.csv"
    )

    pd.DataFrame(
        [
            {
                "order_id": "order-1",
                "price": 100.0,
            },
            {
                "order_id": "order-2",
                "price": 200.0,
            },
        ]
    ).to_csv(
        input_path,
        index=False,
    )

    orders = cli.load_orders(
        input_path
    )

    assert len(
        orders
    ) == 2


def test_load_orders_rejects_unknown_format(
    tmp_path,
):
    input_path = (
        tmp_path
        / "orders.txt"
    )

    input_path.write_text(
        "invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        cli.CliError,
        match="Unsupported input format",
    ):
        cli.load_orders(
            input_path
        )


def test_predict_file_uses_registry_model(
    monkeypatch,
    tmp_path,
):
    input_path = (
        tmp_path
        / "order.json"
    )

    input_path.write_text(
        json.dumps(
            {
                "order_id": "order-1",
            }
        ),
        encoding="utf-8",
    )

    runtime_model = SimpleNamespace(
        version="7",
    )

    monkeypatch.setattr(
        cli,
        "load_registered_inference_model",
        lambda: runtime_model,
    )

    def fake_predict(
        raw_orders,
        runtime_model,
    ):
        assert len(
            raw_orders
        ) == 1

        assert (
            runtime_model.version
            == "7"
        )

        return pd.DataFrame(
            [
                {
                    "order_id": "order-1",
                    "late_probability": 0.75,
                    "predicted_is_late": 1,
                }
            ]
        )

    monkeypatch.setattr(
        cli,
        "predict_orders_with_logging",
        fake_predict,
    )

    result = cli.predict_file(
        input_path
    )

    assert (
        result.loc[
            0,
            "model_version",
        ]
        == "7"
    )

    assert (
        result.loc[
            0,
            "predicted_is_late",
        ]
        == 1
    )


def test_main_writes_json_to_stdout(
    monkeypatch,
    capsys,
    tmp_path,
):
    input_path = (
        tmp_path
        / "order.json"
    )

    input_path.write_text(
        "{}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli,
        "predict_file",
        lambda _path: pd.DataFrame(
            [
                {
                    "order_id": "order-1",
                    "late_probability": 0.25,
                    "predicted_is_late": 0,
                    "model_version": "3",
                }
            ]
        ),
    )

    exit_code = cli.main(
        [
            "--input",
            str(
                input_path
            ),
        ]
    )

    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert exit_code == 0

    assert (
        payload[
            0
        ][
            "order_id"
        ]
        == "order-1"
    )

    assert (
        payload[
            0
        ][
            "model_version"
        ]
        == "3"
    )


def test_main_returns_error_for_failure(
    monkeypatch,
    capsys,
    tmp_path,
):
    input_path = (
        tmp_path
        / "order.json"
    )

    input_path.write_text(
        "{}",
        encoding="utf-8",
    )

    def fail_prediction(
        _path: Path,
    ):
        raise cli.CliError(
            "bad input"
        )

    monkeypatch.setattr(
        cli,
        "predict_file",
        fail_prediction,
    )

    exit_code = cli.main(
        [
            "--input",
            str(
                input_path
            ),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1

    assert (
        "Prediction failed: bad input"
        in captured.err
    )
