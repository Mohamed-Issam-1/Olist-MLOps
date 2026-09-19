from types import SimpleNamespace

import pytest

from olist_ml.mlflow_registry import (
    MlflowRegistryError,
    _build_metric_payload,
    _load_runtime_requirements,
    _set_and_verify_alias,
)


def make_summary():
    return {
        "final_validation": {
            "average_precision": 0.11,
            "roc_auc": 0.75,
            "f1": 0.21,
            "precision": 0.14,
            "recall": 0.42,
            "balanced_accuracy": 0.65,
            "accuracy": 0.86,
        },
        "final_test": {
            "average_precision": 0.076,
            "roc_auc": 0.636,
            "f1": 0.106,
            "precision": 0.064,
            "recall": 0.308,
            "balanced_accuracy": 0.554,
            "accuracy": 0.779,
        },
    }


def test_build_metric_payload():
    metrics = _build_metric_payload(make_summary())

    assert metrics["validation_average_precision"] == pytest.approx(0.11)

    assert metrics["test_roc_auc"] == pytest.approx(0.636)

    assert len(metrics) == 14


def test_build_metric_payload_rejects_missing_section():
    summary = make_summary()

    del summary["final_test"]

    with pytest.raises(
        MlflowRegistryError,
        match="final_test",
    ):
        _build_metric_payload(summary)


def test_load_runtime_requirements(
    tmp_path,
):
    requirements_dir = tmp_path / "requirements"

    requirements_dir.mkdir()

    (requirements_dir / "runtime.txt").write_text(
        ("numpy==2.5.2\n\n# comment\nmlflow-skinny==3.16.0\n"),
        encoding="utf-8",
    )

    result = _load_runtime_requirements(tmp_path)

    assert result == [
        "numpy==2.5.2",
        "mlflow-skinny==3.16.0",
    ]


class FakeClient:
    def __init__(
        self,
        resolved_version,
    ):
        self.resolved_version = resolved_version

        self.alias_calls = []

    def set_registered_model_alias(
        self,
        name,
        alias,
        version,
    ):
        self.alias_calls.append(
            (
                name,
                alias,
                version,
            )
        )

    def get_model_version_by_alias(
        self,
        name,
        alias,
    ):
        return SimpleNamespace(version=(self.resolved_version))


def test_set_and_verify_alias():
    client = FakeClient("3")

    _set_and_verify_alias(
        client,
        model_name="olist",
        alias="champion",
        version="3",
    )

    assert client.alias_calls == [
        (
            "olist",
            "champion",
            "3",
        )
    ]


def test_set_and_verify_alias_rejects_mismatch():
    client = FakeClient("4")

    with pytest.raises(
        MlflowRegistryError,
        match="alias verification",
    ):
        _set_and_verify_alias(
            client,
            model_name="olist",
            alias="champion",
            version="3",
        )
