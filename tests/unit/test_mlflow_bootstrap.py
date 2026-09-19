from types import SimpleNamespace

import pytest
from mlflow.exceptions import (
    MlflowException,
)
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
)

import olist_ml.mlflow_bootstrap as bootstrap
from olist_ml.mlflow_config import (
    MlflowSettings,
)
from olist_ml.mlflow_registry import (
    RegisteredModelResult,
)


def make_settings(
    tmp_path,
):
    return MlflowSettings(
        backend_database=(tmp_path / "mlflow.db"),
        artifact_directory=(tmp_path / "mlartifacts"),
        experiment_name=("olist-late-delivery"),
        registered_model_name=("olist-late-delivery"),
        model_name=("inference_pipeline"),
        production_alias=("champion"),
    )


def missing_resource_error():
    return MlflowException(
        "Resource does not exist.",
        error_code=(RESOURCE_DOES_NOT_EXIST),
    )


class FakeClient:
    def __init__(
        self,
        *,
        registered_exists,
        alias_version=None,
    ):
        self.registered_exists = registered_exists

        self.alias_version = alias_version

    def get_registered_model(
        self,
        name,
    ):
        if not self.registered_exists:
            raise (missing_resource_error())

        return SimpleNamespace(name=name)

    def get_model_version_by_alias(
        self,
        name,
        alias,
    ):
        if self.alias_version is None:
            raise (missing_resource_error())

        return SimpleNamespace(
            name=name,
            version=(self.alias_version),
            aliases=[alias],
        )


def test_bootstrap_reuses_existing_alias(
    monkeypatch,
    tmp_path,
):
    settings = make_settings(tmp_path)

    client = FakeClient(
        registered_exists=True,
        alias_version="7",
    )

    monkeypatch.setattr(
        bootstrap,
        "configure_mlflow",
        lambda: settings,
    )

    monkeypatch.setattr(
        bootstrap,
        "get_mlflow_client",
        lambda _settings: client,
    )

    def unexpected_registration():
        raise AssertionError("Existing model must not be registered again.")

    monkeypatch.setattr(
        bootstrap,
        "register_task2_model",
        unexpected_registration,
    )

    result = bootstrap.ensure_production_model()

    assert result.registered_model_name == "olist-late-delivery"

    assert result.version == "7"

    assert result.alias == "champion"

    assert result.model_uri == ("models:/olist-late-delivery@champion")

    assert result.created is False


def test_bootstrap_registers_empty_registry(
    monkeypatch,
    tmp_path,
):
    settings = make_settings(tmp_path)

    client = FakeClient(
        registered_exists=False,
    )

    monkeypatch.setattr(
        bootstrap,
        "configure_mlflow",
        lambda: settings,
    )

    monkeypatch.setattr(
        bootstrap,
        "get_mlflow_client",
        lambda _settings: client,
    )

    registered = RegisteredModelResult(
        run_id="run-1",
        registered_model_name=("olist-late-delivery"),
        version="1",
        alias="champion",
        model_uri=("models:/olist-late-delivery@champion"),
    )

    monkeypatch.setattr(
        bootstrap,
        "register_task2_model",
        lambda: registered,
    )

    result = bootstrap.ensure_production_model()

    assert result.version == "1"
    assert result.created is True

    assert result.model_uri == registered.model_uri


def test_bootstrap_rejects_model_without_alias(
    monkeypatch,
    tmp_path,
):
    settings = make_settings(tmp_path)

    client = FakeClient(
        registered_exists=True,
        alias_version=None,
    )

    monkeypatch.setattr(
        bootstrap,
        "configure_mlflow",
        lambda: settings,
    )

    monkeypatch.setattr(
        bootstrap,
        "get_mlflow_client",
        lambda _settings: client,
    )

    with pytest.raises(
        bootstrap.MlflowBootstrapError,
        match=("production alias is missing"),
    ):
        bootstrap.ensure_production_model()
