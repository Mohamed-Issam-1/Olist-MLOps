from pathlib import Path

import mlflow
import pytest

import olist_ml.mlflow_config as mlflow_config

from olist_ml.mlflow_config import MlflowSettings


def test_load_mlflow_settings():
    settings = (
        mlflow_config
        .load_mlflow_settings()
    )

    assert (
        settings.backend_database.name
        == "mlflow.db"
    )

    assert (
        settings.artifact_directory.name
        == "mlartifacts"
    )

    assert (
        settings.experiment_name
        == "olist-late-delivery"
    )

    assert (
        settings.registered_model_name
        == "olist-late-delivery"
    )

    assert (
        settings.model_name
        == "inference_pipeline"
    )

    assert (
        settings.production_alias
        == "champion"
    )


def test_tracking_uri_is_sqlite_uri():
    settings = (
        mlflow_config
        .load_mlflow_settings()
    )

    assert (
        settings.tracking_uri
        .startswith(
            "sqlite:///"
        )
    )

    assert (
        "mlflow.db"
        in settings.tracking_uri
    )


def test_artifact_uri_is_file_uri():
    settings = (
        mlflow_config
        .load_mlflow_settings()
    )

    assert (
        settings.artifact_uri
        .startswith(
            "file:"
        )
    )

    assert (
        "mlartifacts"
        in settings.artifact_uri
    )


def test_resolve_local_path_rejects_absolute_path(
    tmp_path,
):
    absolute_path = (
        tmp_path
        / "outside.db"
    ).resolve()

    with pytest.raises(
        mlflow_config.MlflowConfigurationError,
        match="project-relative",
    ):
        mlflow_config._resolve_local_path(
            tmp_path,
            str(
                absolute_path
            ),
            key="backend_database",
        )


def test_configure_mlflow_sets_tracking_and_registry(
    monkeypatch,
    tmp_path,
):
    settings = (
        mlflow_config.MlflowSettings(
            backend_database=(
                tmp_path
                / "mlflow.db"
            ),
            artifact_directory=(
                tmp_path
                / "mlartifacts"
            ),
            experiment_name="test-experiment",
            registered_model_name="test-model",
            model_name="test-pipeline",
            production_alias="champion",
        )
    )

    monkeypatch.setattr(
        mlflow_config,
        "load_mlflow_settings",
        lambda: settings,
    )

    tracking_calls = []
    registry_calls = []

    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        tracking_calls.append,
    )

    monkeypatch.setattr(
        mlflow,
        "set_registry_uri",
        registry_calls.append,
    )

    result = (
        mlflow_config
        .configure_mlflow()
    )

    assert result == settings

    assert tracking_calls == [
        settings.tracking_uri
    ]

    assert registry_calls == [
        settings.tracking_uri
    ]

    assert (
        settings
        .artifact_directory
        .is_dir()
    )

def test_tracking_uri_preserves_parentheses(
    tmp_path,
):
    special_directory = (
        tmp_path
        / "(01)04"
    )

    settings = MlflowSettings(
        backend_database=(
            special_directory
            / "mlflow.db"
        ),
        artifact_directory=(
            tmp_path
            / "mlartifacts"
        ),
        experiment_name="test-experiment",
        registered_model_name="test-model",
        model_name="test-pipeline",
        production_alias="champion",
    )

    assert (
        "(01)04"
        in settings.tracking_uri
    )

    assert (
        "%28"
        not in settings.tracking_uri
    )

    assert (
        "%29"
        not in settings.tracking_uri
    )
