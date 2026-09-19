from types import SimpleNamespace

import pytest

import olist_ml.mlflow_tracking as tracking


class FakeMlflowClient:
    def __init__(
        self,
        *,
        experiment=None,
        artifact_location=None,
    ):
        self.experiment = experiment
        self.artifact_location = artifact_location
        self.created = []
        self.get_calls = []

    def get_experiment_by_name(
        self,
        name,
    ):
        self.get_calls.append(
            (
                "by_name",
                name,
            )
        )

        return self.experiment

    def create_experiment(
        self,
        name,
        artifact_location=None,
    ):
        self.created.append(
            (
                name,
                artifact_location,
            )
        )

        self.experiment = SimpleNamespace(
            experiment_id="42",
            name=name,
            artifact_location=(artifact_location),
            lifecycle_stage="active",
        )

        return "42"

    def get_experiment(
        self,
        experiment_id,
    ):
        self.get_calls.append(
            (
                "by_id",
                experiment_id,
            )
        )

        return self.experiment


def make_settings(
    tmp_path,
):
    from olist_ml.mlflow_config import (
        MlflowSettings,
    )

    return MlflowSettings(
        backend_database=(tmp_path / "mlflow.db"),
        artifact_directory=(tmp_path / "mlartifacts"),
        experiment_name=("olist-late-delivery"),
        registered_model_name=("olist-late-delivery"),
        model_name=("inference_pipeline"),
        production_alias=("champion"),
    )


def test_ensure_experiment_creates_missing(
    monkeypatch,
    tmp_path,
):
    settings = make_settings(tmp_path)

    fake_client = FakeMlflowClient()

    monkeypatch.setattr(
        tracking,
        "configure_mlflow",
        lambda: settings,
    )

    monkeypatch.setattr(
        tracking,
        "get_mlflow_client",
        lambda _settings: fake_client,
    )

    (
        returned_settings,
        returned_client,
        experiment,
    ) = tracking.ensure_experiment()

    assert returned_settings == settings

    assert returned_client is fake_client

    assert experiment.experiment_id == "42"

    assert fake_client.created == [
        (
            settings.experiment_name,
            settings.artifact_uri,
        )
    ]


def test_ensure_experiment_reuses_existing(
    monkeypatch,
    tmp_path,
):
    settings = make_settings(tmp_path)

    existing = SimpleNamespace(
        experiment_id="7",
        name=settings.experiment_name,
        artifact_location=(settings.artifact_uri),
        lifecycle_stage="active",
    )

    fake_client = FakeMlflowClient(experiment=existing)

    monkeypatch.setattr(
        tracking,
        "configure_mlflow",
        lambda: settings,
    )

    monkeypatch.setattr(
        tracking,
        "get_mlflow_client",
        lambda _settings: fake_client,
    )

    (
        _settings,
        _client,
        experiment,
    ) = tracking.ensure_experiment()

    assert experiment.experiment_id == "7"

    assert fake_client.created == []


def test_validate_experiment_rejects_wrong_artifact_location(
    tmp_path,
):
    settings = make_settings(tmp_path)

    experiment = SimpleNamespace(
        experiment_id="9",
        name=settings.experiment_name,
        artifact_location=("file:///wrong/location"),
        lifecycle_stage="active",
    )

    with pytest.raises(
        tracking.MlflowTrackingError,
        match="artifact location",
    ):
        tracking._validate_experiment(
            experiment,
            settings,
        )


def test_validate_experiment_rejects_deleted_experiment(
    tmp_path,
):
    settings = make_settings(tmp_path)

    experiment = SimpleNamespace(
        experiment_id="9",
        name=settings.experiment_name,
        artifact_location=(settings.artifact_uri),
        lifecycle_stage="deleted",
    )

    with pytest.raises(
        tracking.MlflowTrackingError,
        match="not active",
    ):
        tracking._validate_experiment(
            experiment,
            settings,
        )


def test_get_mlflow_client_uses_registry_uri(
    monkeypatch,
    tmp_path,
):
    from olist_ml.mlflow_config import (
        MlflowSettings,
    )

    settings = MlflowSettings(
        backend_database=(tmp_path / "mlflow.db"),
        artifact_directory=(tmp_path / "mlartifacts"),
        experiment_name=("olist-late-delivery"),
        registered_model_name=("olist-late-delivery"),
        model_name=("inference_pipeline"),
        production_alias=("champion"),
        tracking_uri_override=("http://tracking:5000"),
        registry_uri_override=("http://registry:5000"),
    )

    captured = {}

    class CapturingClient:
        def __init__(
            self,
            *,
            tracking_uri,
            registry_uri,
        ):
            captured["tracking_uri"] = tracking_uri

            captured["registry_uri"] = registry_uri

    monkeypatch.setattr(
        tracking,
        "MlflowClient",
        CapturingClient,
    )

    tracking.get_mlflow_client(settings)

    assert captured == {
        "tracking_uri": "http://tracking:5000",
        "registry_uri": "http://registry:5000",
    }
