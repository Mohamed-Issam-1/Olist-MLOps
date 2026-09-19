from types import SimpleNamespace

import pandas as pd
import pytest

import olist_ml.mlflow_loader as loader
from olist_ml.artifacts import (
    InferenceArtifacts,
)
from olist_ml.mlflow_model import (
    OlistLateDeliveryPythonModel,
)


class DummyModel:
    pass


class FakePyfuncModel:
    def __init__(
        self,
        python_model,
        prediction=None,
    ):
        self.python_model = python_model

        self.prediction = prediction

        self.predict_calls = []

    def unwrap_python_model(
        self,
    ):
        return self.python_model

    def predict(
        self,
        raw_orders,
    ):
        self.predict_calls.append(raw_orders)

        if self.prediction is None:
            return pd.DataFrame()

        return self.prediction.copy()


class FakeClient:
    def __init__(
        self,
        *,
        version="1",
        status="READY",
        run_id="run-1",
    ):
        self.version = version

        self.status = status

        self.run_id = run_id

        self.alias_calls = []

    def get_model_version_by_alias(
        self,
        name,
        alias,
    ):
        self.alias_calls.append(
            (
                name,
                alias,
            )
        )

        return SimpleNamespace(
            version=self.version,
            status=self.status,
            run_id=self.run_id,
        )


def make_artifacts():
    return InferenceArtifacts(
        preprocessor=object(),
        feature_names=[],
        feature_config={},
        model_bundle={
            "model": DummyModel(),
            "classification_threshold": 0.4,
        },
    )


def make_python_model():
    model = OlistLateDeliveryPythonModel()

    model._artifacts = make_artifacts()

    return model


@pytest.fixture(autouse=True)
def clear_cache():
    loader.clear_registered_inference_model_cache()

    yield

    loader.clear_registered_inference_model_cache()


def configure_fake_registry(
    monkeypatch,
    *,
    client=None,
    pyfunc_model=None,
):
    settings = SimpleNamespace(
        registered_model_name=("olist-late-delivery"),
        production_alias=("champion"),
    )

    if client is None:
        client = FakeClient()

    if pyfunc_model is None:
        pyfunc_model = FakePyfuncModel(make_python_model())

    monkeypatch.setattr(
        loader,
        "ensure_experiment",
        lambda: (
            settings,
            client,
            object(),
        ),
    )

    load_calls = []

    def fake_load_model(
        model_uri,
    ):
        load_calls.append(model_uri)

        return pyfunc_model

    monkeypatch.setattr(
        loader.mlflow.pyfunc,
        "load_model",
        fake_load_model,
    )

    return (
        client,
        pyfunc_model,
        load_calls,
    )


def test_load_registered_model_resolves_alias(
    monkeypatch,
):
    (
        client,
        _pyfunc_model,
        load_calls,
    ) = configure_fake_registry(monkeypatch)

    result = loader.load_registered_inference_model()

    assert result.registered_model_name == "olist-late-delivery"

    assert result.version == "1"

    assert result.alias == "champion"

    assert result.run_id == "run-1"

    assert result.model_uri == ("models:/olist-late-delivery@champion")

    assert result.source == "mlflow-registry"

    assert result.model_type == "DummyModel"

    assert result.classification_threshold == pytest.approx(0.4)

    assert client.alias_calls == [
        (
            "olist-late-delivery",
            "champion",
        )
    ]

    assert load_calls == [("models:/olist-late-delivery@champion")]


def test_registered_model_predict_delegates_to_pyfunc(
    monkeypatch,
):
    expected = pd.DataFrame(
        {
            "late_probability": [
                0.2,
            ],
            "predicted_is_late": [
                0,
            ],
        }
    )

    pyfunc_model = FakePyfuncModel(
        make_python_model(),
        prediction=expected,
    )

    configure_fake_registry(
        monkeypatch,
        pyfunc_model=pyfunc_model,
    )

    runtime_model = loader.load_registered_inference_model()

    raw_orders = pd.DataFrame(
        {
            "order_id": [
                "order-1",
            ]
        }
    )

    actual = runtime_model.predict(raw_orders)

    pd.testing.assert_frame_equal(
        actual,
        expected,
    )

    assert pyfunc_model.predict_calls == [raw_orders]


def test_loader_rejects_non_ready_model(
    monkeypatch,
):
    configure_fake_registry(
        monkeypatch,
        client=FakeClient(status="PENDING_REGISTRATION"),
    )

    with pytest.raises(
        loader.MlflowModelLoadError,
        match="not READY",
    ):
        (loader.load_registered_inference_model())


def test_loader_rejects_wrong_python_model(
    monkeypatch,
):
    configure_fake_registry(
        monkeypatch,
        pyfunc_model=(FakePyfuncModel(object())),
    )

    with pytest.raises(
        loader.MlflowModelLoadError,
        match="unexpected",
    ):
        (loader.load_registered_inference_model())


def test_loader_uses_process_cache(
    monkeypatch,
):
    (
        client,
        _pyfunc_model,
        load_calls,
    ) = configure_fake_registry(monkeypatch)

    first = loader.load_registered_inference_model()

    second = loader.load_registered_inference_model()

    assert first is second

    assert len(client.alias_calls) == 1

    assert len(load_calls) == 1


def test_refresh_reloads_alias(
    monkeypatch,
):
    (
        client,
        _pyfunc_model,
        load_calls,
    ) = configure_fake_registry(monkeypatch)

    first = loader.load_registered_inference_model()

    client.version = "2"
    client.run_id = "run-2"

    second = loader.load_registered_inference_model(refresh=True)

    assert first.version == "1"
    assert second.version == "2"
    assert second.run_id == "run-2"

    assert len(load_calls) == 2
