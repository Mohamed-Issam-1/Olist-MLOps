from pathlib import Path

import pytest

import olist_ml.artifacts as artifacts_module
from olist_ml.artifacts import (
    ArtifactValidationError,
    InferenceArtifacts,
    load_inference_artifacts,
    validate_artifact_compatibility,
)


class DummyPreprocessor:
    def __init__(
        self,
        input_features=None,
        output_features=None,
    ):
        self.feature_names_in_ = input_features or [
            "number_a",
            "category_a",
        ]

        self._output_features = output_features or [
            "numeric__number_a",
            "categorical__category_a_A",
        ]

    def get_feature_names_out(self):
        return self._output_features


class DummyModel:
    def __init__(
        self,
        feature_count=2,
    ):
        self.n_features_in_ = feature_count

    def predict_proba(self, values):
        return values


def make_valid_artifacts():
    preprocessor = DummyPreprocessor()

    model = DummyModel()

    feature_config = {
        "numeric_features": [
            "number_a",
        ],
        "categorical_features": [
            "category_a",
        ],
        "raw_feature_count": 2,
        "transformed_feature_count": 2,
    }

    feature_names = [
        "numeric__number_a",
        "categorical__category_a_A",
    ]

    model_bundle = {
        "model": model,
        "classification_threshold": 0.4,
        "model_type": "DummyModel",
        "feature_count": 2,
    }

    return InferenceArtifacts(
        preprocessor=preprocessor,
        feature_names=feature_names,
        feature_config=feature_config,
        model_bundle=model_bundle,
    )


def test_validate_artifact_compatibility_accepts_valid_artifacts():
    artifacts = make_valid_artifacts()

    result = validate_artifact_compatibility(artifacts)

    assert result is artifacts


def test_validation_rejects_raw_feature_order_mismatch():
    artifacts = make_valid_artifacts()

    artifacts.preprocessor.feature_names_in_ = [
        "category_a",
        "number_a",
    ]

    with pytest.raises(
        ArtifactValidationError,
        match="input feature order",
    ):
        validate_artifact_compatibility(artifacts)


def test_validation_rejects_transformed_name_mismatch():
    artifacts = make_valid_artifacts()

    artifacts.feature_names[0] = "wrong_feature_name"

    with pytest.raises(
        ArtifactValidationError,
        match="feature_names",
    ):
        validate_artifact_compatibility(artifacts)


def test_validation_rejects_feature_count_mismatch():
    artifacts = make_valid_artifacts()

    artifacts.model_bundle["feature_count"] = 3

    with pytest.raises(
        ArtifactValidationError,
        match="feature counts",
    ):
        validate_artifact_compatibility(artifacts)


def test_validation_rejects_model_type_mismatch():
    artifacts = make_valid_artifacts()

    artifacts.model_bundle["model_type"] = "WrongModel"

    with pytest.raises(
        ArtifactValidationError,
        match="Model type metadata",
    ):
        validate_artifact_compatibility(artifacts)


def test_validation_rejects_invalid_threshold():
    artifacts = make_valid_artifacts()

    artifacts.model_bundle["classification_threshold"] = 1.5

    with pytest.raises(
        ArtifactValidationError,
        match="threshold",
    ):
        validate_artifact_compatibility(artifacts)


def test_load_inference_artifacts_loads_and_validates(
    monkeypatch,
):
    expected = make_valid_artifacts()

    fake_paths = {
        "preprocessor": Path("preprocessor.joblib"),
        "feature_names": Path("feature_names.json"),
        "feature_config": Path("feature_config.json"),
        "model_bundle": Path("model_bundle.joblib"),
    }

    monkeypatch.setattr(
        artifacts_module,
        "validate_artifact_paths",
        lambda: fake_paths,
    )

    def fake_joblib_load(path):
        if path.name == ("preprocessor.joblib"):
            return expected.preprocessor

        if path.name == ("model_bundle.joblib"):
            return expected.model_bundle

        raise AssertionError(f"Unexpected joblib path: {path}")

    def fake_json_load(path):
        if path.name == ("feature_names.json"):
            return expected.feature_names

        if path.name == ("feature_config.json"):
            return expected.feature_config

        raise AssertionError(f"Unexpected JSON path: {path}")

    monkeypatch.setattr(
        artifacts_module.joblib,
        "load",
        fake_joblib_load,
    )

    monkeypatch.setattr(
        artifacts_module,
        "_load_json",
        fake_json_load,
    )

    load_inference_artifacts.cache_clear()

    result = load_inference_artifacts()

    load_inference_artifacts.cache_clear()

    assert result.model is expected.model

    assert result.classification_threshold == pytest.approx(0.4)
