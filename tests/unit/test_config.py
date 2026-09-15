from pathlib import Path

import pytest

import olist_ml.config as config_module
from olist_ml.config import (
    ConfigurationError,
    REQUIRED_ARTIFACT_KEYS,
    REQUIRED_CONFIG_SECTIONS,
    find_project_root,
    load_config,
    resolve_project_path,
    validate_artifact_paths,
)


def test_find_project_root_detects_repository():
    root = find_project_root()

    assert root.is_dir()
    assert (root / "config" / "config.json").is_file()
    assert (root / "compose.yaml").is_file()


def test_load_config_contains_required_sections():
    config = load_config()

    assert REQUIRED_CONFIG_SECTIONS.issubset(config.keys())


def test_load_config_contains_required_artifacts():
    config = load_config()

    assert REQUIRED_ARTIFACT_KEYS.issubset(
        config["artifacts"].keys()
    )


def test_resolve_project_path_returns_absolute_path():
    resolved = resolve_project_path(
        "artifacts/06_model/model_bundle.joblib"
    )

    assert isinstance(resolved, Path)
    assert resolved.is_absolute()
    assert resolved.name == "model_bundle.joblib"


def test_find_project_root_raises_for_invalid_location(tmp_path):
    fake_location = tmp_path / "outside_project"
    fake_location.mkdir()

    with pytest.raises(ConfigurationError):
        find_project_root(fake_location)


def test_validate_artifact_paths_returns_existing_files(
    tmp_path,
    monkeypatch,
):
    artifact_paths = {}

    for artifact_name in REQUIRED_ARTIFACT_KEYS:
        artifact_path = tmp_path / f"{artifact_name}.test"
        artifact_path.write_text("test", encoding="utf-8")
        artifact_paths[artifact_name] = artifact_path

    monkeypatch.setattr(
        config_module,
        "get_artifact_paths",
        lambda: artifact_paths,
    )

    result = validate_artifact_paths()

    assert result == artifact_paths


def test_validate_artifact_paths_raises_for_missing_file(
    tmp_path,
    monkeypatch,
):
    missing_path = tmp_path / "missing_model.joblib"

    artifact_paths = {
        "model_bundle": missing_path,
    }

    monkeypatch.setattr(
        config_module,
        "get_artifact_paths",
        lambda: artifact_paths,
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_artifact_paths()

    error_message = str(exc_info.value)

    assert "model_bundle" in error_message
    assert str(missing_path) in error_message
