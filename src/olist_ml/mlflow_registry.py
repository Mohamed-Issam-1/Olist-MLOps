from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import mlflow

from olist_ml.artifacts import (
    InferenceArtifacts,
    load_inference_artifacts,
)
from olist_ml.config import (
    find_project_root,
    load_config,
    validate_artifact_paths,
)
from olist_ml.mlflow_model import (
    OlistLateDeliveryPythonModel,
)
from olist_ml.mlflow_tracking import (
    ensure_experiment,
)


class MlflowRegistryError(RuntimeError):
    """Raised when MLflow model registration fails validation."""


@dataclass(frozen=True)
class RegisteredModelResult:
    run_id: str
    registered_model_name: str
    version: str
    alias: str
    model_uri: str


METRIC_NAMES = (
    "average_precision",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "accuracy",
)


def _load_results_summary(
    model_bundle_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """
    Load the Task 2 evaluation summary located beside
    the configured model bundle.
    """

    summary_path = model_bundle_path.with_name("results_summary.json")

    if not summary_path.is_file():
        raise MlflowRegistryError(
            f"Task 2 results summary was not found: {summary_path}"
        )

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise MlflowRegistryError("Could not read Task 2 results summary.") from exc

    if not isinstance(
        summary,
        dict,
    ):
        raise MlflowRegistryError("Task 2 results summary must be a JSON object.")

    return (
        summary_path,
        summary,
    )


def _build_metric_payload(
    summary: dict[str, Any],
) -> dict[str, float]:
    """
    Extract the existing final validation and test metrics.

    No metrics are recalculated here.
    """

    metrics = {}

    for section_name, prefix in (
        (
            "final_validation",
            "validation",
        ),
        (
            "final_test",
            "test",
        ),
    ):
        section = summary.get(section_name)

        if not isinstance(
            section,
            dict,
        ):
            raise MlflowRegistryError(
                f"Missing results summary section: {section_name}"
            )

        for metric_name in METRIC_NAMES:
            value = section.get(metric_name)

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    Real,
                )
                or not math.isfinite(float(value))
            ):
                raise MlflowRegistryError(
                    f"Invalid metric {section_name}.{metric_name}"
                )

            metrics[f"{prefix}_{metric_name}"] = float(value)

    return metrics


def _build_parameter_payload(
    artifacts: InferenceArtifacts,
) -> dict[str, Any]:
    """
    Build MLflow parameters from the fitted Task 2 artifacts.
    """

    bundle = artifacts.model_bundle

    feature_config = artifacts.feature_config

    class_weight = bundle.get("best_class_weight")

    return {
        "model_type": bundle["model_type"],
        "best_C": bundle.get("best_C"),
        "best_class_weight": ("None" if class_weight is None else str(class_weight)),
        "classification_threshold": artifacts.classification_threshold,
        "primary_metric": bundle.get("primary_metric"),
        "raw_feature_count": feature_config["raw_feature_count"],
        "transformed_feature_count": feature_config["transformed_feature_count"],
        "prediction_point": feature_config.get("prediction_point"),
    }


def _load_runtime_requirements(
    project_root: Path,
) -> list[str]:
    """
    Use the project's production dependency set
    as the MLflow model environment metadata.
    """

    path = project_root / "requirements" / "runtime.txt"

    if not path.is_file():
        raise MlflowRegistryError("Runtime requirements file was not found.")

    requirements = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if (line.strip() and not line.strip().startswith("#"))
    ]

    if not requirements:
        raise MlflowRegistryError("Runtime requirements file is empty.")

    return requirements


def _build_model_artifact_mapping(
    artifact_paths: dict[str, Path],
) -> dict[str, str]:
    required = (
        "preprocessor",
        "model_bundle",
        "feature_names",
        "feature_config",
    )

    missing = [name for name in required if name not in artifact_paths]

    if missing:
        raise MlflowRegistryError(
            "Missing configured inference artifacts: " + ", ".join(missing)
        )

    return {name: str(artifact_paths[name].resolve()) for name in required}


def _set_and_verify_alias(
    client,
    *,
    model_name: str,
    alias: str,
    version: str,
) -> None:
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version,
    )

    resolved = client.get_model_version_by_alias(
        model_name,
        alias,
    )

    if str(resolved.version) != str(version):
        raise MlflowRegistryError("MLflow alias verification failed.")


def register_task2_model() -> RegisteredModelResult:
    """
    Register the existing fitted Task 2 inference model.

    This function never fits or retrains a model.
    """

    (
        settings,
        client,
        experiment,
    ) = ensure_experiment()

    project_root = find_project_root()

    project_config = load_config()

    model_config = project_config["model"]

    artifact_paths = validate_artifact_paths()

    artifacts = load_inference_artifacts()

    (
        summary_path,
        summary,
    ) = _load_results_summary(artifact_paths["model_bundle"])

    metrics = _build_metric_payload(summary)

    params = _build_parameter_payload(artifacts)

    model_artifacts = _build_model_artifact_mapping(artifact_paths)

    runtime_requirements = _load_runtime_requirements(project_root)

    code_path = project_root / "src" / "olist_ml"

    if not code_path.is_dir():
        raise MlflowRegistryError("Production package directory was not found.")

    with mlflow.start_run(
        experiment_id=(experiment.experiment_id),
        run_name=(f"{settings.registered_model_name}-registration"),
        tags={
            "source": str(model_config["source"]),
            "training_performed": "false",
            "model_version": str(model_config["version"]),
        },
    ) as run:
        mlflow.log_params(params)

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            str(summary_path),
            artifact_path=("evaluation"),
        )

        model_info = mlflow.pyfunc.log_model(
            name=(settings.model_name),
            python_model=(OlistLateDeliveryPythonModel()),
            artifacts=(model_artifacts),
            code_paths=[str(code_path)],
            pip_requirements=(runtime_requirements),
            registered_model_name=(settings.registered_model_name),
            metadata={
                "classification_threshold": artifacts.classification_threshold,
                "model_type": artifacts.model_bundle["model_type"],
                "training_performed": False,
            },
        )

        run_id = run.info.run_id

    version = getattr(
        model_info,
        "registered_model_version",
        None,
    )

    if version is None:
        raise MlflowRegistryError("MLflow did not return a registered model version.")

    version = str(version)

    _set_and_verify_alias(
        client,
        model_name=(settings.registered_model_name),
        alias=(settings.production_alias),
        version=version,
    )

    model_uri = f"models:/{settings.registered_model_name}@{settings.production_alias}"

    return RegisteredModelResult(
        run_id=run_id,
        registered_model_name=(settings.registered_model_name),
        version=version,
        alias=(settings.production_alias),
        model_uri=model_uri,
    )
