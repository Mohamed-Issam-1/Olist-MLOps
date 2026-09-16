from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class HealthResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    status: str = Field(
        description=(
            "Current service health status."
        )
    )

    service: str = Field(
        description=(
            "Configured API service name."
        )
    )

    api_version: str = Field(
        description=(
            "API contract version."
        )
    )


class ModelInfoResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    source: str

    registered_model_name: str

    alias: str

    version: str

    run_id: str

    model_uri: str

    model_type: str

    classification_threshold: float = Field(
        gt=0.0,
        lt=1.0,
    )
