from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from backend.api.features.library import model as library_model
from backend.data import graph as graph_db


class DeleteGraphResponse(TypedDict):
    version_counts: int


class UpdateGraphResponse(BaseModel):
    """Response for creating/activating a new graph version.

    Carries the new graph version plus any webhook presets that were left
    pinned to their old version because the new version's trigger block is
    incompatible and needs to be reconfigured.
    """

    graph: graph_db.GraphModel
    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = Field(
        default_factory=list
    )


class SetActiveGraphVersionResponse(BaseModel):
    """Response for activating an existing graph version.

    Carries any webhook presets that were left pinned to their old version
    because the newly activated version's trigger block is incompatible and
    needs to be reconfigured.
    """

    skipped_webhook_presets: list[library_model.SkippedWebhookPreset] = Field(
        default_factory=list
    )
