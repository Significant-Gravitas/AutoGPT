import datetime
import json
import typing

import prisma.models
import pydantic

import backend.data.block
import backend.server.model


class LibraryAgent(pydantic.BaseModel):
    id: str  # Changed from agent_id to match GraphMeta
    version: int  # Changed from agent_version to match GraphMeta
    is_active: bool  # Added to match GraphMeta
    name: str
    description: str

    isCreatedByUser: bool
    # Made input_schema and output_schema match GraphMeta's type
    input_schema: dict[str, typing.Any]  # Should be BlockIOObjectSubSchema in frontend
    output_schema: dict[str, typing.Any]  # Should be BlockIOObjectSubSchema in frontend


class LibraryAgentPreset(pydantic.BaseModel):
    id: str
    updated_at: datetime.datetime

    agent_id: str
    agent_version: int

    name: str
    description: str

    is_active: bool
    inputs: dict[str, backend.data.block.BlockInput]

    @staticmethod
    def from_db(preset: prisma.models.AgentPreset):
        input_data = {}

        for data in preset.InputPresets or []:
            input_data[data.name] = json.loads(data.data)

        return LibraryAgentPreset(
            id=preset.id,
            updated_at=preset.updatedAt,
            agent_id=preset.agentId,
            agent_version=preset.agentVersion,
            name=preset.name,
            description=preset.description,
            is_active=preset.isActive,
            inputs=input_data,
        )


class LibraryAgentPresetResponse(pydantic.BaseModel):
    presets: list[LibraryAgentPreset]
    pagination: backend.server.model.Pagination


class CreateLibraryAgentPresetRequest(pydantic.BaseModel):
    name: str
    description: str
    inputs: dict[str, backend.data.block.BlockInput]
    agent_id: str
    agent_version: int
    is_active: bool
