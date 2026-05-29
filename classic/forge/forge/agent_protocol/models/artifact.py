from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Artifact(BaseModel):
    created_at: datetime = Field(
        description="The creation datetime of the task.",
        examples=["2023-01-01T00:00:00Z"],
    )
    modified_at: datetime = Field(
        description="The modification datetime of the task.",
        examples=["2023-01-01T00:00:00Z"],
    )
    artifact_id: str = Field(
        description="ID of the artifact.",
        examples=["b225e278-8b4c-4f99-a696-8facf19f0e56"],
    )
    agent_created: bool = Field(
        description="Whether the artifact has been created by the agent.",
        examples=[False],
    )
    relative_path: str = Field(
        description="Relative path of the artifact in the agents workspace.",
        examples=["/my_folder/my_other_folder/"],
    )
    file_name: str = Field(
        description="Filename of the artifact.",
        examples=["main.py"],
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )
