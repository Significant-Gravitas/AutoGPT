# coding: utf-8


from __future__ import annotations

import json
import pprint
import re  # noqa: F401
from typing import Optional

from pydantic import BaseModel, Field, StrictStr


class Artifact(BaseModel):
    """
    Artifact that the task has produced.
    """

    artifact_id: StrictStr = Field(..., description="ID of the artifact.")
    file_name: StrictStr = Field(..., description="Filename of the artifact.")
    relative_path: Optional[StrictStr] = Field(
        None, description="Relative path of the artifact in the agent's workspace."
    )
    __properties = ["artifact_id", "file_name", "relative_path"]
    created_at: StrictStr = Field(..., description="Creation date of the artifact.")
    # modified_at: StrictStr = Field(..., description="Modification date of the artifact.")
    agent_created: bool = Field(..., description="True if created by the agent")

    class Config:
        """Pydantic configuration"""

        allow_population_by_field_name = True
        validate_assignment = True

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Artifact:
        """Create an instance of Artifact from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True, exclude={}, exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> Artifact:
        """Create an instance of Artifact from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return Artifact.parse_obj(obj)

        _obj = Artifact.parse_obj(
            {
                "artifact_id": obj.get("artifact_id"),
                "file_name": obj.get("file_name"),
                "relative_path": obj.get("relative_path"),
                "created_at": obj.get("created_at"),
                "modified_at": obj.get("modified_at"),
                "agent_created": obj.get("agent_created"),
            }
        )
        return _obj
