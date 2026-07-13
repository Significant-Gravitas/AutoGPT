"""Typed tool responses for the Menlo robot copilot tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.copilot.tools.models import ResponseType, ToolResponseBase


class MenloRobotConnectedResponse(ToolResponseBase):
    type: ResponseType = ResponseType.MENLO_ROBOT_CONNECTED
    robot_id: str
    model: str
    viewer_url: str = Field(
        description="Open this in Chrome to start the 3D simulation runtime"
    )


class MenloRobotDisconnectedResponse(ToolResponseBase):
    type: ResponseType = ResponseType.MENLO_ROBOT_DISCONNECTED
    robot_id: str | None = None


class MenloSkillInfo(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class MenloSkillsDiscoveredResponse(ToolResponseBase):
    type: ResponseType = ResponseType.MENLO_SKILLS_DISCOVERED
    count: int
    skills: list[MenloSkillInfo]


class MenloSkillResultResponse(ToolResponseBase):
    type: ResponseType = ResponseType.MENLO_SKILL_RESULT
    skill: str
    status: str = Field(description='"done" or "failed" — a terminal outcome')
    action_id: str | None = None
    error: str | None = None
    result: Any | None = None


class MenloRobotStateResponse(ToolResponseBase):
    type: ResponseType = ResponseType.MENLO_ROBOT_STATE
    key: str
    state: Any


class MenloVisionResponse(ToolResponseBase):
    type: ResponseType = ResponseType.MENLO_VISION
    camera: str
    file_id: str
    filename: str
    width: int
    height: int
