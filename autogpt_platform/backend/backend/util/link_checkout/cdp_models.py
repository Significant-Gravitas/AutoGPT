from typing import Literal

from pydantic import BaseModel, Field

from backend.util.link_checkout.models import BoundField
from backend.util.link_checkout.network import NetworkMetadata


class Target(BaseModel):
    targetId: str
    type: str
    url: str


class Frame(BaseModel):
    id: str
    url: str
    loaderId: str = ""
    parentId: str = ""


class FrameTree(BaseModel):
    frame: Frame
    childFrames: list["FrameTree"] = Field(default_factory=list)


# What ``cdp._CHECK_CONTROL`` reports about a payment control.
ControlVerdict = Literal["ok", "not_ready", "not_card_field"]


class RemoteValue(BaseModel):
    value: bool | ControlVerdict | Literal["invalid_selector"] | None = None
    objectId: str = ""


class Node(BaseModel):
    backendNodeId: int


class Result(BaseModel):
    targetInfos: list[Target] = Field(default_factory=list)
    sessionId: str = ""
    executionContextId: int = 0
    frameTree: FrameTree | None = None
    result: RemoteValue | None = None
    object: RemoteValue | None = None
    node: Node | None = None
    exceptionDetails: dict | None = None


class Response(BaseModel):
    id: int = 0
    result: Result = Field(default_factory=Result)
    error: dict | None = None
    method: str = ""
    sessionId: str = ""
    params: NetworkMetadata = Field(default_factory=NetworkMetadata)


class Control(BaseModel):
    binding: BoundField
    session: str
    object_id: str


class FrameContext(BaseModel):
    frame: Frame
    target_id: str
    session: str
