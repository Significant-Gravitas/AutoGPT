"""Provider-neutral requests used by both translating adapters."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An inline image, expressed as a base64 data URL."""

    type: Literal["image"] = "image"
    url: str


class DocumentPart(BaseModel):
    type: Literal["document"] = "document"
    media_type: str
    data: str

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: object


class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content: list[TextPart | ImagePart] = Field(default_factory=list)
    is_error: bool = False

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.content if isinstance(p, TextPart))

    @property
    def images(self) -> list[ImagePart]:
        return [p for p in self.content if isinstance(p, ImagePart)]


class ReasoningPart(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    text: str
    signature: str | None = None


Part = Annotated[
    TextPart | ImagePart | DocumentPart | ToolCallPart | ToolResultPart | ReasoningPart,
    Field(discriminator="type"),
]


class NeutralMessage(BaseModel):
    role: Literal["user", "assistant"]
    parts: list[Part]


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, object]


ToolChoiceMode = Literal["auto", "any", "none", "tool"]


class ToolChoice(BaseModel):
    mode: ToolChoiceMode = "auto"
    name: str | None = None
    parallel: bool = True


class NeutralRequest(BaseModel):
    system: str
    messages: list[NeutralMessage]
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] = Field(default_factory=list)
    session_id: str | None = None
