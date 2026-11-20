"""The cards Otto can show in a message, and the limits on each.

Blocks carry plain text, numbers and ``workspace://`` image refs only. There is
no URL or markup field, so nothing a model writes here can leave the page.
"""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .models import ResponseType, ToolResponseBase

MAX_BLOCKS = 6
# Tool data replays into context on later turns.
MAX_PAYLOAD_CHARS = 16_000

WORKSPACE_IMAGE_PATTERN = r"^workspace://[A-Za-z0-9_-]+(#image/[a-z0-9.+-]+)?$"

Short = Annotated[str, StringConstraints(min_length=1, max_length=60)]
Medium = Annotated[str, StringConstraints(min_length=1, max_length=120)]
Long = Annotated[str, StringConstraints(min_length=1, max_length=200)]
Tone = Literal["neutral", "positive", "negative", "warning"]
Status = Literal["ok", "warning", "error", "idle"]


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, allow_inf_nan=False
    )


class StatItem(_Strict):
    label: Short
    value: Short
    change: Short | None = None
    tone: Tone | None = None


class StatsBlock(_Strict):
    """Headline numbers."""

    type: Literal["stats"]
    items: list[StatItem] = Field(min_length=1, max_length=4)


class ChartSeries(_Strict):
    name: Short
    values: list[float] = Field(min_length=1, max_length=50)


class ChartAnnotation(_Strict):
    x: Short = Field(description="One of x_labels.")
    label: Short


class ChartBlock(_Strict):
    """A trend or comparison over labelled points."""

    type: Literal["chart"]
    title: Medium | None = None
    kind: Literal["line", "area", "bar"]
    x_labels: list[Short] = Field(min_length=1, max_length=50)
    series: list[ChartSeries] = Field(min_length=1, max_length=3)
    unit: Short | None = None
    annotation: ChartAnnotation | None = None

    @model_validator(mode="after")
    def _points_line_up(self) -> Self:
        if any(len(line.values) != len(self.x_labels) for line in self.series):
            raise ValueError("every series needs one value per entry in x_labels")
        if self.annotation and self.annotation.x not in self.x_labels:
            raise ValueError("annotation x must be one of x_labels")
        return self


class WorkflowStep(_Strict):
    label: Short
    detail: Short | None = None
    status: Status = "idle"


class WorkflowDiagramBlock(_Strict):
    """Steps of a workflow in order, showing where things stand."""

    type: Literal["workflow_diagram"]
    title: Medium | None = None
    steps: list[WorkflowStep] = Field(min_length=2, max_length=8)


class GalleryImage(_Strict):
    src: str = Field(
        pattern=WORKSPACE_IMAGE_PATTERN, description="workspace://<file_id> image."
    )
    caption: Medium


class GalleryBlock(_Strict):
    """Images from the user's workspace."""

    type: Literal["gallery"]
    title: Medium | None = None
    images: list[GalleryImage] = Field(min_length=1, max_length=6)


class TimelineItem(_Strict):
    date: Short | None = None
    title: Medium
    detail: Long | None = None
    status: Status | None = None


class TimelineBlock(_Strict):
    """Events or plan steps over time."""

    type: Literal["timeline"]
    title: Medium | None = None
    items: list[TimelineItem] = Field(min_length=1, max_length=10)


class OptionItem(_Strict):
    title: Medium = Field(description="Sent as the user's reply when clicked.")
    description: Long | None = None
    tone: Tone | None = None


class OptionsBlock(_Strict):
    """Next steps the user can pick with one click."""

    type: Literal["options"]
    items: list[OptionItem] = Field(min_length=2, max_length=4)


class TableBlock(_Strict):
    """Rows of exact values, when nothing more visual fits."""

    type: Literal["table"]
    title: Medium | None = None
    columns: list[Short] = Field(min_length=1, max_length=6)
    rows: list[list[Long]] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _rows_fill_columns(self) -> Self:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("every row needs one cell per entry in columns")
        return self


Block = Annotated[
    StatsBlock
    | ChartBlock
    | WorkflowDiagramBlock
    | GalleryBlock
    | TimelineBlock
    | OptionsBlock
    | TableBlock,
    Field(discriminator="type"),
]


class RenderUIInput(_Strict):
    blocks: list[Block] = Field(
        min_length=1,
        max_length=MAX_BLOCKS,
        description=f"Cards to show, in order. At most {MAX_BLOCKS}.",
    )

    @model_validator(mode="after")
    def _fits_in_context(self) -> Self:
        if len(self.model_dump_json(exclude_none=True)) > MAX_PAYLOAD_CHARS:
            raise ValueError(f"blocks exceed {MAX_PAYLOAD_CHARS} characters")
        return self


class UIBlocksResponse(ToolResponseBase):
    """The cards the chat draws for this call."""

    type: ResponseType = ResponseType.UI_BLOCKS
    blocks: list[Block] = Field(default_factory=list)


def render_ui_parameters() -> dict[str, Any]:
    """The input schema, flattened: this is sent to the model on every turn."""
    schema = RenderUIInput.model_json_schema()
    definitions = schema.pop("$defs", {})
    return _compact(schema, definitions)


# String limits and the like are enforced on the way in and reported back, so
# they don't need to ride along in every turn's prompt.
_DROPPED_KEYWORDS = frozenset(
    {
        "title",
        "default",
        "discriminator",
        "additionalProperties",
        "minLength",
        "maxLength",
        "pattern",
    }
)


def _compact(node: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_compact(item, definitions) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        rest = {key: value for key, value in node.items() if key != "$ref"}
        return _compact({**definitions[name], **rest}, definitions)
    if (branch := _only_value_branch(node)) is not None:
        rest = {key: value for key, value in node.items() if key != "anyOf"}
        return _allow_null(_compact({**branch, **rest}, definitions))
    compacted: dict[str, Any] = {}
    for key, value in node.items():
        if key == "properties":
            # These keys are field names, and a field may be called "title".
            compacted[key] = {
                name: _compact(field, definitions) for name, field in value.items()
            }
        elif key == "oneOf":
            compacted["anyOf"] = _compact(value, definitions)
        elif key not in _DROPPED_KEYWORDS:
            compacted[key] = _compact(value, definitions)
    return compacted


def _only_value_branch(node: dict[str, Any]) -> dict[str, Any] | None:
    """The ``X`` of an ``anyOf: [X, null]``, which is how Pydantic writes optional."""
    branches = node.get("anyOf")
    if not isinstance(branches, list) or len(branches) != 2:
        return None
    kept = [branch for branch in branches if branch != {"type": "null"}]
    return kept[0] if len(kept) == 1 else None


def _allow_null(field: dict[str, Any]) -> dict[str, Any]:
    """Keep null valid: models send it for fields they leave empty, and the SDK
    checks calls against this schema before the tool can read them."""
    widened = dict(field)
    if isinstance(widened.get("type"), str):
        widened["type"] = [widened["type"], "null"]
    if "enum" in widened:
        widened["enum"] = [*widened["enum"], None]
    return widened
