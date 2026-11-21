"""What Otto may put on a workflow page, and the check against the workflow.

A design names the workflow's own inputs and outputs and nothing else: no
values of its own beyond labels and defaults, no URL or markup field.
"""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from backend.copilot.tools.render_ui_models import Long, Medium, Short

MAX_FORM_GROUPS = 3

IOName = Annotated[str, StringConstraints(min_length=1, max_length=100)]
Body = Annotated[str, StringConstraints(min_length=1, max_length=600)]


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, allow_inf_nan=False
    )


class TextSection(_Strict):
    kind: Literal["text"]
    heading: Medium | None = None
    body: Body = Field(description="Plain text, no markup.")


class FormField(_Strict):
    input: IOName = Field(description="One of the workflow's input names.")
    label: Short | None = None
    help: Long | None = None
    default: str | float | bool | None = None
    group: Short | None = Field(None, description="Fields in one group show together.")


class RunFormSection(_Strict):
    kind: Literal["run_form"]
    fields: list[FormField] = Field(
        min_length=1, max_length=30, description="In display order."
    )
    button_label: Short | None = None


class StatBinding(_Strict):
    label: Short
    output: IOName


class StatsCard(_Strict):
    type: Literal["stats"]
    items: list[StatBinding] = Field(min_length=1, max_length=4)


class TableColumn(_Strict):
    label: Short
    key: Short | None = Field(
        None, description="Key in each row object; omit for a list of plain values."
    )


class TableCard(_Strict):
    type: Literal["table"]
    title: Medium | None = None
    output: IOName = Field(description="An output holding a list.")
    columns: list[TableColumn] = Field(min_length=1, max_length=6)


class ChartCard(_Strict):
    type: Literal["chart"]
    title: Medium | None = None
    kind: Literal["line", "area", "bar"]
    output: IOName = Field(description="An output holding a list of objects.")
    x_key: Short
    y_keys: list[Short] = Field(min_length=1, max_length=3)
    unit: Short | None = None


Card = Annotated[StatsCard | TableCard | ChartCard, Field(discriminator="type")]


class LatestResultSection(_Strict):
    kind: Literal["latest_result"]
    cards: list[Card] = Field(
        default_factory=list,
        max_length=4,
        description="Outputs no card names show below the cards.",
    )


class RecentRunsSection(_Strict):
    kind: Literal["recent_runs"]
    count: int = Field(5, ge=3, le=10)


class ScheduleSection(_Strict):
    kind: Literal["schedule"]


Section = Annotated[
    TextSection
    | RunFormSection
    | LatestResultSection
    | RecentRunsSection
    | ScheduleSection,
    Field(discriminator="kind"),
]


class WorkflowPageDesign(_Strict):
    title: Medium
    sections: list[Section] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _one_of_each_kind(self) -> Self:
        kinds = [section.kind for section in self.sections if section.kind != "text"]
        if len(kinds) != len(set(kinds)):
            raise ValueError("each section kind but text may appear once")
        return self


class DesignProblem(BaseModel):
    section: int
    message: str


def check_design(
    design: WorkflowPageDesign,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any],
) -> list[DesignProblem]:
    """Every way *design* fails the workflow it is for, by section index.

    Run when Otto proposes a page, when one is saved and on every read, where a
    section with a problem is left out instead of refused.
    """
    problems: list[DesignProblem] = []
    for index, section in enumerate(design.sections):
        if isinstance(section, RunFormSection):
            messages = _form_problems(section, input_schema)
        elif isinstance(section, LatestResultSection):
            messages = _result_problems(section, output_schema)
        else:
            messages = []
        problems += [DesignProblem(section=index, message=m) for m in messages]
    return problems


def _form_problems(section: RunFormSection, input_schema: dict[str, Any]) -> list[str]:
    inputs: dict[str, Any] = input_schema.get("properties") or {}
    named = [field.input for field in section.fields]
    problems = [
        f"the workflow has no input named '{n}'" for n in named if n not in inputs
    ]
    problems += [
        f"required input '{name}' is missing from the form"
        for name in input_schema.get("required") or []
        if name not in named
    ]
    if len(named) != len(set(named)):
        problems.append("an input appears in the form twice")
    groups = {field.group for field in section.fields if field.group}
    if len(groups) > MAX_FORM_GROUPS:
        problems.append(f"the form has more than {MAX_FORM_GROUPS} groups")
    problems += [
        f"the default for '{field.input}' does not fit that input"
        for field in section.fields
        if field.input in inputs
        and field.default is not None
        and not _default_fits(field.default, inputs[field.input])
    ]
    return problems


def _default_fits(value: str | float | bool, schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    if kind == "boolean":
        return isinstance(value, bool)
    if kind in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return kind == "number" or float(value).is_integer()
    if kind == "string" and isinstance(value, str):
        return value in schema["enum"] if "enum" in schema else True
    return False


def _result_problems(
    section: LatestResultSection, output_schema: dict[str, Any]
) -> list[str]:
    outputs: dict[str, Any] = output_schema.get("properties") or {}
    problems: list[str] = []
    for card in section.cards:
        bound = (
            [i.output for i in card.items]
            if isinstance(card, StatsCard)
            else [card.output]
        )
        for name in bound:
            if name not in outputs:
                problems.append(f"the workflow has no output named '{name}'")
            elif not _type_fits(card, outputs[name].get("type")):
                problems.append(f"a {card.type} card cannot show output '{name}'")
    return problems


def _type_fits(card: StatsCard | TableCard | ChartCard, declared: str | None) -> bool:
    # Most outputs declare no type; those are checked against the run's value.
    if declared is None:
        return True
    if isinstance(card, StatsCard):
        return declared not in ("array", "object")
    return declared == "array"
