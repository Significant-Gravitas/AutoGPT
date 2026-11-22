"""What Otto may put on a page, and the check of a design against its sources.

A section names its own source: one of the user's workflows (a library entry
id) or experts. Text and cards hold the only values a design stores; nothing a
model writes is a URL or markup.
"""

from datetime import date
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator

from backend.copilot.tools.render_ui_models import (
    ChartBlock,
    GalleryBlock,
    Long,
    Medium,
    Short,
    StatsBlock,
    TableBlock,
    TimelineBlock,
    WorkflowDiagramBlock,
)

from .collection_design import (
    MAX_COLLECTIONS_PER_PAGE,
    CollectionSection,
    RowButton,
    collection_problems,
    section_buttons,
    section_collection,
)
from .design_base import Ref
from .design_base import Strict as _Strict

if TYPE_CHECKING:
    from .sources import PageSources

MAX_SECTIONS = 12
MAX_WORKFLOWS = 6
MAX_EXPERTS = 4
MAX_FORM_GROUPS = 3

IOName = Annotated[str, StringConstraints(min_length=1, max_length=100)]
Body = Annotated[str, StringConstraints(min_length=1, max_length=600)]


class _OnWorkflow(_Strict):
    workflow: Ref = Field(description="A library workflow id.")


class _OnExpert(_Strict):
    expert: Ref = Field(description="An expert id.")


class OutputRef(_Strict):
    workflow: Ref = Field(description="A workflow on this page.")
    output: IOName


class FormField(_Strict):
    input: IOName = Field(description="One of the workflow's input names.")
    label: Short | None = None
    help: Long | None = None
    default: str | float | bool | None = None
    group: Short | None = Field(None, description="Fields in one group show together.")
    from_output: OutputRef | None = Field(
        None, description="Start from another workflow's latest output."
    )


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


class RowAction(_Strict):
    label: Short
    workflow: Ref = Field(description="A workflow on this page to run per row.")
    input: IOName
    column: Short = Field(description="The column key whose value the input takes.")


class TableCard(_Strict):
    type: Literal["table"]
    title: Medium | None = None
    output: IOName = Field(description="An output holding a list.")
    columns: list[TableColumn] = Field(min_length=1, max_length=6)
    row_action: RowAction | None = None


class ChartCard(_Strict):
    type: Literal["chart"]
    title: Medium | None = None
    kind: Literal["line", "area", "bar"]
    output: IOName = Field(description="An output holding a list of objects.")
    x_key: Short
    y_keys: list[Short] = Field(min_length=1, max_length=3)
    unit: Short | None = None


class GalleryCard(_Strict):
    type: Literal["gallery"]
    title: Medium | None = None
    output: IOName = Field(description="An output holding workspace image refs.")


class TimelineCard(_Strict):
    type: Literal["timeline"]
    title: Medium | None = None
    output: IOName = Field(description="An output holding a list of objects.")
    date_key: Short
    title_key: Short
    detail_key: Short | None = None


Card = Annotated[
    StatsCard | TableCard | ChartCard | GalleryCard | TimelineCard,
    Field(discriminator="type"),
]
StoredCard = Annotated[
    StatsBlock
    | ChartBlock
    | TableBlock
    | GalleryBlock
    | TimelineBlock
    | WorkflowDiagramBlock,
    Field(discriminator="type"),
]


class TextSection(_Strict):
    kind: Literal["text"]
    heading: Medium | None = None
    body: Body = Field(description="Plain text, no markup.")
    tone: Literal["takeaway", "warning"] | None = None


class CardsSection(_Strict):
    kind: Literal["cards"]
    blocks: list[StoredCard] = Field(min_length=1, max_length=4)
    written_on: date | None = Field(None, description="Set when saved.")


class RunFormSection(_OnWorkflow):
    kind: Literal["run_form"]
    fields: list[FormField] = Field(min_length=1, max_length=30)
    button_label: Short | None = None


class QuickRunsSection(_OnWorkflow):
    kind: Literal["quick_runs"]


class LatestResultSection(_OnWorkflow):
    kind: Literal["latest_result"]
    cards: list[Card] = Field(default_factory=list, max_length=4)


class RecentRunsSection(_OnWorkflow):
    kind: Literal["recent_runs"]
    count: int = Field(5, ge=3, le=10)


class HistoryColumn(_Strict):
    label: Short
    output: IOName


class RunHistorySection(_OnWorkflow):
    kind: Literal["run_history"]
    columns: list[HistoryColumn] = Field(min_length=1, max_length=4)
    count: int = Field(10, ge=3, le=20)


class RunNumbersSection(_OnWorkflow):
    kind: Literal["run_numbers"]
    period: Literal["today", "7d", "30d"] = "7d"


class ScheduleSection(_OnWorkflow):
    kind: Literal["schedule"]


class RunTrendSection(_OnWorkflow):
    kind: Literal["run_trend"]
    value: Literal["cost", "duration", "output"]
    output: IOName | None = Field(
        None, description="A numeric output, for value 'output'."
    )
    runs: int = Field(15, ge=5, le=30)


class ExpertStatusSection(_OnExpert):
    kind: Literal["expert_status"]


class ExpertWorkSection(_OnExpert):
    kind: Literal["expert_work"]
    count: int = Field(5, ge=3, le=10)


class ExpertLearnedSection(_OnExpert):
    kind: Literal["expert_learned"]
    count: int = Field(5, ge=3, le=10)


class ExpertActivitySection(_OnExpert):
    kind: Literal["expert_activity"]


class WaitingSection(_Strict):
    kind: Literal["waiting"]
    workflow: Ref | None = None
    expert: Ref | None = None

    @model_validator(mode="after")
    def _one_scope(self) -> Self:
        if self.workflow and self.expert:
            raise ValueError("waiting takes a workflow or an expert, not both")
        return self


WorkflowSection = (
    RunFormSection
    | QuickRunsSection
    | LatestResultSection
    | RecentRunsSection
    | RunHistorySection
    | RunNumbersSection
    | ScheduleSection
    | RunTrendSection
)
ExpertSection = (
    ExpertStatusSection
    | ExpertWorkSection
    | ExpertLearnedSection
    | ExpertActivitySection
)
Section = Annotated[
    TextSection
    | CardsSection
    | WorkflowSection
    | ExpertSection
    | WaitingSection
    | CollectionSection,
    Field(discriminator="kind"),
]


class PageDesign(_Strict):
    title: Medium
    sections: list[Section] = Field(min_length=1, max_length=MAX_SECTIONS)

    @model_validator(mode="after")
    def _within_limits(self) -> Self:
        if len(self.workflow_ids()) > MAX_WORKFLOWS:
            raise ValueError(f"a page draws on at most {MAX_WORKFLOWS} workflows")
        if len(self.expert_ids()) > MAX_EXPERTS:
            raise ValueError(f"a page draws on at most {MAX_EXPERTS} experts")
        if len(self.collection_ids()) > MAX_COLLECTIONS_PER_PAGE:
            raise ValueError(
                f"a page draws on at most {MAX_COLLECTIONS_PER_PAGE} collections"
            )
        return self

    def workflow_ids(self) -> list[str]:
        return _distinct(section_workflow(s) for s in self.sections)

    def expert_ids(self) -> list[str]:
        return _distinct(section_expert(s) for s in self.sections)

    def collection_ids(self) -> list[str]:
        return _distinct(section_collection(s) for s in self.sections)


def section_workflow(section: Any) -> str | None:
    if isinstance(section, (_OnWorkflow, WaitingSection)):
        return section.workflow
    return None


def section_expert(section: Any) -> str | None:
    if isinstance(section, (_OnExpert, WaitingSection)):
        return section.expert
    return None


class DesignProblem(BaseModel):
    section: int
    message: str


UNAVAILABLE = "no longer available"


def check_design(design: PageDesign, sources: "PageSources") -> list[DesignProblem]:
    """Every way *design* fails its sources, by section index.

    Run when Otto proposes a page, when one is saved and on every read, where a
    section with a problem is left out instead of refused. A source that could
    not be loaded just now is not a problem here; the read reports it.
    """
    on_page = set(design.workflow_ids())
    problems: list[DesignProblem] = []
    for index, section in enumerate(design.sections):
        problems += [
            DesignProblem(section=index, message=message)
            for message in _section_problems(section, sources, on_page)
        ]
    return problems


def _section_problems(
    section: Any, sources: "PageSources", on_page: set[str]
) -> list[str]:
    workflow, expert = section_workflow(section), section_expert(section)
    if (
        workflow
        and workflow not in sources.failed
        and not sources.workflows.get(workflow)
    ):
        return [f"its workflow is {UNAVAILABLE}"]
    if expert and expert not in sources.failed and not sources.experts.get(expert):
        return [f"its expert is {UNAVAILABLE}"]
    if collection := section_collection(section):
        found = sources.collections.get(collection)
        if collection in sources.failed:
            return []
        if found is None:
            return [f"its collection is {UNAVAILABLE}"]
        return collection_problems(section, found) + _button_problems(
            section_buttons(section), sources, on_page
        )
    agent = sources.workflows.get(workflow or "")
    if agent is None:
        return []
    if isinstance(section, RunFormSection):
        return _form_problems(section, agent.input_schema, sources, on_page)
    if isinstance(section, LatestResultSection):
        return _result_problems(section, agent.output_schema, sources, on_page)
    if isinstance(section, RunHistorySection):
        return _outputs_missing(
            [c.output for c in section.columns], agent.output_schema
        )
    if isinstance(section, RunTrendSection) and section.value == "output":
        if not section.output:
            return ["an output trend names an output"]
        return _outputs_missing([section.output], agent.output_schema)
    return []


def _form_problems(
    section: RunFormSection,
    input_schema: dict[str, Any],
    sources: "PageSources",
    on_page: set[str],
) -> list[str]:
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
    if len({f.group for f in section.fields if f.group}) > MAX_FORM_GROUPS:
        problems.append(f"the form has more than {MAX_FORM_GROUPS} groups")
    problems += [
        f"the default for '{field.input}' does not fit that input"
        for field in section.fields
        if field.input in inputs
        and field.default is not None
        and not _default_fits(field.default, inputs[field.input])
    ]
    for field in section.fields:
        if field.from_output:
            ref = _ref_problems(field.from_output.workflow, on_page, sources)
            target = sources.workflows.get(field.from_output.workflow)
            if not ref and target is not None:
                ref = _outputs_missing([field.from_output.output], target.output_schema)
            problems += ref
    return problems


def _result_problems(
    section: LatestResultSection,
    output_schema: dict[str, Any],
    sources: "PageSources",
    on_page: set[str],
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
        if isinstance(card, TableCard) and card.row_action:
            problems += _row_action_problems(card, sources, on_page)
    return problems


def _row_action_problems(
    card: TableCard, sources: "PageSources", on_page: set[str]
) -> list[str]:
    action = card.row_action
    assert action is not None
    problems = _ref_problems(action.workflow, on_page, sources)
    if action.column not in {c.key for c in card.columns if c.key}:
        problems.append(
            f"the row button reads column '{action.column}', which the table lacks"
        )
    target = sources.workflows.get(action.workflow)
    if target is not None and action.input not in (
        target.input_schema.get("properties") or {}
    ):
        problems.append(
            f"the row button's workflow has no input named '{action.input}'"
        )
    return problems


def _button_problems(
    buttons: list[RowButton], sources: "PageSources", on_page: set[str]
) -> list[str]:
    problems: list[str] = []
    for button in buttons:
        problems += _ref_problems(button.workflow, on_page, sources)
        target = sources.workflows.get(button.workflow)
        inputs = (target.input_schema.get("properties") or {}) if target else {}
        problems += [
            f"the '{button.label}' button's workflow has no input named '{name}'"
            for name in button.inputs
            if target is not None and name not in inputs
        ]
    return problems


def _ref_problems(
    workflow: str, on_page: set[str], sources: "PageSources"
) -> list[str]:
    if workflow not in on_page:
        return ["a binding names a workflow that has no section on this page"]
    if workflow not in sources.failed and sources.workflows.get(workflow) is None:
        return [f"a bound workflow is {UNAVAILABLE}"]
    return []


def _outputs_missing(names: list[str], output_schema: dict[str, Any]) -> list[str]:
    outputs = output_schema.get("properties") or {}
    return [
        f"the workflow has no output named '{n}'" for n in names if n not in outputs
    ]


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


def _type_fits(card: Any, declared: str | None) -> bool:
    # Most outputs declare no type; those are checked against the run's value.
    if declared is None:
        return True
    if isinstance(card, StatsCard):
        return declared not in ("array", "object")
    return declared == "array"


def _distinct(values: Any) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen
