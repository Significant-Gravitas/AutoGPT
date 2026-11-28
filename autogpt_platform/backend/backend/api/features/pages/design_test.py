from typing import Any

import pytest
from pydantic import ValidationError

from backend.api.features.library.model import LibraryAgent

from .design import PageDesign, check_design
from .sources import PageSources


def agent(
    library_id: str,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    required: list[str] = (),
) -> LibraryAgent:
    return LibraryAgent.model_construct(
        id=library_id,
        graph_id=f"graph-{library_id}",
        graph_version=1,
        name=library_id.title(),
        input_schema={
            "type": "object",
            "properties": inputs or {},
            "required": list(required),
        },
        output_schema={"type": "object", "properties": outputs or {}},
    )


IDEAS = agent(
    "ideas", {"topic": {"type": "string"}}, {"keywords": {"type": "array"}, "topic": {}}
)
DRAFT = agent(
    "draft", {"keyword": {"type": "string"}}, {"title": {}, "draft": {}}, ["keyword"]
)
SOURCES = PageSources(workflows={"ideas": IDEAS, "draft": DRAFT})


def page(*sections: dict[str, Any]) -> PageDesign:
    return PageDesign.model_validate({"title": "Posts", "sections": list(sections)})


def table(**row_action: Any) -> dict[str, Any]:
    card: dict[str, Any] = {
        "type": "table",
        "output": "keywords",
        "columns": [{"label": "Keyword", "key": "keyword"}],
    }
    if row_action:
        card["row_action"] = {"label": "Draft", "workflow": "draft", **row_action}
    return {"kind": "latest_result", "workflow": "ideas", "cards": [card]}


DRAFT_FORM = {"kind": "run_form", "workflow": "draft", "fields": [{"input": "keyword"}]}


def problems(
    design: PageDesign, sources: PageSources = SOURCES
) -> list[tuple[int, str]]:
    return [(p.section, p.message) for p in check_design(design, sources)]


def test_a_page_binding_two_workflows_correctly_has_no_problems() -> None:
    design = page(
        table(input="keyword", column="keyword"),
        {
            "kind": "run_form",
            "workflow": "draft",
            "fields": [
                {
                    "input": "keyword",
                    "from_output": {"workflow": "ideas", "output": "topic"},
                }
            ],
        },
        {
            "kind": "run_history",
            "workflow": "draft",
            "columns": [{"label": "T", "output": "title"}],
        },
        {"kind": "waiting", "workflow": "ideas"},
    )

    assert problems(design) == []
    assert design.workflow_ids() == ["ideas", "draft"]


@pytest.mark.parametrize(
    "section, expected",
    [
        (
            {"kind": "run_form", "workflow": "draft", "fields": [{"input": "nope"}]},
            "the workflow has no input named 'nope'",
        ),
        (
            {
                "kind": "run_form",
                "workflow": "ideas",
                "fields": [{"input": "topic", "default": 3}],
            },
            "the default for 'topic' does not fit that input",
        ),
        (
            {
                "kind": "run_history",
                "workflow": "draft",
                "columns": [{"label": "X", "output": "gone"}],
            },
            "the workflow has no output named 'gone'",
        ),
        (
            {"kind": "run_trend", "workflow": "draft", "value": "output"},
            "an output trend names an output",
        ),
        (
            {"kind": "recent_runs", "workflow": "deleted"},
            "its workflow is no longer available",
        ),
        (
            {"kind": "expert_status", "expert": "someone-elses"},
            "its expert is no longer available",
        ),
    ],
)
def test_a_section_that_does_not_fit_its_source_is_reported(
    section: dict[str, Any], expected: str
) -> None:
    sources = SOURCES.model_copy(
        update={
            "workflows": {**SOURCES.workflows, "deleted": None},
            "experts": {"someone-elses": None},
        }
    )
    assert problems(page(DRAFT_FORM, section), sources)[0] == (1, expected)


@pytest.mark.parametrize(
    "row_action, sections, expected",
    [
        (
            {"input": "keyword", "column": "keyword"},
            [],
            "a binding names a workflow that has no section on this page",
        ),
        (
            {"input": "keyword", "column": "missing"},
            [DRAFT_FORM],
            "the row button reads column 'missing', which the table lacks",
        ),
        (
            {"input": "nope", "column": "keyword"},
            [DRAFT_FORM],
            "the row button's workflow has no input named 'nope'",
        ),
    ],
)
def test_a_row_button_must_run_a_workflow_on_the_page_with_a_real_input(
    row_action: dict[str, Any], sections: list[dict[str, Any]], expected: str
) -> None:
    assert problems(page(table(**row_action), *sections)) == [(0, expected)]


def test_a_bound_field_must_read_an_output_of_a_workflow_on_the_page() -> None:
    bound = {
        "kind": "run_form",
        "workflow": "draft",
        "fields": [
            {"input": "keyword", "from_output": {"workflow": "ideas", "output": "nope"}}
        ],
    }

    assert problems(page(bound)) == [
        (0, "a binding names a workflow that has no section on this page")
    ]
    assert problems(page(bound, table())) == [
        (0, "the workflow has no output named 'nope'")
    ]


def test_a_source_that_could_not_be_read_is_not_a_design_problem() -> None:
    sources = PageSources(workflows={"draft": None}, failed={"draft"})

    assert problems(page(DRAFT_FORM), sources) == []


@pytest.mark.parametrize(
    "sections, message",
    [
        ([{"kind": "text", "body": "x"}] * 13, "at most 12"),
        (
            [{"kind": "schedule", "workflow": f"w{i}"} for i in range(7)],
            "at most 6 workflows",
        ),
        (
            [{"kind": "expert_status", "expert": f"e{i}"} for i in range(5)],
            "at most 4 experts",
        ),
        ([{"kind": "waiting", "workflow": "a", "expert": "b"}], "not both"),
        ([{"kind": "text", "body": "x", "html": "<b>"}], "Extra inputs"),
    ],
)
def test_the_page_limits_hold(sections: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        page(*sections)
