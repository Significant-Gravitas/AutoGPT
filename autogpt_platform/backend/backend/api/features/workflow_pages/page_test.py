import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.api.features.library.model import LibraryAgent
from backend.copilot.tools.design_workflow_page import WorkflowPageResponse
from backend.data.execution import ExecutionStatus, GraphExecution, GraphExecutionMeta

from .design import WorkflowPageDesign
from .page import PageLatestResult, PageRunForm, resolve_page

INPUTS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company": {"type": "string", "title": "Company"},
        "notes": {"type": "string", "title": "Notes"},
    },
    "required": ["company"],
}
OUTPUTS: dict[str, Any] = {
    "type": "object",
    "properties": {"leads": {}, "lead_count": {}, "report": {}},
}
DESIGN = WorkflowPageDesign.model_validate(
    {
        "title": "Lead research",
        "sections": [
            {
                "kind": "run_form",
                "fields": [{"input": "company", "label": "Company name"}],
            },
            {
                "kind": "latest_result",
                "cards": [
                    {
                        "type": "stats",
                        "items": [{"label": "Leads", "output": "lead_count"}],
                    },
                    {
                        "type": "table",
                        "output": "leads",
                        "columns": [
                            {"label": "Name", "key": "name"},
                            {"label": "Role", "key": "role"},
                        ],
                    },
                ],
            },
            {"kind": "recent_runs", "count": 3},
        ],
    }
)


def _agent(inputs: dict[str, Any] = INPUTS) -> LibraryAgent:
    return LibraryAgent.model_construct(
        id="lib-1",
        graph_id="graph-1",
        input_schema=inputs,
        output_schema=OUTPUTS,
        is_scheduled=False,
        next_scheduled_run=None,
        schedule_info_unavailable=False,
    )


def _run(outputs: dict[str, list[Any]]) -> GraphExecution:
    return GraphExecution.model_construct(
        id="run-1",
        status=ExecutionStatus.COMPLETED,
        ended_at=datetime(2026, 11, 20, tzinfo=UTC),
        outputs=outputs,
    )


def _result(page_sections: list[Any]) -> PageLatestResult:
    return next(s for s in page_sections if isinstance(s, PageLatestResult))


def test_a_table_binding_turns_a_list_output_into_table_rows() -> None:
    leads = [{"name": "Ada", "role": "CTO"}, {"name": "Lin", "role": "VP Sales"}]
    run = _run({"leads": [leads], "lead_count": [2], "report": ["Two leads."]})

    result = _result(resolve_page(DESIGN, _agent(), run, []).sections)

    stats, table = result.blocks
    assert stats.model_dump()["items"][0] == {
        "label": "Leads",
        "value": "2",
        "change": None,
        "tone": None,
    }
    assert table.model_dump()["rows"] == [["Ada", "CTO"], ["Lin", "VP Sales"]]
    assert result.other_outputs == {"report": ["Two leads."]}


def test_a_value_a_card_cannot_show_falls_back_to_the_plain_outputs() -> None:
    run = _run({"leads": ["not a list of rows"], "lead_count": [2]})

    result = _result(resolve_page(DESIGN, _agent(), run, []).sections)

    assert [b.type for b in result.blocks] == ["stats"]
    assert result.other_outputs == {"leads": ["not a list of rows"]}


def test_a_workflow_with_no_runs_gets_an_empty_result_section() -> None:
    result = _result(resolve_page(DESIGN, _agent(), None, []).sections)

    assert result.run_id is None
    assert result.blocks == []
    assert result.other_outputs == {}


def test_a_section_the_workflow_no_longer_fits_is_left_out_and_named() -> None:
    without_company = {"type": "object", "properties": {"notes": {"type": "string"}}}

    page = resolve_page(DESIGN, _agent(without_company), None, [])

    assert [s.kind for s in page.sections] == ["latest_result", "recent_runs"]
    assert [(s.index, s.kind) for s in page.left_out] == [(0, "run_form")]
    assert "company" in page.left_out[0].problem


def test_the_form_keeps_the_inputs_the_design_leaves_out() -> None:
    form = next(
        s
        for s in resolve_page(DESIGN, _agent(), None, []).sections
        if isinstance(s, PageRunForm)
    )

    assert [(f.input, f.label, f.group) for f in form.fields] == [
        ("company", "Company name", None),
        ("notes", "Notes", "More options"),
    ]


def test_recent_runs_are_capped_at_the_count_the_design_asks_for() -> None:
    runs = [
        GraphExecutionMeta.model_construct(
            id=f"run-{i}",
            status=ExecutionStatus.COMPLETED,
            started_at=None,
            ended_at=None,
            stats=None,
        )
        for i in range(6)
    ]

    page = resolve_page(DESIGN, _agent(), None, runs)

    recent = page.sections[-1]
    assert recent.kind == "recent_runs"
    assert [run.id for run in recent.runs] == ["run-0", "run-1", "run-2"]


FIXTURE = (
    Path(__file__).parents[5]
    / "frontend/src/app/(platform)/copilot/tools/DesignWorkflowPage/__tests__"
    / "workflowPageOutput.json"
)


def test_the_tool_output_matches_the_frontend_fixture() -> None:
    """The chat card's test renders this same file, so the two can't drift."""
    leads = [{"name": "Ada", "role": "CTO"}, {"name": "Lin", "role": "VP Sales"}]
    run = _run({"leads": [leads], "lead_count": [2], "report": ["Two leads."]})
    response = WorkflowPageResponse(
        message="Preview shown to the user.",
        page=resolve_page(DESIGN, _agent(), run, []),
    )
    output = json.loads(response.model_dump_json(exclude_none=True))

    if os.environ.get("UPDATE_WORKFLOW_PAGE_FIXTURE"):
        FIXTURE.write_text(json.dumps(output, indent=2) + "\n")

    assert output == json.loads(FIXTURE.read_text())
