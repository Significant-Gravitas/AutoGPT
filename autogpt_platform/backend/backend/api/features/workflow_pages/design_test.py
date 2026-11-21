from typing import Any

import pytest
from pydantic import ValidationError

from .design import WorkflowPageDesign, check_design

INPUTS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company": {"type": "string", "title": "Company"},
        "depth": {"type": "string", "enum": ["quick", "deep"]},
        "max_leads": {"type": "integer"},
    },
    "required": ["company"],
}
OUTPUTS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "leads": {"title": "leads"},
        "lead_count": {"title": "lead_count"},
        "summary": {"type": "string"},
    },
}


def _design(
    form: list[dict[str, Any]], cards: list[dict[str, Any]]
) -> WorkflowPageDesign:
    return WorkflowPageDesign.model_validate(
        {
            "title": "Lead research",
            "sections": [
                {"kind": "text", "body": "Finds leads for one company."},
                {"kind": "run_form", "fields": form},
                {"kind": "latest_result", "cards": cards},
            ],
        }
    )


VALID_FORM = [{"input": "company", "label": "Company name"}]
VALID_CARDS = [
    {"type": "stats", "items": [{"label": "Leads", "output": "lead_count"}]},
    {"type": "table", "output": "leads", "columns": [{"label": "Name", "key": "name"}]},
]


def test_a_design_that_fits_the_workflow_has_no_problems() -> None:
    form = [*VALID_FORM, {"input": "depth", "default": "deep"}]
    assert check_design(_design(form, VALID_CARDS), INPUTS, OUTPUTS) == []


@pytest.mark.parametrize(
    "form,cards,section,expected",
    [
        ([*VALID_FORM, {"input": "region"}], VALID_CARDS, 1, "no input named 'region'"),
        ([{"input": "max_leads"}], VALID_CARDS, 1, "required input 'company'"),
        (
            VALID_FORM,
            [{"type": "table", "output": "people", "columns": [{"label": "N"}]}],
            2,
            "no output named 'people'",
        ),
        (
            VALID_FORM,
            [{"type": "table", "output": "summary", "columns": [{"label": "N"}]}],
            2,
            "a table card cannot show output 'summary'",
        ),
        (
            [*VALID_FORM, {"input": "depth", "default": "medium"}],
            VALID_CARDS,
            1,
            "the default for 'depth' does not fit",
        ),
        (
            [*VALID_FORM, {"input": "max_leads", "default": 2.5}],
            VALID_CARDS,
            1,
            "the default for 'max_leads' does not fit",
        ),
    ],
    ids=[
        "unknown input",
        "missing required input",
        "unknown output",
        "card bound to an output of the wrong type",
        "default outside a dropdown's options",
        "fractional default for a whole number",
    ],
)
def test_a_design_that_does_not_fit_is_refused(
    form: list[dict[str, Any]], cards: list[dict[str, Any]], section: int, expected: str
) -> None:
    problems = check_design(_design(form, cards), INPUTS, OUTPUTS)

    assert [p.section for p in problems] == [section]
    assert expected in problems[0].message


def test_each_section_kind_but_text_appears_once() -> None:
    with pytest.raises(ValidationError, match="appear once"):
        WorkflowPageDesign.model_validate(
            {
                "title": "Twice",
                "sections": [{"kind": "schedule"}, {"kind": "schedule"}],
            }
        )


def test_a_design_has_no_field_for_markup_or_links() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        WorkflowPageDesign.model_validate(
            {
                "title": "Page",
                "sections": [{"kind": "text", "body": "Hi", "html": "<b>hi</b>"}],
            }
        )
