"""Tests for parsing Exa Agent API responses."""

from backend.blocks.exa.agent_fixtures import (
    SAMPLE_COMPLETED_RUN,
    SAMPLE_EVENT_LIST,
    SAMPLE_FAILED_EVENT_LIST,
    SAMPLE_QUEUED_RUN,
)
from backend.blocks.exa.agent_models import (
    AgentCitation,
    AgentGrounding,
    failure_message_from_events,
    parse_event,
    parse_page,
    parse_run,
)


def test_parse_completed_run():
    run = parse_run(SAMPLE_COMPLETED_RUN)
    assert run.run_id == SAMPLE_COMPLETED_RUN["id"]
    assert run.status == "completed"
    assert run.is_terminal
    assert run.stop_reason == "schema_satisfied"
    assert run.query == SAMPLE_COMPLETED_RUN["request"]["query"]
    assert run.effort == "medium"
    assert run.text == SAMPLE_COMPLETED_RUN["output"]["text"]
    assert run.structured == SAMPLE_COMPLETED_RUN["output"]["structured"]
    assert run.grounding == [
        AgentGrounding(
            field="assistants[0].price",
            citations=[
                AgentCitation(url="https://example.com/pricing", title="Pricing")
            ],
            confidence="high",
        )
    ]
    assert run.usage.agent_compute_units == 0.6
    assert run.usage.searches == 4
    assert run.cost.total == 0.1
    assert run.cost.search == 0.02
    assert run.completed_at == SAMPLE_COMPLETED_RUN["completedAt"]


def test_parse_queued_run_is_not_terminal():
    run = parse_run(SAMPLE_QUEUED_RUN)
    assert not run.is_terminal
    assert run.stop_reason is None
    assert run.completed_at is None


def test_parse_run_tolerates_null_request_and_data_source_breakdowns():
    run = parse_run(
        {
            **SAMPLE_COMPLETED_RUN,
            "request": None,
            "usage": {**SAMPLE_COMPLETED_RUN["usage"], "dataSources": {"fiber": 3}},
            "costDollars": {
                **SAMPLE_COMPLETED_RUN["costDollars"],
                "dataSources": {"fiber": 0.06},
            },
        }
    )
    assert run.query == ""
    assert run.effort is None
    assert run.usage.data_sources == {"fiber": 3}
    assert run.cost.data_sources == {"fiber": 0.06}


def test_parse_run_reads_an_error_object_if_exa_includes_one():
    run = parse_run(
        {**SAMPLE_COMPLETED_RUN, "error": {"code": "E", "message": "Broke"}}
    )
    assert run.error == "E: Broke"


def test_failure_message_from_failed_event():
    events = [parse_event(e) for e in SAMPLE_FAILED_EVENT_LIST["data"]]
    assert failure_message_from_events(events) == "INTERNAL_ERROR: Upstream timeout"


def test_failure_message_absent_without_failed_event():
    events = [parse_event(e) for e in SAMPLE_EVENT_LIST["data"]]
    assert failure_message_from_events(events) is None


def test_failure_message_ignores_unexpected_event_payloads():
    events = [parse_event({"id": "9", "event": "agent_run.failed", "data": "oops"})]
    assert failure_message_from_events(events) is None


def test_parse_page():
    page = parse_page({"hasMore": True, "nextCursor": "abc"})
    assert page.has_more is True
    assert page.next_cursor == "abc"
    assert parse_page({}).has_more is False
