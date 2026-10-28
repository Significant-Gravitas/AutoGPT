"""Tests for SmartSearchBlock: needs_input continuation flow and initial-vs-
continuation input validation."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from pytest_snapshot.plugin import Snapshot

from backend.blocks.dataforb2b._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_META_INPUT,
)
from backend.blocks.dataforb2b.reasoning import MAX_RESULTS, SmartSearchBlock


async def _run_and_collect(input_data, response=None):
    block = SmartSearchBlock()
    outputs = {}
    if response is None:
        response = {
            "status": "needs_input",
            "session_id": "sess-1",
            "questions": [{"id": "q1", "text": "Which region?"}],
        }
    with patch.object(
        SmartSearchBlock,
        "reasoning_search",
        new=AsyncMock(return_value=response),
    ) as mock_search:
        async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS):
            outputs[name] = value
    return outputs, mock_search


@pytest.mark.asyncio
async def test_needs_input_status_surfaces_questions_and_session_id(
    snapshot: Snapshot,
):
    """Thread ED6Z: the needs_input turn must surface status/questions/session_id
    so a caller can resolve it in a follow-up call."""
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        query="marketing directors",
        category="people",
    )
    outputs, _ = await _run_and_collect(input_data)
    assert outputs["status"] == "needs_input"
    assert outputs["session_id"] == "sess-1"
    assert outputs["questions"] == [{"id": "q1", "text": "Which region?"}]
    assert outputs["results"] == []
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(outputs, indent=2, sort_keys=True),
        "smart_search_needs_input_outputs",
    )


@pytest.mark.asyncio
async def test_continuation_call_with_session_id_and_answers():
    """A follow-up call resolving a needs_input turn omits 'query' and supplies
    'session_id' + 'answers' instead."""
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        session_id="sess-1",
        answers={"q1": "Europe"},
    )
    outputs, mock_search = await _run_and_collect(input_data)
    payload = mock_search.await_args.args[0]
    assert payload["session_id"] == "sess-1"
    assert payload["answers"] == {"q1": "Europe"}
    assert "query" not in payload
    assert outputs["status"] == "needs_input"


@pytest.mark.asyncio
async def test_missing_query_and_continuation_data_raises():
    """Thread ED7X/DxBo: initial calls need 'query'; continuation calls need both
    'session_id' and 'answers'. Neither present must raise."""
    block = SmartSearchBlock()
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
    )
    with pytest.raises(ValueError):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_session_id_without_answers_raises():
    """Thread ED7X/DxBo: partial continuation data (only session_id, no answers)
    must raise rather than silently proceeding."""
    block = SmartSearchBlock()
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        session_id="sess-1",
    )
    with pytest.raises(ValueError):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_query_and_continuation_data_together_raises():
    """Thread ED7X/DxBo: providing 'query' together with 'session_id'/'answers'
    is ambiguous and must raise rather than picking one silently."""
    block = SmartSearchBlock()
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        query="marketing directors",
        session_id="sess-1",
        answers={"q1": "Europe"},
    )
    with pytest.raises(ValueError):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_max_results_clamped_to_upper_bound():
    """Thread ED6B: max_results above MAX_RESULTS must be clamped."""
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        query="engineers",
        max_results=999999,
    )
    _, mock_search = await _run_and_collect(input_data)
    assert mock_search.await_args.args[0]["max_results"] == MAX_RESULTS


@pytest.mark.asyncio
async def test_category_output_echoes_normalized_input_category():
    """Thread ED7H/ED57: category output must echo the (validated) input
    category, not raw API response data."""
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        query="engineers",
        category="company",
    )
    outputs, _ = await _run_and_collect(input_data)
    assert outputs["category"] == "company"


@pytest.mark.asyncio
async def test_max_results_clamped_to_lower_bound():
    """max_results of 0 (or negative) must clamp up to 1, not reach the API as 0."""
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        query="engineers",
        max_results=0,
    )
    _, mock_search = await _run_and_collect(input_data)
    assert mock_search.await_args.args[0]["max_results"] == 1


@pytest.mark.asyncio
async def test_success_status_is_passed_through_verbatim():
    """The live API returns status='complete' on success, not 'ok'. Verified
    against api.dataforb2b.ai on 2026-08-17. Graph authors branch on this
    value, so it must reach them unaltered rather than being normalized."""
    input_data = SmartSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        query="marketing directors",
    )
    outputs, _ = await _run_and_collect(
        input_data,
        response={"status": "complete", "total": 121, "results": [{"id": "1"}]},
    )
    assert outputs["status"] == "complete"
