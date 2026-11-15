"""Payload and polling tests for the Websets search blocks."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa.websets_fixtures import (
    SEARCH_ID,
    WEBSET_ID,
    collect_outputs,
    sample_search,
    sample_webset,
)
from backend.blocks.exa.websets_search import (
    ExaCreateWebsetSearchBlock,
    ExaFindOrCreateSearchBlock,
    ExaGetWebsetSearchBlock,
)


@pytest.mark.asyncio
async def test_create_search_payload_and_legacy_merge_behavior():
    block = ExaCreateWebsetSearchBlock()
    api = AsyncMock(return_value=sample_search(status="pending"))
    with patch.object(block, "_api", api):
        await collect_outputs(
            block,
            webset_id=WEBSET_ID,
            query="CTOs",
            count=20,
            entity_type="person",
            behavior="merge",
            max_people_per_company=1,
            scope_source_ids=["ws_1"],
            scope_source_types=["webset"],
            scope_relationships=["CTOs of"],
            scope_relationship_limits=[1],
        )

    assert api.call_args.args[1:3] == ("POST", f"/websets/{WEBSET_ID}/searches")
    assert api.call_args.kwargs["json"] == {
        "query": "CTOs",
        "count": 20,
        "behavior": "append",
        "recall": True,
        "entity": {"type": "person"},
        "scope": [
            {
                "source": "webset",
                "id": "ws_1",
                "relationship": {"definition": "CTOs of", "limit": 1},
            }
        ],
        "maxPeoplePerCompany": 1,
    }


@pytest.mark.asyncio
async def test_wait_stops_on_canceled_search():
    """'canceled' (Exa's spelling) must end the wait, not run to the timeout."""
    block = ExaCreateWebsetSearchBlock()
    api = AsyncMock(
        side_effect=[
            sample_search(status="pending"),
            sample_search(status="canceled", canceledReason="out_of_credits"),
        ]
    )
    with patch.object(block, "_api", api):
        outputs = dict(
            await collect_outputs(
                block, webset_id=WEBSET_ID, query="CTOs", wait_for_completion=True
            )
        )

    assert api.await_count == 2
    assert outputs["status"] == "canceled"
    assert outputs["items_found"] == 5


@pytest.mark.asyncio
async def test_create_search_expected_results_from_recall():
    block = ExaCreateWebsetSearchBlock()
    recall = {
        "expected": {
            "total": 120,
            "confidence": "high",
            "bounds": {"min": 100, "max": 150},
        },
        "reasoning": "Many matches",
    }
    with patch.object(
        block, "_api", AsyncMock(return_value=sample_search(recall=recall))
    ):
        outputs = dict(await collect_outputs(block, webset_id=WEBSET_ID, query="CTOs"))

    assert outputs["expected_results"] == {
        "total": 120,
        "confidence": "high",
        "min": 100,
        "max": 150,
        "reasoning": "Many matches",
    }


@pytest.mark.asyncio
async def test_get_search_reports_new_fields_and_pending():
    block = ExaGetWebsetSearchBlock()
    data = sample_search(status="pending", maxPeoplePerCompany=3, behavior="append")
    with patch.object(block, "_api", AsyncMock(return_value=data)):
        outputs = dict(
            await collect_outputs(block, webset_id=WEBSET_ID, search_id=SEARCH_ID)
        )

    assert outputs["status"] == "pending"
    assert outputs["max_people_per_company"] == 3
    assert outputs["behavior"] == "append"
    assert outputs["progress"]["found"] == 5


@pytest.mark.asyncio
async def test_find_or_create_reuses_matching_query():
    block = ExaFindOrCreateSearchBlock()
    api = AsyncMock(return_value=sample_webset())
    with patch.object(block, "_api", api):
        outputs = dict(
            await collect_outputs(
                block, webset_id=WEBSET_ID, query="  ai startups in europe "
            )
        )

    assert api.await_count == 1
    assert outputs["was_created"] is False
    assert outputs["search_id"] == SEARCH_ID
