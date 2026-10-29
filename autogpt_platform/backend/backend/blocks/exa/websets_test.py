"""Payload and response tests for the core Websets blocks."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa._exa_rest import ExaApiError
from backend.blocks.exa.websets import (
    ExaCreateOrFindWebsetBlock,
    ExaCreateWebsetBlock,
    ExaGetWebsetBlock,
    ExaListWebsetsBlock,
    ExaPreviewWebsetBlock,
    ExaUpdateWebsetBlock,
    ExaWebsetStatusBlock,
    ExaWebsetSummaryBlock,
)
from backend.blocks.exa.websets_fixtures import (
    WEBSET_ID,
    collect_outputs,
    sample_preview,
    sample_webset,
)


def _api(*responses):
    return AsyncMock(side_effect=list(responses))


@pytest.mark.asyncio
async def test_create_webset_sends_every_spec_field():
    block = ExaCreateWebsetBlock()
    api = _api(sample_webset(status="pending"))
    with patch.object(block, "_api", api):
        await collect_outputs(
            block,
            search_query="VPs of engineering at AI labs",
            title="AI leaders",
            search_count=25,
            search_entity_type="custom",
            search_entity_description="Engineering leaders",
            search_criteria=["Works at an AI lab"],
            search_recall=True,
            search_max_people_per_company=2,
            search_exclude_sources=["imp_1"],
            search_scope_sources=["ws_1"],
            search_scope_types=["webset"],
            search_scope_relationships=["employees of"],
            search_scope_relationship_limits=[5],
            import_sources=["imp_2"],
            import_evaluate=[True],
            exclude_sources=["ws_2"],
            exclude_types=["webset"],
            enrichment_descriptions=["LinkedIn profile", "Seniority"],
            enrichment_formats=["url", "options"],
            enrichment_options=[[], ["VP", "Director"]],
            external_id="ext-1",
            metadata={"team": "growth"},
            wait_for_initial_results=False,
        )

    method, path = api.call_args.args[1:3]
    assert (method, path) == ("POST", "/websets")
    assert api.call_args.kwargs["json"] == {
        "search": {
            "query": "VPs of engineering at AI labs",
            "count": 25,
            "entity": {"type": "custom", "description": "Engineering leaders"},
            "criteria": [{"description": "Works at an AI lab"}],
            "exclude": [{"source": "import", "id": "imp_1"}],
            "scope": [
                {
                    "source": "webset",
                    "id": "ws_1",
                    "relationship": {"definition": "employees of", "limit": 5},
                }
            ],
            "maxPeoplePerCompany": 2,
            "recall": True,
        },
        "import": [{"source": "import", "id": "imp_2", "evaluate": True}],
        "enrichments": [
            {"description": "LinkedIn profile", "format": "url"},
            {
                "description": "Seniority",
                "format": "options",
                "options": [{"label": "VP"}, {"label": "Director"}],
            },
        ],
        "exclude": [{"source": "webset", "id": "ws_2"}],
        "title": "AI leaders",
        "externalId": "ext-1",
        "metadata": {"team": "growth"},
    }


@pytest.mark.asyncio
async def test_create_webset_waits_until_idle_and_counts_items():
    block = ExaCreateWebsetBlock()
    api = _api(sample_webset(status="pending"), sample_webset(status="idle"))
    with patch.object(block, "_api", api):
        outputs = dict(await collect_outputs(block, search_query="AI startups"))

    assert outputs["webset"].status == "pending"
    assert outputs["initial_item_count"] == 5
    assert api.call_args_list[1].args[1:3] == ("GET", f"/websets/{WEBSET_ID}")


@pytest.mark.asyncio
async def test_create_or_find_creates_only_on_404():
    block = ExaCreateOrFindWebsetBlock()
    api = _api(ExaApiError("not found", 404), sample_webset(externalId="e1"))
    with patch.object(block, "_api", api):
        outputs = dict(
            await collect_outputs(block, external_id="e1", search_query="AI startups")
        )

    assert outputs["was_created"] is True
    assert api.call_args_list[1].kwargs["json"] == {
        "externalId": "e1",
        "search": {"query": "AI startups", "count": 10},
    }


@pytest.mark.asyncio
async def test_create_or_find_propagates_other_errors():
    block = ExaCreateOrFindWebsetBlock()
    with patch.object(block, "_api", _api(ExaApiError("bad key", 401))), pytest.raises(
        ExaApiError, match="bad key"
    ):
        await collect_outputs(block, external_id="e1")


@pytest.mark.asyncio
async def test_update_webset_sends_title():
    block = ExaUpdateWebsetBlock()
    api = _api(sample_webset(title="Renamed"))
    with patch.object(block, "_api", api):
        outputs = dict(await collect_outputs(block, webset_id="ext/1", title="Renamed"))

    assert api.call_args.args[1:3] == ("POST", "/websets/ext%2F1")
    assert api.call_args.kwargs["json"] == {"title": "Renamed"}
    assert outputs["title"] == "Renamed"


@pytest.mark.asyncio
async def test_list_websets_passes_search_filter():
    block = ExaListWebsetsBlock()
    api = _api({"data": [sample_webset()], "hasMore": True, "nextCursor": "c2"})
    with patch.object(block, "_api", api):
        outputs = dict(await collect_outputs(block, search="startups", limit=5))

    assert api.call_args.kwargs["params"] == {
        "cursor": None,
        "limit": 5,
        "search": "startups",
    }
    assert outputs["has_more"] is True
    assert outputs["next_cursor"] == "c2"


@pytest.mark.asyncio
async def test_get_webset_expands_items_and_tolerates_pending():
    block = ExaGetWebsetBlock()
    data = sample_webset(status="pending", items=[{"id": "item_1"}])
    api = _api(data)
    with patch.object(block, "_api", api):
        outputs = dict(
            await collect_outputs(block, webset_id=WEBSET_ID, expand_items=True)
        )

    assert api.call_args.kwargs["params"] == {"expand": "items"}
    assert outputs["status"] == "pending"
    assert outputs["items"] == [{"id": "item_1"}]


@pytest.mark.asyncio
async def test_preview_wraps_query_in_search_and_requests_items():
    block = ExaPreviewWebsetBlock()
    preview = sample_preview(items=[{"id": "preview_1", "properties": {}}])
    api = _api(preview)
    with patch.object(block, "_api", api):
        outputs = dict(
            await collect_outputs(
                block,
                query="AI startups",
                entity_type="company",
                include_sample_items=True,
                sample_count=3,
            )
        )

    assert api.call_args.kwargs["json"] == {
        "search": {"query": "AI startups", "entity": {"type": "company"}, "count": 3}
    }
    assert api.call_args.kwargs["params"] == {"search": True}
    assert outputs["items"] == preview["items"]
    assert outputs["enrichment_columns"][0].format == "url"


@pytest.mark.asyncio
async def test_preview_without_items_sends_no_search_flag():
    block = ExaPreviewWebsetBlock()
    api = _api(sample_preview())
    with patch.object(block, "_api", api):
        await collect_outputs(block, query="AI startups")

    assert api.call_args.kwargs["json"] == {"search": {"query": "AI startups"}}
    assert api.call_args.kwargs["params"] == {"search": None}


@pytest.mark.asyncio
async def test_status_reports_pending_as_processing():
    block = ExaWebsetStatusBlock()
    with patch.object(block, "_api", _api(sample_webset(status="pending"))):
        outputs = dict(await collect_outputs(block, webset_id=WEBSET_ID))

    assert outputs["is_processing"] is True
    assert outputs["item_count"] == 5


@pytest.mark.asyncio
async def test_summary_reads_monitor_next_run():
    block = ExaWebsetSummaryBlock()
    webset = sample_webset(
        monitors=[
            {"id": "m1", "status": "enabled", "nextRunAt": "2024-02-01T09:00:00Z"}
        ]
    )
    api = _api(webset, {"data": [{"id": "item_1"}]})
    with patch.object(block, "_api", api):
        outputs = dict(await collect_outputs(block, webset_id=WEBSET_ID))

    assert outputs["monitor_summary"].active_monitors == 1
    assert outputs["monitor_summary"].next_run.isoformat().startswith("2024-02-01")
    assert outputs["sample_items"] == [{"id": "item_1"}]
    assert outputs["entity_type"] == "company"
