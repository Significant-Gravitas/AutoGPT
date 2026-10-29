"""Paging tests for the Websets item blocks (Exa caps item pages at 100)."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from exa_py.websets.types import ListWebsetItemResponse, WebsetItem

from backend.blocks.exa.websets_fixtures import WEBSET_ID, collect_outputs
from backend.blocks.exa.websets_items import (
    ExaGetNewItemsBlock,
    ExaListWebsetItemsBlock,
)


def _item(idx: int) -> WebsetItem:
    return WebsetItem.model_validate(
        {
            "id": f"item_{idx}",
            "object": "webset_item",
            "source": "search",
            "sourceId": "search_1",
            "websetId": WEBSET_ID,
            "properties": {
                "type": "company",
                "url": f"https://example{idx}.com",
                "description": "A company",
                "company": {"name": f"Company {idx}"},
            },
            "evaluations": [],
            "enrichments": [],
            "createdAt": datetime(2024, 1, 1),
            "updatedAt": datetime(2024, 1, 1),
        }
    )


def _page(start: int, size: int, has_more: bool) -> ListWebsetItemResponse:
    return ListWebsetItemResponse.model_validate(
        {
            "data": [_item(i) for i in range(start, start + size)],
            "hasMore": has_more,
            "nextCursor": f"cursor_{start + size}" if has_more else None,
        }
    )


@pytest.mark.asyncio
async def test_get_new_items_pages_in_chunks_of_100():
    list_items = AsyncMock(side_effect=[_page(0, 100, True), _page(100, 50, True)])
    client = MagicMock()
    client.websets.items.list = list_items
    with patch("backend.blocks.exa.websets_items.AsyncExa", return_value=client):
        outputs = await collect_outputs(
            ExaGetNewItemsBlock(), webset_id=WEBSET_ID, max_items=150
        )

    limits = [call.kwargs["limit"] for call in list_items.call_args_list]
    cursors = [call.kwargs["cursor"] for call in list_items.call_args_list]
    assert limits == [100, 50]
    assert cursors == [None, "cursor_100"]
    result = dict(outputs)
    assert result["count"] == 150
    assert result["next_cursor"] == "cursor_150"
    assert result["new_items"][0].source_id == "search_1"


@pytest.mark.asyncio
async def test_list_items_passes_source_filter():
    list_items = AsyncMock(return_value=_page(0, 1, False))
    client = MagicMock()
    client.websets.items.list = list_items
    with patch("backend.blocks.exa.websets_items.AsyncExa", return_value=client):
        await collect_outputs(
            ExaListWebsetItemsBlock(), webset_id=WEBSET_ID, source_id="import_9"
        )

    assert list_items.call_args.kwargs["source_id"] == "import_9"
