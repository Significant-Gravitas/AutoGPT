import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web

from backend.blocks.anysearch.extract import AnySearchExtractBlock
from backend.blocks.anysearch.parallel_search import AnySearchParallelSearchBlock
from backend.blocks.anysearch.search import AnySearchBlock
from backend.blocks.anysearch.test_http_helpers import (
    QUOTA_RESPONSE,
    REQUEST_ID,
    anysearch_server,
    assert_no_credentials,
)
from backend.copilot.tools import helpers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_class,input_data",
    [
        (AnySearchBlock, {"query": "q"}),
        (AnySearchExtractBlock, {"url": "https://example.com"}),
        (AnySearchParallelSearchBlock, {"queries": ["q"]}),
    ],
)
async def test_copilot_http_402_retains_correlation_without_logging_credentials(
    monkeypatch, caplog, block_class, input_data
):
    async def handler(request):
        return web.json_response(QUOTA_RESPONSE, status=402)

    workspace = SimpleNamespace(
        get_or_create_workspace=AsyncMock(
            return_value=SimpleNamespace(id="test-workspace")
        )
    )
    user = SimpleNamespace(
        get_user_by_id=AsyncMock(return_value=SimpleNamespace(timezone="UTC"))
    )
    block = block_class()
    async with anysearch_server(monkeypatch, handler) as calls:
        with (
            patch.object(helpers, "workspace_db", return_value=workspace),
            patch.object(helpers, "user_db", return_value=user),
            patch.object(helpers, "credit_db", return_value=object()),
        ):
            output = await asyncio.wait_for(
                helpers.execute_block(
                    block=block,
                    block_id=block.id,
                    input_data=input_data,
                    user_id="test-user",
                    session_id="test-session",
                    node_exec_id="test-node",
                    matched_credentials={},
                    dry_run=False,
                ),
                timeout=2,
            )

    assert len(calls) == 1
    serialized = output.model_dump_json()
    assert "HTTP 402" in serialized
    assert REQUEST_ID in serialized
    assert_no_credentials(serialized)
    assert_no_credentials(caplog.text)
