import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web

from backend.blocks.slant3d._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    OrderItem,
)
from backend.blocks.slant3d.filament import TEST_FILAMENT, Slant3DFilamentBlock
from backend.blocks.slant3d.order import Slant3DCreateOrderBlock
from backend.blocks.slant3d.order_status import Slant3DCancelOrderBlock
from backend.blocks.slant3d.slicing import Slant3DSlicerBlock
from backend.data.execution import ExecutionContext
from backend.util.request import Requests

CONTEXT = ExecutionContext(user_id="user-1", graph_exec_id="run-1")


async def test_lowercase_pla_keeps_legacy_color_tag():
    block = Slant3DFilamentBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(return_value={"data": [{**TEST_FILAMENT, "profile": "pla"}]}),
    ):
        outputs = dict(
            [
                x
                async for x in block.run(
                    block.Input(credentials=TEST_CREDENTIALS_INPUT),
                    credentials=TEST_CREDENTIALS,
                )
            ]
        )
    assert outputs["filaments"][0]["colorTag"] == "black"


@pytest.mark.parametrize(
    "response", [{"success": True}, {"success": True, "message": None}]
)
async def test_cancellation_without_message_returns_success(response):
    block = Slant3DCancelOrderBlock()
    with patch.object(block, "_make_request", AsyncMock(return_value=response)):
        outputs = dict(
            [
                x
                async for x in block.run(
                    block.Input(
                        credentials=TEST_CREDENTIALS_INPUT, order_id="SLANT_123"
                    ),
                    credentials=TEST_CREDENTIALS,
                )
            ]
        )
    assert outputs == {"status": "Order cancelled"}


@pytest.mark.parametrize("quantity", [1, 2])
async def test_missing_total_is_rejected_before_outputs(quantity):
    block = Slant3DSlicerBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(
            return_value={"message": "Estimated", "data": {"quantity": quantity}}
        ),
    ):
        outputs = block.run(
            block.Input(
                credentials=TEST_CREDENTIALS_INPUT, file_id="file-1", quantity=quantity
            ),
            credentials=TEST_CREDENTIALS,
            execution_context=CONTEXT,
        )
        with pytest.raises(ValueError, match="printing total"):
            await anext(outputs)


async def test_item_failure_cancels_inflight_filament_lookup():
    block = Slant3DCreateOrderBlock()
    lookup_started = asyncio.Event()
    lookup_cancelled = asyncio.Event()

    async def lookup(*args):
        lookup_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            lookup_cancelled.set()

    async def upload(*args, **kwargs):
        await lookup_started.wait()
        raise ValueError("Upload rejected")

    with patch.object(
        block, "_resolve_filament_id", AsyncMock(side_effect=lookup)
    ) as resolver, patch.object(block, "_upload_file", AsyncMock(side_effect=upload)):
        with pytest.raises(ValueError, match="Upload rejected"):
            await asyncio.wait_for(
                block._format_order_items(
                    [
                        OrderItem(
                            file_url="https://example.com/failure.stl",
                            quantity=1,
                            filament_id="explicit",
                        ),
                        OrderItem(file_id="existing", quantity=1),
                    ],
                    "platform-1",
                    "key",
                    execution_context=CONTEXT,
                ),
                2,
            )
    assert lookup_cancelled.is_set()
    resolver.assert_awaited_once()


async def test_upload_passes_a_stream_and_closes_it(tmp_path):
    path = tmp_path / "model.stl"
    content = b"x" * 131073
    path.write_bytes(content)
    block = Slant3DSlicerBlock()
    streams = []

    async def put(url, *, data, headers):
        streams.append(data)
        chunks = []
        while chunk := data.read(65536):
            chunks.append(chunk)
        assert b"".join(chunks) == content
        assert len(chunks) == 3
        assert headers["Content-Type"] == "application/octet-stream"

    with patch(
        "backend.blocks.slant3d.base.store_media_file",
        AsyncMock(return_value="model.stl"),
    ), patch(
        "backend.blocks.slant3d.base.get_exec_file_path", return_value=str(path)
    ), patch(
        "backend.blocks.slant3d.base.Requests"
    ) as requests, patch.object(
        block,
        "_make_request",
        AsyncMock(
            side_effect=[
                {
                    "data": {
                        "presignedUrl": "https://example.com/upload",
                        "filePlaceholder": "placeholder",
                    }
                },
                {"data": {"publicFileServiceId": "uploaded"}},
            ]
        ),
    ):
        requests.return_value.put = AsyncMock(side_effect=put)
        result = await block._upload_file(
            "https://example.com/model.stl",
            "platform-1",
            "key",
            execution_context=CONTEXT,
        )
    assert result == "uploaded"
    assert streams[0].closed
    assert path.read_bytes() == content


@pytest.mark.parametrize("execution_id", [None, ""])
async def test_upload_requires_execution_id(execution_id):
    block = Slant3DSlicerBlock()
    context = ExecutionContext(user_id="user-1", graph_exec_id=execution_id)
    with patch.object(block, "_make_request", AsyncMock()) as api:
        with pytest.raises(ValueError, match="graph_exec_id is required"):
            await block._upload_file(
                "https://example.com/model.stl",
                "platform-1",
                "key",
                execution_context=context,
            )
    api.assert_not_awaited()


async def test_stream_upload_uses_content_length_and_preserves_bytes(tmp_path):
    path = tmp_path / "model.stl"
    content = b"stl" * 50000
    path.write_bytes(content)
    received = {}

    async def receive(request):
        received["body"] = await request.read()
        received["headers"] = request.headers
        return web.Response()

    app = web.Application()
    app.router.add_put("/upload", receive)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    url = f"http://127.0.0.1:{runner.addresses[0][1]}/upload"
    block = Slant3DSlicerBlock()
    try:
        with patch(
            "backend.blocks.slant3d.base.store_media_file",
            AsyncMock(return_value="model.stl"),
        ), patch(
            "backend.blocks.slant3d.base.get_exec_file_path", return_value=str(path)
        ), patch(
            "backend.blocks.slant3d.base.Requests",
            return_value=Requests(trusted_origins=["127.0.0.1"], retry_max_attempts=1),
        ), patch.object(
            block,
            "_make_request",
            AsyncMock(
                side_effect=[
                    {"data": {"presignedUrl": url, "filePlaceholder": "placeholder"}},
                    {"data": {"publicFileServiceId": "uploaded"}},
                ]
            ),
        ):
            assert (
                await block._upload_file(
                    "https://example.com/model.stl",
                    "platform-1",
                    "key",
                    execution_context=CONTEXT,
                )
                == "uploaded"
            )
    finally:
        await runner.cleanup()
    assert received["body"] == content
    assert received["headers"]["Content-Length"] == str(len(content))
    assert "Authorization" not in received["headers"]
    assert "Transfer-Encoding" not in received["headers"]
