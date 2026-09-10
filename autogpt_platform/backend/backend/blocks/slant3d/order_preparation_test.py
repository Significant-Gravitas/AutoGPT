import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.slant3d._api import TEST_CREDENTIALS, OrderItem, Profile
from backend.blocks.slant3d._order import TEST_ORDER_INPUT, OrderInput
from backend.blocks.slant3d.order import Slant3DCreateOrderBlock
from backend.data.execution import ExecutionContext

CONTEXT = ExecutionContext(user_id="user-1", graph_exec_id="run-1")


async def test_order_uploads_overlap_and_keep_input_order():
    block = Slant3DCreateOrderBlock()
    inputs = OrderInput(**TEST_ORDER_INPUT)
    items = [
        OrderItem(file_url=f"https://example.com/{i}.stl", quantity=i + 1)
        for i in range(3)
    ]
    started = asyncio.Event()
    release = [asyncio.Event() for _ in items]
    finished = [asyncio.Event() for _ in items]
    uploads = []

    async def upload(file_url, *args, **kwargs):
        index = int(file_url.rsplit("/", 1)[1].split(".")[0])
        uploads.append(index)
        if len(uploads) == len(items):
            started.set()
        await release[index].wait()
        finished[index].set()
        return f"file-{index}"

    with (
        patch.object(block, "_resolve_filament_id", AsyncMock(return_value="filament")),
        patch.object(block, "_upload_file", AsyncMock(side_effect=upload)),
    ):
        task = asyncio.create_task(
            block._format_order_data(
                inputs.customer,
                inputs.order_number,
                items,
                "key",
                inputs.platform_id,
                execution_context=CONTEXT,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            for index in reversed(range(len(items))):
                release[index].set()
                await asyncio.wait_for(finished[index].wait(), 2)
            result = await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert result["items"] == [
        {
            "type": "PRINT",
            "publicFileServiceId": f"file-{index}",
            "filamentId": "filament",
            "quantity": index + 1,
        }
        for index in range(3)
    ]


async def test_filament_lookups_are_shared_per_pair_and_isolated_per_order():
    block = Slant3DCreateOrderBlock()
    inputs = OrderInput(**TEST_ORDER_INPUT)
    items = [
        OrderItem(file_id="one", quantity=1, color="black"),
        OrderItem(file_id="two", quantity=2, color="BLACK"),
        OrderItem(file_id="three", quantity=1, color="black", profile=Profile.PETG),
        OrderItem(file_id="four", quantity=1, color="white"),
        OrderItem(file_id="five", quantity=1, filament_id="explicit"),
    ]

    async def request(method, endpoint, api_key, **kwargs):
        assert (method, endpoint) == ("GET", "filaments")
        profile, color = kwargs["params"]["profile"], kwargs["params"]["color"]
        return {
            "data": [
                {
                    "profile": profile,
                    "color": color,
                    "publicId": f"{api_key}-{profile}-{color.casefold()}",
                }
            ]
        }

    with (
        patch.object(block, "_make_request", AsyncMock(side_effect=request)) as api,
        patch.object(block, "_upload_file", AsyncMock()) as upload,
    ):
        results = await asyncio.gather(
            *(
                block._format_order_data(
                    inputs.customer,
                    inputs.order_number,
                    items,
                    key,
                    inputs.platform_id,
                    execution_context=CONTEXT,
                )
                for key in ("first", "second")
            )
        )
        results.append(
            await block._format_order_data(
                inputs.customer,
                inputs.order_number,
                items,
                "first",
                inputs.platform_id,
                execution_context=CONTEXT,
            )
        )
    assert api.await_count == 9
    upload.assert_not_awaited()
    for key, result in zip(("first", "second", "first"), results):
        assert [item["filamentId"] for item in result["items"]] == [
            f"{key}-PLA-black",
            f"{key}-PLA-black",
            f"{key}-PETG-black",
            f"{key}-PLA-white",
            "explicit",
        ]
    assert [item.filament_id for item in items] == ["", "", "", "", "explicit"]


async def test_preparation_failure_cancels_uploads_without_drafting():
    block = Slant3DCreateOrderBlock()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def upload(file_url, *args, **kwargs):
        if file_url.endswith("bad.stl"):
            await started.wait()
            raise ValueError("Upload rejected")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    inputs = block.Input(
        **{
            **TEST_ORDER_INPUT,
            "items": [
                {"file_url": f"https://example.com/{name}.stl", "quantity": 1}
                for name in ("bad", "pending")
            ],
        }
    )
    with (
        patch.object(block, "_resolve_filament_id", AsyncMock(return_value="filament")),
        patch.object(block, "_upload_file", AsyncMock(side_effect=upload)),
        patch.object(block, "_make_request", AsyncMock()) as api,
        patch.object(block, "_process_order", AsyncMock()) as process,
    ):
        with pytest.raises(ValueError, match="Upload rejected"):
            await asyncio.wait_for(
                anext(
                    block.run(
                        inputs, credentials=TEST_CREDENTIALS, execution_context=CONTEXT
                    )
                ),
                2,
            )
        assert cancelled.is_set()
        api.assert_not_awaited()
        process.assert_not_awaited()


async def test_parallel_uploads_preserve_different_files_with_the_same_name(tmp_path):
    block = Slant3DCreateOrderBlock()
    inputs = OrderInput(**TEST_ORDER_INPUT)
    started = asyncio.Event()
    uploaded = []
    count = 0

    async def request(method, endpoint, api_key, **kwargs):
        nonlocal count
        if endpoint == "files/direct-upload":
            count += 1
            index = count
            if count == 2:
                started.set()
            await started.wait()
            return {
                "data": {
                    "presignedUrl": f"https://upload.example.com/{index}",
                    "filePlaceholder": index,
                }
            }
        assert endpoint == "files/confirm-upload"
        return {"data": {"publicFileServiceId": str(kwargs["json"]["filePlaceholder"])}}

    async def put(url, **kwargs):
        uploaded.append(kwargs["data"].read())

    downloader = Mock()
    downloader.get = AsyncMock(
        side_effect=[Mock(content=b"first STL"), Mock(content=b"second STL")]
    )
    with (
        patch("backend.util.file.TEMP_DIR", tmp_path),
        patch(
            "backend.util.file.get_cloud_storage_handler",
            AsyncMock(return_value=Mock(is_cloud_path=Mock(return_value=False))),
        ),
        patch("backend.util.file.scan_content_safe", AsyncMock()),
        patch("backend.util.file.Requests", return_value=downloader),
        patch("backend.blocks.slant3d.base.Requests") as requests,
        patch.object(block, "_make_request", AsyncMock(side_effect=request)),
    ):
        requests.return_value.put = AsyncMock(side_effect=put)
        await asyncio.wait_for(
            block._format_order_data(
                inputs.customer,
                inputs.order_number,
                [
                    OrderItem(
                        file_url=f"https://example.com/{folder}/part.stl",
                        filament_id="filament",
                        quantity=1,
                    )
                    for folder in ("one", "two")
                ],
                "key",
                inputs.platform_id,
                execution_context=CONTEXT,
            ),
            2,
        )
    assert sorted(uploaded) == [b"first STL", b"second STL"]


async def test_large_orders_limit_active_uploads():
    block = Slant3DCreateOrderBlock()
    inputs = OrderInput(**TEST_ORDER_INPUT)
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum = 0

    async def upload(file_url, *args, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 4:
            started.set()
        try:
            await release.wait()
            return file_url.rsplit("/", 1)[-1]
        finally:
            active -= 1

    with patch.object(block, "_upload_file", AsyncMock(side_effect=upload)):
        task = asyncio.create_task(
            block._format_order_data(
                inputs.customer,
                inputs.order_number,
                [
                    OrderItem(
                        file_url=f"https://example.com/{i}.stl",
                        filament_id="filament",
                        quantity=1,
                    )
                    for i in range(12)
                ],
                "key",
                inputs.platform_id,
                execution_context=CONTEXT,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert active == 4
            release.set()
            result = await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert len(result["items"]) == 12
    assert maximum == 4
