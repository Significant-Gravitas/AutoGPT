from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from backend.blocks.slant3d._api import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.slant3d.filament import Slant3DFilamentBlock
from backend.blocks.slant3d.order import (
    Slant3DCreateOrderBlock,
    Slant3DEstimateOrderBlock,
)
from backend.data.execution import ExecutionContext

CUSTOMER = {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "123-456-7890",
    "address": "123 Test St",
    "city": "Test City",
    "state": "TS",
    "zip": "12345",
}
ORDER_INPUT = {
    "credentials": TEST_CREDENTIALS_INPUT,
    "platform_id": "platform-1",
    "order_number": "TEST-001",
    "customer": CUSTOMER,
    "items": [{"file_id": "file-1", "filament_id": "filament-1", "quantity": "2"}],
}
DRAFT = {
    "order": {"publicId": "SLANT_123", "status": "DRAFT"},
    "totals": {"printingCost": 3.75, "deliveryCost": 5.56, "totalCost": 9.31},
}


TEST_EXECUTION_CONTEXT = ExecutionContext(user_id="user-1", graph_exec_id="run-1")


async def test_v2_auth_and_response_envelope():
    block = Slant3DFilamentBlock()
    response = Mock(ok=True, status=200)
    response.json.return_value = {"success": True, "message": "OK", "data": []}
    with patch("backend.blocks.slant3d.base.Requests") as requests:
        requests.return_value.request = AsyncMock(return_value=response)
        assert await block._make_request("GET", "filaments", "key") == response.json()
    requests.return_value.request.assert_awaited_once_with(
        method="GET",
        url="https://slant3dapi.com/v2/api/filaments",
        headers={"Authorization": "Bearer key", "Content-Type": "application/json"},
    )


async def test_estimate_uses_draft_totals_without_processing():
    block = Slant3DEstimateOrderBlock()
    with patch.object(
        block, "_make_request", AsyncMock(return_value={"data": DRAFT})
    ) as request:
        result = dict(
            [
                output
                async for output in block.run(
                    block.Input(**ORDER_INPUT),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    assert result == {
        "total_price": 9.31,
        "shipping_cost": 5.56,
        "printing_cost": 3.75,
        "order_id": "SLANT_123",
    }
    request.assert_awaited_once()
    assert request.call_args.args[:2] == ("POST", "orders")
    body = request.call_args.kwargs["json"]
    assert body["customer"] == {
        "platformId": "platform-1",
        "details": {
            "email": "john@example.com",
            "address": {
                "name": "John Doe",
                "line1": "123 Test St",
                "line2": "",
                "city": "Test City",
                "state": "TS",
                "zip": "12345",
                "country": "US",
            },
        },
    }
    assert body["items"] == [
        {
            "type": "PRINT",
            "quantity": 2,
            "publicFileServiceId": "file-1",
            "filamentId": "filament-1",
        }
    ]
    assert body["metadata"] == {"orderNumber": "TEST-001"}


async def test_create_drafts_then_processes_order():
    block = Slant3DCreateOrderBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(
            side_effect=[
                {"data": DRAFT},
                {"data": {"publicId": "SLANT_123", "status": "PAID"}},
            ]
        ),
    ) as request:
        result = dict(
            [
                output
                async for output in block.run(
                    block.Input(**ORDER_INPUT),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    assert result == {"order_id": "SLANT_123"}
    assert [call.args[:2] for call in request.await_args_list] == [
        ("POST", "orders"),
        ("POST", "orders/SLANT_123"),
    ]
    assert block.is_sensitive_action


@pytest.mark.parametrize(
    "status,payload",
    [
        (400, {"success": False, "error": {"message": "Invalid filament"}}),
        (200, {"success": False, "message": "Invalid filament"}),
        (500, {"error": "Invalid filament"}),
    ],
)
async def test_api_errors_are_raised(status, payload):
    response = Mock(ok=status < 400, status=status)
    response.json.return_value = payload
    with patch("backend.blocks.slant3d.base.Requests") as requests:
        requests.return_value.request = AsyncMock(return_value=response)
        with pytest.raises(
            RuntimeError, match="^Slant3D API request failed: Invalid filament$"
        ):
            await Slant3DFilamentBlock()._make_request("POST", "orders", "key")
        assert requests.call_args.kwargs["retry_max_attempts"] == 1


async def test_non_json_errors_are_readable():
    response = Mock(ok=False, status=502)
    response.json.side_effect = ValueError("HTML response")
    with patch("backend.blocks.slant3d.base.Requests") as requests:
        requests.return_value.request = AsyncMock(return_value=response)
        with pytest.raises(RuntimeError, match="HTTP 502"):
            await Slant3DFilamentBlock()._make_request("GET", "filaments", "key")


@pytest.mark.parametrize(
    "platforms,expected",
    [
        ([{"id": "one", "enabled": True}], "one"),
        ([{"id": "one", "enabled": True}, {"id": "two", "enabled": False}], "one"),
        ([], None),
        ([{"id": "one"}, {"id": "two"}], None),
    ],
)
async def test_platform_selection_is_unambiguous(platforms, expected):
    block = Slant3DFilamentBlock()
    with patch.object(
        block, "_make_request", AsyncMock(return_value={"data": platforms})
    ):
        if expected:
            assert await block._resolve_platform_id("", "key") == expected
        else:
            with pytest.raises(ValueError, match="platform_id"):
                await block._resolve_platform_id("", "key")


async def test_upload_confirms_exact_placeholder_without_forwarding_api_key(tmp_path):
    source = tmp_path / "model.stl"
    source.write_bytes(b"STL bytes")
    uploaded = []

    async def put(url, **kwargs):
        uploaded.append(kwargs["data"].read())

    block = Slant3DFilamentBlock()
    placeholder = {
        "publicFileServiceId": "file-1",
        "name": "model.stl",
        "platformId": "platform-1",
    }
    with patch.object(
        block,
        "_make_request",
        AsyncMock(
            side_effect=[
                {
                    "data": {
                        "presignedUrl": "https://upload.example.com/object",
                        "filePlaceholder": placeholder,
                    }
                },
                {"data": {"publicFileServiceId": "file-1"}},
            ]
        ),
    ) as api, patch("backend.blocks.slant3d.base.Requests") as requests, patch(
        "backend.blocks.slant3d.base.store_media_file",
        AsyncMock(return_value=str(source)),
    ) as media:
        requests.return_value.put = AsyncMock(side_effect=put)
        assert (
            await block._upload_file(
                "https://files.example.com/model.stl?download=1",
                "platform-1",
                "secret",
                execution_context=TEST_EXECUTION_CONTEXT,
            )
            == "file-1"
        )
    media.assert_awaited_once_with(
        file="https://files.example.com/model.stl?download=1",
        execution_context=TEST_EXECUTION_CONTEXT,
        return_format="for_local_processing",
    )
    requests.return_value.put.assert_awaited_once_with(
        "https://upload.example.com/object",
        data=ANY,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert uploaded == [b"STL bytes"]
    assert api.await_args_list[0].kwargs["json"] == {
        "name": "model.stl",
        "platformId": "platform-1",
    }
    assert api.await_args_list[1].args[:2] == ("POST", "files/confirm-upload")
    assert api.await_args_list[1].kwargs["json"]["filePlaceholder"] is placeholder


async def test_failed_upload_is_not_confirmed(tmp_path):
    source = tmp_path / "model.stl"
    source.write_bytes(b"STL bytes")
    block = Slant3DFilamentBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(
            return_value={
                "data": {
                    "presignedUrl": "https://upload.example.com/object",
                    "filePlaceholder": {},
                }
            }
        ),
    ) as api, patch("backend.blocks.slant3d.base.Requests") as requests, patch(
        "backend.blocks.slant3d.base.store_media_file",
        AsyncMock(return_value=str(source)),
    ) as media:
        requests.return_value.put = AsyncMock(side_effect=RuntimeError("Upload failed"))
        with pytest.raises(RuntimeError, match="Upload failed"):
            await block._upload_file(
                "https://example.com/model.stl",
                "platform-1",
                "key",
                execution_context=TEST_EXECUTION_CONTEXT,
            )
    media.assert_awaited_once()
    api.assert_awaited_once()
