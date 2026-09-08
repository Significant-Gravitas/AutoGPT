from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from backend.blocks.slant3d._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    OrderItem,
    Profile,
)
from backend.blocks.slant3d._order import TEST_DRAFT, TEST_ORDER_INPUT, OrderInput
from backend.blocks.slant3d.filament import TEST_FILAMENT, Slant3DFilamentBlock
from backend.blocks.slant3d.order import (
    Slant3DCreateOrderBlock,
    Slant3DEstimateShippingBlock,
)
from backend.blocks.slant3d.order_status import (
    Slant3DCancelOrderBlock,
    Slant3DGetOrdersBlock,
    Slant3DProcessOrderBlock,
    Slant3DTrackingBlock,
)
from backend.blocks.slant3d.slicing import Slant3DSlicerBlock
from backend.data.execution import ExecutionContext

TEST_EXECUTION_CONTEXT = ExecutionContext(user_id="user-1", graph_exec_id="run-1")


async def test_filament_filters_and_legacy_output_fields():
    block = Slant3DFilamentBlock()
    with patch.object(
        block, "_make_request", AsyncMock(return_value={"data": [TEST_FILAMENT]})
    ) as request:
        result = dict(
            [
                out
                async for out in block.run(
                    block.Input(
                        credentials=TEST_CREDENTIALS_INPUT,
                        profiles=[Profile.PLA, Profile.OPM],
                        colors=["black", "blue"],
                    ),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    assert request.call_args.args[:2] == ("GET", "filaments")
    assert request.call_args.kwargs["params"] == {
        "profile": "PLA,OPM",
        "color": "black,blue",
    }
    assert result["filaments"][0]["publicId"] == TEST_FILAMENT["publicId"]
    assert result["filaments"][0]["colorTag"] == "black"
    assert result["filaments"][0]["hexColor"] == "000000"


async def test_exact_filament_match_ignores_partial_and_unavailable_matches():
    block = Slant3DFilamentBlock()
    filaments = [
        {**TEST_FILAMENT, "publicId": "matte", "color": "matte black"},
        {**TEST_FILAMENT, "publicId": "unavailable", "available": False},
        TEST_FILAMENT,
    ]
    with patch.object(
        block, "_make_request", AsyncMock(return_value={"data": filaments})
    ):
        assert (
            await block._resolve_filament_id(Profile.PLA, "black", "key")
            == TEST_FILAMENT["publicId"]
        )


@pytest.mark.parametrize(
    "filaments",
    [[], [{**TEST_FILAMENT, "available": False}], [TEST_FILAMENT, TEST_FILAMENT]],
)
async def test_unavailable_or_ambiguous_filaments_require_explicit_choice(filaments):
    block = Slant3DFilamentBlock()
    with patch.object(
        block, "_make_request", AsyncMock(return_value={"data": filaments})
    ):
        with pytest.raises(ValueError, match="filament_id"):
            await block._resolve_filament_id(Profile.PLA, "black", "key")


@pytest.mark.parametrize("quantity", [0, -1, "invalid", 1.5])
def test_invalid_print_quantity_is_rejected(quantity):
    with pytest.raises(ValidationError):
        OrderItem(file_id="file-1", quantity=quantity)


def test_missing_file_and_empty_orders_are_rejected():
    with pytest.raises(ValidationError, match="file_id or file_url"):
        OrderItem(quantity=1)
    with pytest.raises(ValidationError):
        OrderInput(**{**TEST_ORDER_INPUT, "items": []})


async def test_legacy_url_items_are_uploaded_and_use_numeric_quantity():
    block = Slant3DCreateOrderBlock()
    item = OrderItem(
        file_url="https://example.com/model.stl", quantity="2", color="black"
    )
    with patch.object(
        block, "_resolve_filament_id", AsyncMock(return_value="filament-1")
    ) as filament, patch.object(
        block, "_upload_file", AsyncMock(return_value="file-1")
    ) as upload:
        result = await block._format_order_item(
            item, "platform-1", "key", execution_context=TEST_EXECUTION_CONTEXT
        )
    filament.assert_awaited_once_with(Profile.PLA, "black", "key")
    upload.assert_awaited_once_with(
        "https://example.com/model.stl",
        "platform-1",
        "key",
        execution_context=TEST_EXECUTION_CONTEXT,
    )
    assert result == {
        "type": "PRINT",
        "quantity": 2,
        "publicFileServiceId": "file-1",
        "filamentId": "filament-1",
    }


async def test_order_processing_failure_identifies_draft_and_does_not_retry():
    block = Slant3DCreateOrderBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(side_effect=[{"data": TEST_DRAFT}, RuntimeError("Payment failed")]),
    ) as request:
        with pytest.raises(RuntimeError, match="SLANT_1234567890.*Check its status"):
            _ = [
                out
                async for out in block.run(
                    block.Input(**TEST_ORDER_INPUT),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
    assert request.await_count == 2


async def test_shipping_estimate_does_not_process_order():
    block = Slant3DEstimateShippingBlock()
    with patch.object(
        block, "_make_request", AsyncMock(return_value={"data": TEST_DRAFT})
    ) as request:
        result = dict(
            [
                out
                async for out in block.run(
                    block.Input(**TEST_ORDER_INPUT),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    assert result == {
        "shipping_cost": 5.56,
        "currency_code": "usd",
        "order_id": "SLANT_1234567890",
    }
    assert request.await_count == 1
    assert request.call_args.args[:2] == ("POST", "orders")


async def test_get_orders_fetches_every_page():
    block = Slant3DGetOrdersBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(
            side_effect=[
                {"data": [{"publicId": "one"}], "pagination": {"totalPages": 2}},
                {"data": [{"publicId": "two"}], "pagination": {"totalPages": 2}},
            ]
        ),
    ) as request:
        result = dict(
            [
                out
                async for out in block.run(
                    block.Input(credentials=TEST_CREDENTIALS_INPUT),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    assert result == {"orders": ["one", "two"]}
    assert [call.kwargs["params"]["page"] for call in request.await_args_list] == [1, 2]


@pytest.mark.parametrize(
    "fulfillment,tracking",
    [(None, []), ({"trackingNumbers": ["track-1"]}, ["track-1"])],
)
async def test_tracking_reads_fulfillment(fulfillment, tracking):
    block = Slant3DTrackingBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(
            return_value={
                "data": {"order": {"status": "SHIPPED", "fulfillment": fulfillment}}
            }
        ),
    ) as request:
        result = dict(
            [
                out
                async for out in block.run(
                    block.Input(
                        credentials=TEST_CREDENTIALS_INPUT, order_id="SLANT_123"
                    ),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    assert request.call_args.args[:2] == ("GET", "orders/SLANT_123")
    assert result == {"status": "SHIPPED", "tracking_numbers": tracking}


async def test_cancellation_reads_envelope_message():
    block = Slant3DCancelOrderBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(return_value={"success": True, "message": "Order cancelled"}),
    ) as request:
        result = dict(
            [
                out
                async for out in block.run(
                    block.Input(
                        credentials=TEST_CREDENTIALS_INPUT, order_id="SLANT_123"
                    ),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    assert request.call_args.args[:2] == ("DELETE", "orders/SLANT_123")
    assert result == {"status": "Order cancelled"}


async def test_process_existing_draft_is_sensitive_and_does_not_create_another():
    block = Slant3DProcessOrderBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(return_value={"data": {"publicId": "SLANT_123"}}),
    ) as request:
        result = dict(
            [
                out
                async for out in block.run(
                    block.Input(
                        credentials=TEST_CREDENTIALS_INPUT, order_id="SLANT_123"
                    ),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    request.assert_awaited_once_with(
        "POST", "orders/SLANT_123", TEST_CREDENTIALS.api_key.get_secret_value()
    )
    assert result == {"order_id": "SLANT_123"}
    assert block.is_sensitive_action


@pytest.mark.parametrize("use_existing", [True, False])
async def test_slicer_upload_or_reuse_and_quantity(use_existing):
    block = Slant3DSlicerBlock()
    inputs = (
        {"file_id": "file-1"}
        if use_existing
        else {"file_url": "https://example.com/file.stl", "platform_id": "platform-1"}
    )
    with patch.object(
        block,
        "_make_request",
        AsyncMock(
            return_value={"message": "File Price Estimated", "data": {"total": 17.75}}
        ),
    ) as request, patch.object(
        block, "_upload_file", AsyncMock(return_value="file-1")
    ) as upload:
        result = dict(
            [
                out
                async for out in block.run(
                    block.Input(
                        credentials=TEST_CREDENTIALS_INPUT,
                        quantity=5,
                        filament_id="filament-1",
                        **inputs
                    ),
                    credentials=TEST_CREDENTIALS,
                    execution_context=TEST_EXECUTION_CONTEXT,
                )
            ]
        )
    assert upload.await_count == (0 if use_existing else 1)
    assert request.call_args.args[:2] == ("POST", "files/file-1/estimate")
    assert request.call_args.kwargs["json"] == {
        "options": {"filamentId": "filament-1", "quantity": 5}
    }
    assert result["price"] == 17.75
    assert result["file_id"] == "file-1"
