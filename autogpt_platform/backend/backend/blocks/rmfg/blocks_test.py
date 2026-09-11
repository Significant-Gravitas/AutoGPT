"""Block-level tests for behaviour the standard harness can't reach.

The harness exercises one canned input per block. These cover the input
shaping and guard paths it misses.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.blocks.rmfg import carts, images
from backend.blocks.rmfg._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.rmfg._inputs import build_items
from backend.blocks.rmfg._testdata import (
    TEST_CART,
    TEST_CONFIGURATION,
    TEST_DESIGN,
    TEST_PAID_CART,
    TEST_PART,
    TEST_SHIP_TO,
)
from backend.blocks.rmfg._types import (
    CartStatus,
    ManufacturingConfiguration,
    PartConfiguration,
    PaymentType,
    QuoteItemRequest,
    QuoteStatus,
)
from backend.blocks.rmfg.carts import RMFGUpdateCartBlock, is_payable
from backend.blocks.rmfg.designs import RMFGAnalyzeDesignBlock
from backend.blocks.rmfg.images import PNG_SIGNATURE, RMFGGetImageBlock
from backend.blocks.rmfg.pay_cart import RMFGPayCartBlock
from backend.blocks.rmfg.quotes import RMFGCreateQuoteBlock
from backend.blocks.rmfg.triggers import RMFGEventTriggerBlock
from backend.data.execution import ExecutionContext


async def _run(block, **inputs) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    async for name, value in block.run(
        block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(graph_exec_id="exec-1"),
        node_exec_id="node-exec-1",
    ):
        collected.setdefault(name, value)
    return collected


class TestBuildItems:
    def test_material_id_becomes_the_default_material(self):
        inputs = RMFGCreateQuoteBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            design_id="dsn_1",
            quantity=3,
            material_id="mat_9",
        )

        [item] = build_items(inputs)

        assert item.design_id == "dsn_1"
        assert item.quantity == 3
        assert item.configuration.defaults.material_id == "mat_9"
        assert item.client_reference_id is None

    def test_material_id_overrides_the_configuration_default(self):
        inputs = RMFGCreateQuoteBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            design_id="dsn_1",
            material_id="mat_new",
            configuration=TEST_CONFIGURATION,
        )

        [item] = build_items(inputs)

        assert item.configuration.defaults.material_id == "mat_new"
        # Per-part choices survive untouched.
        assert item.configuration.parts == TEST_CONFIGURATION.parts
        # The block input itself is not mutated.
        assert TEST_CONFIGURATION.defaults.material_id == "mat_5052_0125"

    def test_empty_material_id_leaves_the_configuration_alone(self):
        configuration = ManufacturingConfiguration(
            parts=[PartConfiguration(part_id="prt_1", tube_profile_id="tube_1")]
        )
        inputs = RMFGCreateQuoteBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            design_id="dsn_1",
            configuration=configuration,
        )

        [item] = build_items(inputs)

        assert item.configuration == configuration

    def test_additional_items_follow_the_primary_one(self):
        extra = QuoteItemRequest(design_id="dsn_2", quantity=5)
        inputs = RMFGCreateQuoteBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            design_id="dsn_1",
            additional_items=[extra],
            client_reference_id="ref-1",
            quantity_options=[1, 25],
        )

        items = build_items(inputs)

        assert [i.design_id for i in items] == ["dsn_1", "dsn_2"]
        assert items[0].client_reference_id == "ref-1"
        assert items[0].quantity_options == [1, 25]
        assert items[1] is extra

    @pytest.mark.parametrize("options", [[0], [5, -1], list(range(1, 12))])
    def test_quantity_options_are_bounded(self, options: list[int]):
        with pytest.raises(ValidationError):
            QuoteItemRequest(design_id="dsn_1", quantity_options=options)
        with pytest.raises(ValidationError):
            RMFGCreateQuoteBlock.Input(
                credentials=TEST_CREDENTIALS_INPUT,
                design_id="dsn_1",
                quantity_options=options,
            )


class TestQuoteBlock:
    async def test_uses_node_exec_id_as_idempotency_key(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = RMFGCreateQuoteBlock()
        create = AsyncMock(return_value=TEST_CART.quote)
        monkeypatch.setattr(block, "create_quote", create)

        await _run(block, design_id="dsn_1")

        assert create.await_args.args[3] == "node-exec-1"

    async def test_explicit_idempotency_key_wins(self, monkeypatch: pytest.MonkeyPatch):
        block = RMFGCreateQuoteBlock()
        create = AsyncMock(return_value=TEST_CART.quote)
        monkeypatch.setattr(block, "create_quote", create)

        await _run(block, design_id="dsn_1", idempotency_key="bracket-quote-001")

        assert create.await_args.args[3] == "bracket-quote-001"

    async def test_reports_requirements_from_items(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from backend.blocks.rmfg._models import Requirement

        quote = TEST_CART.quote.model_copy(
            deep=True, update={"status": QuoteStatus.REQUIRES_INPUT}
        )
        quote.items[0].requirements = [
            Requirement(code="material_required", message="Pick a material")
        ]
        block = RMFGCreateQuoteBlock()
        monkeypatch.setattr(block, "create_quote", AsyncMock(return_value=quote))

        out = await _run(block, design_id="dsn_1")

        assert out["is_ready"] is False
        assert [r.code for r in out["requirements"]] == ["material_required"]


class TestCartPayability:
    def test_ready_open_cart_with_address_and_shipping_is_payable(self):
        assert is_payable(TEST_CART) is True

    def test_missing_shipping_option_is_not_payable(self):
        cart = TEST_CART.model_copy(update={"shipping_option_id": None})
        assert is_payable(cart) is False

    def test_missing_address_is_not_payable(self):
        cart = TEST_CART.model_copy(update={"ship_to": None})
        assert is_payable(cart) is False

    def test_unready_quote_is_not_payable(self):
        quote = TEST_CART.quote.model_copy(update={"status": QuoteStatus.BLOCKED})
        assert is_payable(TEST_CART.model_copy(update={"quote": quote})) is False

    def test_checked_out_cart_is_not_payable_again(self):
        assert TEST_PAID_CART.status == CartStatus.CHECKED_OUT
        assert is_payable(TEST_PAID_CART) is False


class TestUpdateCartBlock:
    async def test_refuses_an_empty_update(self, monkeypatch: pytest.MonkeyPatch):
        block = RMFGUpdateCartBlock()
        update = AsyncMock(return_value=TEST_CART)
        monkeypatch.setattr(block, "update_cart", update)

        with pytest.raises(ValueError, match="Nothing to update"):
            await _run(block, cart_id="crt_1")
        update.assert_not_awaited()

    async def test_address_alone_is_a_valid_update(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = RMFGUpdateCartBlock()
        monkeypatch.setattr(block, "update_cart", AsyncMock(return_value=TEST_CART))

        out = await _run(block, cart_id="crt_1", ship_to=TEST_SHIP_TO.model_dump())

        assert out["cart_id"] == TEST_CART.id
        assert out["is_payable"] is True

    async def test_unset_items_do_not_empty_the_basket(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # The input's empty default must reach the client as "not set";
        # an empty list there is an instruction to clear the cart.
        update = AsyncMock(return_value=TEST_CART)
        monkeypatch.setattr(
            carts, "RMFGClient", lambda _: SimpleNamespace(update_cart=update)
        )

        await _run(RMFGUpdateCartBlock(), cart_id="crt_1", shipping_option_id="s")

        assert update.await_args.kwargs["items"] is None


class TestPayCartBlock:
    def test_is_a_sensitive_action(self):
        assert RMFGPayCartBlock().is_sensitive_action is True

    async def test_uses_node_exec_id_as_idempotency_key(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # The double-charge protection rests on this default.
        block = RMFGPayCartBlock()
        pay = AsyncMock(return_value=TEST_PAID_CART)
        monkeypatch.setattr(block, "pay_cart", pay)

        await _run(block, cart_id="crt_1")

        assert pay.await_args.args[2] == "node-exec-1"

    async def test_explicit_idempotency_key_wins(self, monkeypatch: pytest.MonkeyPatch):
        block = RMFGPayCartBlock()
        pay = AsyncMock(return_value=TEST_PAID_CART)
        monkeypatch.setattr(block, "pay_cart", pay)

        await _run(block, cart_id="crt_1", idempotency_key="pay-bracket-001")

        assert pay.await_args.args[2] == "pay-bracket-001"

    async def test_payment_method_requires_an_id(self, monkeypatch: pytest.MonkeyPatch):
        block = RMFGPayCartBlock()
        pay = AsyncMock(return_value=TEST_PAID_CART)
        monkeypatch.setattr(block, "pay_cart", pay)

        with pytest.raises(ValueError, match="payment_method_id"):
            await _run(block, cart_id="crt_1", payment_type=PaymentType.PAYMENT_METHOD)
        pay.assert_not_awaited()

    async def test_processing_payment_is_reported_not_hidden(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # A 202 leaves the cart open with no payment object yet; the graph must
        # see "processing" so it re-checks the cart rather than paying again.
        block = RMFGPayCartBlock()
        monkeypatch.setattr(block, "pay_cart", AsyncMock(return_value=TEST_CART))

        out = await _run(block, cart_id="crt_1")

        assert out["payment_status"].value == "processing"
        assert out["checked_out"] is False
        assert "order_id" not in out


class TestAnalyzeDesignBlock:
    async def test_step_extension_is_added_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = RMFGAnalyzeDesignBlock()
        monkeypatch.setattr(
            block, "read_step_file", AsyncMock(return_value=("upload", b"ISO"))
        )
        analyze = AsyncMock(return_value=TEST_DESIGN)
        monkeypatch.setattr(block, "analyze", analyze)

        await _run(block, file="data:application/step;base64,SVNP")

        assert analyze.await_args.args[1] == "upload.step"

    async def test_file_name_override_and_stp_are_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = RMFGAnalyzeDesignBlock()
        monkeypatch.setattr(
            block, "read_step_file", AsyncMock(return_value=("x.step", b"ISO"))
        )
        analyze = AsyncMock(return_value=TEST_DESIGN)
        monkeypatch.setattr(block, "analyze", analyze)

        await _run(
            block, file="data:application/step;base64,SVNP", file_name="Bracket.STP"
        )

        assert analyze.await_args.args[1] == "Bracket.STP"


class TestGetImageBlock:
    def _client(self, monkeypatch: pytest.MonkeyPatch, content: bytes, kind: str):
        get_image = AsyncMock(return_value=(content, kind))
        monkeypatch.setattr(
            images, "RMFGClient", lambda _: SimpleNamespace(get_image=get_image)
        )

    async def test_png_bytes_become_a_png_data_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._client(monkeypatch, PNG_SIGNATURE + b"...", "image/png; charset=binary")
        block = RMFGGetImageBlock()

        image = await block.fetch_image(
            TEST_CREDENTIALS,
            block.input_schema(
                credentials=TEST_CREDENTIALS_INPUT,
                design_id=TEST_DESIGN.id,
                part_id=TEST_PART.id,
            ),
        )

        assert image.startswith("data:image/png;base64,iVBORw0KGgo")

    @pytest.mark.parametrize(
        "content, kind",
        [
            (b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/svg+xml"),
            (b"<html><script>alert(1)</script></html>", "text/html"),
            (b"<html>oops</html>", "image/png"),
        ],
    )
    async def test_anything_but_png_bytes_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, content: bytes, kind: str
    ):
        # A stored SVG or HTML page would execute in the viewer's browser.
        self._client(monkeypatch, content, kind)
        block = RMFGGetImageBlock()

        with pytest.raises(ValueError, match="instead of a PNG"):
            await block.fetch_image(
                TEST_CREDENTIALS,
                block.input_schema(
                    credentials=TEST_CREDENTIALS_INPUT, design_id=TEST_DESIGN.id
                ),
            )


class TestEventTrigger:
    async def test_malformed_data_does_not_fail_the_trigger(self):
        block = RMFGEventTriggerBlock()
        payload = {"id": "evt_1", "type": "design.ready", "data": "not-an-object"}

        out = await _run(block, payload=payload)

        assert out["event"] == "design.ready"
        assert out["resource_id"] == ""
        assert out["status_url"] == ""
