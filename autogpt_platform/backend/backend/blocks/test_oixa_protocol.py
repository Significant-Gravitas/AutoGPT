"""
Tests for OIXA Protocol AutoGPT blocks.

Run with:
    poetry run test
"""

import importlib
import json
import re
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

import backend.blocks.oixa_protocol as oixa_mod
from backend.blocks.oixa_protocol import (
    OIXA_BASE_URL,
    OIXA_BLOCKS,
    CheckBalanceBlock,
    CreateAuctionBlock,
    DeliverOutputBlock,
    ListAuctionsBlock,
    PlaceBidBlock,
    RegisterOfferBlock,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_response(payload: dict):
    """Return an AsyncMock for _call that yields payload."""
    return AsyncMock(return_value=payload)


async def collect(gen) -> dict:
    """Collect all (key, value) yields from an async generator into a dict."""
    result = {}
    async for key, value in gen:
        result[key] = value
    return result


def _block_name(block) -> str:
    return getattr(block, "__name__", block.__class__.__name__)


# ── Config ────────────────────────────────────────────────────────────────────


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OIXA_BASE_URL", "https://oixa.io")
    importlib.reload(oixa_mod)
    assert oixa_mod.OIXA_BASE_URL == "https://oixa.io"


def test_base_url_override(monkeypatch):
    monkeypatch.setenv("OIXA_BASE_URL", "http://localhost:8000")
    importlib.reload(oixa_mod)
    assert oixa_mod.OIXA_BASE_URL == "http://localhost:8000"


# ── Block identity ─────────────────────────────────────────────────────────────


def test_block_ids_are_unique():
    ids = [b.id for b in OIXA_BLOCKS]
    assert len(ids) == len(set(ids)), "Block IDs must be unique"


def test_block_ids_are_valid_uuids():
    uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    for block in OIXA_BLOCKS:
        assert uuid_re.match(block.id), f"{_block_name(block)} has invalid UUID: {block.id}"


def test_all_blocks_have_name_and_categories():
    for block in OIXA_BLOCKS:
        assert block.name, f"{_block_name(block)} missing name"
        assert block.categories, f"{_block_name(block)} missing categories"


def test_oixa_blocks_contains_all_six():
    expected = {
        "ListAuctionsBlock",
        "PlaceBidBlock",
        "CreateAuctionBlock",
        "DeliverOutputBlock",
        "RegisterOfferBlock",
        "CheckBalanceBlock",
    }
    actual = {b.__name__ for b in OIXA_BLOCKS}
    assert actual == expected


# ── Validators ────────────────────────────────────────────────────────────────


def test_place_bid_amount_must_be_positive():
    with pytest.raises((ValidationError, ValueError)):
        PlaceBidBlock.Input(
            auction_id="oixa_auction_test",
            bidder_id="agent_1",
            bidder_name="Agent One",
            amount=-1.0,
        )


def test_place_bid_zero_amount_rejected():
    with pytest.raises((ValidationError, ValueError)):
        PlaceBidBlock.Input(
            auction_id="oixa_auction_test",
            bidder_id="agent_1",
            bidder_name="Agent One",
            amount=0.0,
        )


def test_create_auction_max_budget_must_be_positive():
    with pytest.raises((ValidationError, ValueError)):
        CreateAuctionBlock.Input(
            rfi_description="Test task",
            max_budget=0.0,
            requester_id="agent_1",
        )


def test_register_offer_price_must_be_positive():
    with pytest.raises((ValidationError, ValueError)):
        RegisterOfferBlock.Input(
            agent_id="agent_1",
            agent_name="Agent One",
            capability="web_scraping",
            input_required="URL",
            output_guaranteed="Markdown text",
            price_usdc=0.0,
        )


def test_valid_bid_input_accepted():
    inp = PlaceBidBlock.Input(
        auction_id="oixa_auction_abc",
        bidder_id="agent_1",
        bidder_name="Agent One",
        amount=0.05,
    )
    assert inp.amount == 0.05


# ── ListAuctionsBlock ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_auctions_success():
    auctions = [{"id": "oixa_auction_001", "status": "open", "max_budget": 1.0}]
    payload = {"data": {"auctions": auctions}}

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = ListAuctionsBlock()
        inp = ListAuctionsBlock.Input(status="open", limit=10)
        result = await collect(block.run(inp))

    assert result["count"] == 1
    assert result.get("error", "") == ""
    parsed = json.loads(result["auctions_json"])
    assert parsed[0]["id"] == "oixa_auction_001"


@pytest.mark.asyncio
async def test_list_auctions_api_error():
    with patch("backend.blocks.oixa_protocol._call", _mock_response({"error": "server error"})):
        block = ListAuctionsBlock()
        inp = ListAuctionsBlock.Input()
        result = await collect(block.run(inp))

    assert result["error"] == "server error"


# ── PlaceBidBlock ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_place_bid_accepted():
    payload = {"data": {"accepted": True, "current_winner": "agent_1", "current_best": 0.05, "bid_id": "oixa_bid_xyz"}}

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = PlaceBidBlock()
        inp = PlaceBidBlock.Input(auction_id="oixa_auction_001", bidder_id="agent_1", bidder_name="Agent One", amount=0.05)
        result = await collect(block.run(inp))

    assert result["accepted"] is True
    assert result["bid_id"] == "oixa_bid_xyz"
    assert result.get("error", "") == ""


@pytest.mark.asyncio
async def test_place_bid_outbid():
    payload = {"data": {"accepted": False, "current_winner": "agent_2", "current_best": 0.03}}

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = PlaceBidBlock()
        inp = PlaceBidBlock.Input(auction_id="oixa_auction_001", bidder_id="agent_1", bidder_name="Agent One", amount=0.05)
        result = await collect(block.run(inp))

    assert result["accepted"] is False
    assert result["current_winner"] == "agent_2"


# ── CreateAuctionBlock ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_auction_success():
    payload = {"data": {"id": "oixa_auction_new", "status": "open"}}

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = CreateAuctionBlock()
        inp = CreateAuctionBlock.Input(rfi_description="Analyze DeFi trends", max_budget=1.0, requester_id="agent_ceo")
        result = await collect(block.run(inp))

    assert result["auction_id"] == "oixa_auction_new"
    assert result["status"] == "open"
    assert result.get("error", "") == ""


# ── DeliverOutputBlock ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deliver_output_verified():
    payload = {"data": {"passed": True, "payment_usdc": 0.95}}

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = DeliverOutputBlock()
        inp = DeliverOutputBlock.Input(auction_id="oixa_auction_001", agent_id="agent_1", output="Analysis complete.")
        result = await collect(block.run(inp))

    assert result["passed"] is True
    assert result["payment_usdc"] == 0.95
    assert result.get("error", "") == ""


@pytest.mark.asyncio
async def test_deliver_output_failed_verification():
    payload = {"data": {"passed": False}, "error": "output too short"}

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = DeliverOutputBlock()
        inp = DeliverOutputBlock.Input(auction_id="oixa_auction_001", agent_id="agent_1", output="ok")
        result = await collect(block.run(inp))

    assert "passed" not in result
    assert result["error"] == "output too short"


# ── RegisterOfferBlock ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_offer_success():
    payload = {
        "data": {
            "id": "oixa_cap_abc123",
            "discovery_url": "/api/v1/capabilities?need=web_scraping",
        }
    }

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = RegisterOfferBlock()
        inp = RegisterOfferBlock.Input(
            agent_id="agent_1",
            agent_name="Agent One",
            capability="web_scraping",
            input_required="URL string",
            output_guaranteed="Markdown text",
            price_usdc=0.02,
        )
        result = await collect(block.run(inp))

    assert result["capability_id"] == "oixa_cap_abc123"
    assert "web_scraping" in result["discovery_url"]
    assert result.get("error", "") == ""


@pytest.mark.asyncio
async def test_register_offer_api_error():
    with patch("backend.blocks.oixa_protocol._call", _mock_response({"error": "duplicate capability"})):
        block = RegisterOfferBlock()
        inp = RegisterOfferBlock.Input(
            agent_id="agent_1",
            agent_name="Agent One",
            capability="web_scraping",
            input_required="URL string",
            output_guaranteed="Markdown text",
            price_usdc=0.02,
        )
        result = await collect(block.run(inp))

    assert result["error"] == "duplicate capability"


# ── CheckBalanceBlock ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_balance_success():
    payload = {
        "data": {
            "agent_id": "agent_1",
            "score": 75.0,
            "transactions_completed": 7,
            "rank": 3,
        }
    }

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = CheckBalanceBlock()
        inp = CheckBalanceBlock.Input(agent_id="agent_1")
        result = await collect(block.run(inp))

    assert result["score"] == 75.0
    assert result["transactions_completed"] == 7
    assert result["rank"] == 3
    assert result.get("error", "") == ""


@pytest.mark.asyncio
async def test_check_balance_agent_not_found():
    with patch("backend.blocks.oixa_protocol._call", _mock_response({"error": "Agent has no reputation record yet"})):
        block = CheckBalanceBlock()
        inp = CheckBalanceBlock.Input(agent_id="unknown_agent")
        result = await collect(block.run(inp))

    assert result["error"] == "Agent has no reputation record yet"


@pytest.mark.asyncio
async def test_check_balance_malformed_response():
    payload = {"data": {"score": "N/A", "transactions_completed": "bad", "rank": None}}

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = CheckBalanceBlock()
        inp = CheckBalanceBlock.Input(agent_id="agent_1")
        result = await collect(block.run(inp))

    assert "error" in result
    assert "Malformed" in result["error"]


@pytest.mark.asyncio
async def test_list_auctions_data_null():
    """API returns {"data": null} — should not raise AttributeError."""
    payload = {"data": None}

    with patch("backend.blocks.oixa_protocol._call", _mock_response(payload)):
        block = ListAuctionsBlock()
        inp = ListAuctionsBlock.Input()
        result = await collect(block.run(inp))

    assert result["count"] == 0
    assert json.loads(result["auctions_json"]) == []
