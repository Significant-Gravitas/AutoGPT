"""
Tests for OIXA Protocol AutoGPT blocks.

Run with:
    pytest agents/test_oixa_protocol.py -v
"""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

# Ensure fallback path is used (no autogpt runtime needed)
os.environ.setdefault("OIXA_BASE_URL", "https://oixa.io")

from oixa_autogpt import (
    OIXA_BASE_URL,
    OIXA_BLOCKS,
    CreateAuctionBlock,
    DeliverOutputBlock,
    ListAuctionsBlock,
    PlaceBidBlock,
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


# ── Config ────────────────────────────────────────────────────────────────────

def test_base_url_from_env():
    assert OIXA_BASE_URL == "https://oixa.io"


def test_base_url_override(monkeypatch):
    monkeypatch.setenv("OIXA_BASE_URL", "http://localhost:8000")
    import importlib
    import oixa_autogpt
    importlib.reload(oixa_autogpt)
    assert oixa_autogpt.OIXA_BASE_URL == "http://localhost:8000"


# ── Block identity ─────────────────────────────────────────────────────────────

def test_block_ids_are_unique():
    ids = [b.id for b in OIXA_BLOCKS]
    assert len(ids) == len(set(ids)), "Block IDs must be unique"


def test_block_ids_are_valid_uuids():
    import re
    uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    for block in OIXA_BLOCKS:
        assert uuid_re.match(block.id), f"{block.name} has invalid UUID: {block.id}"


def test_all_blocks_have_name_and_categories():
    for block in OIXA_BLOCKS:
        assert block.name, f"{block.__name__} missing name"
        assert block.categories, f"{block.__name__} missing categories"


# ── Validators ────────────────────────────────────────────────────────────────

def test_place_bid_amount_must_be_positive():
    from pydantic import ValidationError
    with pytest.raises((ValidationError, ValueError)):
        PlaceBidBlock.Input(
            auction_id="oixa_auction_test",
            bidder_id="agent_1",
            bidder_name="Agent One",
            amount=-1.0,
        )


def test_place_bid_zero_amount_rejected():
    from pydantic import ValidationError
    with pytest.raises((ValidationError, ValueError)):
        PlaceBidBlock.Input(
            auction_id="oixa_auction_test",
            bidder_id="agent_1",
            bidder_name="Agent One",
            amount=0.0,
        )


def test_create_auction_max_budget_must_be_positive():
    from pydantic import ValidationError
    with pytest.raises((ValidationError, ValueError)):
        CreateAuctionBlock.Input(
            rfi_description="Test task",
            max_budget=0.0,
            requester_id="agent_1",
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

    with patch("oixa_autogpt._call", _mock_response(payload)):
        block = ListAuctionsBlock()
        inp = ListAuctionsBlock.Input(status="open", limit=10)
        result = await collect(block.run(inp))

    assert result["count"] == 1
    assert result["error"] == ""
    parsed = json.loads(result["auctions_json"])
    assert parsed[0]["id"] == "oixa_auction_001"


@pytest.mark.asyncio
async def test_list_auctions_api_error():
    with patch("oixa_autogpt._call", _mock_response({"error": "server error"})):
        block = ListAuctionsBlock()
        inp = ListAuctionsBlock.Input()
        result = await collect(block.run(inp))

    assert result["error"] == "server error"


# ── PlaceBidBlock ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_place_bid_accepted():
    payload = {"data": {"accepted": True, "current_winner": "agent_1", "current_best": 0.05, "bid_id": "oixa_bid_xyz"}}

    with patch("oixa_autogpt._call", _mock_response(payload)):
        block = PlaceBidBlock()
        inp = PlaceBidBlock.Input(auction_id="oixa_auction_001", bidder_id="agent_1", bidder_name="Agent One", amount=0.05)
        result = await collect(block.run(inp))

    assert result["accepted"] is True
    assert result["bid_id"] == "oixa_bid_xyz"
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_place_bid_outbid():
    payload = {"data": {"accepted": False, "current_winner": "agent_2", "current_best": 0.03}}

    with patch("oixa_autogpt._call", _mock_response(payload)):
        block = PlaceBidBlock()
        inp = PlaceBidBlock.Input(auction_id="oixa_auction_001", bidder_id="agent_1", bidder_name="Agent One", amount=0.05)
        result = await collect(block.run(inp))

    assert result["accepted"] is False
    assert result["current_winner"] == "agent_2"


# ── CreateAuctionBlock ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_auction_success():
    payload = {"data": {"id": "oixa_auction_new", "status": "open"}}

    with patch("oixa_autogpt._call", _mock_response(payload)):
        block = CreateAuctionBlock()
        inp = CreateAuctionBlock.Input(rfi_description="Analyze DeFi trends", max_budget=1.0, requester_id="agent_ceo")
        result = await collect(block.run(inp))

    assert result["auction_id"] == "oixa_auction_new"
    assert result["status"] == "open"
    assert result["error"] == ""


# ── DeliverOutputBlock ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deliver_output_verified():
    payload = {"data": {"passed": True, "payment_usdc": 0.95}}

    with patch("oixa_autogpt._call", _mock_response(payload)):
        block = DeliverOutputBlock()
        inp = DeliverOutputBlock.Input(auction_id="oixa_auction_001", agent_id="agent_1", output="Analysis complete.")
        result = await collect(block.run(inp))

    assert result["passed"] is True
    assert result["payment_usdc"] == 0.95
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_deliver_output_failed_verification():
    payload = {"data": {"passed": False}, "error": "output too short"}

    with patch("oixa_autogpt._call", _mock_response(payload)):
        block = DeliverOutputBlock()
        inp = DeliverOutputBlock.Input(auction_id="oixa_auction_001", agent_id="agent_1", output="ok")
        result = await collect(block.run(inp))

    assert result["passed"] is False
    assert result["error"] == "output too short"
