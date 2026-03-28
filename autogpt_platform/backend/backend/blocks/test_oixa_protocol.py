"""
Tests for OIXA Protocol AutoGPT blocks.

Run with:  poetry run pytest backend/blocks/test_oixa_protocol.py -v
"""

import importlib
import json
import re
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

import backend.blocks.oixa_protocol as oixa_mod
from backend.blocks.oixa_protocol import (
    OIXA_BLOCKS,
    CheckBalanceBlock,
    CreateAuctionBlock,
    DeliverOutputBlock,
    ListAuctionsBlock,
    PlaceBidBlock,
    RegisterOfferBlock,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def collect(gen) -> dict:
    """Drain an async generator into a dict of {output_name: value}."""
    result = {}
    async for key, value in gen:
        result[key] = value
    return result


# ── Config ────────────────────────────────────────────────────────────────────


def test_base_url_default():
    assert oixa_mod.OIXA_BASE_URL == "https://oixa.io"


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OIXA_BASE_URL", "http://localhost:8000")
    importlib.reload(oixa_mod)
    assert oixa_mod.OIXA_BASE_URL == "http://localhost:8000"
    monkeypatch.setenv("OIXA_BASE_URL", "https://oixa.io")
    importlib.reload(oixa_mod)


# ── Block identity ─────────────────────────────────────────────────────────────


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


def test_block_ids_are_unique():
    ids = [b().id for b in OIXA_BLOCKS]
    assert len(ids) == len(set(ids)), "Block IDs must be unique"


def test_block_ids_are_valid_uuids():
    uuid_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    for cls in OIXA_BLOCKS:
        block = cls()
        assert uuid_re.match(block.id), f"{cls.__name__} has invalid UUID: {block.id}"


def test_all_blocks_have_test_metadata():
    for cls in OIXA_BLOCKS:
        block = cls()
        assert block.test_input is not None, f"{cls.__name__} missing test_input"
        assert block.test_output is not None, f"{cls.__name__} missing test_output"
        assert block.test_mock is not None, f"{cls.__name__} missing test_mock"


# ── Input validators ──────────────────────────────────────────────────────────


def test_place_bid_zero_amount_rejected():
    with pytest.raises((ValidationError, ValueError)):
        PlaceBidBlock.Input(
            auction_id="oixa_auction_001", bidder_id="a", bidder_name="A", amount=0.0
        )


def test_place_bid_negative_amount_rejected():
    with pytest.raises((ValidationError, ValueError)):
        PlaceBidBlock.Input(
            auction_id="oixa_auction_001", bidder_id="a", bidder_name="A", amount=-1.0
        )


def test_place_bid_positive_amount_accepted():
    inp = PlaceBidBlock.Input(
        auction_id="oixa_auction_001", bidder_id="a", bidder_name="A", amount=0.05
    )
    assert inp.amount == 0.05


def test_create_auction_zero_budget_rejected():
    with pytest.raises((ValidationError, ValueError)):
        CreateAuctionBlock.Input(
            rfi_description="task", max_budget=0.0, requester_id="a"
        )


def test_register_offer_zero_price_rejected():
    with pytest.raises((ValidationError, ValueError)):
        RegisterOfferBlock.Input(
            agent_id="a", agent_name="A", capability="x",
            input_required="y", output_guaranteed="z", price_usdc=0.0,
        )


# ── ListAuctionsBlock ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_auctions_success():
    auctions = [{"id": "oixa_auction_001", "status": "open", "max_budget": 1.0}]
    block = ListAuctionsBlock()
    inp = ListAuctionsBlock.Input(status="open", limit=10)
    with patch.object(block, "fetch_auctions", new=AsyncMock(return_value=auctions)):
        result = await collect(block.run(inp))
    assert result["count"] == 1
    assert json.loads(result["auctions_json"])[0]["id"] == "oixa_auction_001"


@pytest.mark.asyncio
async def test_list_auctions_empty():
    block = ListAuctionsBlock()
    inp = ListAuctionsBlock.Input()
    with patch.object(block, "fetch_auctions", new=AsyncMock(return_value=[])):
        result = await collect(block.run(inp))
    assert result["count"] == 0
    assert json.loads(result["auctions_json"]) == []


@pytest.mark.asyncio
async def test_list_auctions_raises_on_error():
    block = ListAuctionsBlock()
    with patch.object(block, "fetch_auctions", new=AsyncMock(side_effect=RuntimeError("API down"))):
        with pytest.raises(RuntimeError, match="API down"):
            await collect(block.run(ListAuctionsBlock.Input()))


# ── PlaceBidBlock ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_place_bid_accepted():
    mock_data = {"accepted": True, "current_winner": "agent_1", "current_best": 0.05, "bid_id": "oixa_bid_xyz"}
    block = PlaceBidBlock()
    inp = PlaceBidBlock.Input(auction_id="oixa_auction_001", bidder_id="agent_1", bidder_name="Agent One", amount=0.05)
    with patch.object(block, "place_bid", new=AsyncMock(return_value=mock_data)):
        result = await collect(block.run(inp))
    assert result["accepted"] is True
    assert result["bid_id"] == "oixa_bid_xyz"
    assert result["current_best"] == 0.05


@pytest.mark.asyncio
async def test_place_bid_outbid():
    mock_data = {"accepted": False, "current_winner": "agent_2", "current_best": 0.03, "bid_id": ""}
    block = PlaceBidBlock()
    inp = PlaceBidBlock.Input(auction_id="oixa_auction_001", bidder_id="agent_1", bidder_name="Agent One", amount=0.05)
    with patch.object(block, "place_bid", new=AsyncMock(return_value=mock_data)):
        result = await collect(block.run(inp))
    assert result["accepted"] is False
    assert result["current_winner"] == "agent_2"


# ── CreateAuctionBlock ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_auction_success():
    block = CreateAuctionBlock()
    inp = CreateAuctionBlock.Input(rfi_description="Analyze DeFi", max_budget=1.0, requester_id="agent_ceo")
    with patch.object(block, "create_auction", new=AsyncMock(return_value={"id": "oixa_auction_new", "status": "open"})):
        result = await collect(block.run(inp))
    assert result["auction_id"] == "oixa_auction_new"
    assert result["status"] == "open"


@pytest.mark.asyncio
async def test_create_auction_raises_on_error():
    block = CreateAuctionBlock()
    inp = CreateAuctionBlock.Input(rfi_description="Test", max_budget=1.0, requester_id="a")
    with patch.object(block, "create_auction", new=AsyncMock(side_effect=RuntimeError("timeout"))):
        with pytest.raises(RuntimeError, match="timeout"):
            await collect(block.run(inp))


# ── DeliverOutputBlock ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deliver_output_verified():
    block = DeliverOutputBlock()
    inp = DeliverOutputBlock.Input(auction_id="oixa_auction_001", agent_id="agent_1", output="Analysis complete.")
    with patch.object(block, "deliver_output", new=AsyncMock(return_value={"passed": True, "payment_usdc": 0.95})):
        result = await collect(block.run(inp))
    assert result["passed"] is True
    assert result["payment_usdc"] == 0.95


@pytest.mark.asyncio
async def test_deliver_output_failed():
    block = DeliverOutputBlock()
    inp = DeliverOutputBlock.Input(auction_id="oixa_auction_001", agent_id="agent_1", output="too short")
    with patch.object(block, "deliver_output", new=AsyncMock(return_value={"passed": False, "payment_usdc": 0.0})):
        result = await collect(block.run(inp))
    assert result["passed"] is False
    assert result["payment_usdc"] == 0.0


# ── RegisterOfferBlock ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_offer_success():
    mock_data = {"id": "oixa_cap_abc123", "discovery_url": "/api/v1/capabilities?need=web_scraping"}
    block = RegisterOfferBlock()
    inp = RegisterOfferBlock.Input(
        agent_id="agent_1", agent_name="Agent One", capability="web_scraping",
        input_required="URL string", output_guaranteed="Markdown text", price_usdc=0.02,
    )
    with patch.object(block, "register_offer", new=AsyncMock(return_value=mock_data)):
        result = await collect(block.run(inp))
    assert result["capability_id"] == "oixa_cap_abc123"
    assert "web_scraping" in result["discovery_url"]


@pytest.mark.asyncio
async def test_register_offer_raises_on_error():
    block = RegisterOfferBlock()
    inp = RegisterOfferBlock.Input(
        agent_id="a", agent_name="A", capability="x",
        input_required="y", output_guaranteed="z", price_usdc=0.01,
    )
    with patch.object(block, "register_offer", new=AsyncMock(side_effect=RuntimeError("duplicate"))):
        with pytest.raises(RuntimeError, match="duplicate"):
            await collect(block.run(inp))


# ── CheckBalanceBlock ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_balance_success():
    block = CheckBalanceBlock()
    inp = CheckBalanceBlock.Input(agent_id="agent_1")
    with patch.object(block, "fetch_reputation", new=AsyncMock(return_value={"score": 75.0, "transactions_completed": 7, "rank": 3})):
        result = await collect(block.run(inp))
    assert result["score"] == 75.0
    assert result["transactions_completed"] == 7
    assert result["rank"] == 3


@pytest.mark.asyncio
async def test_check_balance_raises_on_error():
    block = CheckBalanceBlock()
    inp = CheckBalanceBlock.Input(agent_id="unknown")
    with patch.object(block, "fetch_reputation", new=AsyncMock(side_effect=RuntimeError("not found"))):
        with pytest.raises(RuntimeError, match="not found"):
            await collect(block.run(inp))


@pytest.mark.asyncio
async def test_check_balance_malformed_response():
    block = CheckBalanceBlock()
    inp = CheckBalanceBlock.Input(agent_id="agent_1")
    with patch.object(block, "fetch_reputation", new=AsyncMock(return_value={"score": "N/A", "transactions_completed": None, "rank": None})):
        with pytest.raises(RuntimeError, match="Malformed"):
            await collect(block.run(inp))


@pytest.mark.asyncio
async def test_check_balance_missing_fields():
    block = CheckBalanceBlock()
    inp = CheckBalanceBlock.Input(agent_id="agent_1")
    with patch.object(block, "fetch_reputation", new=AsyncMock(return_value={})):
        with pytest.raises(RuntimeError, match="Malformed"):
            await collect(block.run(inp))
