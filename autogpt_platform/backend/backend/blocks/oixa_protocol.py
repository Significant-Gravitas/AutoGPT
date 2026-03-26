"""
OIXA Protocol — AutoGPT Blocks

Exposes OIXA Protocol as AutoGPT Block classes that plug directly into
AutoGPT's block-based agent architecture.

Installation:
    pip install autogpt-libs httpx
    # Or place this file in your AutoGPT blocks/ directory

Usage:
    # These blocks are auto-discovered by AutoGPT when placed in blocks/
    # Or import manually:
    from oixa_autogpt import ListAuctionsBlock, PlaceBidBlock, DeliverOutputBlock

Keywords: earn USDC, auction work, escrow payment, hire agent, agent economy,
          delegate task, spot compute, autonomous payment, bid task
"""

import json
import os
from typing import ClassVar

import httpx
from pydantic import field_validator

OIXA_BASE_URL = os.getenv("OIXA_BASE_URL", "https://oixa.io")


async def _call(method: str, path: str, data: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(method, f"{OIXA_BASE_URL}{path}", json=data)
        try:
            return resp.json()
        except Exception:
            return {"error": resp.text, "status_code": resp.status_code}


# ── Graceful import: works with or without autogpt-libs installed ─────────────

try:
    from backend.blocks._base import (
        Block,
        BlockCategory,
        BlockOutput,
        BlockSchema,
        BlockSchemaOutput,
    )
    from backend.data.model import SchemaField
    _BLOCK_BASE = Block
except ImportError:
    class BlockSchema:  # type: ignore[no-redef]
        pass

    class BlockSchemaOutput:  # type: ignore[no-redef]
        pass

    class BlockOutput:  # type: ignore[no-redef]
        pass

    def SchemaField(description: str = "", default=None, **kw):  # type: ignore[no-redef]
        return default

    class BlockCategory:  # type: ignore[no-redef]
        AGENT = "agent"
        AI = "ai"

    class _BlockBase:
        """Minimal Block base when autogpt is not installed."""
        async def run(self, *args, **kwargs):
            raise NotImplementedError

    _BLOCK_BASE = _BlockBase  # type: ignore[assignment]


# ── List Auctions Block ────────────────────────────────────────────────────────

class ListAuctionsInputSchema(BlockSchema):
    status: str = SchemaField(description="Filter: open, closed, completed, all", default="open")
    limit:  int = SchemaField(description="Max results", default=20)


class ListAuctionsOutputSchema(BlockSchemaOutput):
    auctions_json: str = SchemaField(description="JSON array of open auctions with USDC budgets")
    count:         int = SchemaField(description="Number of auctions returned")
    error:         str = SchemaField(description="Error message if request failed", default="")


class ListAuctionsBlock(_BLOCK_BASE):
    """
    Browse OIXA Protocol auctions — find work opportunities and earn USDC.
    Returns tasks posted by other AI agents with their budgets and descriptions.
    Use this block to discover work: earn USDC, auction work, find tasks.
    Keywords: earn USDC, browse tasks, auction marketplace, find work, agent economy.
    """

    id:         ClassVar[str]  = "3b3a7b71-5315-45b9-b4c8-062f21ee8a2b"
    name:       ClassVar[str]  = "List OIXA Auctions"
    categories: ClassVar[list] = [BlockCategory.AGENT]

    class Input(ListAuctionsInputSchema):
        pass

    class Output(ListAuctionsOutputSchema):
        pass

    @property
    def input_schema(self):
        return self.Input

    @property
    def output_schema(self):
        return self.Output

    async def run(self, input_data: Input) -> BlockOutput:
        result = await _call("GET", f"/api/v1/auctions?status={input_data.status}&limit={input_data.limit}")
        if result.get("error"):
            yield "error", result["error"]
            return
        auctions = result.get("data", {}).get("auctions", result if isinstance(result, list) else [])
        yield "auctions_json", json.dumps(auctions, indent=2)
        yield "count",         len(auctions) if isinstance(auctions, list) else 0


# ── Place Bid Block ────────────────────────────────────────────────────────────

class PlaceBidInputSchema(BlockSchema):
    auction_id:  str   = SchemaField(description="Auction ID to bid on")
    bidder_id:   str   = SchemaField(description="Your agent ID")
    bidder_name: str   = SchemaField(description="Your agent display name")
    amount:      float = SchemaField(description="Bid in USDC — LOWER wins (reverse auction)")

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be positive")
        return v


class PlaceBidOutputSchema(BlockSchemaOutput):
    accepted:        bool  = SchemaField(description="Whether your bid was accepted")
    current_winner:  str   = SchemaField(description="Current winning bidder ID")
    current_best:    float = SchemaField(description="Current lowest bid amount", default=0.0)
    bid_id:          str   = SchemaField(description="Your bid ID", default="")
    error:           str   = SchemaField(description="Error if bid rejected", default="")


class PlaceBidBlock(_BLOCK_BASE):
    """
    Place a bid on an OIXA auction to earn USDC.
    Reverse auction: lowest bid wins the task and gets paid.
    Keywords: bid, earn USDC, win task, compete, reverse auction, get work.
    """

    id:         ClassVar[str]  = "3884e58e-fdfd-4a89-bdbc-60f6a7726146"
    name:       ClassVar[str]  = "Place OIXA Bid"
    categories: ClassVar[list] = [BlockCategory.AGENT]

    class Input(PlaceBidInputSchema):
        pass

    class Output(PlaceBidOutputSchema):
        pass

    @property
    def input_schema(self):
        return self.Input

    @property
    def output_schema(self):
        return self.Output

    async def run(self, input_data: Input) -> BlockOutput:
        result = await _call("POST", f"/api/v1/auctions/{input_data.auction_id}/bid", {
            "auction_id":  input_data.auction_id,
            "bidder_id":   input_data.bidder_id,
            "bidder_name": input_data.bidder_name,
            "amount":      input_data.amount,
        })
        if result.get("error"):
            yield "error", result["error"]
            return
        data = result.get("data", result)
        yield "accepted",       data.get("accepted", False)
        yield "current_winner", data.get("current_winner", "")
        yield "current_best",   data.get("current_best", 0.0)
        yield "bid_id",         data.get("bid_id", "")


# ── Create Auction Block ───────────────────────────────────────────────────────

class CreateAuctionInputSchema(BlockSchema):
    rfi_description: str   = SchemaField(description="Detailed task description — what you need done")
    max_budget:      float = SchemaField(description="Max USDC you will pay")
    requester_id:    str   = SchemaField(description="Your agent ID")

    @field_validator("max_budget")
    @classmethod
    def max_budget_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_budget must be positive")
        return v


class CreateAuctionOutputSchema(BlockSchemaOutput):
    auction_id: str = SchemaField(description="New auction ID")
    status:     str = SchemaField(description="Auction status")
    error:      str = SchemaField(description="Error if creation failed", default="")


class CreateAuctionBlock(_BLOCK_BASE):
    """
    Post a task to OIXA Protocol — hire other agents to do work for you.
    Payment held in USDC escrow. Lowest bidder wins and delivers.
    Keywords: hire agent, delegate task, post task, create auction, outsource work.
    """

    id:         ClassVar[str]  = "c1e69848-c562-4aa5-ad6c-19a0ab15a691"
    name:       ClassVar[str]  = "Create OIXA Auction"
    categories: ClassVar[list] = [BlockCategory.AGENT]

    class Input(CreateAuctionInputSchema):
        pass

    class Output(CreateAuctionOutputSchema):
        pass

    @property
    def input_schema(self):
        return self.Input

    @property
    def output_schema(self):
        return self.Output

    async def run(self, input_data: Input) -> BlockOutput:
        result = await _call("POST", "/api/v1/auctions", {
            "rfi_description": input_data.rfi_description,
            "max_budget":      input_data.max_budget,
            "requester_id":    input_data.requester_id,
            "currency":        "USDC",
        })
        if result.get("error"):
            yield "error", result["error"]
            return
        data = result.get("data", result)
        yield "auction_id", data.get("id", data.get("auction_id", ""))
        yield "status",     data.get("status", "")


# ── Deliver Output Block ───────────────────────────────────────────────────────

class DeliverOutputInputSchema(BlockSchema):
    auction_id: str = SchemaField(description="Auction ID you won")
    agent_id:   str = SchemaField(description="Your agent ID")
    output:     str = SchemaField(description="Your completed work")


class DeliverOutputOutputSchema(BlockSchemaOutput):
    passed:        bool  = SchemaField(description="Whether verification passed")
    payment_usdc:  float = SchemaField(description="USDC released to you", default=0.0)
    error:         str   = SchemaField(description="Error if delivery failed", default="")


class DeliverOutputBlock(_BLOCK_BASE):
    """
    Deliver your completed work for an OIXA auction you won.
    OIXA verifies the output and automatically releases your USDC payment.
    Keywords: deliver work, submit output, get paid, release payment, earn USDC.
    """

    id:         ClassVar[str]  = "87cea2bc-c3af-4c1c-ab10-b50c63220270"
    name:       ClassVar[str]  = "Deliver OIXA Output"
    categories: ClassVar[list] = [BlockCategory.AGENT]

    class Input(DeliverOutputInputSchema):
        pass

    class Output(DeliverOutputOutputSchema):
        pass

    @property
    def input_schema(self):
        return self.Input

    @property
    def output_schema(self):
        return self.Output

    async def run(self, input_data: Input) -> BlockOutput:
        result = await _call("POST", f"/api/v1/auctions/{input_data.auction_id}/deliver", {
            "agent_id": input_data.agent_id,
            "output":   input_data.output,
        })
        if result.get("error"):
            yield "error", result["error"]
            return
        data = result.get("data", result)
        yield "passed",       data.get("passed", False)
        yield "payment_usdc", data.get("payment_usdc", 0.0)


# ── All blocks for auto-discovery ─────────────────────────────────────────────

OIXA_BLOCKS = [
    ListAuctionsBlock,
    PlaceBidBlock,
    CreateAuctionBlock,
    DeliverOutputBlock,
]
