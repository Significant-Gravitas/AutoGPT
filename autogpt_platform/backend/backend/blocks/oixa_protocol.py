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
from typing import Optional

import httpx

OIXA_BASE_URL = "http://localhost:8000"


def _call(method: str, path: str, data: Optional[dict] = None) -> dict:
    with httpx.Client(timeout=15) as client:
        resp = client.request(method, f"{OIXA_BASE_URL}{path}", json=data)
        try:
            return resp.json()
        except Exception:
            return {"error": resp.text, "status_code": resp.status_code}


# ── Graceful import: works with or without autogpt-libs installed ─────────────

try:
    from autogpt_libs.supabase_integration_credentials_store.store import BaseBlockConfiguration  # noqa
    _AUTOGPT_AVAILABLE = True
except ImportError:
    _AUTOGPT_AVAILABLE = False

try:
    from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
    from backend.data.model import SchemaField
    _BLOCK_BASE = Block
except ImportError:
    # Fallback: plain class that works without AutoGPT runtime
    class BlockSchema:  # type: ignore
        pass

    class BlockOutput:  # type: ignore
        pass

    def SchemaField(description: str = "", default=None, **kw):  # type: ignore
        return default

    class _BlockBase:
        """Minimal Block base when autogpt is not installed."""
        @property
        def input_schema(self):
            return self._input_schema

        @property
        def output_schema(self):
            return self._output_schema

        def run(self, *args, **kwargs):
            raise NotImplementedError

    _BLOCK_BASE = _BlockBase  # type: ignore


# ── List Auctions Block ────────────────────────────────────────────────────────

class ListAuctionsInputSchema(BlockSchema):
    status: str = SchemaField(description="Filter: open, closed, completed, all", default="open")
    limit:  int = SchemaField(description="Max results", default=20)


class ListAuctionsOutputSchema(BlockSchema):
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

    def run(self, input_data: Input) -> BlockOutput:
        result = _call("GET", f"/api/v1/auctions?status={input_data.status}&limit={input_data.limit}")
        auctions = result.get("data", {}).get("auctions", result if isinstance(result, list) else [])
        yield "auctions_json", json.dumps(auctions, indent=2)
        yield "count",         len(auctions) if isinstance(auctions, list) else 0
        yield "error",         result.get("error", "")


# ── Place Bid Block ────────────────────────────────────────────────────────────

class PlaceBidInputSchema(BlockSchema):
    auction_id:  str   = SchemaField(description="Auction ID to bid on")
    bidder_id:   str   = SchemaField(description="Your agent ID")
    bidder_name: str   = SchemaField(description="Your agent display name")
    amount:      float = SchemaField(description="Bid in USDC — LOWER wins (reverse auction)")


class PlaceBidOutputSchema(BlockSchema):
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

    def run(self, input_data: Input) -> BlockOutput:
        result = _call("POST", f"/api/v1/auctions/{input_data.auction_id}/bid", {
            "auction_id":  input_data.auction_id,
            "bidder_id":   input_data.bidder_id,
            "bidder_name": input_data.bidder_name,
            "amount":      input_data.amount,
        })
        data = result.get("data", result)
        yield "accepted",       data.get("accepted", False)
        yield "current_winner", data.get("current_winner", "")
        yield "current_best",   data.get("current_best", 0.0)
        yield "bid_id",         data.get("bid_id", "")
        yield "error",          result.get("error", "")


# ── Create Auction Block ───────────────────────────────────────────────────────

class CreateAuctionInputSchema(BlockSchema):
    rfi_description: str   = SchemaField(description="Detailed task description — what you need done")
    max_budget:      float = SchemaField(description="Max USDC you will pay")
    requester_id:    str   = SchemaField(description="Your agent ID")


class CreateAuctionOutputSchema(BlockSchema):
    auction_id: str   = SchemaField(description="New auction ID")
    status:     str   = SchemaField(description="Auction status")
    error:      str   = SchemaField(description="Error if creation failed", default="")


class CreateAuctionBlock(_BLOCK_BASE):
    """
    Post a task to OIXA Protocol — hire other agents to do work for you.
    Payment held in USDC escrow. Lowest bidder wins and delivers.
    Keywords: hire agent, delegate task, post task, create auction, outsource work.
    """

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

    def run(self, input_data: Input) -> BlockOutput:
        result = _call("POST", "/api/v1/auctions", {
            "rfi_description": input_data.rfi_description,
            "max_budget":      input_data.max_budget,
            "requester_id":    input_data.requester_id,
            "currency":        "USDC",
        })
        data = result.get("data", result)
        yield "auction_id", data.get("id", data.get("auction_id", ""))
        yield "status",     data.get("status", "")
        yield "error",      result.get("error", "")


# ── Deliver Output Block ───────────────────────────────────────────────────────

class DeliverOutputInputSchema(BlockSchema):
    auction_id: str = SchemaField(description="Auction ID you won")
    agent_id:   str = SchemaField(description="Your agent ID")
    output:     str = SchemaField(description="Your completed work")


class DeliverOutputOutputSchema(BlockSchema):
    passed:        bool  = SchemaField(description="Whether verification passed")
    payment_usdc:  float = SchemaField(description="USDC released to you", default=0.0)
    error:         str   = SchemaField(description="Error if delivery failed", default="")


class DeliverOutputBlock(_BLOCK_BASE):
    """
    Deliver your completed work for an OIXA auction you won.
    OIXA verifies the output and automatically releases your USDC payment.
    Keywords: deliver work, submit output, get paid, release payment, earn USDC.
    """

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

    def run(self, input_data: Input) -> BlockOutput:
        result = _call("POST", f"/api/v1/auctions/{input_data.auction_id}/deliver", {
            "agent_id": input_data.agent_id,
            "output":   input_data.output,
        })
        data = result.get("data", result)
        yield "passed",       data.get("passed", False)
        yield "payment_usdc", data.get("payment_usdc", 0.0)
        yield "error",        result.get("error", "")


# ── All blocks for auto-discovery ─────────────────────────────────────────────

OIXA_BLOCKS = [
    ListAuctionsBlock,
    PlaceBidBlock,
    CreateAuctionBlock,
    DeliverOutputBlock,
]
