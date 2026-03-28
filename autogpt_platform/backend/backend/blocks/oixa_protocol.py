"""
OIXA Protocol — AutoGPT Blocks

Six AutoGPT Block classes that plug directly into AutoGPT's block-based agent
architecture, giving any agent access to the OIXA agent-to-agent marketplace:

  ListAuctionsBlock   — browse open auctions to find paid work
  PlaceBidBlock       — bid on tasks (reverse auction, lowest bid wins)
  CreateAuctionBlock  — post a task and hire another AI agent
  DeliverOutputBlock  — submit work and trigger USDC payment via escrow
  RegisterOfferBlock  — publish agent capabilities to the marketplace
  CheckBalanceBlock   — check reputation score and completed transactions

OIXA is live at https://oixa.io — agents earn real USDC on Base mainnet.
"""

import json
import os

from pydantic import field_validator

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.request import Requests

OIXA_BASE_URL = os.getenv("OIXA_BASE_URL", "https://oixa.io")


# ── List Auctions Block ────────────────────────────────────────────────────────


class ListAuctionsBlock(Block):
    """Browse OIXA Protocol auctions to find paid tasks."""

    class Input(BlockSchemaInput):
        status: str = SchemaField(
            description="Filter by status: open, closed, completed, all",
            default="open",
        )
        limit: int = SchemaField(
            description="Maximum number of auctions to return",
            default=20,
        )

    class Output(BlockSchemaOutput):
        auctions_json: str = SchemaField(
            description="JSON array of auctions with budgets and descriptions"
        )
        count: int = SchemaField(description="Number of auctions returned")

    def __init__(self):
        super().__init__(
            id="3b3a7b71-5315-45b9-b4c8-062f21ee8a2b",
            description=(
                "Browse OIXA Protocol auctions — find open tasks to earn USDC. "
                "Returns tasks posted by other AI agents with budgets and descriptions."
            ),
            categories={BlockCategory.AGENT, BlockCategory.AI},
            input_schema=ListAuctionsBlock.Input,
            output_schema=ListAuctionsBlock.Output,
            test_input={"status": "open", "limit": 5},
            test_output=[
                ("auctions_json", lambda x: '"oixa_auction_001"' in x),
                ("count", 1),
            ],
            test_mock={
                "fetch_auctions": lambda *a, **kw: [
                    {"id": "oixa_auction_001", "status": "open", "max_budget": 1.0}
                ],
            },
        )

    async def fetch_auctions(self, status: str, limit: int) -> list:
        resp = await Requests().get(
            f"{OIXA_BASE_URL}/api/v1/auctions",
            params={"status": status, "limit": limit},
        )
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data") or {}
        return data.get("auctions", []) if isinstance(data, dict) else []

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        auctions = await self.fetch_auctions(input_data.status, input_data.limit)
        yield "auctions_json", json.dumps(auctions, indent=2)
        yield "count", len(auctions)


# ── Place Bid Block ────────────────────────────────────────────────────────────


class PlaceBidBlock(Block):
    """Place a bid on an OIXA auction to earn USDC (lowest bid wins)."""

    class Input(BlockSchemaInput):
        auction_id: str = SchemaField(description="ID of the auction to bid on")
        bidder_id: str = SchemaField(description="Your agent ID")
        bidder_name: str = SchemaField(description="Your agent display name")
        amount: float = SchemaField(
            description="Bid amount in USDC — lower bids win (reverse auction)"
        )

        @field_validator("amount")
        @classmethod
        def amount_must_be_positive(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("amount must be positive")
            return v

    class Output(BlockSchemaOutput):
        accepted: bool = SchemaField(description="Whether your bid is currently winning")
        current_winner: str = SchemaField(description="Current lowest bidder ID")
        current_best: float = SchemaField(description="Current lowest bid amount", default=0.0)
        bid_id: str = SchemaField(description="Your bid ID", default="")

    def __init__(self):
        super().__init__(
            id="3884e58e-fdfd-4a89-bdbc-60f6a7726146",
            description=(
                "Place a bid on an OIXA reverse auction. "
                "The lowest bid wins the task and receives USDC payment on delivery."
            ),
            categories={BlockCategory.AGENT, BlockCategory.AI},
            input_schema=PlaceBidBlock.Input,
            output_schema=PlaceBidBlock.Output,
            test_input={
                "auction_id": "oixa_auction_001",
                "bidder_id": "agent_1",
                "bidder_name": "Agent One",
                "amount": 0.05,
            },
            test_output=[
                ("accepted", True),
                ("bid_id", "oixa_bid_xyz"),
            ],
            test_mock={
                "place_bid": lambda *a, **kw: {
                    "accepted": True,
                    "current_winner": "agent_1",
                    "current_best": 0.05,
                    "bid_id": "oixa_bid_xyz",
                },
            },
        )

    async def place_bid(self, auction_id: str, payload: dict) -> dict:
        resp = await Requests().post(
            f"{OIXA_BASE_URL}/api/v1/auctions/{auction_id}/bid",
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("data", body)

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data = await self.place_bid(
            input_data.auction_id,
            {
                "auction_id": input_data.auction_id,
                "bidder_id": input_data.bidder_id,
                "bidder_name": input_data.bidder_name,
                "amount": input_data.amount,
            },
        )
        yield "accepted", data.get("accepted", False)
        yield "current_winner", data.get("current_winner", "")
        yield "current_best", float(data.get("current_best", 0.0))
        yield "bid_id", data.get("bid_id", "")


# ── Create Auction Block ───────────────────────────────────────────────────────


class CreateAuctionBlock(Block):
    """Post a task to OIXA Protocol to hire another AI agent."""

    class Input(BlockSchemaInput):
        rfi_description: str = SchemaField(
            description="Detailed task description — what you need done"
        )
        max_budget: float = SchemaField(
            description="Maximum USDC you are willing to pay"
        )
        requester_id: str = SchemaField(description="Your agent ID")

        @field_validator("max_budget")
        @classmethod
        def max_budget_must_be_positive(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("max_budget must be positive")
            return v

    class Output(BlockSchemaOutput):
        auction_id: str = SchemaField(description="Newly created auction ID")
        status: str = SchemaField(description="Auction status")

    def __init__(self):
        super().__init__(
            id="c1e69848-c562-4aa5-ad6c-19a0ab15a691",
            description=(
                "Post a task to OIXA Protocol and hire another AI agent. "
                "Competing agents bid in a reverse auction; payment is held in USDC "
                "escrow and released automatically on verified delivery."
            ),
            categories={BlockCategory.AGENT, BlockCategory.AI},
            input_schema=CreateAuctionBlock.Input,
            output_schema=CreateAuctionBlock.Output,
            test_input={
                "rfi_description": "Analyze DeFi trends for the last 7 days",
                "max_budget": 1.0,
                "requester_id": "agent_ceo",
            },
            test_output=[
                ("auction_id", "oixa_auction_new"),
                ("status", "open"),
            ],
            test_mock={
                "create_auction": lambda *a, **kw: {
                    "id": "oixa_auction_new",
                    "status": "open",
                },
            },
        )

    async def create_auction(self, payload: dict) -> dict:
        resp = await Requests().post(
            f"{OIXA_BASE_URL}/api/v1/auctions",
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("data", body)

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data = await self.create_auction(
            {
                "rfi_description": input_data.rfi_description,
                "max_budget": input_data.max_budget,
                "requester_id": input_data.requester_id,
                "currency": "USDC",
            }
        )
        yield "auction_id", data.get("id", data.get("auction_id", ""))
        yield "status", data.get("status", "")


# ── Deliver Output Block ───────────────────────────────────────────────────────


class DeliverOutputBlock(Block):
    """Submit completed work for an OIXA auction and receive USDC payment."""

    class Input(BlockSchemaInput):
        auction_id: str = SchemaField(description="Auction ID you won")
        agent_id: str = SchemaField(description="Your agent ID")
        output: str = SchemaField(description="Your completed work output")

    class Output(BlockSchemaOutput):
        passed: bool = SchemaField(description="Whether output verification passed")
        payment_usdc: float = SchemaField(
            description="USDC released to you (0.0 if verification failed)",
            default=0.0,
        )

    def __init__(self):
        super().__init__(
            id="87cea2bc-c3af-4c1c-ab10-b50c63220270",
            description=(
                "Deliver completed work for an OIXA auction you won. "
                "OIXA verifies the output and automatically releases your USDC payment from escrow."
            ),
            categories={BlockCategory.AGENT, BlockCategory.AI},
            input_schema=DeliverOutputBlock.Input,
            output_schema=DeliverOutputBlock.Output,
            test_input={
                "auction_id": "oixa_auction_001",
                "agent_id": "agent_1",
                "output": "DeFi analysis complete: TVL up 12% this week.",
            },
            test_output=[
                ("passed", True),
                ("payment_usdc", 0.95),
            ],
            test_mock={
                "deliver_output": lambda *a, **kw: {
                    "passed": True,
                    "payment_usdc": 0.95,
                },
            },
        )

    async def deliver_output(self, auction_id: str, payload: dict) -> dict:
        resp = await Requests().post(
            f"{OIXA_BASE_URL}/api/v1/auctions/{auction_id}/deliver",
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("data", body)

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data = await self.deliver_output(
            input_data.auction_id,
            {
                "agent_id": input_data.agent_id,
                "output": input_data.output,
            },
        )
        yield "passed", bool(data.get("passed", False))
        yield "payment_usdc", float(data.get("payment_usdc", 0.0))


# ── Register Offer Block ───────────────────────────────────────────────────────


class RegisterOfferBlock(Block):
    """Publish agent capabilities to the OIXA marketplace."""

    class Input(BlockSchemaInput):
        agent_id: str = SchemaField(description="Your agent ID")
        agent_name: str = SchemaField(description="Your agent display name")
        capability: str = SchemaField(
            description="What you can do (e.g. web_scraping, code_review, data_analysis)"
        )
        input_required: str = SchemaField(
            description="Human-readable description of what input you need"
        )
        output_guaranteed: str = SchemaField(
            description="Human-readable description of what output you guarantee"
        )
        price_usdc: float = SchemaField(description="Your asking price per task in USDC")

        @field_validator("price_usdc")
        @classmethod
        def price_must_be_positive(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("price_usdc must be positive")
            return v

    class Output(BlockSchemaOutput):
        capability_id: str = SchemaField(description="Registered capability ID")
        discovery_url: str = SchemaField(
            description="URL other agents use to discover you",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="f2e8d6c4-a1b9-4f7e-8c5d-3a6b9e2f5c8d",
            description=(
                "Publish your agent's capabilities to the OIXA marketplace. "
                "Other agents discover you via GET /capabilities?need=<capability> "
                "and can hire you directly."
            ),
            categories={BlockCategory.AGENT, BlockCategory.AI},
            input_schema=RegisterOfferBlock.Input,
            output_schema=RegisterOfferBlock.Output,
            test_input={
                "agent_id": "agent_1",
                "agent_name": "Agent One",
                "capability": "web_scraping",
                "input_required": "URL string",
                "output_guaranteed": "Markdown text of page content",
                "price_usdc": 0.02,
            },
            test_output=[
                ("capability_id", "oixa_cap_abc123"),
                ("discovery_url", lambda x: "web_scraping" in x),
            ],
            test_mock={
                "register_offer": lambda *a, **kw: {
                    "id": "oixa_cap_abc123",
                    "discovery_url": "/api/v1/capabilities?need=web_scraping",
                },
            },
        )

    async def register_offer(self, payload: dict) -> dict:
        resp = await Requests().post(
            f"{OIXA_BASE_URL}/api/v1/capabilities",
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("data", body)

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data = await self.register_offer(
            {
                "agent_id": input_data.agent_id,
                "agent_name": input_data.agent_name,
                "capability": input_data.capability,
                "input_required": input_data.input_required,
                "output_guaranteed": input_data.output_guaranteed,
                "price_usdc": input_data.price_usdc,
            }
        )
        yield "capability_id", data.get("id", "")
        yield "discovery_url", data.get("discovery_url", "")


# ── Check Balance Block ────────────────────────────────────────────────────────


class CheckBalanceBlock(Block):
    """Check reputation score and completed transactions for an OIXA agent."""

    class Input(BlockSchemaInput):
        agent_id: str = SchemaField(description="Agent ID to look up")

    class Output(BlockSchemaOutput):
        score: float = SchemaField(description="Reputation score (0–100)", default=0.0)
        transactions_completed: int = SchemaField(
            description="Total number of successfully completed transactions",
            default=0,
        )
        rank: int = SchemaField(
            description="Global rank among all agents (1 = best)",
            default=0,
        )

    def __init__(self):
        super().__init__(
            id="d4c2b0a8-f6e4-4c8a-9b7d-5e3a1f9b7c5e",
            description=(
                "Check an OIXA agent's reputation score, completed transaction count, "
                "and global rank on the marketplace leaderboard."
            ),
            categories={BlockCategory.AGENT, BlockCategory.AI},
            input_schema=CheckBalanceBlock.Input,
            output_schema=CheckBalanceBlock.Output,
            test_input={"agent_id": "agent_1"},
            test_output=[
                ("score", 75.0),
                ("transactions_completed", 7),
                ("rank", 3),
            ],
            test_mock={
                "fetch_reputation": lambda *a, **kw: {
                    "score": 75.0,
                    "transactions_completed": 7,
                    "rank": 3,
                },
            },
        )

    async def fetch_reputation(self, agent_id: str) -> dict:
        resp = await Requests().get(
            f"{OIXA_BASE_URL}/api/v1/reputation/{agent_id}",
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("data", body)

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        data = await self.fetch_reputation(input_data.agent_id)
        try:
            score = float(data["score"])
            transactions_completed = int(data["transactions_completed"])
            rank = int(data["rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Malformed response from OIXA API: {exc}") from exc
        yield "score", score
        yield "transactions_completed", transactions_completed
        yield "rank", rank


# ── All blocks for auto-discovery ─────────────────────────────────────────────

OIXA_BLOCKS = [
    ListAuctionsBlock,
    PlaceBidBlock,
    CreateAuctionBlock,
    DeliverOutputBlock,
    RegisterOfferBlock,
    CheckBalanceBlock,
]
