"""Spraay transaction status block.

Uses the free ``GET /api/v1/tx/decode`` endpoint (Blockscout-backed) to look
up a transaction's on-chain status along with a plain-English summary and its
token transfers. No x402 payment or credentials are required.
"""

from enum import Enum
from typing import Any

from backend.sdk import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    SchemaField,
)

from ._api import spraay_request


class DecodeChain(str, Enum):
    """Chains the gateway's transaction decoder can read."""

    BASE = "base"
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"


class SpraayTransactionStatusBlock(Block):
    """Check the status of an EVM transaction and decode what it did.

    Use it to wait for a batch payment to confirm, branch on failures, or log
    completed payouts.
    """

    class Input(BlockSchemaInput):
        chain: DecodeChain = SchemaField(
            description="Chain the transaction was sent on",
            default=DecodeChain.BASE,
        )
        transaction_hash: str = SchemaField(
            description="Transaction hash (0x followed by 64 hex characters)",
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(
            description="Transaction status reported by the explorer (e.g. ok, error)",
        )
        block_number: int = SchemaField(
            description="Block the transaction was included in (0 if pending)",
            default=0,
        )
        gas_used: str = SchemaField(description="Gas used", default="0")
        fee: str = SchemaField(description="Fee paid, in wei", default="0")
        summary: str = SchemaField(
            description="Plain-English summary of the transaction",
            default="",
        )
        transaction_type: str = SchemaField(
            description="Decoded type (transfer, swap, approval, ...)",
            default="unknown",
        )
        token_transfers: list[dict[str, Any]] = SchemaField(
            description="Token transfers performed by the transaction",
            default_factory=list,
        )

    def __init__(self):
        super().__init__(
            id="0617a4fb-c6b7-4569-8330-8912c37086c5",
            description=(
                "Check the on-chain status of an EVM transaction and decode it "
                "into a plain-English summary with token transfers. Free "
                "Spraay gateway endpoint."
            ),
            categories={BlockCategory.BASIC},
            effect=BlockEffect.READ,
            input_schema=SpraayTransactionStatusBlock.Input,
            output_schema=SpraayTransactionStatusBlock.Output,
            test_input={
                "chain": "base",
                "transaction_hash": "0x" + "ab" * 32,
            },
            test_output=[
                ("status", "ok"),
                ("block_number", 18500000),
                ("gas_used", "85000"),
                ("fee", "1200000000000"),
                ("summary", "Transferred 1.0 USDC to 0x1234...5678"),
                ("transaction_type", "transfer"),
                (
                    "token_transfers",
                    [{"symbol": "USDC", "amount": "1.0"}],
                ),
            ],
            test_mock={
                "decode_transaction": lambda *args, **kwargs: {
                    "hash": "0x" + "ab" * 32,
                    "chain": "base",
                    "status": "ok",
                    "blockNumber": 18500000,
                    "gasUsed": "85000",
                    "fee": "1200000000000",
                    "decoded": {
                        "type": "transfer",
                        "summary": "Transferred 1.0 USDC to 0x1234...5678",
                    },
                    "tokenTransfers": [{"symbol": "USDC", "amount": "1.0"}],
                }
            },
        )

    @staticmethod
    async def decode_transaction(chain: str, transaction_hash: str) -> dict[str, Any]:
        """Decode a transaction via the free Spraay gateway endpoint."""
        return await spraay_request(
            method="GET",
            endpoint="/api/v1/tx/decode",
            params={"hash": transaction_hash, "chain": chain},
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        result = await self.decode_transaction(
            chain=input_data.chain.value,
            transaction_hash=input_data.transaction_hash,
        )

        if not result.get("status"):
            yield "error", "Gateway returned incomplete response: missing 'status'"
            return

        decoded = result.get("decoded") or {}
        yield "status", result["status"]
        yield "block_number", result.get("blockNumber") or 0
        yield "gas_used", result.get("gasUsed") or "0"
        yield "fee", result.get("fee") or "0"
        yield "summary", decoded.get("summary", "")
        yield "transaction_type", decoded.get("type", "unknown")
        yield "token_transfers", result.get("tokenTransfers", [])
