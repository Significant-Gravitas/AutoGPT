"""Spraay transaction status check block."""

import uuid
from typing import Any

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsMetaInput, SchemaField

from ._api import SpraayAPIError, spraay_request
from ._config import spraay_provider
from .batch_payment import ChainNetwork


class SpraayTransactionStatusBlock(Block):
    """
    Check the status of a Spraay transaction.

    Query the on-chain confirmation status of a previously submitted transaction.
    Use this in workflows to wait for confirmations before proceeding, retry
    failed transactions, or log completed payments.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[Any, Any] = spraay_provider.credentials_field(
            description="Spraay API credentials",
        )
        chain: ChainNetwork = SchemaField(
            description="Blockchain network the transaction was sent on",
            default=ChainNetwork.BASE,
        )
        transaction_hash: str = SchemaField(
            description="Transaction hash to check status for",
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="Transaction status: pending, confirmed, or failed",
        )
        confirmations: int = SchemaField(
            description="Number of block confirmations",
        )
        block_number: int = SchemaField(
            description="Block number the transaction was included in (0 if pending)",
        )
        gas_used: str = SchemaField(
            description="Gas used by the transaction",
        )
        error: str = SchemaField(
            description="Error message if the status check failed",
        )

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-defa-234567890123",
            description=(
                "Check the confirmation status of a Spraay transaction. "
                "Returns confirmations, block number, and gas used. "
                "Powered by Spraay's x402 payment gateway."
            ),
            categories={BlockCategory.FINANCE},
            input_schema=SpraayTransactionStatusBlock.Input,
            output_schema=SpraayTransactionStatusBlock.Output,
            test_input={
                "credentials": {
                    "provider": "spraay",
                    "id": str(uuid.uuid4()),
                    "type": "api_key",
                },
                "chain": "base",
                "transaction_hash": "0xabc123def456789...",
            },
            test_output=[
                ("status", "confirmed"),
                ("confirmations", 12),
                ("block_number", 18500000),
                ("gas_used", "85000"),
            ],
            test_mock={
                "fetch_status": lambda *args, **kwargs: {
                    "status": "confirmed",
                    "confirmations": 12,
                    "block_number": 18500000,
                    "gas_used": "85000",
                }
            },
        )

    @staticmethod
    def fetch_status(
        api_key: str,
        chain: str,
        transaction_hash: str,
    ) -> dict:
        return spraay_request(
            method="GET",
            endpoint="/v1/transaction/status",
            api_key=api_key,
            params={
                "chain": chain,
                "txHash": transaction_hash,
            },
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            api_key = input_data.credentials.api_key.get_secret_value()

            result = self.fetch_status(
                api_key=api_key,
                chain=input_data.chain.value,
                transaction_hash=input_data.transaction_hash,
            )

            yield "status", result.get("status", "unknown")
            yield "confirmations", result.get("confirmations", 0)
            yield "block_number", result.get("block_number", 0)
            yield "gas_used", result.get("gas_used", "0")

        except SpraayAPIError as e:
            yield "error", f"Spraay API error: {e.message}"
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
