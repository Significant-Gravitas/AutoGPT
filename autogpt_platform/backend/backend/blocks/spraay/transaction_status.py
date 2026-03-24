"""Spraay transaction status check block.

This module provides the SpraayTransactionStatusBlock, which queries the
on-chain confirmation status of a previously submitted transaction. Use it
in workflows to wait for confirmations, retry failures, or log completions.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import spraay_request
from ._config import spraay_provider
from .batch_payment import ChainNetwork


class SpraayTransactionStatusBlock(Block):
    """Check the status of a Spraay transaction.

    Query the on-chain confirmation status of a previously submitted transaction.
    Use this in workflows to wait for confirmations before proceeding, retry
    failed transactions, or log completed payments.
    """

    class Input(BlockSchema):
        """Input schema for the transaction status block."""

        credentials: CredentialsMetaInput = spraay_provider.credentials_field(
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
        """Output schema for the transaction status block."""

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
        """Initialize the SpraayTransactionStatusBlock with metadata and test fixtures."""
        super().__init__(
            id="0617a4fb-c6b7-4569-8330-8912c37086c5",
            description=(
                "Check the confirmation status of a Spraay transaction. "
                "Returns confirmations, block number, and gas used. "
                "Powered by Spraay's x402 payment gateway."
            ),
            categories={BlockCategory.BASIC},
            input_schema=SpraayTransactionStatusBlock.Input,
            output_schema=SpraayTransactionStatusBlock.Output,
            test_input={
                "credentials": {
                    "provider": "spraay",
                    "id": "test-cred-id",
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
        """Fetch the confirmation status of a transaction via the Spraay gateway.

        Args:
            api_key: Spraay API key for authentication.
            chain: Blockchain network the transaction was submitted on.
            transaction_hash: On-chain transaction hash to look up.

        Returns:
            Dictionary containing status, confirmations, block_number,
            and gas_used.

        Raises:
            SpraayAPIError: If the gateway returns an error.
        """
        return spraay_request(
            method="GET",
            endpoint="/v1/transaction/status",
            api_key=api_key,
            params={
                "chain": chain,
                "txHash": transaction_hash,
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the transaction status check block.

        Args:
            input_data: Validated input containing chain and transaction hash.
            credentials: API key credentials injected by the framework.
            **kwargs: Additional keyword arguments passed by the executor.

        Yields:
            Output fields: status, confirmations, block_number, gas_used,
            or error.
        """
        api_key = credentials.api_key.get_secret_value()

        result = self.fetch_status(
            api_key=api_key,
            chain=input_data.chain.value,
            transaction_hash=input_data.transaction_hash,
        )

        yield "status", result.get("status", "unknown")
        yield "confirmations", result.get("confirmations", 0)
        yield "block_number", result.get("block_number", 0)
        yield "gas_used", result.get("gas_used", "0")
