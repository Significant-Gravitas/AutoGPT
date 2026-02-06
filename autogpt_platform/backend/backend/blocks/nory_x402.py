"""Nory x402 Payment Blocks for AutoGPT.

Blocks for AI agents to make payments using the x402 HTTP protocol.
Supports Solana and 7 EVM chains with sub-400ms settlement.
"""

from typing import Literal
from enum import Enum

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchemaInput
from backend.data.model import SchemaField
from backend.util.request import Requests


NORY_API_BASE = "https://noryx402.com"


class NoryNetwork(str, Enum):
    """Supported blockchain networks."""

    SOLANA_MAINNET = "solana-mainnet"
    SOLANA_DEVNET = "solana-devnet"
    BASE_MAINNET = "base-mainnet"
    POLYGON_MAINNET = "polygon-mainnet"
    ARBITRUM_MAINNET = "arbitrum-mainnet"
    OPTIMISM_MAINNET = "optimism-mainnet"
    AVALANCHE_MAINNET = "avalanche-mainnet"
    SEI_MAINNET = "sei-mainnet"
    IOTEX_MAINNET = "iotex-mainnet"


class NoryGetPaymentRequirementsBlock(Block):
    """
    Get x402 payment requirements for accessing a paid resource.

    Use this when you encounter an HTTP 402 Payment Required response
    and need to know how much to pay and where to send payment.
    """

    class Input(BlockSchemaInput):
        resource: str = SchemaField(
            description="The resource path requiring payment (e.g., /api/premium/data)",
            placeholder="/api/premium/data",
        )
        amount: str = SchemaField(
            description="Amount in human-readable format (e.g., '0.10' for $0.10 USDC)",
            placeholder="0.10",
        )
        network: NoryNetwork | None = SchemaField(
            description="Preferred blockchain network",
            default=None,
        )
        api_key: str | None = SchemaField(
            description="Nory API key (optional for public endpoints)",
            default=None,
        )

    class Output(BlockSchemaInput):
        requirements: dict = SchemaField(
            description="Payment requirements including amount, networks, and wallet address"
        )
        error: str = SchemaField(
            description="Error message if request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description="Get x402 payment requirements for a resource. Returns amount, supported networks, and wallet address.",
            categories={BlockCategory.FINANCE},
            input_schema=NoryGetPaymentRequirementsBlock.Input,
            output_schema=NoryGetPaymentRequirementsBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            params = {
                "resource": input_data.resource,
                "amount": input_data.amount,
            }
            if input_data.network:
                params["network"] = input_data.network.value

            headers = {}
            if input_data.api_key:
                headers["Authorization"] = f"Bearer {input_data.api_key}"

            response = await Requests().get(
                f"{NORY_API_BASE}/api/x402/requirements",
                params=params,
                headers=headers,
            )
            yield "requirements", response.json()
        except Exception as e:
            yield "error", str(e)


class NoryVerifyPaymentBlock(Block):
    """
    Verify a signed payment transaction before settlement.

    Use this to validate that a payment transaction is correct
    before submitting it to the blockchain.
    """

    class Input(BlockSchemaInput):
        payload: str = SchemaField(
            description="Base64-encoded payment payload containing signed transaction",
        )
        api_key: str | None = SchemaField(
            description="Nory API key (optional for public endpoints)",
            default=None,
        )

    class Output(BlockSchemaInput):
        result: dict = SchemaField(
            description="Verification result including validity and payer info"
        )
        error: str = SchemaField(
            description="Error message if verification failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            description="Verify a signed payment transaction before submitting to blockchain.",
            categories={BlockCategory.FINANCE},
            input_schema=NoryVerifyPaymentBlock.Input,
            output_schema=NoryVerifyPaymentBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            headers = {"Content-Type": "application/json"}
            if input_data.api_key:
                headers["Authorization"] = f"Bearer {input_data.api_key}"

            response = await Requests().post(
                f"{NORY_API_BASE}/api/x402/verify",
                json={"payload": input_data.payload},
                headers=headers,
            )
            yield "result", response.json()
        except Exception as e:
            yield "error", str(e)


class NorySettlePaymentBlock(Block):
    """
    Settle a payment on-chain.

    Use this to submit a verified payment transaction to the blockchain.
    Settlement typically completes in under 400ms.
    """

    class Input(BlockSchemaInput):
        payload: str = SchemaField(
            description="Base64-encoded payment payload",
        )
        api_key: str | None = SchemaField(
            description="Nory API key (optional for public endpoints)",
            default=None,
        )

    class Output(BlockSchemaInput):
        result: dict = SchemaField(
            description="Settlement result including transaction ID"
        )
        error: str = SchemaField(
            description="Error message if settlement failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-123456789012",
            description="Submit a verified payment to the blockchain for settlement (~400ms).",
            categories={BlockCategory.FINANCE},
            input_schema=NorySettlePaymentBlock.Input,
            output_schema=NorySettlePaymentBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            headers = {"Content-Type": "application/json"}
            if input_data.api_key:
                headers["Authorization"] = f"Bearer {input_data.api_key}"

            response = await Requests().post(
                f"{NORY_API_BASE}/api/x402/settle",
                json={"payload": input_data.payload},
                headers=headers,
            )
            yield "result", response.json()
        except Exception as e:
            yield "error", str(e)


class NoryTransactionLookupBlock(Block):
    """
    Look up transaction status.

    Use this to check the status of a previously submitted payment.
    """

    class Input(BlockSchemaInput):
        transaction_id: str = SchemaField(
            description="Transaction ID or signature",
        )
        network: NoryNetwork = SchemaField(
            description="Network where the transaction was submitted",
        )
        api_key: str | None = SchemaField(
            description="Nory API key (optional for public endpoints)",
            default=None,
        )

    class Output(BlockSchemaInput):
        transaction: dict = SchemaField(
            description="Transaction details including status and confirmations"
        )
        error: str = SchemaField(
            description="Error message if lookup failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-def0-234567890123",
            description="Look up the status and details of a transaction.",
            categories={BlockCategory.FINANCE},
            input_schema=NoryTransactionLookupBlock.Input,
            output_schema=NoryTransactionLookupBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            headers = {}
            if input_data.api_key:
                headers["Authorization"] = f"Bearer {input_data.api_key}"

            response = await Requests().get(
                f"{NORY_API_BASE}/api/x402/transactions/{input_data.transaction_id}",
                params={"network": input_data.network.value},
                headers=headers,
            )
            yield "transaction", response.json()
        except Exception as e:
            yield "error", str(e)


class NoryHealthCheckBlock(Block):
    """
    Check Nory service health.

    Use this to verify the payment service is operational
    and see supported networks.
    """

    class Input(BlockSchemaInput):
        pass

    class Output(BlockSchemaInput):
        health: dict = SchemaField(
            description="Health status and supported networks"
        )
        error: str = SchemaField(
            description="Error message if health check failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-ef01-345678901234",
            description="Check health status of Nory x402 payment service.",
            categories={BlockCategory.FINANCE},
            input_schema=NoryHealthCheckBlock.Input,
            output_schema=NoryHealthCheckBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            response = await Requests().get(f"{NORY_API_BASE}/api/x402/health")
            yield "health", response.json()
        except Exception as e:
            yield "error", str(e)
