"""Nory x402 Payment Blocks for AutoGPT.

Blocks for AI agents to make payments using the x402 HTTP protocol.
Supports Solana and 7 EVM chains with sub-400ms settlement.
"""

from enum import Enum
from urllib.parse import quote

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
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

    class Input(BlockSchema):
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
            secret=True,
        )

    class Output(BlockSchema):
        requirements: dict = SchemaField(
            description="Payment requirements including amount, networks, and wallet address"
        )
        error: str = SchemaField(
            description="Error message if request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="2bd9a224-4bd9-4280-bd17-bbe6c970bc9a",
            description="Get x402 payment requirements for a resource. Returns amount, supported networks, and wallet address.",
            categories={BlockCategory.DATA},
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

    class Input(BlockSchema):
        payload: str = SchemaField(
            description="Base64-encoded payment payload containing signed transaction",
        )
        api_key: str | None = SchemaField(
            description="Nory API key (optional for public endpoints)",
            default=None,
            secret=True,
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="Verification result including validity and payer info"
        )
        error: str = SchemaField(
            description="Error message if verification failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="a8160ae9-876c-45b1-a23c-0c89608dcb01",
            description="Verify a signed payment transaction before submitting to blockchain.",
            categories={BlockCategory.DATA},
            input_schema=NoryVerifyPaymentBlock.Input,
            output_schema=NoryVerifyPaymentBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            headers = {}
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

    class Input(BlockSchema):
        payload: str = SchemaField(
            description="Base64-encoded payment payload",
        )
        api_key: str | None = SchemaField(
            description="Nory API key (optional for public endpoints)",
            default=None,
            secret=True,
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
            description="Settlement result including transaction ID"
        )
        error: str = SchemaField(
            description="Error message if settlement failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="787dad1d-c64c-4996-89b6-8390b73b17f8",
            description="Submit a verified payment to the blockchain for settlement (~400ms).",
            categories={BlockCategory.DATA},
            input_schema=NorySettlePaymentBlock.Input,
            output_schema=NorySettlePaymentBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            headers = {}
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

    class Input(BlockSchema):
        transaction_id: str = SchemaField(
            description="Transaction ID or signature",
        )
        network: NoryNetwork = SchemaField(
            description="Network where the transaction was submitted",
        )
        api_key: str | None = SchemaField(
            description="Nory API key (optional for public endpoints)",
            default=None,
            secret=True,
        )

    class Output(BlockSchema):
        transaction: dict = SchemaField(
            description="Transaction details including status and confirmations"
        )
        error: str = SchemaField(
            description="Error message if lookup failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="1334cf87-2b48-4e70-87d2-6e4807b78e02",
            description="Look up the status and details of a transaction.",
            categories={BlockCategory.DATA},
            input_schema=NoryTransactionLookupBlock.Input,
            output_schema=NoryTransactionLookupBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            headers = {}
            if input_data.api_key:
                headers["Authorization"] = f"Bearer {input_data.api_key}"

            encoded_tx_id = quote(input_data.transaction_id, safe="")
            response = await Requests().get(
                f"{NORY_API_BASE}/api/x402/transactions/{encoded_tx_id}",
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

    class Input(BlockSchema):
        pass

    class Output(BlockSchema):
        health: dict = SchemaField(description="Health status and supported networks")
        error: str = SchemaField(
            description="Error message if health check failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="3b2595e1-5950-4094-b8b1-733dddd3b16c",
            description="Check health status of Nory x402 payment service.",
            categories={BlockCategory.DATA},
            input_schema=NoryHealthCheckBlock.Input,
            output_schema=NoryHealthCheckBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            response = await Requests().get(f"{NORY_API_BASE}/api/x402/health")
            yield "health", response.json()
        except Exception as e:
            yield "error", str(e)
