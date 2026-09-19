from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.request import Requests

API_BASE = "https://shieldz.cash"


class ShieldzCreatePaymentLinkBlock(Block):
    """Create a one-time, non-custodial crypto payment link with Shieldz.

    Keyless: no account and no API key. Just provide a destination wallet
    address and an amount. Funds settle directly to the address; Shieldz never
    holds them. Every link is OFAC sanctions-screened and rate-limited.
    """

    class Input(BlockSchemaInput):
        address: str = SchemaField(
            description="Destination wallet (0x EVM address). Funds settle here; Shieldz never holds them."
        )
        amount_usd: float = SchemaField(description="Amount in USD, e.g. 49 for $49.00")
        chain: str = SchemaField(
            default="base",
            description="Settlement chain: base, arbitrum, optimism, polygon, or ethereum",
        )
        asset: str = SchemaField(default="USDC", description="Stablecoin: USDC or USDT")
        memo: str = SchemaField(
            default="", description="Description shown on the checkout"
        )
        email: str = SchemaField(
            default="", description="Optional; lets the owner claim a dashboard later"
        )

    class Output(BlockSchemaOutput):
        pay_url: str = SchemaField(
            description="Hosted checkout URL to send the payer to"
        )
        manage_url: str = SchemaField(
            description="Capability URL to read status later (keep private)"
        )
        embed: str = SchemaField(description="Embeddable <script> button snippet")
        error: str = SchemaField(
            description="Error message if the link could not be created"
        )

    def __init__(self):
        super().__init__(
            id="96877aef-b0eb-4e63-a381-d6ecd36aa227",
            input_schema=ShieldzCreatePaymentLinkBlock.Input,
            output_schema=ShieldzCreatePaymentLinkBlock.Output,
            description="Create a keyless, non-custodial crypto payment link with Shieldz (no API key).",
            categories={BlockCategory.OUTPUT},
            test_input={
                "address": "0x66f6794168758d2e146c898E22ea2c4Ca5f30000",
                "amount_usd": 49,
                "chain": "base",
                "asset": "USDC",
                "memo": "Pro plan",
            },
            test_output=[
                ("pay_url", "https://shieldz.cash/pay/8kQ2x"),
                ("manage_url", "https://shieldz.cash/a/cap_8kQ2x"),
                (
                    "embed",
                    '<script src="https://shieldz.cash/embed/button.js" data-href="https://shieldz.cash/pay/8kQ2x"></script>',
                ),
            ],
            test_mock={
                "create_link": lambda *args, **kwargs: {
                    "pay_url": "https://shieldz.cash/pay/8kQ2x",
                    "manage_url": "https://shieldz.cash/a/cap_8kQ2x",
                    "embed": '<script src="https://shieldz.cash/embed/button.js" data-href="https://shieldz.cash/pay/8kQ2x"></script>',
                }
            },
        )

    async def create_link(self, body: dict) -> dict:
        response = await Requests().post(f"{API_BASE}/api/v1/links", json=body)
        return response.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        body: dict = {
            "settlement": {
                "chain": input_data.chain,
                "asset": input_data.asset,
                "address": input_data.address,
            },
            "amount_usd_cents": round(input_data.amount_usd * 100),
        }
        if input_data.memo:
            body["memo"] = input_data.memo
        if input_data.email:
            body["email"] = input_data.email

        data = await self.create_link(body)
        if "pay_url" not in data:
            yield "error", (data.get("error") or {}).get(
                "message", "could not create payment link"
            )
            return
        yield "pay_url", data["pay_url"]
        yield "manage_url", data["manage_url"]
        yield "embed", data["embed"]


class ShieldzCreateTipJarBlock(Block):
    """Create a reusable, non-custodial "pay what you want" tip jar with Shieldz.

    Keyless: no account and no API key. The payer chooses the amount. Idempotent
    per wallet address. Funds settle directly to the address; Shieldz never holds
    them.
    """

    class Input(BlockSchemaInput):
        address: str = SchemaField(
            description="Destination wallet (0x EVM address). Funds settle here; Shieldz never holds them."
        )
        chain: str = SchemaField(
            default="base",
            description="Settlement chain: base, arbitrum, optimism, polygon, or ethereum",
        )
        asset: str = SchemaField(default="USDC", description="Stablecoin: USDC or USDT")
        title: str = SchemaField(
            default="", description="Heading shown on the tip page"
        )
        suggested_amounts_usd: list[float] = SchemaField(
            default_factory=list,
            description="Preset amount buttons in USD, e.g. [3, 5, 10]",
        )
        email: str = SchemaField(
            default="", description="Optional; lets the owner claim a dashboard later"
        )

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="Reusable tip-jar URL")
        manage_url: str = SchemaField(
            description="Capability URL to read status later (keep private)"
        )
        error: str = SchemaField(
            description="Error message if the tip jar could not be created"
        )

    def __init__(self):
        super().__init__(
            id="8ba7e938-b0ea-4942-aee9-2d5f5be70db7",
            input_schema=ShieldzCreateTipJarBlock.Input,
            output_schema=ShieldzCreateTipJarBlock.Output,
            description="Create a keyless, non-custodial 'pay what you want' tip jar with Shieldz.",
            categories={BlockCategory.OUTPUT},
            test_input={
                "address": "0x66f6794168758d2e146c898E22ea2c4Ca5f30000",
                "chain": "base",
                "asset": "USDC",
                "title": "Buy me a coffee",
                "suggested_amounts_usd": [3, 5, 10],
            },
            test_output=[
                ("url", "https://shieldz.cash/tip/tip-10851092bec481a4"),
                ("manage_url", "https://shieldz.cash/a/cap_8kQ2x"),
            ],
            test_mock={
                "create_tip_jar": lambda *args, **kwargs: {
                    "url": "https://shieldz.cash/tip/tip-10851092bec481a4",
                    "manage_url": "https://shieldz.cash/a/cap_8kQ2x",
                }
            },
        )

    async def create_tip_jar(self, body: dict) -> dict:
        response = await Requests().post(f"{API_BASE}/api/v1/tip-jars", json=body)
        return response.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        body: dict = {
            "settlement": {
                "chain": input_data.chain,
                "asset": input_data.asset,
                "address": input_data.address,
            },
        }
        if input_data.title:
            body["title"] = input_data.title
        if input_data.suggested_amounts_usd:
            body["suggested_amounts_usd_cents"] = [
                round(a * 100) for a in input_data.suggested_amounts_usd
            ]
        if input_data.email:
            body["email"] = input_data.email

        data = await self.create_tip_jar(body)
        if "url" not in data:
            yield "error", (data.get("error") or {}).get(
                "message", "could not create tip jar"
            )
            return
        yield "url", data["url"]
        yield "manage_url", data["manage_url"]
