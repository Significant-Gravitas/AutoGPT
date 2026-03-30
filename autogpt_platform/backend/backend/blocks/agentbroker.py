"""AgentBroker blocks for crypto trading via Jupiter DEX / Solana.

AgentBroker (https://agentbroker.polsia.app) is a no-KYC crypto exchange
designed for AI agents. Supports market/limit orders, real-time prices,
OHLCV candles, and account balance queries.

API docs: https://agentbroker.polsia.app/api
"""

from enum import Enum
from typing import Any

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.request import Requests

AGENTBROKER_BASE_URL = "https://agentbroker.polsia.app"


class AgentBrokerGetPriceBlock(Block):
    """Get the current price of a crypto token pair from AgentBroker (no API key required)."""

    class Input(BlockSchemaInput):
        pair: str = SchemaField(
            description="Token pair to query (e.g. SOL-USDC, BTC-USDC, BONK-USDC)",
            placeholder="SOL-USDC",
            default="SOL-USDC",
        )

    class Output(BlockSchemaOutput):
        pair: str = SchemaField(description="Token pair queried")
        price_usdc: float = SchemaField(description="Current price in USDC")
        change_24h_pct: float = SchemaField(description="24-hour price change (%)")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-4a7b-8c9d-e0f1a2b3c4d5",
            description=(
                "Fetch the real-time price of any crypto token pair from AgentBroker "
                "(Jupiter DEX / Solana). No API key required. "
                "Supports 1000+ pairs including SOL, BTC, ETH, and meme coins."
            ),
            categories={BlockCategory.DATA},
            input_schema=AgentBrokerGetPriceBlock.Input,
            output_schema=AgentBrokerGetPriceBlock.Output,
            test_input={"pair": "SOL-USDC"},
            test_output=[
                ("pair", "SOL-USDC"),
                ("price_usdc", 170.42),
                ("change_24h_pct", 1.5),
            ],
            test_mock={
                "fetch_price": lambda *args, **kwargs: {
                    "pair": "SOL-USDC",
                    "price_usdc": 170.42,
                    "change_24h_pct": 1.5,
                }
            },
        )

    @staticmethod
    async def fetch_price(pair: str) -> dict[str, Any]:
        api = Requests()
        resp = await api.get(f"{AGENTBROKER_BASE_URL}/v1/prices/{pair}")
        return resp.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            data = await self.fetch_price(input_data.pair)
            yield "pair", data.get("pair", input_data.pair)
            yield "price_usdc", float(data.get("price_usdc", 0))
            yield "change_24h_pct", float(data.get("change_24h_pct", 0))
        except Exception as e:
            yield "error", str(e)


class CandleInterval(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"


class AgentBrokerGetCandlesBlock(Block):
    """Get OHLCV candlestick data for technical analysis from AgentBroker (no API key required)."""

    class Input(BlockSchemaInput):
        pair: str = SchemaField(
            description="Token pair (e.g. SOL-USDC, BTC-USDC)",
            placeholder="SOL-USDC",
            default="SOL-USDC",
        )
        interval: CandleInterval = SchemaField(
            description="Candle interval (1m, 5m, 15m, 1h, 4h, 1d)",
            default=CandleInterval.ONE_HOUR,
        )
        limit: int = SchemaField(
            description="Number of candles to retrieve (max 1000)",
            default=100,
        )

    class Output(BlockSchemaOutput):
        candles: list = SchemaField(
            description="List of OHLCV candles with timestamp, open, high, low, close, volume"
        )
        pair: str = SchemaField(description="Token pair")
        interval: str = SchemaField(description="Candle interval used")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-4b8c-9d0e-f1a2b3c4d5e6",
            description=(
                "Fetch OHLCV candlestick data from AgentBroker for technical analysis. "
                "Supports intervals from 1 minute to 1 day. No API key required."
            ),
            categories={BlockCategory.DATA},
            input_schema=AgentBrokerGetCandlesBlock.Input,
            output_schema=AgentBrokerGetCandlesBlock.Output,
            test_input={"pair": "SOL-USDC", "interval": "1h", "limit": 5},
            test_output=[
                ("pair", "SOL-USDC"),
                ("interval", "1h"),
                ("candles", []),
            ],
            test_mock={
                "fetch_candles": lambda *args, **kwargs: {
                    "pair": "SOL-USDC",
                    "interval": "1h",
                    "candles": [],
                }
            },
        )

    @staticmethod
    async def fetch_candles(pair: str, interval: str, limit: int) -> dict[str, Any]:
        api = Requests()
        resp = await api.get(
            f"{AGENTBROKER_BASE_URL}/v1/candles",
            params={"pair": pair, "interval": interval, "limit": limit},
        )
        return resp.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            data = await self.fetch_candles(
                input_data.pair,
                input_data.interval.value,
                input_data.limit,
            )
            yield "pair", data.get("pair", input_data.pair)
            yield "interval", data.get("interval", input_data.interval.value)
            yield "candles", data.get("candles", [])
        except Exception as e:
            yield "error", str(e)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class AgentBrokerPlaceOrderBlock(Block):
    """Place a buy or sell order on AgentBroker (requires API key)."""

    class Input(BlockSchemaInput):
        api_key: str = SchemaField(
            description="Your AgentBroker API key (get one at https://agentbroker.polsia.app)",
            placeholder="ab_live_...",
        )
        pair: str = SchemaField(
            description="Token pair to trade (e.g. SOL-USDC, BONK-USDC)",
            placeholder="SOL-USDC",
        )
        side: OrderSide = SchemaField(
            description="Order side: buy or sell",
            default=OrderSide.BUY,
        )
        order_type: OrderType = SchemaField(
            description="Order type: market (immediate) or limit (at specific price)",
            default=OrderType.MARKET,
        )
        quantity: float = SchemaField(
            description="Amount in USDC to spend (buy) or tokens to sell",
        )
        price: float = SchemaField(
            description="Limit price in USDC per token (required for limit orders)",
            default=0.0,
        )

    class Output(BlockSchemaOutput):
        order_id: str = SchemaField(description="Unique order ID")
        status: str = SchemaField(description="Order status (filled, open, cancelled)")
        filled_quantity: float = SchemaField(description="Amount filled")
        fees: float = SchemaField(description="Trading fees charged in USDC")
        balance_usdc: float = SchemaField(description="Remaining USDC balance after order")
        error: str = SchemaField(description="Error message if the order failed")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-4c9d-0e1f-a2b3c4d5e6f7",
            description=(
                "Place a market or limit order on AgentBroker (Jupiter DEX / Solana). "
                "No KYC required. Supports 1000+ token pairs including SOL, meme coins, "
                "and any Solana token. Requires an AgentBroker API key."
            ),
            categories={BlockCategory.OUTPUT},
            input_schema=AgentBrokerPlaceOrderBlock.Input,
            output_schema=AgentBrokerPlaceOrderBlock.Output,
            test_input={
                "api_key": "test-api-key",
                "pair": "SOL-USDC",
                "side": "buy",
                "order_type": "market",
                "quantity": 10.0,
                "price": 0.0,
            },
            test_output=[
                ("order_id", "ord_test123"),
                ("status", "filled"),
                ("filled_quantity", 10.0),
                ("fees", 0.03),
                ("balance_usdc", 990.0),
            ],
            test_mock={
                "place_order": lambda *args, **kwargs: {
                    "order_id": "ord_test123",
                    "status": "filled",
                    "filled_quantity": 10.0,
                    "fees": 0.03,
                    "balance_usdc": 990.0,
                }
            },
        )

    @staticmethod
    async def place_order(
        api_key: str,
        pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float,
    ) -> dict[str, Any]:
        api = Requests()
        body: dict[str, Any] = {
            "pair": pair,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        if order_type == "limit" and price > 0:
            body["price"] = price
        resp = await api.post(
            f"{AGENTBROKER_BASE_URL}/v1/orders",
            json=body,
            headers={"X-API-Key": api_key},
        )
        return resp.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            data = await self.place_order(
                api_key=input_data.api_key,
                pair=input_data.pair,
                side=input_data.side.value,
                order_type=input_data.order_type.value,
                quantity=input_data.quantity,
                price=input_data.price,
            )
            yield "order_id", data.get("order_id", "")
            yield "status", data.get("status", "")
            yield "filled_quantity", float(data.get("filled_quantity", 0))
            yield "fees", float(data.get("fees", 0))
            yield "balance_usdc", float(data.get("balance_usdc", 0))
        except Exception as e:
            yield "error", str(e)


class AgentBrokerGetBalanceBlock(Block):
    """Get account balance and portfolio summary from AgentBroker (requires API key)."""

    class Input(BlockSchemaInput):
        api_key: str = SchemaField(
            description="Your AgentBroker API key (get one at https://agentbroker.polsia.app)",
            placeholder="ab_live_...",
        )

    class Output(BlockSchemaOutput):
        balance_usdc: float = SchemaField(description="Available USDC balance")
        total_portfolio_value: float = SchemaField(
            description="Total portfolio value in USDC (cash + holdings)"
        )
        trade_count: int = SchemaField(description="Total number of trades executed")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-4d0e-1f2a-b3c4d5e6f7a8",
            description=(
                "Retrieve account balance and portfolio summary from AgentBroker. "
                "Returns USDC balance, total portfolio value, and trade statistics. "
                "Requires an AgentBroker API key."
            ),
            categories={BlockCategory.DATA},
            input_schema=AgentBrokerGetBalanceBlock.Input,
            output_schema=AgentBrokerGetBalanceBlock.Output,
            test_input={"api_key": "test-api-key"},
            test_output=[
                ("balance_usdc", 1000.0),
                ("total_portfolio_value", 1250.0),
                ("trade_count", 42),
            ],
            test_mock={
                "fetch_balance": lambda *args, **kwargs: {
                    "balance_usdc": 1000.0,
                    "total_portfolio_value": 1250.0,
                    "trade_count": 42,
                }
            },
        )

    @staticmethod
    async def fetch_balance(api_key: str) -> dict[str, Any]:
        api = Requests()
        resp = await api.get(
            f"{AGENTBROKER_BASE_URL}/v1/account",
            headers={"X-API-Key": api_key},
        )
        return resp.json()

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            data = await self.fetch_balance(input_data.api_key)
            yield "balance_usdc", float(data.get("balance_usdc", 0))
            yield "total_portfolio_value", float(data.get("total_portfolio_value", 0))
            yield "trade_count", int(data.get("trade_count", 0))
        except Exception as e:
            yield "error", str(e)
