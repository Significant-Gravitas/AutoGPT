import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from eth_account import Account
from eth_account.messages import encode_typed_data

from backend.blocks.spraay._api import (
    SpraayAPIError,
    _typed_data,
    build_payment_signature,
    spraay_request,
)

PAYER = Account.from_key("0x" + "11" * 32)

BASE_OPTION = {
    "scheme": "exact",
    "network": "eip155:8453",
    "amount": "5000",
    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "payTo": "0xAd62f03C7514bb8c51f1eA70C2b75C37404695c8",
    "maxTimeoutSeconds": 300,
    "extra": {"name": "USD Coin", "version": "2"},
}
ROBINHOOD_OPTION = {
    **BASE_OPTION,
    "network": "eip155:4663",
    "asset": "0x5fc5360D0400a0Fd4f2af552ADD042D716F1d168",
    "payTo": "0xdAA0fb4fb470AA8fb53A0c301EF9AADC89949F33",
    "extra": {"name": "Global Dollar", "version": "1"},
}
SOLANA_OPTION = {**BASE_OPTION, "network": "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"}


def _payment_required(*accepts: dict) -> dict:
    return {
        "x402Version": 2,
        "resource": {"url": "https://gateway.spraay.app/api/v1/balances"},
        "accepts": list(accepts),
    }


def _decode(header: str) -> dict:
    return json.loads(base64.b64decode(header))


def _response(status: int, body: dict, headers: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.headers = headers or {}
    response.json.return_value = body
    return response


def test_payment_signature_recovers_to_payer():
    payload = _decode(
        build_payment_signature(
            _payment_required(SOLANA_OPTION, BASE_OPTION), PAYER.key.hex()
        )
    )

    assert payload["accepted"] == BASE_OPTION
    authorization = payload["payload"]["authorization"]
    assert authorization["from"] == PAYER.address
    assert authorization["to"] == BASE_OPTION["payTo"]
    assert authorization["value"] == "5000"
    recovered = Account.recover_message(
        encode_typed_data(full_message=_typed_data(BASE_OPTION, authorization)),
        signature=payload["payload"]["signature"],
    )
    assert recovered == PAYER.address


def test_payment_falls_back_to_robinhood_usdg():
    payload = _decode(
        build_payment_signature(
            _payment_required(SOLANA_OPTION, ROBINHOOD_OPTION), PAYER.key.hex()
        )
    )

    assert payload["accepted"]["network"] == "eip155:4663"


def test_payment_rejects_unsupported_networks():
    with pytest.raises(SpraayAPIError, match="No supported x402 payment option"):
        build_payment_signature(_payment_required(SOLANA_OPTION), PAYER.key.hex())


@pytest.mark.asyncio
async def test_request_pays_402_and_retries():
    challenge = base64.b64encode(
        json.dumps(_payment_required(BASE_OPTION)).encode()
    ).decode()
    client = MagicMock()
    client.request = AsyncMock(
        side_effect=[
            _response(402, {}, {"PAYMENT-REQUIRED": challenge}),
            _response(200, {"balances": []}),
        ]
    )

    with patch("backend.blocks.spraay._api.Requests", return_value=client):
        result = await spraay_request(
            "GET",
            "/api/v1/balances",
            private_key=PAYER.key.hex(),
            params={"address": PAYER.address},
        )

    assert result == {"balances": []}
    retry_headers = client.request.call_args_list[1].kwargs["headers"]
    assert _decode(retry_headers["PAYMENT-SIGNATURE"])["accepted"] == BASE_OPTION


@pytest.mark.asyncio
async def test_request_without_key_fails_on_402():
    client = MagicMock()
    client.request = AsyncMock(return_value=_response(402, {}))

    with patch("backend.blocks.spraay._api.Requests", return_value=client):
        with pytest.raises(SpraayAPIError, match="requires x402 payment"):
            await spraay_request("GET", "/api/v1/balances")


@pytest.mark.asyncio
async def test_request_raises_gateway_error():
    client = MagicMock()
    client.request = AsyncMock(
        return_value=_response(400, {"error": "Unsupported chain: peaq"})
    )

    with patch("backend.blocks.spraay._api.Requests", return_value=client):
        with pytest.raises(SpraayAPIError, match="Unsupported chain: peaq"):
            await spraay_request("GET", "/api/v1/tx/decode")
