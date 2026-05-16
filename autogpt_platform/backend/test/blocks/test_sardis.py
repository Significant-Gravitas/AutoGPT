from typing import Any, cast

import pytest
from pydantic import ValidationError

from backend.blocks.sardis._api import SardisClient, get_client
from backend.blocks.sardis._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    SardisCredentialsInput,
)
from backend.blocks.sardis.balance import SardisBalanceBlock
from backend.blocks.sardis.payment import SardisPayBlock
from backend.blocks.sardis.policy import SardisPolicyCheckBlock


class FakeResponse:
    def __init__(
        self,
        *,
        ok: bool,
        status: int,
        payload: Any = None,
        body: str = "",
        json_error: Exception | None = None,
    ):
        self.ok = ok
        self.status = status
        self._payload = payload
        self._body = body
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload

    def text(self) -> str:
        return self._body


def credentials_input() -> SardisCredentialsInput:
    return cast(SardisCredentialsInput, TEST_CREDENTIALS_INPUT)


def test_normalize_response_does_not_return_non_json_body():
    response = FakeResponse(
        ok=False,
        status=502,
        body="secret upstream body",
        json_error=ValueError("not json"),
    )

    result = SardisClient._normalize_response(
        cast(Any, response),
        default_error="Sardis payment request failed",
    )

    assert result == {
        "error": "Sardis payment request failed: HTTP 502",
        "status": 502,
    }
    assert "secret upstream body" not in result["error"]


@pytest.mark.asyncio
async def test_get_client_returns_fresh_client_instances():
    first = await get_client(TEST_CREDENTIALS)
    second = await get_client(TEST_CREDENTIALS)

    assert isinstance(first, SardisClient)
    assert isinstance(second, SardisClient)
    assert first is not second


@pytest.mark.asyncio
async def test_pay_block_yields_approved_result_and_forwards_node_exec_id(
    monkeypatch,
):
    block = SardisPayBlock()
    input_data = block.Input(
        wallet_id="wal_test123",
        destination="0x1234567890abcdef1234567890abcdef12345678",
        amount="10.00",
        token="USDC",
        chain="base",
        purpose="Invoice 123",
        credentials=credentials_input(),
    )
    observed: dict[str, str] = {}

    async def fake_send_payment(
        client,
        wallet_id,
        destination,
        amount,
        token,
        chain,
        purpose,
        idempotency_key="",
    ):
        observed.update(
            {
                "wallet_id": wallet_id,
                "destination": destination,
                "amount": amount,
                "token": token,
                "chain": chain,
                "purpose": purpose,
                "idempotency_key": idempotency_key,
            }
        )
        return {
            "success": True,
            "tx_id": "tx_123",
            "message": "Payment approved",
            "amount": "10.00",
        }

    monkeypatch.setattr(SardisPayBlock, "send_payment", staticmethod(fake_send_payment))

    results = [
        output
        async for output in block.run(
            input_data=input_data,
            credentials=TEST_CREDENTIALS,
            node_exec_id="node-exec-1",
        )
    ]

    assert results == [
        ("status", "APPROVED"),
        ("tx_id", "tx_123"),
        ("amount", "10.00"),
        ("message", "Payment approved"),
    ]
    assert observed == {
        "wallet_id": "wal_test123",
        "destination": "0x1234567890abcdef1234567890abcdef12345678",
        "amount": "10.00",
        "token": "USDC",
        "chain": "base",
        "purpose": "Invoice 123",
        "idempotency_key": "node-exec-1",
    }


@pytest.mark.asyncio
async def test_pay_block_yields_error_result(monkeypatch):
    block = SardisPayBlock()
    input_data = block.Input(
        wallet_id="wal_test123",
        destination="merchant_123",
        amount="10.00",
        credentials=credentials_input(),
    )

    async def fake_send_payment(*args, **kwargs):
        return {"error": "insufficient funds"}

    monkeypatch.setattr(SardisPayBlock, "send_payment", staticmethod(fake_send_payment))

    results = [
        output
        async for output in block.run(
            input_data=input_data,
            credentials=TEST_CREDENTIALS,
        )
    ]

    assert results == [
        ("status", "ERROR"),
        ("message", "insufficient funds"),
    ]


@pytest.mark.asyncio
async def test_pay_block_yields_blocked_result(monkeypatch):
    block = SardisPayBlock()
    input_data = block.Input(
        wallet_id="wal_test123",
        destination="merchant_123",
        amount="10.00",
        credentials=credentials_input(),
    )

    async def fake_send_payment(*args, **kwargs):
        return {"message": "Exceeds daily spending limit"}

    monkeypatch.setattr(SardisPayBlock, "send_payment", staticmethod(fake_send_payment))

    results = [
        output
        async for output in block.run(
            input_data=input_data,
            credentials=TEST_CREDENTIALS,
        )
    ]

    assert results == [
        ("status", "BLOCKED"),
        ("message", "Exceeds daily spending limit"),
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wallet_id", "wal_" + "a" * 129),
        ("destination", "bad destination with spaces"),
        ("amount", "1000000.01"),
        ("purpose", "x" * 501),
    ],
)
def test_pay_block_rejects_unbounded_or_invalid_inputs(field, value):
    payload = {
        "wallet_id": "wal_test123",
        "destination": "merchant_123",
        "amount": "10.00",
        "purpose": "Invoice 123",
        "credentials": credentials_input(),
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        SardisPayBlock.Input(**payload)


@pytest.mark.asyncio
async def test_balance_block_yields_error_result(monkeypatch):
    block = SardisBalanceBlock()
    input_data = block.Input(
        wallet_id="wal_test123",
        credentials=credentials_input(),
    )

    async def fake_get_balance(*args, **kwargs):
        return {"error": "wallet not found"}

    monkeypatch.setattr(
        SardisBalanceBlock, "get_balance", staticmethod(fake_get_balance)
    )

    results = [
        output
        async for output in block.run(
            input_data=input_data,
            credentials=TEST_CREDENTIALS,
        )
    ]

    assert results == [("error", "wallet not found")]


@pytest.mark.asyncio
async def test_policy_block_yields_policy_decision(monkeypatch):
    block = SardisPolicyCheckBlock()
    input_data = block.Input(
        wallet_id="wal_test123",
        destination="merchant_123",
        amount="10.00",
        credentials=credentials_input(),
    )

    async def fake_check_policy(*args, **kwargs):
        return {
            "allowed": False,
            "reason": "Merchant not approved",
            "remaining_limit": "500.00",
        }

    monkeypatch.setattr(
        SardisPolicyCheckBlock,
        "check_policy",
        staticmethod(fake_check_policy),
    )

    results = [
        output
        async for output in block.run(
            input_data=input_data,
            credentials=TEST_CREDENTIALS,
        )
    ]

    assert results == [
        ("allowed", False),
        ("reason", "Merchant not approved"),
        ("remaining_limit", "500.00"),
    ]
