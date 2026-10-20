"""Spraay x402 Gateway API client.

Paid gateway endpoints answer an unpaid request with HTTP 402 and a
base64-encoded ``PAYMENT-REQUIRED`` header listing the payment options it
accepts. This client handles that exchange transparently: it signs an EIP-3009
``TransferWithAuthorization`` for the requested USDC/USDG amount with the payer
wallet key, retries with a ``PAYMENT-SIGNATURE`` header, and returns the JSON
body of the paid response. Free endpoints never return 402 and pass straight
through.
"""

import base64
import json
import logging
import secrets
import time
from typing import Any
from urllib.parse import urlencode

from eth_account import Account
from eth_account.messages import encode_typed_data

from backend.sdk import Requests
from backend.util.request import Response

from ._config import SPRAAY_GATEWAY_BASE_URL

logger = logging.getLogger(__name__)

X402_VERSION = 2

# EVM networks the payer can settle x402 fees on, in order of preference:
# USDC on Base, then USDG on Robinhood Chain. Both use EIP-3009.
X402_PAYMENT_NETWORKS = ("eip155:8453", "eip155:4663")

# Allow for clock skew between this host and the facilitator.
VALID_AFTER_SKEW_SECONDS = 600

TRANSFER_WITH_AUTHORIZATION_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ],
}


class SpraayAPIError(Exception):
    """Raised when a Spraay gateway API call fails.

    Attributes:
        status_code: HTTP status code returned by the gateway.
        message: Human-readable error description.
    """

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Spraay API error {status_code}: {message}")


async def spraay_request(
    method: str,
    endpoint: str,
    private_key: str | None = None,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the Spraay gateway, paying any x402 challenge automatically.

    Args:
        method: HTTP method (GET, POST, etc.).
        endpoint: API path including the ``/api/v1`` prefix.
        private_key: Payer wallet private key. Required for paid endpoints.
        json_body: Optional JSON request body.
        params: Optional query parameters.

    Returns:
        Parsed JSON response body.

    Raises:
        SpraayAPIError: On HTTP errors, malformed JSON, or an unpayable 402.
    """
    url = f"{SPRAAY_GATEWAY_BASE_URL}{endpoint}"
    if params:
        url = f"{url}?{urlencode(params)}"
    headers = {"Content-Type": "application/json"}
    client = Requests(raise_for_status=False)

    response = await client.request(method, url, headers=headers, json=json_body)

    if response.status == 402:
        if not private_key:
            raise SpraayAPIError(402, "Endpoint requires x402 payment")
        headers["PAYMENT-SIGNATURE"] = build_payment_signature(
            _read_payment_required(response), private_key
        )
        response = await client.request(method, url, headers=headers, json=json_body)
        _log_settlement(endpoint, response)

    if response.status >= 400:
        raise SpraayAPIError(response.status, _error_message(response))

    return _parse_json(response)


def build_payment_signature(
    payment_required: dict[str, Any],
    private_key: str,
) -> str:
    """Build the base64 ``PAYMENT-SIGNATURE`` header for a 402 challenge.

    Args:
        payment_required: Decoded ``PAYMENT-REQUIRED`` payload from the gateway.
        private_key: Hex private key of the payer wallet.

    Returns:
        Base64-encoded x402 payment payload.
    """
    option = _select_payment_option(payment_required)
    now = int(time.time())
    authorization = {
        "from": Account.from_key(private_key).address,
        "to": option["payTo"],
        "value": str(option["amount"]),
        "validAfter": str(now - VALID_AFTER_SKEW_SECONDS),
        "validBefore": str(now + int(option.get("maxTimeoutSeconds") or 300)),
        "nonce": "0x" + secrets.token_hex(32),
    }
    signed = Account.sign_message(
        encode_typed_data(full_message=_typed_data(option, authorization)),
        private_key,
    )
    return _b64_json_encode(
        {
            "x402Version": payment_required.get("x402Version", X402_VERSION),
            "resource": payment_required.get("resource"),
            "accepted": option,
            "payload": {
                "signature": "0x" + bytes(signed.signature).hex(),
                "authorization": authorization,
            },
        }
    )


def _select_payment_option(payment_required: dict[str, Any]) -> dict[str, Any]:
    """Pick the first accepted x402 option this client can pay."""
    accepts = payment_required.get("accepts") or []
    for network in X402_PAYMENT_NETWORKS:
        for option in accepts:
            if option.get("scheme") == "exact" and option.get("network") == network:
                return option
    offered = ", ".join(str(o.get("network")) for o in accepts) or "none"
    raise SpraayAPIError(
        402, f"No supported x402 payment option offered (offered: {offered})"
    )


def _typed_data(
    option: dict[str, Any], authorization: dict[str, str]
) -> dict[str, Any]:
    """EIP-712 payload for an EIP-3009 TransferWithAuthorization."""
    extra = option.get("extra") or {}
    return {
        "types": TRANSFER_WITH_AUTHORIZATION_TYPES,
        "primaryType": "TransferWithAuthorization",
        "domain": {
            "name": extra.get("name", "USD Coin"),
            "version": extra.get("version", "2"),
            "chainId": int(option["network"].split(":", 1)[1]),
            "verifyingContract": option["asset"],
        },
        "message": {
            "from": authorization["from"],
            "to": authorization["to"],
            "value": int(authorization["value"]),
            "validAfter": int(authorization["validAfter"]),
            "validBefore": int(authorization["validBefore"]),
            "nonce": bytes.fromhex(authorization["nonce"][2:]),
        },
    }


def _read_payment_required(response: Response) -> dict[str, Any]:
    header = response.headers.get("PAYMENT-REQUIRED")
    if header:
        return _b64_json_decode(header)
    return _parse_json(response)


def _log_settlement(endpoint: str, response: Response) -> None:
    settlement = response.headers.get("PAYMENT-RESPONSE")
    if settlement:
        logger.info(
            f"Spraay x402 payment settled for {endpoint}: "
            f"{_b64_json_decode(settlement)}"
        )


def _parse_json(response: Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        raise SpraayAPIError(
            response.status,
            f"Invalid JSON in response: {response.text()[:200]}",
        )


def _error_message(response: Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text()[:500]
    return str(data.get("error") or data.get("message") or data)


def _b64_json_decode(value: str) -> dict[str, Any]:
    return json.loads(base64.b64decode(value))


def _b64_json_encode(value: dict[str, Any]) -> str:
    return base64.b64encode(json.dumps(value).encode()).decode()
