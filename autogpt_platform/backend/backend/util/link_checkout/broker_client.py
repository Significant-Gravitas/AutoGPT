"""The controller's client for a remote checkout broker (mutual TLS)."""

import httpx
from pydantic import BaseModel, ValidationError

from backend.util.link_checkout.broker_protocol import Principal
from backend.util.link_checkout.broker_routing import route_for
from backend.util.link_checkout.refusals import MESSAGES, CheckoutRefused

_OPERATIONS = frozenset(
    {
        "browser",
        "checkout/create",
        "checkout/get",
        "checkout/complete",
        "checkout/status",
        "checkout/reset",
    }
)
# Screenshots are the largest responses, returned base64-encoded.
_MAX_RESPONSE_BYTES = 12_000_000


async def request(operation: str, payload: Principal) -> dict:
    if operation not in _OPERATIONS:
        raise ValueError("Unsupported checkout operation")
    try:
        # The route comes from operator configuration for the authenticated
        # user; nothing a tool argument names can select it.
        route = route_for(payload.user_id)
        context, secret = route.tls_context(), route.credential()
        async with httpx.AsyncClient(
            verify=context, follow_redirects=False, trust_env=False, timeout=120
        ) as client:
            async with client.stream(
                "POST",
                f"{route.url}/v1/{operation}",
                json=payload.model_dump(mode="json", context={"reveal_secrets": True}),
                headers={"Authorization": f"Bearer {secret}"},
            ) as response:
                if response.status_code == 422:
                    await _raise_refusal(response)
                if response.status_code != 200:
                    raise RuntimeError("Checkout broker unavailable")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_RESPONSE_BYTES:
                        raise RuntimeError("Checkout response exceeds its bound")
                return httpx.Response(200, content=bytes(body)).json()
    except CheckoutRefused:
        raise
    except Exception:
        # The cause can name a route, a certificate path or the broker's reply.
        raise RuntimeError(
            "Checkout broker unavailable; check payment status before retrying"
        ) from None


class _Refusal(BaseModel):
    detail: str


async def _raise_refusal(response: httpx.Response) -> None:
    """Pass on a broker's refusal, but only one of the fixed texts: anything
    else from the broker is reported generically."""
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > 4096:
            return
    try:
        detail = _Refusal.model_validate_json(bytes(body)).detail
    except ValidationError:
        return
    if detail in MESSAGES:
        raise CheckoutRefused(detail)
