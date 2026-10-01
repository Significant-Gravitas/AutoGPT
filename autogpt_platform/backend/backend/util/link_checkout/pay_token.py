"""Stripe checkouts that take a Link Pay Token instead of a card.

A Stripe-hosted or embedded checkout carries a visually hidden "I am an AI
agent" option (``.AiAgentPaymentSteering``). Once it is ticked, the page shows
a field that accepts a Link Pay Token and names its Stripe account; Link then
pays the whole checkout from the customer's wallet, with no card form. The
engine finds and ticks the option when the checkout is prepared and pins the
token field like any other payment field. Only the worker ever holds the token.
https://docs.stripe.com/agentic-commerce/link-agent-wallet/use-link-wallet-pay-online#pay-with-a-link-pay-token
"""

import asyncio
import time
from typing import Protocol

from backend.util.link_checkout.cdp_models import FrameContext, Result
from backend.util.link_checkout.models import BrowserBinding, PayToken
from backend.util.link_checkout.refusals import PAY_TOKEN_UNAVAILABLE, CheckoutRefused

PAY_TOKEN_SELECTOR = 'input[name="link_pay_token"]'
# How long the page may take to show the token field once the option is ticked.
_READY_SECONDS = 5

_HAS_STEERING = (
    "function() { return document.querySelectorAll('.AiAgentPaymentSteering')"
    ".length === 1; }"
)
# The option is hidden from the keyboard, so it is ticked with a DOM click,
# as Stripe's guide does. Its own label decides; nothing else on the page is
# touched.
_TICK_STEERING = """function() {
    const box = document.querySelector(
        '.AiAgentPaymentSteering input[type="checkbox"]');
    if (!box) return false;
    if (!box.checked) box.click();
    return box.checked;
}"""
# The Stripe account the token pays, once the token field is there too.
_READY = """function() {
    const block = document.querySelector('.AiAgentPaymentSteering');
    const account = block && block.querySelector('[data-stripe-merchant-account]');
    const field = document.querySelectorAll('input[name="link_pay_token"]');
    const id = account && account.getAttribute('data-stripe-merchant-account');
    return field.length === 1 && id && /^acct_[A-Za-z0-9]+$/.test(id) ? id : null;
}"""


class Browser(Protocol):
    """What this needs of ``cdp.CDP``, which calls it while preparing."""

    async def frames(self, binding: BrowserBinding) -> list[FrameContext]: ...

    async def call(self, method: str, params: dict, session: str = "") -> Result: ...


async def enable_pay_token(cdp: Browser, binding: BrowserBinding) -> PayToken:
    """Tick the page's agent option and return where its token goes. Refused
    unless exactly one frame of the tab carries the option."""
    frames = [f for f in await cdp.frames(binding) if await _steers(cdp, f)]
    if len(frames) != 1:
        raise CheckoutRefused(PAY_TOKEN_UNAVAILABLE)
    context = frames[0]
    if await _evaluate(cdp, context, _TICK_STEERING) is not True:
        raise CheckoutRefused(PAY_TOKEN_UNAVAILABLE)
    deadline = time.monotonic() + _READY_SECONDS
    while True:
        account = await _evaluate(cdp, context, _READY)
        if account and account is not True:
            return PayToken(
                frame_url=context.frame.url, merchant_account_id=str(account)
            )
        if time.monotonic() >= deadline:
            raise CheckoutRefused(PAY_TOKEN_UNAVAILABLE)
        await asyncio.sleep(0.25)


async def _steers(cdp: Browser, context: FrameContext) -> bool:
    try:
        return await _evaluate(cdp, context, _HAS_STEERING) is True
    except RuntimeError:
        # A frame that went away or can't run script has no option to tick.
        return False


async def _evaluate(
    cdp: Browser, context: FrameContext, function: str
) -> bool | str | None:
    world = await cdp.call(
        "Page.createIsolatedWorld",
        {"frameId": context.frame.id, "worldName": "autogpt-private-checkout"},
        context.session,
    )
    result = await cdp.call(
        "Runtime.callFunctionOn",
        {
            "functionDeclaration": function,
            "executionContextId": world.executionContextId,
            "returnByValue": True,
        },
        context.session,
    )
    return result.result.value if result.result else None
