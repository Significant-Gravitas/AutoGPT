"""Which frame of a checkout tab may name the Link Pay Token's payee."""

import pytest

from backend.util.link_checkout import pay_token
from backend.util.link_checkout.cdp_models import (
    Frame,
    FrameContext,
    RemoteValue,
    Result,
)
from backend.util.link_checkout.models import BrowserBinding
from backend.util.link_checkout.refusals import PAY_TOKEN_UNAVAILABLE, CheckoutRefused

TAB = "tab"
CHECKOUT_URL = "https://shop.example/checkout"
STRIPE_FRAME = "https://js.stripe.com/v3/embedded-checkout-inner.html#id=1"
BINDING = BrowserBinding(
    endpoint="ws://127.0.0.1:9222/x", target_id=TAB, url=CHECKOUT_URL
)


def top(account: str | None = None) -> tuple[FrameContext, str | None]:
    frame = Frame(id="top", url=CHECKOUT_URL)
    return FrameContext(frame=frame, target_id=TAB, session="s0"), account


def iframe(url: str, account: str | None, *, own_target: bool = True):
    """A child frame: a cross-process one in its own target, or one in the tab's."""
    frame = Frame(id=f"frame:{url}", url=url, parentId="top")
    target = f"target:{url}" if own_target else TAB
    return FrameContext(frame=frame, target_id=target, session="s1"), account


class Tab:
    """Answers the calls ``enable_pay_token`` makes: each frame may carry a
    steering block naming an account, and its token field shows once ticked."""

    def __init__(self, *frames: tuple[FrameContext, str | None]):
        self.frames_ = list(frames)
        self.ticked: list[str] = []
        self._worlds: dict[int, int] = {}

    async def frames(self, binding: BrowserBinding) -> list[FrameContext]:
        return [context for context, _ in self.frames_]

    async def call(self, method: str, params: dict, session: str = "") -> Result:
        if method == "Page.createIsolatedWorld":
            index = next(
                i
                for i, (c, _) in enumerate(self.frames_)
                if c.frame.id == params["frameId"]
            )
            world = len(self._worlds) + 1
            self._worlds[world] = index
            return Result(executionContextId=world)
        context, account = self.frames_[self._worlds[params["executionContextId"]]]
        code = params["functionDeclaration"]
        if code == pay_token._HAS_STEERING:
            return Result(result=RemoteValue(value=account is not None))
        if code == pay_token._TICK_STEERING:
            self.ticked.append(context.frame.url)
            return Result(result=RemoteValue(value=True))
        return Result(result=RemoteValue(value=account))


@pytest.mark.asyncio
async def test_a_stripe_frame_names_the_payee():
    tab = Tab(top(), iframe(STRIPE_FRAME, "acct_merchant"))

    token = await pay_token.enable_pay_token(tab, BINDING)

    assert (token.merchant_account_id, token.frame_url) == (
        "acct_merchant",
        STRIPE_FRAME,
    )


@pytest.mark.asyncio
async def test_the_checkout_page_itself_may_name_the_payee():
    tab = Tab(top("acct_merchant"))

    token = await pay_token.enable_pay_token(tab, BINDING)

    assert token.merchant_account_id == "acct_merchant"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,own_target",
    [
        ("https://ads.example/slot", True),
        ("https://ads.example/slot", False),
        ("https://js.stripe.com.ads.example/slot", True),
        ("http://js.stripe.com/v3/", True),
    ],
    ids=["other_origin", "other_origin_in_process", "lookalike_host", "plain_http"],
)
async def test_a_block_from_any_other_frame_is_never_the_payee(url, own_target):
    tab = Tab(top(), iframe(url, "acct_mallory", own_target=own_target))

    with pytest.raises(CheckoutRefused) as refused:
        await pay_token.enable_pay_token(tab, BINDING)
    assert str(refused.value) == PAY_TOKEN_UNAVAILABLE
    assert tab.ticked == []


@pytest.mark.asyncio
async def test_a_block_beside_stripes_is_ignored_rather_than_refusing_both():
    tab = Tab(
        top(),
        iframe("https://ads.example/slot", "acct_mallory"),
        iframe(STRIPE_FRAME, "acct_merchant"),
    )

    token = await pay_token.enable_pay_token(tab, BINDING)

    assert token.merchant_account_id == "acct_merchant"
    assert tab.ticked == [STRIPE_FRAME]
