"""The worker's fill-and-submit path against a scripted DevTools socket.

``browser_test.py`` drives a real Chromium but needs the private runtime, which
CI's runners (swap on) don't have; this runs everywhere and pins the same
contract: the pinned fields get the card, the pinned button is clicked once,
and a changed page stops everything before the card is asked for.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from backend.util.link_checkout import cdp, network, worker
from backend.util.link_checkout.models import Card, CheckoutPlan, WorkerJob
from backend.util.link_checkout.refusals import (
    FRAME_NOT_FOUND,
    INVALID_SELECTOR,
    NOT_CARD_FIELDS,
    PAY_TOKEN_UNAVAILABLE,
    CheckoutRefused,
)
from backend.util.link_checkout.synthetic_link import (
    SYNTHETIC_PAY_TOKEN,
    synthetic_spend,
)

CHECKOUT_URL = "https://shop.example/checkout"
SELECTORS = {
    "#number": 11,
    "#cvc": 12,
    "#expiry": 13,
    "#pay": 14,
    "#note": 15,
    "#month": 16,
    "#year": 17,
    'input[name="link_pay_token"]': 18,
}


class ScriptedBrowser:
    """Answers the DevTools calls the checkout makes, and records the fill."""

    def __init__(self):
        self.nodes = dict(SELECTORS)
        # What each element declares; the pay control is the one button.
        self.autocomplete = {
            "#number": "cc-number",
            "#cvc": "cc-csc",
            "#expiry": "cc-exp",
            "#month": "cc-exp-month",
            "#year": "cc-exp-year",
            "#note": "",
        }
        self.buttons = {"#pay"}
        # Fields with a maxlength; the rest take any length.
        self.max_length: dict[str, int] = {}
        # Other targets the browser lists, and which of them fail to attach.
        self.other_targets: list[dict] = []
        self.broken_targets: set[str] = set()
        self.filled: dict[str, str] = {}
        self.clicks = 0
        self.closed = False
        # Pay buttons the page enables only once the card is in, or never.
        self.disabled_until_filled: set[str] = set()
        self.always_disabled: set[str] = set()
        # Frames inside the checkout tab besides its own document.
        self.child_frames: list[dict] = []
        # A Stripe checkout's "I am an AI agent" option, and what it took.
        self.steering = False
        self.steering_ticked = False
        self.pay_token = ""
        self._replies: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        params = message.get("params", {})
        if (
            message["method"] == "Target.attachToTarget"
            and params.get("targetId") in self.broken_targets
        ):
            reply = {"id": message["id"], "error": {"message": "No target"}}
        else:
            reply = {
                "id": message["id"],
                "result": self._answer(message["method"], params),
            }
        await self._replies.put(json.dumps(reply))

    async def recv(self) -> str:
        return await self._replies.get()

    def _answer(self, method: str, params: dict) -> dict:
        if method == "Target.getTargets":
            return {
                "targetInfos": [
                    {"targetId": "tab", "type": "page", "url": CHECKOUT_URL},
                    *self.other_targets,
                ]
            }
        if method == "Target.attachToTarget":
            return {"sessionId": "session"}
        if method == "Page.getFrameTree":
            return {
                "frameTree": {
                    "frame": {"id": "frame", "url": CHECKOUT_URL, "loaderId": "doc"},
                    "childFrames": [
                        {"frame": {**frame, "parentId": "frame"}}
                        for frame in self.child_frames
                    ],
                }
            }
        if method == "Page.createIsolatedWorld":
            return {"executionContextId": 1}
        if method == "DOM.describeNode":
            selector = params["objectId"].removeprefix("node:")
            return {"node": {"backendNodeId": self.nodes[selector]}}
        if method == "Runtime.callFunctionOn":
            return {"result": self._call(params)}
        if method == "Browser.close":
            self.closed = True
        return {}

    def _call(self, params: dict) -> dict:
        code = params["functionDeclaration"]
        if ".AiAgentPaymentSteering').length" in code:
            return {"value": self.steering}
        if "box.click()" in code:
            self.steering_ticked = self.steering
            return {"value": self.steering}
        if "data-stripe-merchant-account" in code:
            return {"value": "acct_test123" if self.steering_ticked else None}
        if "set.call(el, token)" in code:
            self.pay_token = params["arguments"][0]["value"]
            return {"value": True}
        if "this.name !== 'link_pay_token'" in code:
            return {"value": "not_ready" if self.pay_token else "ok"}
        if "querySelectorAll" in code:
            selector = params["arguments"][0]["value"]
            if ":visible" in selector:
                return {"type": "string", "value": "invalid_selector"}
            return {"objectId": f"node:{selector}"}
        if "getBoundingClientRect" in code:
            return {"value": self._check(params)}
        if "set.call(el, value)" in code:
            selector = params["objectId"].removeprefix("node:")
            limit = self.max_length.get(selector, -1)
            fitting = [
                value
                for value in params["arguments"][0]["value"]
                if limit < 0 or len(value) <= limit
            ]
            if not fitting:
                return {"value": False}
            self.filled[selector] = fitting[0]
            return {"value": True}
        if "this.click()" in code:
            self.clicks += 1
        return {"value": True}

    def _check(self, params: dict) -> str:
        selector = params["objectId"].removeprefix("node:")
        names = params["arguments"][0]["value"]
        allow_disabled = params["arguments"][1]["value"]
        if names is None:
            if selector not in self.buttons:
                return "not_card_field"
            disabled = selector in self.always_disabled or (
                selector in self.disabled_until_filled and len(self.filled) < 3
            )
            return "not_ready" if disabled and not allow_disabled else "ok"
        if self.autocomplete.get(selector) not in names:
            return "not_card_field"
        return "not_ready" if selector in self.filled else "ok"


@pytest.fixture
def browser(monkeypatch):
    scripted = ScriptedBrowser()

    @asynccontextmanager
    async def connect(endpoint: str):
        yield scripted

    for module in (cdp, worker):
        monkeypatch.setattr(module, "connection", connect)
    monkeypatch.setattr(worker, "retire_payment_browser", AsyncMock(return_value=True))
    # No page requests in flight, so the post-submit wait can end at once.
    monkeypatch.setattr(network.NetworkDrain, "settled", lambda self: not self.pending)
    return scripted


async def prepared_job(intent) -> WorkerJob:
    intent.browser = await cdp.prepare_browser("ws://127.0.0.1:9222/x", intent.plan)
    return WorkerJob(intent=intent, access_token=SecretStr("synthetic-token"))


@pytest.mark.asyncio
async def test_pinned_fields_get_the_card_and_the_button_is_clicked_once(
    browser, intent, monkeypatch
):
    monkeypatch.setattr(worker, "request_spend", synthetic_spend)
    job = await prepared_job(intent)

    receipt = await worker.pay(job)

    assert receipt.status == "submitted"
    assert receipt.browser_closed
    assert browser.filled == {
        "#number": "4242424242424242",
        "#cvc": "987",
        "#expiry": "12/30",
    }
    assert browser.clicks == 1
    assert browser.closed


@pytest.mark.asyncio
async def test_a_changed_field_stops_before_the_card_is_requested(
    browser, intent, monkeypatch
):
    job = await prepared_job(intent)
    browser.nodes["#number"] = 99  # the page swapped the input for another
    requested = AsyncMock()
    monkeypatch.setattr(worker, "request_spend", requested)

    receipt = await worker.pay(job)

    assert receipt.status == "not_submitted"
    requested.assert_not_awaited()
    assert browser.filled == {}
    assert browser.clicks == 0
    assert browser.closed


@pytest.mark.asyncio
async def test_an_expired_card_is_never_filled(browser, intent, monkeypatch):
    async def expired_card(job, include_credential=False):
        spend = await synthetic_spend(job, include_credential)
        spend.card = Card(
            number=SecretStr("4242424242424242"),
            cvc=SecretStr("987"),
            exp_month=12,
            exp_year=2030,
            valid_until=datetime.now(UTC) - timedelta(minutes=1),
        )
        return spend

    monkeypatch.setattr(worker, "request_spend", expired_card)
    job = await prepared_job(intent)

    receipt = await worker.pay(job)

    assert receipt.status == "not_submitted"
    assert browser.filled == {}
    assert browser.clicks == 0


@pytest.mark.asyncio
async def test_a_field_that_is_not_a_card_input_is_never_pinned(browser, intent):
    with pytest.raises(CheckoutRefused) as refused:
        await cdp.prepare_browser(
            "ws://127.0.0.1:9222/x", intent.plan.model_copy(update={"number": "#note"})
        )
    assert str(refused.value) == NOT_CARD_FIELDS


@pytest.mark.asyncio
async def test_a_pay_control_that_is_not_a_button_is_never_pinned(browser, intent):
    with pytest.raises(CheckoutRefused):
        await cdp.prepare_browser(
            "ws://127.0.0.1:9222/x", intent.plan.model_copy(update={"submit": "#note"})
        )


@pytest.mark.asyncio
async def test_a_field_that_stops_declaring_itself_a_card_input_gets_no_card(
    browser, intent, monkeypatch
):
    job = await prepared_job(intent)
    browser.autocomplete["#number"] = "street-address"
    requested = AsyncMock()
    monkeypatch.setattr(worker, "request_spend", requested)

    receipt = await worker.pay(job)

    assert receipt.status == "not_submitted"
    requested.assert_not_awaited()
    assert browser.filled == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("max_length,year", [(2, "30"), (4, "2030"), (None, "2030")])
async def test_a_separate_year_field_gets_the_spelling_that_fits(
    browser, intent, monkeypatch, max_length, year
):
    intent.plan = intent.plan.model_copy(
        update={"expiry": None, "exp_month": "#month", "exp_year": "#year"}
    )
    if max_length is not None:
        browser.max_length["#year"] = max_length
    monkeypatch.setattr(worker, "request_spend", synthetic_spend)
    job = await prepared_job(intent)

    receipt = await worker.pay(job)

    assert receipt.status == "submitted"
    assert browser.filled["#month"] == "12"
    assert browser.filled["#year"] == year


@pytest.mark.asyncio
async def test_a_combined_expiry_without_room_for_a_slash_gets_mmyy(
    browser, intent, monkeypatch
):
    browser.max_length["#expiry"] = 4
    monkeypatch.setattr(worker, "request_spend", synthetic_spend)
    job = await prepared_job(intent)

    await worker.pay(job)

    assert browser.filled["#expiry"] == "1230"


@pytest.mark.asyncio
async def test_a_field_that_changes_after_the_card_is_fetched_leaves_it_unused(
    browser, intent, monkeypatch
):
    """Nothing was typed, so the attempt reads as one whose card never reached
    the page, and the broker cancels the unused card."""

    async def fetch_then_change(job, include_credential=False):
        spend = await synthetic_spend(job, include_credential)
        browser.autocomplete["#number"] = "street-address"
        return spend

    monkeypatch.setattr(worker, "request_spend", fetch_then_change)
    job = await prepared_job(intent)

    receipt = await worker.pay(job)

    assert receipt.status == "not_submitted"
    assert browser.filled == {}
    assert browser.clicks == 0


@pytest.mark.asyncio
async def test_an_iframe_elsewhere_that_cannot_be_attached_is_skipped(
    browser, intent, monkeypatch
):
    browser.other_targets.append(
        {"targetId": "ad", "type": "iframe", "url": "https://ads.example/frame"}
    )
    browser.broken_targets.add("ad")
    monkeypatch.setattr(worker, "request_spend", synthetic_spend)
    job = await prepared_job(intent)

    receipt = await worker.pay(job)

    assert receipt.status == "submitted"
    assert browser.clicks == 1


@pytest.mark.asyncio
async def test_a_selector_the_page_cannot_parse_is_refused_by_name(browser, intent):
    with pytest.raises(CheckoutRefused) as refused:
        await cdp.prepare_browser(
            "ws://127.0.0.1:9222/x",
            intent.plan.model_copy(update={"number": "#number:visible"}),
        )
    assert str(refused.value) == INVALID_SELECTOR


@pytest.mark.asyncio
async def test_a_frame_url_naming_no_loaded_frame_is_refused_by_name(browser, intent):
    plan = intent.plan.model_copy(
        update={"frame_urls": {"number": "https://shop.example/pay"}}
    )
    with pytest.raises(CheckoutRefused) as refused:
        await cdp.prepare_browser("ws://127.0.0.1:9222/x", plan)
    assert str(refused.value) == FRAME_NOT_FOUND


CARD_FRAME = "https://js.stripe.example/v3/elements-inner-card"


@pytest.mark.asyncio
async def test_a_frame_url_without_its_query_finds_the_one_frame_at_that_address(
    browser, intent, monkeypatch
):
    """The agent can rarely read an iframe's full address; the address without
    its query is enough when only one loaded frame has it."""
    browser.child_frames = [
        {"id": "card", "url": f"{CARD_FRAME}?key=pk_test&id=1", "loaderId": "c1"}
    ]
    intent.plan = intent.plan.model_copy(
        update={"frame_urls": {"number": CARD_FRAME, "cvc": CARD_FRAME}}
    )
    monkeypatch.setattr(worker, "request_spend", synthetic_spend)
    job = await prepared_job(intent)

    receipt = await worker.pay(job)

    assert receipt.status == "submitted"
    assert {f.frame_id for f in job.intent.browser.fields if f.role == "number"} == {
        "card"
    }


@pytest.mark.asyncio
async def test_a_frame_url_without_its_query_is_refused_when_frames_share_it(
    browser, intent
):
    browser.child_frames = [
        {"id": "a", "url": f"{CARD_FRAME}?id=1", "loaderId": "a1"},
        {"id": "b", "url": f"{CARD_FRAME}?id=2", "loaderId": "b1"},
    ]
    plan = intent.plan.model_copy(update={"frame_urls": {"number": CARD_FRAME}})
    with pytest.raises(CheckoutRefused) as refused:
        await cdp.prepare_browser("ws://127.0.0.1:9222/x", plan)
    assert str(refused.value) == FRAME_NOT_FOUND


@pytest.mark.asyncio
async def test_a_pay_button_enabled_only_once_the_card_is_in_is_clicked_then(
    browser, intent, monkeypatch
):
    browser.disabled_until_filled.add("#pay")
    monkeypatch.setattr(worker, "request_spend", synthetic_spend)
    job = await prepared_job(intent)

    receipt = await worker.pay(job)

    assert receipt.status == "submitted"
    assert browser.clicks == 1


@pytest.mark.asyncio
async def test_a_pay_button_that_never_enables_is_never_clicked(
    browser, intent, monkeypatch
):
    browser.always_disabled.add("#pay")
    monkeypatch.setattr(worker, "PAY_BUTTON_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(worker, "request_spend", synthetic_spend)
    job = await prepared_job(intent)

    receipt = await worker.pay(job)

    # The card was typed, so the outcome stays for Link to settle.
    assert receipt.status == "outcome_unknown"
    assert browser.clicks == 0
    assert browser.closed


def pay_token_plan(plan) -> CheckoutPlan:
    fields = plan.model_dump(exclude={"number", "cvc", "expiry", "exp_month"})
    return CheckoutPlan.model_validate(
        {**fields, "execution": "link_pay_token", "test_mode": False}
    )


@pytest.mark.asyncio
async def test_a_stripe_checkout_gets_the_pay_token_and_never_a_card(
    browser, intent, monkeypatch
):
    browser.steering = True
    intent.plan = pay_token_plan(intent.plan)
    requested: list[bool] = []

    async def link(job, include_credential=False):
        requested.append(include_credential)
        return await synthetic_spend(job, include_credential)

    monkeypatch.setattr(worker, "request_spend", link)
    job = await prepared_job(intent)

    assert browser.steering_ticked
    pay_token = job.intent.browser.pay_token
    assert pay_token is not None
    assert pay_token.merchant_account_id == "acct_test123"
    assert pay_token.frame_url == CHECKOUT_URL

    receipt = await worker.pay(job)

    assert receipt.status == "submitted"
    assert requested == [True]
    assert browser.pay_token == SYNTHETIC_PAY_TOKEN
    assert browser.filled == {}
    assert browser.clicks == 1
    assert browser.closed


@pytest.mark.asyncio
async def test_a_page_without_the_agent_option_cannot_take_a_pay_token(browser, intent):
    with pytest.raises(CheckoutRefused) as refused:
        await cdp.prepare_browser("ws://127.0.0.1:9222/x", pay_token_plan(intent.plan))
    assert str(refused.value) == PAY_TOKEN_UNAVAILABLE
    assert not browser.steering_ticked
