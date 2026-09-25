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
from backend.util.link_checkout.models import Card, WorkerJob
from backend.util.link_checkout.refusals import NOT_CARD_FIELDS, CheckoutRefused
from backend.util.link_checkout.synthetic_link import synthetic_spend

CHECKOUT_URL = "https://shop.example/checkout"
SELECTORS = {
    "#number": 11,
    "#cvc": 12,
    "#expiry": 13,
    "#pay": 14,
    "#note": 15,
    "#month": 16,
    "#year": 17,
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
        self.filled: dict[str, str] = {}
        self.clicks = 0
        self.closed = False
        self._replies: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        result = self._answer(message["method"], message.get("params", {}))
        await self._replies.put(json.dumps({"id": message["id"], "result": result}))

    async def recv(self) -> str:
        return await self._replies.get()

    def _answer(self, method: str, params: dict) -> dict:
        if method == "Target.getTargets":
            return {
                "targetInfos": [
                    {"targetId": "tab", "type": "page", "url": CHECKOUT_URL}
                ]
            }
        if method == "Target.attachToTarget":
            return {"sessionId": "session"}
        if method == "Page.getFrameTree":
            return {
                "frameTree": {
                    "frame": {"id": "frame", "url": CHECKOUT_URL, "loaderId": "doc"}
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
        if "querySelectorAll" in code:
            return {"objectId": f"node:{params['arguments'][0]['value']}"}
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
        if names is None:
            return "ok" if selector in self.buttons else "not_card_field"
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
    async def expired_card(job, include_card=False):
        spend = await synthetic_spend(job, include_card)
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
