import asyncio
import uuid

import pytest

from backend.util.link_checkout import runtime
from backend.util.link_checkout.browser_test import browse, browser_endpoint, merchant
from backend.util.link_checkout.cdp import CDP, connection, prepare_browser
from backend.util.link_checkout.models import WorkerJob
from backend.util.link_checkout.refusals import CheckoutRefused
from backend.util.link_checkout.worker import pay

pytestmark = pytest.mark.skipif(
    not runtime.local_runtime_ready(),
    reason="needs the private browser runtime (Linux, agent-browser, no swap)",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["replace", "reload"])
async def test_changed_field_or_document_rejected_before_card_retrieval(
    monkeypatch, plan, intent, mutation
):
    monkeypatch.setenv("COPILOT_LINK_PRIVATE_CHECKOUT", "true")
    session_id = f"binding-{uuid.uuid4().hex}"
    await browse(session_id, "open", "about:blank")
    endpoint = await browser_endpoint(session_id)
    async with connection(endpoint) as observer:
        cdp = CDP(observer)
        binding = await cdp.bind("about:blank")
        session = await cdp.attach(binding)
        await cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*"}]}, session)
        serving = asyncio.create_task(merchant(observer, session, asyncio.Event()))
        await browse(session_id, "open", plan.checkout_url)
        intent.session_id = session_id
        intent.browser = await prepare_browser(endpoint, plan)
        async with connection(endpoint) as editor:
            control = CDP(editor)
            editing_session = await control.attach(intent.browser)
            if mutation == "replace":
                await control.call(
                    "Runtime.evaluate",
                    {
                        "expression": "document.querySelector('#number').replaceWith(document.querySelector('#number').cloneNode()); true",
                        "returnByValue": True,
                    },
                    editing_session,
                )
            else:
                await browse(session_id, "reload")
        requested = []

        async def retrieve(*args, **kwargs):
            requested.append(True)
            raise RuntimeError("Provider must not be reached")

        monkeypatch.setattr("backend.util.link_checkout.worker.request_spend", retrieve)
        try:
            await pay(WorkerJob(intent=intent, access_token="test-only"))
            assert requested == []
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
            await runtime.retire_payment_browser(session_id)


@pytest.mark.asyncio
async def test_a_field_that_is_not_a_card_input_is_refused(monkeypatch, plan):
    """A selector aimed at an order note, address or search box would have the
    card typed where the page keeps or shows it."""
    monkeypatch.setenv("COPILOT_LINK_PRIVATE_CHECKOUT", "true")
    session_id = f"binding-{uuid.uuid4().hex}"
    await browse(session_id, "open", "about:blank")
    endpoint = await browser_endpoint(session_id)
    async with connection(endpoint) as observer:
        cdp = CDP(observer)
        session = await cdp.attach(await cdp.bind("about:blank"))
        await cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*"}]}, session)
        serving = asyncio.create_task(merchant(observer, session, asyncio.Event()))
        await browse(session_id, "open", plan.checkout_url)
        try:
            with pytest.raises(CheckoutRefused, match="own card inputs"):
                await prepare_browser(
                    endpoint, plan.model_copy(update={"number": "#note"})
                )
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
            await runtime.retire_payment_browser(session_id)
