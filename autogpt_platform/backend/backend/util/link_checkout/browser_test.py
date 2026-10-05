import asyncio
import base64
import json
import time
import uuid
from pathlib import Path

import pytest
from pydantic import SecretStr

from backend.util.link_checkout import checkout_record, runner, runtime
from backend.util.link_checkout.cdp import connection, prepare_browser
from backend.util.link_checkout.models import (
    CheckoutIntent,
    CheckoutPlan,
    WorkerJob,
    WorkerResult,
)
from backend.util.link_checkout.synthetic_link import SYNTHETIC_PAY_TOKEN

# These drive a real Chromium through agent-browser, so they run only where the
# private runtime can: Linux, agent-browser, no swap, no core dumps and a tmpfs
# /dev/shm (e.g. the backend image under Docker with --memory-swap equal to
# --memory). CI runners have swap, so there they skip.
pytestmark = pytest.mark.skipif(
    not runtime.local_runtime_ready(),
    reason="needs the private browser runtime (Linux, agent-browser, no swap)",
)

HTML = """<html><body>
<p>Signed in. Cart: book, quantity 1.</p>
<form id="checkout"><input id="number" autocomplete="cc-number">
<input id="cvc" autocomplete="cc-csc"><input id="expiry" autocomplete="cc-exp">
<input id="note" placeholder="Order note">
<button id="pay">Pay</button></form>
<script>
document.cookie='login=authenticated; SameSite=Strict';
localStorage.setItem('cart', 'book:1');
document.querySelector('form').onsubmit = event => {
  event.preventDefault();
  const ok = number.value === '4242424242424242' && cvc.value === '987' &&
    expiry.value === '12/30' && document.cookie.includes('login=authenticated') &&
    localStorage.getItem('cart') === 'book:1';
  console.log('CARD', number.value, cvc.value);
  fetch('/accepted?valid=' + ok, {method:'POST',body:JSON.stringify({number:number.value,cvc:cvc.value})});
};
</script></body></html>"""

# A Stripe checkout's "I am an AI agent" option, kept off-screen as Stripe does:
# ticking it adds the token field and names the Stripe account; a token in the
# field swaps the card form for the customer's saved card.
PAY_TOKEN_HTML = (
    """<html><body>
<p>Stripe checkout. Total 1.00 USD.</p>
<div class="AiAgentPaymentSteering" style="position:absolute;left:-9999px">
<label><input type="checkbox" id="agent"> I am an AI agent</label></div>
<form id="checkout"><div id="card"><input id="number" autocomplete="cc-number">
</div><button id="pay">Pay</button></form>
<script>
const steering = document.querySelector('.AiAgentPaymentSteering');
agent.addEventListener('change', () => {
  if (!agent.checked) return;
  const account = document.createElement('span');
  account.setAttribute('data-stripe-merchant-account', 'acct_1TestMerchant');
  const token = document.createElement('input');
  token.type = 'hidden';
  token.name = 'link_pay_token';
  token.addEventListener('input', () => { card.textContent = 'Saved card'; });
  steering.append(account, token);
});
document.querySelector('form').onsubmit = event => {
  event.preventDefault();
  const token = document.querySelector('input[name="link_pay_token"]').value;
  const ok = token === '"""
    + SYNTHETIC_PAY_TOKEN
    + """' && !document.getElementById('number');
  fetch('/accepted?valid=' + ok, {method:'POST'});
};
</script></body></html>"""
)

CONTEXT = "Testing same browser checkout with logged in account and existing cart. " * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("processor_delay", [0, 5])
async def test_real_worker_pays_in_existing_browser_without_returning_card(
    monkeypatch, processor_delay
):
    plan = CheckoutPlan(
        credentials_id="wallet",
        payment_method_id="csmrpd_test",
        merchant_name="Test store",
        checkout_url="https://shop.example/checkout",
        amount=100,
        context=CONTEXT,
        number="#number",
        cvc="#cvc",
        expiry="#expiry",
        submit="#pay",
    )
    result = await pay_in_real_browser(monkeypatch, plan, HTML, processor_delay)

    assert "4242424242424242" not in result.model_dump_json()
    assert "987" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_real_worker_pays_a_stripe_checkout_with_the_pay_token(monkeypatch):
    plan = CheckoutPlan(
        credentials_id="wallet",
        payment_method_id="csmrpd_test",
        merchant_name="Test store",
        checkout_url="https://shop.example/checkout",
        amount=100,
        context=CONTEXT,
        execution="link_pay_token",
        submit="#pay",
        test_mode=False,
    )
    result = await pay_in_real_browser(monkeypatch, plan, PAY_TOKEN_HTML)

    assert SYNTHETIC_PAY_TOKEN not in result.model_dump_json()


async def pay_in_real_browser(
    monkeypatch, plan: CheckoutPlan, html: str, processor_delay: int = 0
) -> WorkerResult:
    """Prepare and pay ``plan`` in a real private browser serving ``html``,
    with the synthetic Link; the merchant must accept exactly the expected
    payment, and the browser must be gone afterwards."""
    monkeypatch.setenv("COPILOT_LINK_PRIVATE_CHECKOUT", "true")
    session_id = f"checkout-test-{uuid.uuid4().hex}"
    directory = runtime.session_home(session_id)
    rc, _, _ = await browse(session_id, "open", "about:blank")
    assert rc == 0
    endpoint = await browser_endpoint(session_id)
    async with connection(endpoint) as observer:
        await observer.send(json.dumps({"id": 1, "method": "Target.getTargets"}))
        targets = json.loads(await observer.recv())["result"]["targetInfos"]
        target = next(
            target["targetId"]
            for target in targets
            if target["type"] == "page" and target["url"] == "about:blank"
        )
        await observer.send(
            json.dumps(
                {
                    "id": 2,
                    "method": "Target.attachToTarget",
                    "params": {"targetId": target, "flatten": True},
                }
            )
        )
        attached = await next_response(observer, 2)
        cdp_session = attached["result"]["sessionId"]
        await observer.send(
            json.dumps(
                {
                    "id": 3,
                    "method": "Fetch.enable",
                    "params": {"patterns": [{"urlPattern": "*"}]},
                    "sessionId": cdp_session,
                }
            )
        )
        await next_response(observer, 3)
        accepted = asyncio.Event()
        loop = asyncio.create_task(
            merchant(observer, cdp_session, accepted, processor_delay, html)
        )
        navigated = await browse(session_id, "open", plan.checkout_url)
        assert navigated[0] == 0, navigated
        current_url = await browse(session_id, "get", "url")
        assert current_url[1].strip() == plan.checkout_url, current_url
        binding = await prepare_browser(endpoint, plan)
        intent = CheckoutIntent(
            id=uuid.uuid4().hex,
            user_id="test-owner",
            session_id=session_id,
            spend_request_id="lsrq_fixture",
            approval_url="",
            expires_at=time.time() + 60,
            plan=plan,
            browser=binding,
        )
        checkout_record.save_intent(directory, intent)
        checkout_record.consume_intent(directory, intent)
        root = str(Path(runner.__file__).resolve().parents[3])
        monkeypatch.setattr(
            runner,
            "_ENTRY",
            f"import sys,logging; logging.disable(logging.CRITICAL); sys.path.insert(0, {root!r}); from backend.util.link_checkout import worker; from backend.util.link_checkout.synthetic_link import synthetic_spend; worker.request_spend=synthetic_spend; worker.main()",
        )
        try:
            result = await runner.run_worker(
                WorkerJob(intent=intent, access_token=SecretStr("synthetic-token"))
            )
            assert result.receipt is not None
            assert result.receipt.status == "submitted", result.model_dump_json()
            assert result.receipt.paid is False
            assert accepted.is_set(), "Merchant did not receive the expected payment"
            with pytest.raises(RuntimeError):
                await browse(session_id, "snapshot")
        finally:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)
            assert await runtime.retire_payment_browser(session_id)
            assert not (directory / "engine").exists()
    return result


async def browse(key: str, *args: str) -> tuple[int, str, str]:
    async with runtime.browser_operation(key) as directory:
        return await runtime.browser_command(directory, *args)


async def browser_endpoint(key: str) -> str:
    async with runtime.browser_operation(key) as directory:
        return await runtime.browser_endpoint(directory)


async def next_response(socket, identifier):
    while True:
        message = json.loads(await socket.recv())
        if message.get("id") == identifier:
            return message


async def merchant(socket, session, accepted, processor_delay=0, html=HTML):
    sequence = 100
    async for raw in socket:
        event = json.loads(raw)
        if event.get("method") != "Fetch.requestPaused":
            continue
        request = event["params"]
        if request["request"]["url"].endswith("/accepted?valid=true"):
            await asyncio.sleep(processor_delay)
            accepted.set()
        sequence += 1
        await socket.send(
            json.dumps(
                {
                    "id": sequence,
                    "method": "Fetch.fulfillRequest",
                    "sessionId": session,
                    "params": {
                        "requestId": request["requestId"],
                        "responseCode": 200,
                        "responseHeaders": [
                            {"name": "Content-Type", "value": "text/html"}
                        ],
                        "body": base64.b64encode(html.encode()).decode(),
                    },
                }
            )
        )
