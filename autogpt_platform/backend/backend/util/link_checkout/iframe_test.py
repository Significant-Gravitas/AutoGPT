import asyncio
import base64
import json
import uuid

import pytest

from backend.util.link_checkout import runtime
from backend.util.link_checkout.browser_test import browse, browser_endpoint
from backend.util.link_checkout.cdp import CDP, connection, prepare_browser
from backend.util.link_checkout.models import WorkerJob
from backend.util.link_checkout.synthetic_link import synthetic_spend
from backend.util.link_checkout.worker import pay

pytestmark = pytest.mark.skipif(
    not runtime.local_runtime_ready(),
    reason="needs the private browser runtime (Linux, agent-browser, no swap)",
)

PAGE = """<iframe src="https://processor.example/card"></iframe><button id="pay">Pay</button>
<script>let ready=false; onmessage=e=>{if(e.origin==='https://processor.example')ready=e.data==='ready'};
document.querySelector('button').onclick=()=>{if(ready)fetch('/accepted',{method:'POST'})};</script>"""
FRAME = """<input id="number" autocomplete="cc-number">
<input id="cvc" autocomplete="cc-csc"><input id="expiry" autocomplete="cc-exp"><script>
onchange=()=>{if(number.value==='4242424242424242'&&cvc.value==='987'&&expiry.value==='12/30')
parent.postMessage('ready','https://shop.example')};</script>"""


async def serve(socket, accepted):
    sequence = 100
    async for raw in socket:
        event = json.loads(raw)
        method, params = event.get("method"), event.get("params", {})
        commands = []
        if method == "Target.attachedToTarget":
            session = params["sessionId"]
            commands = [
                ("Fetch.enable", {"patterns": [{"urlPattern": "*"}]}),
                ("Runtime.runIfWaitingForDebugger", {}),
            ]
        elif method == "Fetch.requestPaused":
            session = event["sessionId"]
            url = params["request"]["url"]
            if url == "https://shop.example/accepted":
                accepted.set()
            html = FRAME if url == "https://processor.example/card" else PAGE
            commands = [
                (
                    "Fetch.fulfillRequest",
                    {
                        "requestId": params["requestId"],
                        "responseCode": 200,
                        "responseHeaders": [
                            {"name": "Content-Type", "value": "text/html"}
                        ],
                        "body": base64.b64encode(html.encode()).decode(),
                    },
                )
            ]
        else:
            continue
        for command, arguments in commands:
            sequence += 1
            await socket.send(
                json.dumps(
                    {
                        "id": sequence,
                        "method": command,
                        "params": arguments,
                        "sessionId": session,
                    }
                )
            )


@pytest.mark.asyncio
async def test_real_cross_process_payment_frame_keeps_field_binding(
    monkeypatch, intent, plan
):
    monkeypatch.setenv("COPILOT_LINK_PRIVATE_CHECKOUT", "true")
    key = "iframe-" + uuid.uuid4().hex
    await browse(key, "open", "about:blank")
    endpoint = await browser_endpoint(key)
    plan.frame_urls = {
        role: "https://processor.example/card" for role in ("number", "cvc", "expiry")
    }
    intent.session_id, intent.plan = key, plan
    intent.spend_request_id = "lsrq_fixture"
    async with connection(endpoint) as observer:
        cdp = CDP(observer)
        session = await cdp.attach(await cdp.bind("about:blank"))
        await cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*"}]}, session)
        await cdp.call(
            "Target.setAutoAttach",
            {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True},
            session,
        )
        accepted = asyncio.Event()
        serving = asyncio.create_task(serve(observer, accepted))
        try:
            opened = await browse(key, "open", plan.checkout_url)
            assert opened[0] == 0
            intent.browser = await prepare_browser(endpoint, plan)
            fields = {field.role: field for field in intent.browser.fields}
            assert fields["number"].target_id != fields["submit"].target_id
            monkeypatch.setattr(
                "backend.util.link_checkout.worker.request_spend", synthetic_spend
            )
            result = await pay(WorkerJob(intent=intent, access_token="synthetic"))
            assert result.status == "submitted"
            assert accepted.is_set()
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
            await runtime.retire_payment_browser(key)
