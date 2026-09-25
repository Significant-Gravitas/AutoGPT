"""The payment worker: the only process that ever holds a card number.

Spawned per job with a scrubbed environment by ``runner``. It reads one job
from stdin and writes one bounded result to stdout; the card never leaves it.
For ``pay`` it retrieves the single-use card, fills the fields pinned when the
checkout was prepared, submits once, waits for the page's requests to settle
and retires the whole browser, so nothing it held can be read back.
"""

import asyncio
import ctypes
import logging
import sys
import time

from backend.util.link_checkout.cdp import CDP, Control, connection
from backend.util.link_checkout.link import (
    LinkDuplicate,
    LinkRejected,
    request_spend,
    validate_spend,
)
from backend.util.link_checkout.models import (
    Card,
    WorkerJob,
    WorkerReceipt,
    WorkerResult,
)
from backend.util.link_checkout.runtime import require_runtime, retire_payment_browser


async def execute(job: WorkerJob) -> WorkerResult:
    require_runtime()
    # PR_SET_DUMPABLE 0: no core dump, and no same-user process may attach
    # to this one or read its memory through /proc.
    if ctypes.CDLL(None).prctl(4, 0, 0, 0, 0) != 0:
        raise RuntimeError("Cannot disable process dumps")
    if job.action != "pay":
        return WorkerResult(spend=await request_spend(job))
    return WorkerResult(receipt=await pay(job))


async def pay(job: WorkerJob) -> WorkerReceipt:
    receipt = WorkerReceipt(status="not_submitted")
    async with connection(job.intent.browser.endpoint) as socket:
        cdp = CDP(socket)
        try:
            async with asyncio.timeout(55):
                controls = await cdp.controls(job.intent.browser, job.intent.plan)
                await cdp.monitor_network(controls)
                if job.intent.expires_at <= time.time():
                    return receipt
                spend = await request_spend(job, include_card=True)
                validate_spend(job.intent, spend, require_card=True)
                if spend.card is None:
                    return receipt
                await cdp.attach(job.intent.browser)
                receipt.status = "outcome_unknown"
                await fill(cdp, controls, spend.card)
                await submit(cdp, controls["submit"], job)
                if await cdp.drain_network():
                    receipt.status = "submitted"
        except Exception:
            pass
        finally:
            try:
                await cdp.call("Browser.close", {})
                receipt.browser_closed = True
            except Exception:
                pass
            receipt.browser_closed = await retire_payment_browser(job.intent.session_id)
    return receipt


async def fill(cdp: CDP, controls: dict[str, Control], card: Card) -> None:
    fields = [
        ("number", card.number.get_secret_value()),
        ("cvc", card.cvc.get_secret_value()),
        ("expiry", f"{card.exp_month:02d}/{card.exp_year % 100:02d}"),
        ("exp_month", f"{card.exp_month:02d}"),
        ("exp_year", str(card.exp_year)),
    ]
    for role, value in fields:
        if role not in controls:
            continue
        control = controls[role]
        await cdp.check_control(control, role)
        result = await cdp.call(
            "Runtime.callFunctionOn",
            {
                "functionDeclaration": """function(value) {
                const el = this;
                if (!el.isConnected || !(el instanceof HTMLInputElement) || el.value !== '' || el.disabled) return false;
                Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, value);
                el.dispatchEvent(new Event('input', {bubbles: true}));
                el.dispatchEvent(new Event('change', {bubbles: true}));
                return true;
            }""",
                "arguments": [{"value": value}],
                "objectId": control.object_id,
                "returnByValue": True,
            },
            control.session,
        )
        if result.result is None or result.result.value is not True:
            raise RuntimeError("Private fill failed")


async def submit(cdp: CDP, control: Control, job: WorkerJob) -> None:
    await cdp.attach(job.intent.browser)
    await cdp.check_control(control, "submit")
    result = await cdp.call(
        "Runtime.callFunctionOn",
        {
            "functionDeclaration": """function() {
            if (!this.isConnected || this.disabled) return false;
            this.click();
            return true;
        }""",
            "objectId": control.object_id,
            "returnByValue": True,
        },
        control.session,
    )
    if result.result is None or result.result.value is not True:
        raise RuntimeError("Private submit failed")


def main() -> None:
    logging.disable(logging.CRITICAL)
    result = WorkerResult(error="private_checkout_failed")
    try:
        job = WorkerJob.model_validate_json(sys.stdin.buffer.read(32_768))
        result = asyncio.run(execute(job))
    except LinkDuplicate:
        result = WorkerResult(error="link_duplicate")
    except LinkRejected:
        result = WorkerResult(error="link_rejected")
    except BaseException:
        # Nothing from the failure is reported: its message could carry
        # what the worker was holding.
        pass
    sys.stdout.write(result.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
