import time
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from backend.util.link_checkout import (
    approval,
    broker_checkout,
    broker_link,
    engine,
    runtime,
)
from backend.util.link_checkout.broker_protocol import Principal
from backend.util.link_checkout.checkout_record import current_intent
from backend.util.link_checkout.models import (
    BrowserBinding,
    CheckoutIntent,
    CheckoutPlan,
    WorkerJob,
    WorkerReceipt,
    WorkerResult,
)
from backend.util.link_checkout.synthetic_link import synthetic_spend


@pytest.fixture
def plan():
    return CheckoutPlan(
        credentials_id="wallet",
        payment_method_id="csmrpd_test",
        merchant_name="Test store",
        checkout_url="https://shop.example/checkout",
        amount=100,
        context="Buying a test product from this store. " * 4,
        number="#number",
        cvc="#cvc",
        expiry="#expiry",
        submit="#pay",
    )


@pytest.fixture
def intent(plan):
    return CheckoutIntent(
        id="a" * 32,
        user_id="owner",
        session_id="chat",
        spend_request_id="lsrq_test",
        approval_url="https://app.link.com/activity/approve/lsrq_test",
        expires_at=time.time() + 60,
        plan=plan,
        browser=BrowserBinding(
            endpoint="ws://127.0.0.1:9222/abc", target_id="tab", url=plan.checkout_url
        ),
    )


@pytest.fixture
def principal():
    return Principal(user_id="owner", session_id="chat")


class FakeRedis:
    """The two commands the approval store uses."""

    def __init__(self):
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, nx: bool = False, ex: int = 0) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True


@pytest.fixture
def fake_redis(monkeypatch) -> FakeRedis:
    redis = FakeRedis()
    monkeypatch.setattr(approval, "get_redis_async", AsyncMock(return_value=redis))
    return redis


class FakeWorker:
    """Stands in for the payment worker; records each job's action."""

    def __init__(self):
        self.calls: list[str] = []
        # Whether the chat was sealed, with the attempt recorded, each time
        # the worker was asked for the card.
        self.sealed_when_paying: list[bool] = []
        self.status = "approved"
        self.receipt = WorkerReceipt(status="submitted", browser_closed=True)
        # Link's refusal of the next create: "link_rejected", "link_duplicate".
        self.create_error: Literal["link_rejected", "link_duplicate"] | None = None
        self.delegated_error: Literal["link_rejected", "link_duplicate"] | None = None

    async def __call__(self, job: WorkerJob) -> WorkerResult:
        self.calls.append(job.action)
        if job.action == "pay":
            directory = runtime.session_home(job.intent.session_id)
            self.sealed_when_paying.append(
                (directory / "sensitive").exists()
                and current_intent(directory).attempted
            )
            return WorkerResult(receipt=self.receipt.model_copy())
        if job.action == "create_delegated" and self.delegated_error:
            return WorkerResult(error=self.delegated_error)
        if job.action == "create" and self.create_error:
            return WorkerResult(error=self.create_error)
        spend = await synthetic_spend(job)
        if job.action == "cancel":
            spend.status = "canceled"
        elif job.action != "create":
            spend.status = self.status
        return WorkerResult(spend=spend)


@pytest.fixture
def local_broker(monkeypatch, tmp_path) -> FakeWorker:
    """The in-process broker over temp directories and a fake worker, with no
    browser: everything above the CDP layer runs for real."""
    monkeypatch.setenv("COPILOT_LINK_PRIVATE_CHECKOUT", "true")
    for name in ("CHECKOUT_BROKER_URL", "CHECKOUT_BROKER_ROUTES_FILE"):
        monkeypatch.delenv(name, raising=False)

    def home(key: str) -> Path:
        path = tmp_path / key
        path.mkdir(exist_ok=True)
        return path

    monkeypatch.setattr(runtime, "session_home", home)
    monkeypatch.setattr(engine, "local_runtime_ready", lambda: True)
    monkeypatch.setattr(
        broker_checkout,
        "browser_endpoint",
        AsyncMock(return_value="ws://127.0.0.1:9222/browser"),
    )
    monkeypatch.setattr(
        broker_checkout,
        "prepare_browser",
        AsyncMock(
            return_value=BrowserBinding(
                endpoint="ws://127.0.0.1:9222/browser",
                target_id="tab",
                url="https://shop.example/checkout",
            )
        ),
    )
    for module in (broker_checkout, broker_link):
        monkeypatch.setattr(
            module, "retire_payment_browser", AsyncMock(return_value=True)
        )
    worker = FakeWorker()
    monkeypatch.setattr(broker_link, "run_worker", worker)
    return worker
