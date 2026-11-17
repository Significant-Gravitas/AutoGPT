"""A stand-in for Link's spend-request API, for tests.

Importable by the worker subprocess, which is why it lives beside the code
rather than in a conftest: the real-browser tests swap it in for
``worker.request_spend`` inside the spawned worker.
"""

from datetime import UTC, datetime, timedelta

from pydantic import SecretStr

from backend.util.link_checkout.models import Card, SpendRequest, WorkerJob

SYNTHETIC_CARD_NUMBER = "4242424242424242"
SYNTHETIC_CVC = "987"
SYNTHETIC_PAY_TOKEN = "lpt_synthetic_0123456789"


async def synthetic_spend(
    job: WorkerJob, include_credential: bool = False
) -> SpendRequest:
    plan = job.intent.plan
    pay_token = plan.execution == "link_pay_token"
    return SpendRequest(
        id=job.intent.spend_request_id or "lsrq_fixture",
        status="pending_approval" if job.action == "create" else "approved",
        merchant_url=plan.merchant_url(),
        amount=plan.amount,
        currency=plan.currency,
        approval_url="https://app.link.com/activity/approve/lsrq_fixture",
        card=(
            Card(
                number=SecretStr(SYNTHETIC_CARD_NUMBER),
                cvc=SecretStr(SYNTHETIC_CVC),
                exp_month=12,
                exp_year=2030,
                valid_until=datetime.now(UTC) + timedelta(minutes=1),
            )
            if include_credential and not pay_token
            else None
        ),
        link_pay_token=(
            SecretStr(SYNTHETIC_PAY_TOKEN) if include_credential and pay_token else None
        ),
    )
