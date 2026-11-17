"""Every charge of a batch dream pass goes by the usage attribution (#15208)
the pass started under, kept on its ``DreamPass`` row, against a real
database and Redis: the trial's spend stays the trial's after the account
converts, whichever path settles it (the reaper, the end of a pass whose
result could not be attributed, the drop of a closed pass's batch), each
phase once; and a row that keeps no attribution is never charged, under the
account's current one or any other.

A trial user is real (its trial row, then on conversion its paid usage
activation), and so are the pass's row, its Redis state and claims, the
charge's ledgers (the trial's cost, the usage counters, the cost log). The
model, the provider's batch API and apply are stubbed at their edges."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from prisma import Json
from prisma.enums import SubscriptionTier
from prisma.models import PaidUsageActivation, PlatformCostLog, SubscriptionTrial, User
from pydantic import BaseModel

from backend.copilot import token_tracking
from backend.copilot.config import ChatConfig
from backend.copilot.dream import apply as apply_mod
from backend.copilot.dream import (
    batch_callbacks,
    batch_deliveries,
    batch_submit,
    cancel,
    cleanup,
    guard,
)
from backend.copilot.dream import orchestrator as orchestrator_mod
from backend.copilot.dream import reaper
from backend.copilot.dream.fetch import DreamInput, EpisodeRow, FactRow
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.usage_activation import UsageActivationUnavailable, usage_keys
from backend.data.db import execute_raw_with_schema
from backend.data.dream_pass import get_dream_pass
from backend.data.redis_client import get_redis_async
from backend.executor.batch_executor import PendingEntry
from backend.util.json import SafeJson
from backend.util.llm.providers import BatchResultRow, BatchSubmissionRef

pytestmark = pytest.mark.asyncio(loop_scope="session")

_CONFIG = ChatConfig(
    fast_standard_model="anthropic/claude-sonnet-5",
    fast_advanced_model="anthropic/claude-opus-5",
    direct_anthropic_api_key="test-key",
)
_OUTPUTS = {
    "consolidate": {"facts": [{"content": "A likes B", "confidence": 0.8}]},
    "recombine": {"proposals": []},
    "sanitize": {"summary_for_user": "quiet night"},
}


class Owner(BaseModel):
    user_id: str
    trial_id: str


class Ledger(BaseModel):
    """What the account was charged: the cost rows (each one's trial), the
    trial's own cost, and the weekly usage counters, untrialed and paid."""

    row_trials: list[str | None]
    trial_cost: int
    weekly: int
    paid_weekly: int


@pytest.fixture
async def owner() -> AsyncIterator[Owner]:
    """A user in a card-verified trial; deleting it afterwards cascades to
    its trial, its activation, its passes and its cost rows."""
    user_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"dream-pass-attribution-{user_id}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
            "subscriptionTier": SubscriptionTier.TRIAL,
        }
    )
    trial = await SubscriptionTrial.prisma().create(
        data={
            "userId": user_id,
            "offer": Json({}),
            "stripeCustomerId": f"cus_{user_id}",
            "checkoutSuccessUrl": "https://example.invalid/ok",
            "checkoutCancelUrl": "https://example.invalid/no",
            "status": "trialing",
            "cardVerifiedAt": now,
            "consumedAt": now,
            "startedAt": now,
            "endsAt": now + timedelta(days=7),
        }
    )
    yield Owner(user_id=user_id, trial_id=trial.id)
    await _flush_cost_rows()
    await User.prisma().delete(where={"id": user_id})


@pytest.fixture
def batch_route() -> Iterator[list[PendingEntry]]:
    """A pass that takes the batch route, everything but the model and the
    provider's batch API real: the batches it submits, queued in the order
    they were submitted."""
    queued: list[PendingEntry] = []

    async def submit(**kwargs) -> BatchSubmissionRef:
        return BatchSubmissionRef(
            provider="anthropic",
            provider_batch_id=f"batch-{uuid.uuid4()}",
            custom_id=kwargs["custom_id"],
            submitted_at=datetime.now(UTC),
        )

    async def enqueue(entry: PendingEntry) -> None:
        queued.append(entry)

    applied = AsyncMock(
        return_value={"session_id": "s", "ingestion_drain_status": "drained"}
    )
    with ExitStack() as stack:
        for target, name, value in (
            (
                orchestrator_mod,
                "check_dream_budget",
                AsyncMock(return_value=(True, None)),
            ),
            (orchestrator_mod, "_route_for", AsyncMock(return_value="anthropic_batch")),
            (orchestrator_mod, "is_feature_enabled", AsyncMock(return_value=True)),
            (orchestrator_mod, "apply_operations", applied),
            (apply_mod, "apply_operations", applied),
            (guard, "evaluate_feature_flag", AsyncMock(return_value=(True, True))),
            (batch_submit, "call_provider", submit),
            (batch_submit, "enqueue_pending", enqueue),
            (cleanup, "provider_batch_stopped", AsyncMock(return_value=True)),
            (cancel, "cancel_provider_batch", AsyncMock(return_value=True)),
        ):
            stack.enter_context(patch.object(target, name, value))
        stack.enter_context(
            patch.object(batch_callbacks, "anthropic_api_key", return_value="test-key")
        )
        yield queued


async def test_the_reaper_charges_a_landed_phase_to_the_trial_it_ran_in(
    owner, batch_route
):
    """R1: consolidate lands, the account converts, and the pass dies; the
    reaper's cleanup charges consolidate to the trial, not the paid
    generation the account now reads."""
    pass_id = await _start(owner, batch_route)
    await _deliver(batch_route[0])
    await _convert(owner)

    await _reap(owner, pass_id)

    _assert_charged_to_the_trial(await _ledger(owner), owner, rows=1)


@pytest.mark.parametrize("row_attribution", [True, False], ids=["kept", "none"])
async def test_a_cancelled_pass_is_charged_once_by_its_row_after_an_unattributed_result(
    owner, batch_route, row_attribution
):
    """R2: consolidate lands and the account converts; the pass is
    cancelled, its next result comes back without the batch's copy of its
    attribution, and the delete of its state fails, so the reaper cleans up
    after it. Consolidate is charged once, to the trial, by the row; never
    when the row keeps no attribution."""
    pass_id = await _start(owner, batch_route)
    await _deliver(batch_route[0])
    if not row_attribution:
        await _forget_the_attribution(pass_id)
    await _convert(owner)
    cancelled = await cancel.cancel_dream_pass(
        pass_id, user_id=owner.user_id, reason="testing"
    )
    assert cancelled.cancelled

    await _deliver_unattributed(batch_route[1])
    await _reap(owner, pass_id)

    ledger = await _ledger(owner)
    if row_attribution:
        _assert_charged_to_the_trial(ledger, owner, rows=1)
    else:
        assert ledger == _CHARGED_NOTHING


@pytest.mark.parametrize("row_attribution", [True, False], ids=["kept", "none"])
async def test_a_dropped_batch_of_an_unattributed_pass_is_charged_once_by_its_row(
    owner, batch_route, row_attribution
):
    """R2: consolidate lands, the account converts, and the next result
    comes back without the batch's copy of the attribution: the pass ends
    ERRORED, the delete of its state failing. The executor then drops the
    closed pass's batch. Consolidate is charged once, to the trial, by the
    row; never when the row keeps no attribution."""
    pass_id = await _start(owner, batch_route)
    await _deliver(batch_route[0])
    if not row_attribution:
        await _forget_the_attribution(pass_id)
    await _convert(owner)

    await _deliver_unattributed(batch_route[1])
    await batch_deliveries.drop_closed(batch_route[1])

    row = await get_dream_pass(pass_id)
    assert row is not None and row.status.value == "ERRORED"
    ledger = await _ledger(owner)
    if row_attribution:
        _assert_charged_to_the_trial(ledger, owner, rows=1)
    else:
        assert ledger == _CHARGED_NOTHING


async def test_a_pass_whose_row_keeps_no_attribution_is_never_charged(
    owner, batch_route
):
    """A pass in flight when rows began keeping the attribution: consolidate
    landed before, and its row keeps none. Its next result ends it unapplied,
    and its landed phase is charged nothing, then or by the reaper after the
    account converts."""
    pass_id = await _start(owner, batch_route)
    await _deliver(batch_route[0])
    await _forget_the_attribution(pass_id)

    with pytest.raises(UsageActivationUnavailable):
        await _deliver(batch_route[1])
    await _convert(owner)
    await _reap(owner, pass_id)

    row = await get_dream_pass(pass_id)
    assert row is not None and row.status.value == "ERRORED"
    assert await _ledger(owner) == _CHARGED_NOTHING


async def _start(owner: Owner, queued: list[PendingEntry]) -> str:
    """Run a pass to its handoff: its row and its first batch, consolidate's."""
    with patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input(owner.user_id)),
    ):
        result = await orchestrator_mod._execute_dream_pass_async(
            owner.user_id, expert_id=None, config=_CONFIG
        )
    assert result.error is None and len(queued) == 1, result
    return result.pass_id


async def _deliver(entry: PendingEntry) -> None:
    """The phase's batch comes back with its output."""
    phase = entry.payload["phase"]
    await batch_callbacks.handle_dream_batch_result(
        entry,
        [
            BatchResultRow(
                custom_id=entry.payload["custom_ids"][0],
                content=json.dumps(_OUTPUTS[phase]),
                input_tokens=1000,
                output_tokens=100,
            )
        ],
    )


async def _deliver_unattributed(entry: PendingEntry) -> None:
    """The phase's batch comes back without the batch's copy of the pass's
    attribution, and the cleanup after the pass cannot delete its state."""
    entry.payload.pop("cost_context")
    with patch.object(cleanup, "_deleted", AsyncMock(return_value=False)):
        with pytest.raises(UsageActivationUnavailable):
            await _deliver(entry)


async def _reap(owner: Owner, pass_id: str) -> None:
    """The reaper's cleanup after the pass, as its run does it."""
    row = await get_dream_pass(pass_id)
    assert row is not None
    await reaper._clean_up(row, MemoryScope.for_user(owner.user_id), release=True)


async def _forget_the_attribution(pass_id: str) -> None:
    """The row as one written before rows kept the attribution."""
    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"DreamPass" SET "costContext" = NULL WHERE "id" = $1',
        pass_id,
    )


async def _convert(owner: Owner) -> None:
    """The account converts: paid, its trial converted, a ready paid usage
    generation."""
    now = datetime.now(UTC)
    await User.prisma().update(
        where={"id": owner.user_id}, data={"subscriptionTier": SubscriptionTier.PRO}
    )
    await SubscriptionTrial.prisma().update(
        where={"id": owner.trial_id}, data={"convertedAt": now, "status": "active"}
    )
    await PaidUsageActivation.prisma().create(
        data={
            "userId": owner.user_id,
            "stripeSubscriptionId": f"sub_{owner.user_id}",
            "stripeInvoiceId": f"inv_{owner.user_id}",
            "readyAt": now,
        }
    )


async def _ledger(owner: Owner) -> Ledger:
    await _flush_cost_rows()
    rows = await PlatformCostLog.prisma().find_many(where={"userId": owner.user_id})
    trial = await SubscriptionTrial.prisma().find_unique(where={"id": owner.trial_id})
    activation = await PaidUsageActivation.prisma().find_unique(
        where={"userId": owner.user_id}
    )
    redis = await get_redis_async()
    now = datetime.now(UTC)
    weekly = await redis.get(usage_keys(owner.user_id, None, now)[1])
    paid_weekly = (
        await redis.get(usage_keys(owner.user_id, activation.id, now)[1])
        if activation is not None
        else None
    )
    return Ledger(
        row_trials=[
            dict(row.metadata or {}).get("subscription_trial_id") for row in rows
        ],
        trial_cost=int(trial.costMicrodollars) if trial is not None else 0,
        weekly=int(weekly or 0),
        paid_weekly=int(paid_weekly or 0),
    )


def _assert_charged_to_the_trial(ledger: Ledger, owner: Owner, *, rows: int) -> None:
    """*rows* cost rows, each the trial's; the trial and its counters charged,
    the paid generation the account converted to untouched."""
    assert ledger.row_trials == [owner.trial_id] * rows, ledger
    assert ledger.trial_cost > 0 and ledger.weekly > 0, ledger
    assert ledger.paid_weekly == 0, ledger


_CHARGED_NOTHING = Ledger(row_trials=[], trial_cost=0, weekly=0, paid_weekly=0)


async def _flush_cost_rows() -> None:
    """The cost rows the charges scheduled, written."""
    tasks = list(token_tracking._pending_log_tasks)
    if tasks:
        await asyncio.gather(*tasks)


def _input(user_id: str) -> DreamInput:
    now = datetime.now(UTC)
    return DreamInput(
        user_id=user_id,
        group_id=MemoryScope.for_user(user_id).group_id,
        window_start=now - timedelta(days=1),
        window_end=now,
        episodes=[
            EpisodeRow(
                uuid="e1",
                name=None,
                content="hello",
                source_description=None,
                valid_at=None,
                created_at=None,
            )
        ],
        facts=[
            FactRow(
                uuid="f1",
                source="A",
                target="B",
                name="likes",
                fact="A likes B",
                scope="real:global",
                confidence=0.7,
                status="active",
                created_at=None,
            )
        ],
        recent_sessions=[],
        known_fact_uuids={"f1"},
        known_episode_uuids={"e1"},
    )
