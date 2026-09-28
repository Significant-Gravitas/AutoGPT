"""The DreamPass table against a real database: the column mapping, the JSON
merges, the owner check, and that a finished pass's row is final."""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta, timezone

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from prisma.models import DreamPass as PrismaDreamPass
from prisma.models import Expert, User

from backend.copilot.dream.fetch import DreamInput, EpisodeRow, FactRow, SessionRow
from backend.copilot.dream.input_bundle import input_bundle_to_dict
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DemotionSummary,
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPassUsage,
    EntityInvalidationSummary,
    PhaseUsage,
    RecombinationOutput,
)
from backend.util.json import SafeJson

from .dream_pass import (
    create_dream_pass,
    get_dream_pass,
    get_dream_pass_for_user,
    list_dream_passes,
    list_open_dream_passes,
    update_dream_pass,
)
from .dream_pass_models import (
    DreamPassApplied,
    DreamPassDraft,
    DreamPassOperations,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

# Postgres keeps TIMESTAMP(3): whole milliseconds read back exactly.
_NOW = datetime.now(timezone.utc).replace(microsecond=0)


@pytest.fixture
async def make_user() -> AsyncIterator[Callable[[], Awaitable[str]]]:
    """Create throwaway users; deleting them afterwards cascades to their
    passes."""
    created: list[str] = []

    async def make() -> str:
        user_id = str(uuid.uuid4())
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"dream-pass-{user_id}@example.com",
                "topUpConfig": SafeJson({}),
                "timezone": "UTC",
            }
        )
        created.append(user_id)
        return user_id

    yield make
    await User.prisma().delete_many(where={"id": {"in": created}})


def _draft(user_id: str, **overrides) -> DreamPassDraft:
    fields = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "scope_key": user_id,
        "route": DreamPassRoute.SYNC,
        "trigger": DreamPassTrigger.CRON,
        "started_at": _NOW,
    }
    return DreamPassDraft(**{**fields, **overrides})


async def test_a_new_row_reads_back_running_and_gathering(make_user):
    owner = await make_user()
    draft = _draft(owner)

    created = await create_dream_pass(draft)

    fetched = await get_dream_pass(draft.id)
    assert fetched == created
    assert fetched is not None
    assert (fetched.status, fetched.phase, fetched.route, fetched.trigger) == (
        DreamPassStatus.RUNNING,
        DreamPassPhase.GATHER,
        DreamPassRoute.SYNC,
        DreamPassTrigger.CRON,
    )
    assert fetched.started_at == _NOW
    assert fetched.cancel_generation == 0
    assert fetched.phase_outputs == DreamPhaseOutputs()
    assert fetched.operations == DreamPassOperations()
    assert fetched.input_bundle is None and fetched.usage is None


async def test_an_update_writes_only_the_columns_it_sets(make_user):
    owner = await make_user()
    draft = _draft(owner)
    await create_dream_pass(draft)

    assert await update_dream_pass(
        draft.id,
        DreamPassUpdate(
            phase=DreamPassPhase.CONSOLIDATE,
            window_start=_NOW - timedelta(days=14),
            window_end=_NOW,
        ),
    )
    assert await update_dream_pass(
        draft.id,
        DreamPassUpdate(phase=DreamPassPhase.CONSOLIDATE, provider_batch_id="b1"),
    )

    row = await get_dream_pass(draft.id)
    assert row is not None
    assert row.phase is DreamPassPhase.CONSOLIDATE
    assert (row.window_start, row.window_end) == (_NOW - timedelta(days=14), _NOW)
    assert row.provider_batch_id == "b1"
    assert row.status is DreamPassStatus.RUNNING


async def test_phase_outputs_and_operations_merge_one_field_at_a_time(make_user):
    owner = await make_user()
    draft = _draft(owner)
    await create_dream_pass(draft)
    consolidated = ConsolidationOutput(
        facts=[ConsolidatedFact(content="Nick ships on Fridays", confidence=0.8)]
    )
    planned = DreamOperations(summary_for_user="clamped")
    applied = DreamPassApplied(
        consolidated_count=1,
        protected_demotions=2,
        indeterminate_demotion_writes=1,
        demotion_accounting_complete=False,
        dream_session_id="s1",
        snapshot=DreamOperationsSnapshot(
            demotions=[
                DemotionSummary(
                    edge_uuid="a",
                    reason="user_signal",
                    new_status="superseded",
                    applied=False,
                    indeterminate=True,
                )
            ],
            entity_invalidations=[
                EntityInvalidationSummary(
                    entity_uuid="hub", reason="stale_fact", indeterminate=True
                )
            ],
        ),
    )

    for update in (
        DreamPassUpdate(phase_outputs=DreamPhaseOutputs(consolidate=consolidated)),
        DreamPassUpdate(
            phase_outputs=DreamPhaseOutputs(recombine=RecombinationOutput())
        ),
        DreamPassUpdate(operations=DreamPassOperations(planned=planned)),
        DreamPassUpdate(operations=DreamPassOperations(applied=applied)),
    ):
        assert await update_dream_pass(draft.id, update)

    row = await get_dream_pass(draft.id)
    assert row is not None
    assert row.phase_outputs == DreamPhaseOutputs(
        consolidate=consolidated, recombine=RecombinationOutput()
    )
    assert row.operations == DreamPassOperations(planned=planned, applied=applied)


async def test_the_input_bundle_keeps_the_batch_paths_format(make_user):
    owner = await make_user()
    draft = _draft(owner, route=DreamPassRoute.ANTHROPIC_BATCH)
    await create_dream_pass(draft)
    bundle = DreamInput(
        user_id=owner,
        group_id=f"user_{owner}",
        window_start=_NOW - timedelta(days=14),
        window_end=_NOW,
        episodes=[
            EpisodeRow(
                uuid="e1",
                name="chat",
                content="I ship on Fridays",
                source_description=None,
                valid_at="2026-09-25T10:00:00Z",
                created_at=None,
            )
        ],
        facts=[
            FactRow(
                uuid="f1",
                source="Nick",
                target="Fridays",
                name="ships_on",
                fact="Nick ships on Fridays",
                scope="real:global",
                confidence=0.7,
                status="active",
                created_at=None,
                recall_count=3,
                last_recalled_at="2026-09-27T09:30:00.000000+00:00",
                prev_recalled_at="2026-09-20T18:00:00.000000+00:00",
            )
        ],
        recent_sessions=[
            SessionRow(session_id="s1", title="Release", created_at=_NOW, body="hi")
        ],
        known_fact_uuids={"f1"},
        known_episode_uuids={"e1"},
    )
    usage = DreamPassUsage(
        phases=[PhaseUsage(phase="consolidate", model="claude-sonnet-5")],
        total_cost_usd=0.01,
        discount_applied=0.5,
    )

    assert await update_dream_pass(
        draft.id, DreamPassUpdate(input_bundle=bundle, usage=usage)
    )

    raw = await PrismaDreamPass.prisma().find_unique(where={"id": draft.id})
    assert raw is not None
    assert raw.inputBundle == input_bundle_to_dict(bundle)
    row = await get_dream_pass(draft.id)
    assert row is not None
    assert row.input_bundle == bundle
    assert row.usage == usage


async def test_a_finished_pass_is_final(make_user):
    owner = await make_user()
    draft = _draft(owner)
    await create_dream_pass(draft)
    assert await update_dream_pass(
        draft.id,
        DreamPassUpdate(
            status=DreamPassStatus.COMPLETE,
            phase=DreamPassPhase.DONE,
            completed_at=_NOW,
        ),
    )

    assert not await update_dream_pass(
        draft.id,
        DreamPassUpdate(status=DreamPassStatus.ERRORED, error="late crash"),
    )
    assert not await update_dream_pass(
        draft.id,
        DreamPassUpdate(
            phase_outputs=DreamPhaseOutputs(consolidate=ConsolidationOutput())
        ),
    )
    row = await get_dream_pass(draft.id)
    assert row is not None
    assert (row.status, row.error) == (DreamPassStatus.COMPLETE, None)
    assert row.phase_outputs == DreamPhaseOutputs()


async def test_a_missing_pass_is_not_written():
    assert not await update_dream_pass(
        "no-such-pass", DreamPassUpdate(phase=DreamPassPhase.CONSOLIDATE)
    )
    assert not await update_dream_pass(
        "no-such-pass",
        DreamPassUpdate(
            phase_outputs=DreamPhaseOutputs(recombine=RecombinationOutput())
        ),
    )


async def test_only_the_owner_reads_a_pass(make_user):
    owner, stranger = await make_user(), await make_user()
    draft = _draft(owner)
    await create_dream_pass(draft)

    owned = await get_dream_pass_for_user(draft.id, owner)

    assert owned is not None and owned.id == draft.id
    assert await get_dream_pass_for_user(draft.id, stranger) is None
    assert await get_dream_pass_for_user("no-such-pass", owner) is None


async def test_open_passes_are_listed_per_scope(make_user):
    owner = await make_user()
    scope, other_scope = f"{owner}:a", f"{owner}:b"
    running = _draft(owner, scope_key=scope)
    submitted = _draft(owner, scope_key=scope, status=DreamPassStatus.SUBMITTED)
    finished = _draft(owner, scope_key=scope)
    elsewhere = _draft(owner, scope_key=other_scope)
    for draft in (running, submitted, finished, elsewhere):
        await create_dream_pass(draft)
    await update_dream_pass(
        finished.id, DreamPassUpdate(status=DreamPassStatus.SKIPPED)
    )

    open_passes = await list_open_dream_passes(scope)

    assert {row.id for row in open_passes} == {running.id, submitted.id}


async def test_a_users_passes_list_newest_first(make_user):
    owner = await make_user()
    drafts = [_draft(owner, scope_key=f"{owner}:{i}") for i in range(3)]
    for draft in drafts:
        await create_dream_pass(draft)

    rows = await list_dream_passes(owner)

    assert {row.id for row in rows} == {draft.id for draft in drafts}
    created = [row.created_at for row in rows]
    assert created == sorted(created, reverse=True)
    assert len(await list_dream_passes(owner, limit=1)) == 1


async def test_deleting_the_expert_or_the_user_removes_their_passes(make_user):
    owner = await make_user()
    expert = await Expert.prisma().create(
        data={
            "ownerUserId": owner,
            "name": "Scout",
            "role": "Research",
            "identity": "Finds things out.",
        }
    )
    experts_pass = _draft(owner, expert_id=expert.id, scope_key=f"{owner}:expert")
    accounts_pass = _draft(owner)
    await create_dream_pass(experts_pass)
    await create_dream_pass(accounts_pass)

    await Expert.prisma().delete(where={"id": expert.id})
    assert await get_dream_pass(experts_pass.id) is None
    assert await get_dream_pass(accounts_pass.id) is not None

    await User.prisma().delete(where={"id": owner})
    assert await get_dream_pass(accounts_pass.id) is None


async def test_text_loses_the_control_characters_postgres_rejects(make_user):
    owner = await make_user()
    draft = _draft(owner)
    await create_dream_pass(draft)

    assert await update_dream_pass(
        draft.id,
        DreamPassUpdate(status=DreamPassStatus.ERRORED, error="bad\x00byte"),
    )

    row = await get_dream_pass(draft.id)
    assert row is not None and row.error == "badbyte"
