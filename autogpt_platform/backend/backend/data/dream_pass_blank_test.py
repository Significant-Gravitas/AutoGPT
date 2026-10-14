"""A hard forget's writes to the DreamPass table against a real database
(``copilot/dream/forget_copies.py``): which of a scope's passes
``erase_dream_passes`` marks and blanks, never another user's or scope's;
every failed pass's error kept to its code; and every transition that
closes a marked pass blanking it in the same statement
(``dream_pass_update.py``), one that leaves it open or an unmarked pass
untouched."""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from prisma.models import DreamPass, User

from backend.copilot.dream.pass_record import (
    cancelled,
    cleanup_finished,
    expired,
    failed,
    reaped,
)
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
)
from backend.util.json import SafeJson

from .dream_pass import (
    create_dream_pass,
    erase_dream_passes,
    get_dream_pass,
    update_dream_pass,
)
from .dream_pass_models import (
    ERASED_TEXT,
    DreamPassApplied,
    DreamPassDraft,
    DreamPassOperations,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

_NOW = datetime.now(timezone.utc).replace(microsecond=0)
_SENTENCE = "Alice runs the Atlas launch"
_FORGOTTEN = "fact-forgotten"


@pytest.fixture
async def make_user() -> AsyncIterator[Callable[[], Awaitable[str]]]:
    created: list[str] = []

    async def make() -> str:
        user_id = str(uuid.uuid4())
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"dream-blank-{user_id}@example.com",
                "topUpConfig": SafeJson({}),
                "timezone": "UTC",
            }
        )
        created.append(user_id)
        return user_id

    yield make
    await User.prisma().delete_many(where={"id": {"in": created}})


def _outputs(content: str, cites: str = _FORGOTTEN) -> DreamPhaseOutputs:
    write = ConsolidatedFact(content=content, confidence=0.8, source_fact_uuids=[cites])
    return DreamPhaseOutputs(
        consolidate=ConsolidationOutput(facts=[write]),
        sanitize=DreamOperations(writes=[write], summary_for_user=content),
    )


def _operations(content: str) -> DreamPassOperations:
    return DreamPassOperations(
        planned=DreamOperations(summary_for_user=content),
        applied=DreamPassApplied(consolidated_count=1, summary_for_user=content),
    )


async def _pass(
    owner: str,
    scope_key: str,
    *,
    status: DreamPassStatus = DreamPassStatus.RUNNING,
    cites: str = _FORGOTTEN,
) -> str:
    """A pass that consolidated ``_SENTENCE`` citing ``cites``, then moved
    to ``status`` (closed with no cleanup pending, for a closed status)."""
    draft = DreamPassDraft(
        id=str(uuid.uuid4()),
        user_id=owner,
        scope_key=scope_key,
        route=DreamPassRoute.SYNC,
        trigger=DreamPassTrigger.CRON,
        started_at=_NOW,
    )
    await create_dream_pass(draft)
    assert await update_dream_pass(
        draft.id,
        DreamPassUpdate(
            phase_outputs=_outputs(_SENTENCE, cites),
            operations=_operations(_SENTENCE),
        ),
    )
    if status != DreamPassStatus.RUNNING:
        await DreamPass.prisma().update(
            where={"id": draft.id},
            data={"status": status, "phase": DreamPassPhase.DONE},
        )
    return draft.id


async def _text(pass_id: str) -> str:
    row = await get_dream_pass(pass_id)
    assert row is not None
    return row.phase_outputs.model_dump_json() + row.operations.model_dump_json()


async def test_a_reached_closed_pass_is_marked_and_blanked(make_user):
    owner = await make_user()
    pass_id = await _pass(owner, owner, status=DreamPassStatus.COMPLETE)

    marked = await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[])

    row = await get_dream_pass(pass_id)
    assert marked == [pass_id] and row is not None and row.erased_at is not None
    assert _SENTENCE not in await _text(pass_id)
    assert row.phase_outputs.consolidate is not None
    [write] = row.phase_outputs.consolidate.facts
    assert (write.content, write.source_fact_uuids) == (ERASED_TEXT, [_FORGOTTEN])
    assert row.operations.applied is not None
    assert row.operations.applied.consolidated_count == 1, "counts stay"
    assert row.status is DreamPassStatus.COMPLETE, "nothing else moves"


async def test_an_open_pass_is_marked_and_kept_until_it_closes(make_user):
    """Its next phase reads its row; whatever it names, it is marked."""
    owner = await make_user()
    running = await _pass(owner, owner, cites="fact-unrelated")

    assert await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]) == [
        running
    ]

    row = await get_dream_pass(running)
    assert row is not None and row.erased_at is not None
    assert _SENTENCE in await _text(running)


async def test_a_named_pass_or_one_whose_cleanup_is_pending_is_marked(make_user):
    """A dream write the forget erased names its pass, which may cite nothing
    else it reached; a pass closed from outside with its cleanup pending may
    still be applying."""
    owner = await make_user()
    named = await _pass(
        owner, owner, status=DreamPassStatus.COMPLETE, cites="fact-unrelated"
    )
    pending = await _pass(
        owner, owner, status=DreamPassStatus.CANCELLED, cites="fact-unrelated"
    )
    await DreamPass.prisma().update(
        where={"id": pending}, data={"cleanupPendingAt": _NOW}
    )

    marked = await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[named])

    assert sorted(marked) == sorted([named, pending])
    for pass_id in (named, pending):
        assert _SENTENCE not in await _text(pass_id)


async def test_a_pass_not_reached_another_scopes_or_another_users_is_left(
    make_user,
):
    owner, stranger = await make_user(), await make_user()
    unreached = await _pass(
        owner, owner, status=DreamPassStatus.COMPLETE, cites="fact-unrelated"
    )
    other_scope = await _pass(owner, f"{owner}:expert", status=DreamPassStatus.COMPLETE)
    theirs = await _pass(stranger, stranger, status=DreamPassStatus.COMPLETE)
    their_open = await _pass(stranger, stranger)

    assert await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]) == []

    for pass_id in (unreached, other_scope, theirs, their_open):
        row = await get_dream_pass(pass_id)
        assert row is not None and row.erased_at is None
        assert _SENTENCE in await _text(pass_id)


async def test_every_failed_pass_of_the_scope_keeps_only_its_errors_code(make_user):
    """An error from before errors were codes may quote the model and names
    no uuid: every ERRORED pass of the scope loses what follows its code,
    reached or not; a code alone, and another scope's error, stay."""
    owner = await make_user()
    old = await _pass(owner, owner, status=DreamPassStatus.ERRORED, cites="x")
    coded = await _pass(owner, owner, status=DreamPassStatus.ERRORED, cites="x")
    bare = await _pass(owner, owner, status=DreamPassStatus.ERRORED, cites="x")
    elsewhere = await _pass(
        owner, f"{owner}:expert", status=DreamPassStatus.ERRORED, cites="x"
    )
    quoted = f"recombine: LLM returned non-JSON content — first 200 chars: {_SENTENCE}"
    for pass_id, error in (
        (old, quoted),
        (coded, "consolidate: schema_mismatch"),
        (bare, f"1 validation error: input_value='{_SENTENCE}'"),
        (elsewhere, quoted),
    ):
        await DreamPass.prisma().update(where={"id": pass_id}, data={"error": error})

    assert await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]) == []

    errors = {
        pass_id: (row.error if (row := await get_dream_pass(pass_id)) else None)
        for pass_id in (old, coded, bare, elsewhere)
    }
    assert errors == {
        old: f"recombine: {ERASED_TEXT}",
        coded: "consolidate: schema_mismatch",
        bare: ERASED_TEXT,
        elsewhere: quoted,
    }


async def test_every_pass_of_the_scope_is_reached_however_many(make_user):
    """The review's 1,001 passes: the affected one is the oldest, behind a
    thousand newer, and no page or limit leaves it out."""
    owner = await make_user()
    oldest = await _pass(owner, owner, status=DreamPassStatus.COMPLETE)
    await DreamPass.prisma().create_many(
        data=[
            {
                "id": str(uuid.uuid4()),
                "userId": owner,
                "scopeKey": owner,
                "route": DreamPassRoute.SYNC,
                "trigger": DreamPassTrigger.ADMIN,
                "status": DreamPassStatus.SKIPPED,
                "phase": DreamPassPhase.DONE,
            }
            for _ in range(1000)
        ]
    )

    assert await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]) == [
        oldest
    ]
    assert _SENTENCE not in await _text(oldest)


async def test_erasing_again_changes_nothing_more(make_user):
    owner = await make_user()
    pass_id = await _pass(owner, owner, status=DreamPassStatus.COMPLETE)
    await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[])
    first = await get_dream_pass(pass_id)

    await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[])

    again = await get_dream_pass(pass_id)
    assert first is not None and again is not None
    assert again.erased_at == first.erased_at
    assert (again.phase_outputs, again.operations) == (
        first.phase_outputs,
        first.operations,
    )


_CLOSES: dict[str, DreamPassUpdate] = {
    "applied": DreamPassUpdate(
        status=DreamPassStatus.COMPLETE,
        phase=DreamPassPhase.DONE,
        operations=DreamPassOperations(
            applied=DreamPassApplied(summary_for_user=f"I noticed {_SENTENCE}.")
        ),
        completed_at=_NOW,
    ),
    "failed": failed(f"recombine: non-JSON — {_SENTENCE}", None, _NOW, marked=True),
    "expired": expired("superseded", not_updated_since=None),
    "reaped": reaped("lease lapsed", not_updated_since=_NOW.replace(year=2100)),
}


@pytest.mark.parametrize("close", [*sorted(_CLOSES), "cancelled"])
async def test_every_transition_that_closes_a_marked_pass_blanks_it(make_user, close):
    """The pass's own end (applied or failed, the update's own text
    included), a cancel, an expiry and the reaper's close: each blanks the
    marked row in the statement that closes it."""
    owner = await make_user()
    pass_id = await _pass(owner, owner, cites="fact-unrelated")
    await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[])
    update = _CLOSES.get(close) or cancelled("owner cancelled", owner_user_id=owner)

    assert await update_dream_pass(pass_id, update)

    row = await get_dream_pass(pass_id)
    assert row is not None and row.status is not DreamPassStatus.RUNNING
    assert _SENTENCE not in await _text(pass_id)
    assert _SENTENCE not in (row.error or "")


async def test_the_cleanups_end_blanks_a_marked_pass_again(make_user):
    owner = await make_user()
    pass_id = await _pass(owner, owner, status=DreamPassStatus.COMPLETE)
    await DreamPass.prisma().update(
        where={"id": pass_id},
        data={"cleanupPendingAt": _NOW, "erasedAt": _NOW},
    )

    assert await update_dream_pass(pass_id, cleanup_finished())

    assert _SENTENCE not in await _text(pass_id)


async def test_an_update_that_leaves_a_marked_pass_open_keeps_its_text(make_user):
    """Its next phase reads what it wrote."""
    owner = await make_user()
    pass_id = await _pass(owner, owner)
    await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[])

    assert await update_dream_pass(
        pass_id,
        DreamPassUpdate(
            phase=DreamPassPhase.RECOMBINE,
            phase_outputs=DreamPhaseOutputs(
                consolidate=ConsolidationOutput(
                    facts=[
                        ConsolidatedFact(
                            content=f"{_SENTENCE}, again",
                            confidence=0.8,
                            source_fact_uuids=["x"],
                        )
                    ]
                )
            ),
        ),
    )

    assert f"{_SENTENCE}, again" in await _text(pass_id)


async def test_an_unmarked_pass_closes_as_it_did(make_user):
    owner = await make_user()
    pass_id = await _pass(owner, owner)

    assert await update_dream_pass(pass_id, _CLOSES["applied"])

    row = await get_dream_pass(pass_id)
    assert row is not None and row.erased_at is None
    assert _SENTENCE in await _text(pass_id)
