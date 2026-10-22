"""A hard forget's writes to the DreamPass table against a real database
(``copilot/dream/forget_copies.py``): which of a scope's passes
``erase_dream_passes`` marks and blanks (a pass from before read sets were
recorded with any forget in its scope), never another user's or scope's,
and returns again once marked; every error kept only as a fixed code; its
statement timeout; its cost over 3,000 passes; and every transition that
closes a marked pass blanking it in the same statement
(``dream_pass_update.py``), one that leaves it open or an unmarked pass
untouched, the read set only growing, and none of them touching a pass's
usage attribution."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from prisma.models import DreamPass, User

from backend.copilot.dream.pass_errors import UNRECORDED
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
from backend.data import db, dream_pass
from backend.util.json import SafeJson

from .dream_pass import (
    ERASE_STATEMENT_TIMEOUT_MS,
    create_dream_pass,
    erase_dream_passes,
    get_dream_pass,
    update_dream_pass,
)
from .dream_pass_models import (
    ERASED_TEXT,
    UNKNOWN_READS,
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
    cost_context: dict[str, Any] | None = None,
) -> str:
    """A pass that read ``cites`` and consolidated ``_SENTENCE`` citing it,
    then moved to ``status`` (closed with no cleanup pending, for a closed
    status)."""
    draft = DreamPassDraft(
        id=str(uuid.uuid4()),
        user_id=owner,
        scope_key=scope_key,
        route=DreamPassRoute.SYNC,
        trigger=DreamPassTrigger.CRON,
        started_at=_NOW,
        cost_context=cost_context,
    )
    await create_dream_pass(draft)
    assert await update_dream_pass(
        draft.id,
        DreamPassUpdate(
            read_uuids=[cites],
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


async def _legacy_pass(owner: str, scope_key: str) -> str:
    """A closed pass written as a release without read sets wrote one: its
    insert names no read set, so the column's default applies, as it does to
    every row that existed when the column was added."""
    pass_id = str(uuid.uuid4())
    await db.execute_raw_with_schema(
        'INSERT INTO {schema_prefix}"DreamPass" ("id", "userId", "scopeKey", '
        '"route", "trigger", "status", "phase", "updatedAt") '
        "VALUES ($1, $2, $3, 'SYNC', 'CRON', 'RUNNING', 'GATHER', now())",
        pass_id,
        owner,
        scope_key,
    )
    assert await update_dream_pass(
        pass_id,
        DreamPassUpdate(
            phase_outputs=_outputs(_SENTENCE, "fact-unrelated"),
            operations=_operations(_SENTENCE),
        ),
    )
    await DreamPass.prisma().update(
        where={"id": pass_id},
        data={"status": DreamPassStatus.COMPLETE, "phase": DreamPassPhase.DONE},
    )
    return pass_id


async def _text(pass_id: str) -> str:
    row = await get_dream_pass(pass_id)
    assert row is not None
    return row.phase_outputs.model_dump_json() + row.operations.model_dump_json()


async def test_a_closed_pass_that_read_a_reached_uuid_is_marked_and_blanked(
    make_user,
):
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


async def test_a_pass_from_before_read_sets_goes_with_any_forget_in_its_scope(
    make_user,
):
    """What such a pass read is unknown, so it may rest on anything: any hard
    forget in its scope marks it and blanks its text, whatever it reached and
    whatever its writes cite. One with a known read set holding none of it
    is left, and so is an unknown one of another scope or user."""
    owner, stranger = await make_user(), await make_user()
    legacy = await _legacy_pass(owner, owner)
    other_scope = await _legacy_pass(owner, f"{owner}:expert")
    theirs = await _legacy_pass(stranger, stranger)
    known = await _pass(
        owner, owner, status=DreamPassStatus.COMPLETE, cites="fact-unrelated"
    )
    row = await get_dream_pass(legacy)
    assert row is not None and row.read_uuids == [UNKNOWN_READS]

    marked = await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[])

    row = await get_dream_pass(legacy)
    assert marked == [legacy] and row is not None and row.erased_at is not None
    assert _SENTENCE not in await _text(legacy)
    for pass_id in (other_scope, theirs, known):
        row = await get_dream_pass(pass_id)
        assert row is not None and row.erased_at is None
        assert _SENTENCE in await _text(pass_id)


async def test_a_row_whose_writer_names_no_read_set_may_have_read_anything(
    make_user,
):
    """The column's default, which every row that existed when the column
    was added reads, and every row a release without it writes during a
    rollout: ``UNKNOWN_READS``."""
    owner = await make_user()

    row = await get_dream_pass(await _legacy_pass(owner, owner))

    assert row is not None and row.read_uuids == [UNKNOWN_READS]


async def test_a_pass_this_release_creates_has_read_nothing_until_it_gathers(
    make_user,
):
    """Its create states its read set, so it never reads ``UNKNOWN_READS``:
    gathering adds what it read, and only that reaches it once it closes."""
    owner = await make_user()
    draft = DreamPassDraft(
        id=str(uuid.uuid4()),
        user_id=owner,
        scope_key=owner,
        route=DreamPassRoute.SYNC,
        trigger=DreamPassTrigger.CRON,
        started_at=_NOW,
    )

    created = await create_dream_pass(draft)
    assert created.read_uuids == []
    assert await update_dream_pass(draft.id, DreamPassUpdate(read_uuids=["fact-a"]))
    await DreamPass.prisma().update(
        where={"id": draft.id},
        data={"status": DreamPassStatus.COMPLETE, "phase": DreamPassPhase.DONE},
    )

    row = await get_dream_pass(draft.id)
    assert row is not None and row.read_uuids == ["fact-a"]
    assert await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]) == []


async def test_every_error_of_the_scope_is_kept_only_as_a_fixed_code(make_user):
    """An error the store did not write (it records codes only) may quote
    anything and names no uuid: every pass of the scope, reached or not and
    whatever its status, keeps its error only when it is one of the fixed
    codes, its stage when that is one; another scope's error stays. The
    review's probe, ``apply: SecretToken123``, looks like a code and is not."""
    owner = await make_user()
    errored = DreamPassStatus.ERRORED
    old = await _pass(owner, owner, status=errored, cites="x")
    coded = await _pass(owner, owner, status=errored, cites="x")
    lookalike = await _pass(owner, owner, status=errored, cites="x")
    bare = await _pass(owner, owner, status=errored, cites="x")
    stopped = await _pass(owner, owner, status=DreamPassStatus.CANCELLED, cites="x")
    elsewhere = await _pass(owner, f"{owner}:expert", status=errored, cites="x")
    quoted = f"recombine: LLM returned non-JSON content — first 200 chars: {_SENTENCE}"
    for pass_id, error in (
        (old, quoted),
        (coded, "consolidate: schema_mismatch"),
        (lookalike, "apply: SecretToken123"),
        (bare, f"1 validation error: input_value='{_SENTENCE}'"),
        (stopped, f"cancelled by an admin: {_SENTENCE}"),
        (elsewhere, quoted),
    ):
        await DreamPass.prisma().update(where={"id": pass_id}, data={"error": error})

    assert await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]) == []

    errors = {
        pass_id: (row.error if (row := await get_dream_pass(pass_id)) else None)
        for pass_id in (old, coded, lookalike, bare, stopped, elsewhere)
    }
    assert errors == {
        old: f"recombine: {ERASED_TEXT}",
        coded: "consolidate: schema_mismatch",
        lookalike: f"apply: {ERASED_TEXT}",
        bare: ERASED_TEXT,
        stopped: ERASED_TEXT,
        elsewhere: quoted,
    }


async def test_an_error_that_is_not_a_fixed_code_is_never_recorded(make_user):
    owner = await make_user()
    pass_id = await _pass(owner, owner)

    assert await update_dream_pass(
        pass_id, failed(f"recombine: the model said {_SENTENCE}", None, _NOW)
    )

    row = await get_dream_pass(pass_id)
    assert row is not None and row.error == UNRECORDED


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
                "readUuids": [],
            }
            for _ in range(1000)
        ]
    )

    assert await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]) == [
        oldest
    ]
    assert _SENTENCE not in await _text(oldest)


async def test_a_pass_marked_before_is_returned_and_not_written_again(make_user):
    """The round-2 review's retry: a later forget, whatever it reached,
    returns every pass marked before so its narrative is blanked again, and
    leaves the row as it was (it was blanked as it closed or was marked)."""
    owner = await make_user()
    pass_id = await _pass(owner, owner, status=DreamPassStatus.COMPLETE)
    assert await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]) == [
        pass_id
    ]
    first = await get_dream_pass(pass_id)

    again = await erase_dream_passes(owner, owner, reached=["fact-other"], named=[])

    row = await get_dream_pass(pass_id)
    assert again == [pass_id]
    assert first is not None and row is not None
    assert row == first, "not even updatedAt moves"


async def test_the_read_set_only_grows_and_outlives_the_bundle(make_user):
    owner = await make_user()
    pass_id = await _pass(owner, owner, cites="fact-a")

    assert await update_dream_pass(
        pass_id, DreamPassUpdate(read_uuids=["fact-c", "fact-a", "episode-b"])
    )
    assert await update_dream_pass(pass_id, _CLOSES["applied"])

    row = await get_dream_pass(pass_id)
    assert row is not None and row.read_uuids == ["episode-b", "fact-a", "fact-c"]


async def test_an_erase_waiting_on_a_locked_row_gives_up_at_its_timeout(
    make_user, monkeypatch
):
    """The database cancels the statement itself, just under the store's
    deadline, rather than leaving it running after the caller gave up; the
    forget reports a cleanup error to repeat."""
    monkeypatch.setattr(dream_pass, "ERASE_STATEMENT_TIMEOUT_MS", 300)
    owner = await make_user()
    pass_id = await _pass(owner, owner, status=DreamPassStatus.COMPLETE)

    async with db.transaction() as tx:
        await db.query_raw_with_schema(
            'SELECT 1 FROM {schema_prefix}"DreamPass" WHERE "id" = $1 FOR UPDATE',
            pass_id,
            client=tx,
        )
        started = time.monotonic()
        with pytest.raises(Exception, match="statement timeout"):
            await asyncio.wait_for(
                erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[]), 10
            )
        assert time.monotonic() - started < 5

    row = await get_dream_pass(pass_id)
    assert row is not None and row.erased_at is None, "nothing was written"


async def test_the_erase_over_3000_passes_and_1000_reached_uuids_is_in_time(
    make_user,
):
    """The round-2 review's cost check, by the gather's caps: 3,000 closed
    passes of a scope, each with ~10 KB of phase output and 550 uuids read,
    against 1,000 reached uuids, first read by none of them (the selection's
    cost), then one read by every pass (all 3,000 marked and blanked in one
    statement), well inside the statement's timeout."""
    owner = await make_user()
    shared = "fact-read-by-every-pass"
    facts = [
        {
            "content": "A sample retained dream sentence. " * 12,
            "confidence": 0.9,
            "source_fact_uuids": [str(uuid.uuid4())],
        }
        for _ in range(20)
    ]
    await db.execute_raw_with_schema(
        'INSERT INTO {schema_prefix}"DreamPass" ("id", "userId", "scopeKey", '
        '"route", "trigger", "status", "phase", "phaseOutputs", "readUuids") '
        "SELECT gen_random_uuid()::text, $1, $1, 'SYNC', 'CRON', 'COMPLETE', "
        "'DONE', $2::jsonb, ARRAY(SELECT gen_random_uuid()::text "
        "FROM generate_series(1, 549) WHERE pass.n > 0) || $3::text "
        "FROM generate_series(1, 3000) AS pass(n)",
        owner,
        json.dumps({"consolidate": {"facts": facts}}),
        shared,
    )
    budget = ERASE_STATEMENT_TIMEOUT_MS / 1000 / 2
    unread = [str(uuid.uuid4()) for _ in range(1000)]

    started = time.monotonic()
    assert await erase_dream_passes(owner, owner, reached=unread, named=[]) == []
    selection = time.monotonic() - started
    started = time.monotonic()
    marked = await erase_dream_passes(
        owner, owner, reached=[*unread[1:], shared], named=[]
    )
    all_marked = time.monotonic() - started

    assert len(marked) == 3000
    assert selection < budget and all_marked < budget, (selection, all_marked)
    blanked = await DreamPass.prisma().count(
        where={"userId": owner, "erasedAt": {"not": None}}
    )
    assert blanked == 3000


_CLOSES: dict[str, DreamPassUpdate] = {
    "applied": DreamPassUpdate(
        status=DreamPassStatus.COMPLETE,
        phase=DreamPassPhase.DONE,
        operations=DreamPassOperations(
            applied=DreamPassApplied(summary_for_user=f"I noticed {_SENTENCE}.")
        ),
        completed_at=_NOW,
    ),
    "failed": failed("recombine: non_json_content", None, _NOW, marked=True),
    "expired": expired("expired: forced", not_updated_since=None),
    "reaped": reaped("recombine: reaped", not_updated_since=_NOW.replace(year=2100)),
}


@pytest.mark.parametrize("close", [*sorted(_CLOSES), "cancelled"])
async def test_every_transition_that_closes_a_marked_pass_blanks_it(make_user, close):
    """The pass's own end (applied or failed, the update's own text
    included), a cancel, an expiry and the reaper's close: each blanks the
    marked row in the statement that closes it."""
    owner = await make_user()
    pass_id = await _pass(owner, owner, cites="fact-unrelated")
    await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[])
    await DreamPass.prisma().update(
        where={"id": pass_id}, data={"error": f"consolidate: said {_SENTENCE}"}
    )
    update = _CLOSES.get(close) or cancelled("cancelled: by_admin", owner_user_id=owner)

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


async def test_a_hard_forget_leaves_a_passs_usage_attribution(make_user):
    """``costContext`` is the billing attribution a pass started under, not
    the dream's words: the erase of a closed pass, the blanking of a marked
    pass as it closes and the cleanup's end that blanks it again all leave
    it as it was."""
    owner = await make_user()
    attribution = {"user_id": owner, "trial_id": "trial-1", "generation": 3}
    closed = await _pass(
        owner, owner, status=DreamPassStatus.COMPLETE, cost_context=attribution
    )
    running = await _pass(owner, owner, cost_context=attribution)

    marked = await erase_dream_passes(owner, owner, reached=[_FORGOTTEN], named=[])
    assert await update_dream_pass(running, _CLOSES["failed"])
    assert await update_dream_pass(running, cleanup_finished())

    assert sorted(marked) == sorted([closed, running])
    for pass_id in (closed, running):
        row = await get_dream_pass(pass_id)
        assert row is not None and row.erased_at is not None
        assert _SENTENCE not in await _text(pass_id)
        assert row.cost_context == attribution
