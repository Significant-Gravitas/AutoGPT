"""A hard forget's writes to the DreamPass table against a real database
(``copilot/dream/forget_copies.py``): a closed pass's phase outputs and
operations replaced whole, never an open pass's or another user's, and a
user's passes read for one scope."""

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
from prisma.models import User

from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
)
from backend.util.json import SafeJson

from .dream_pass import (
    blank_dream_pass_text,
    create_dream_pass,
    get_dream_pass,
    list_dream_passes,
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

_NOW = datetime.now(timezone.utc).replace(microsecond=0)
_SENTENCE = "Alice runs the Atlas launch"


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


def _outputs(content: str) -> DreamPhaseOutputs:
    write = ConsolidatedFact(content=content, confidence=0.8, source_fact_uuids=["f1"])
    return DreamPhaseOutputs(
        consolidate=ConsolidationOutput(facts=[write]),
        sanitize=DreamOperations(writes=[write], summary_for_user=content),
    )


def _operations(content: str) -> DreamPassOperations:
    return DreamPassOperations(
        planned=DreamOperations(summary_for_user=content),
        applied=DreamPassApplied(consolidated_count=1, summary_for_user=content),
    )


async def _pass(owner: str, scope_key: str, *, closed: bool) -> str:
    """A pass that consolidated ``_SENTENCE``, finished when ``closed``."""
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
            phase_outputs=_outputs(_SENTENCE), operations=_operations(_SENTENCE)
        ),
    )
    if closed:
        assert await update_dream_pass(
            draft.id,
            DreamPassUpdate(
                status=DreamPassStatus.COMPLETE,
                phase=DreamPassPhase.DONE,
                completed_at=_NOW,
            ),
        )
    return draft.id


async def test_a_closed_passes_text_is_replaced_whole(make_user):
    owner = await make_user()
    pass_id = await _pass(owner, owner, closed=True)

    written = await blank_dream_pass_text(
        pass_id, owner, _outputs("[forgotten]"), _operations("[forgotten]")
    )

    row = await get_dream_pass(pass_id)
    assert written and row is not None
    assert row.phase_outputs == _outputs("[forgotten]")
    assert row.operations == _operations("[forgotten]")
    assert row.status is DreamPassStatus.COMPLETE, "nothing else moves"


async def test_an_open_pass_or_another_users_is_left(make_user):
    owner, stranger = await make_user(), await make_user()
    running = await _pass(owner, owner, closed=False)
    closed = await _pass(owner, owner, closed=True)

    assert not await blank_dream_pass_text(
        running, owner, _outputs("[forgotten]"), _operations("[forgotten]")
    )
    assert not await blank_dream_pass_text(
        closed, stranger, _outputs("[forgotten]"), _operations("[forgotten]")
    )
    for pass_id in (running, closed):
        row = await get_dream_pass(pass_id)
        assert row is not None and row.phase_outputs == _outputs(_SENTENCE)


async def test_a_users_passes_are_listed_for_one_scope(make_user):
    owner = await make_user()
    mine = await _pass(owner, f"{owner}:expert", closed=True)
    other = await _pass(owner, owner, closed=True)

    listed = await list_dream_passes(owner, scope_key=f"{owner}:expert")

    assert [row.id for row in listed] == [mine]
    assert {row.id for row in await list_dream_passes(owner)} == {mine, other}
