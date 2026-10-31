"""A chat-message insert on an id that is already saved is the idempotency
signal: every writer must recognise it and skip quietly, not retry it as a
``(sessionId, sequence)`` race or fail the request.

The mock-only tests build the error in the two shapes it reaches us in: from
the local Prisma client (``code`` and ``meta`` set) and after the database
manager's RPC (message only). The ``server`` tests insert real rows, so a
Prisma upgrade that changes the error format fails here.
"""

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import User

from backend.copilot import db as copilot_db
from backend.copilot import turn_queue
from backend.copilot.model import ChatMessage, append_and_save_message
from backend.data.db import prisma as db_client
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)

_ID_MESSAGE = "Unique constraint failed on the fields: (`id`)"
_SEQUENCE_MESSAGE = "Unique constraint failed on the fields: (`sessionId`,`sequence`)"


def _local_error(message: str, target: list[str]) -> UniqueViolationError:
    return UniqueViolationError(
        {
            "user_facing_error": {
                "message": message,
                "error_code": "P2002",
                "meta": {"target": target},
            }
        }
    )


def _rpc_error(message: str) -> UniqueViolationError:
    """The error as ``backend.util.service`` rebuilds it after the RPC."""
    return UniqueViolationError({"user_facing_error": {"message": message}})


@asynccontextmanager
async def _held_lock(_session_id: str):
    yield True


# ---------- the helper ----------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_local_error(_ID_MESSAGE, ["id"]), True),
        (_rpc_error(_ID_MESSAGE), True),
        (_local_error("An error occurred", ["id"]), True),
        (_local_error(_SEQUENCE_MESSAGE, ["sessionId", "sequence"]), False),
        (_rpc_error(_SEQUENCE_MESSAGE), False),
        (_local_error(_ID_MESSAGE, ["sessionId", "sequence"]), False),
        (_rpc_error("An error occurred while processing data."), False),
        (ValueError(_ID_MESSAGE), False),
    ],
    ids=[
        "local-id",
        "rpc-id",
        "local-id-meta-only",
        "local-sequence",
        "rpc-sequence",
        "meta-wins-over-message",
        "rpc-no-fields",
        "not-a-unique-violation",
    ],
)
def test_is_duplicate_chat_message_id_error(exc: BaseException, expected: bool):
    assert copilot_db.is_duplicate_chat_message_id_error(exc) is expected


# ---------- the writers, mocked ----------


@pytest.mark.asyncio
async def test_batch_raises_a_duplicate_id_at_once():
    create_many = AsyncMock(side_effect=_local_error(_ID_MESSAGE, ["id"]))
    next_sequence = AsyncMock(return_value=5)
    with (
        patch.object(
            PrismaChatMessage, "prisma", return_value=AsyncMock(create_many=create_many)
        ),
        patch.object(PrismaChatSession, "prisma", return_value=AsyncMock()),
        patch("backend.copilot.db.db.transaction") as transaction,
        patch("backend.copilot.db.get_next_sequence", new=next_sequence),
    ):
        transaction.return_value.__aenter__ = AsyncMock(return_value=None)
        transaction.return_value.__aexit__ = AsyncMock(return_value=None)
        with pytest.raises(UniqueViolationError):
            await copilot_db.add_chat_messages_batch(
                session_id="s1",
                messages=[{"id": "m1", "role": "user", "content": "hi"}],
                start_sequence=0,
            )

    create_many.assert_awaited_once()
    next_sequence.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_still_retries_a_sequence_collision():
    create_many = AsyncMock(
        side_effect=[_rpc_error(_SEQUENCE_MESSAGE), None],
    )
    with (
        patch.object(
            PrismaChatMessage, "prisma", return_value=AsyncMock(create_many=create_many)
        ),
        patch.object(PrismaChatSession, "prisma", return_value=AsyncMock()),
        patch("backend.copilot.db.db.transaction") as transaction,
        patch("backend.copilot.db.get_next_sequence", new=AsyncMock(return_value=5)),
    ):
        transaction.return_value.__aenter__ = AsyncMock(return_value=None)
        transaction.return_value.__aexit__ = AsyncMock(return_value=None)
        used = await copilot_db.add_chat_messages_batch(
            session_id="s1",
            messages=[{"role": "user", "content": "hi"}],
            start_sequence=0,
        )

    assert used == 5
    assert create_many.await_count == 2


@pytest.mark.asyncio
async def test_expert_run_post_skips_a_duplicate_id_without_retrying():
    add_message = AsyncMock(side_effect=_local_error(_ID_MESSAGE, ["id"]))
    session = PrismaChatSession.model_construct(id="sess-1")
    with (
        patch.object(
            PrismaChatMessage,
            "prisma",
            return_value=AsyncMock(find_unique=AsyncMock(return_value=None)),
        ),
        patch.object(
            PrismaChatSession,
            "prisma",
            return_value=AsyncMock(find_first=AsyncMock(return_value=session)),
        ),
        patch("backend.copilot.db.add_chat_message", new=add_message),
        patch("backend.copilot.db.get_next_sequence", new=AsyncMock(return_value=3)),
        patch("backend.copilot.db._get_session_lock", new=_held_lock),
        patch("backend.copilot.db.invalidate_session_cache", new=AsyncMock()),
    ):
        result = await copilot_db.append_expert_run_message(
            user_id="u1", expert_id="e1", content="done", message_id="m1"
        )

    assert result is None
    add_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_append_and_save_message_skips_a_duplicate_id_from_the_rpc():
    """Outside the database manager the batch insert's error arrives through
    the RPC with only its message."""
    from backend.copilot import model as copilot_model

    db = AsyncMock()
    db.get_next_sequence = AsyncMock(return_value=1)
    db.add_chat_messages_batch = AsyncMock(side_effect=_rpc_error(_ID_MESSAGE))
    session = copilot_model.ChatSession.new(user_id="u1", dry_run=False)
    with (
        patch.object(copilot_model, "chat_db", return_value=db),
        patch.object(copilot_model, "_get_session_lock", new=_held_lock),
        patch.object(
            copilot_model, "get_chat_session", new=AsyncMock(return_value=session)
        ),
        patch.object(copilot_model, "cache_chat_session", new=AsyncMock()),
    ):
        result = await append_and_save_message(
            session.session_id,
            ChatMessage(id="m1", role="user", content="hi"),
            dedupe_content=False,
        )

    assert result is None
    db.add_chat_messages_batch.assert_awaited_once()


# ---------- real inserts ----------


@pytest_asyncio.fixture
async def database(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a closed
    loop; only the first query on the new loop fails. Spend it here."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise


async def _create_user(user_id: str) -> None:
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"duplicate-id-{user_id}@example.com",
            "name": "Duplicate Id Test",
        }
    )


async def _cleanup(user_id: str) -> None:
    try:
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


async def _new_session(user_id: str) -> str:
    created = await copilot_db.create_chat_session(
        session_id=str(uuid4()), user_id=user_id
    )
    return created.session_id


async def _insert(session_id: str, message_id: str, sequence: int) -> None:
    await copilot_db.add_chat_message(
        session_id=session_id,
        role="user",
        sequence=sequence,
        content="hi",
        message_id=message_id,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_real_duplicate_id_is_told_apart_from_a_sequence_collision(
    database: None,
):
    user_id = f"duplicate-id-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await _new_session(user_id)
        message_id = str(uuid4())
        await _insert(session_id, message_id, sequence=0)

        with pytest.raises(UniqueViolationError) as same_id:
            await _insert(session_id, message_id, sequence=1)
        assert copilot_db.is_duplicate_chat_message_id_error(same_id.value)
        # The database manager's RPC keeps only the message.
        assert copilot_db.is_duplicate_chat_message_id_error(
            _rpc_error(str(same_id.value))
        )

        with pytest.raises(UniqueViolationError) as same_sequence:
            await _insert(session_id, str(uuid4()), sequence=0)
        assert not copilot_db.is_duplicate_chat_message_id_error(same_sequence.value)
        assert not copilot_db.is_duplicate_chat_message_id_error(
            _rpc_error(str(same_sequence.value))
        )
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_real_duplicate_id_in_a_batch_is_not_retried(database: None):
    user_id = f"duplicate-id-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await _new_session(user_id)
        message_id = str(uuid4())
        await _insert(session_id, message_id, sequence=0)

        next_sequence = AsyncMock(wraps=copilot_db.get_next_sequence)
        with (
            patch("backend.copilot.db.get_next_sequence", new=next_sequence),
            pytest.raises(UniqueViolationError),
        ):
            await copilot_db.add_chat_messages_batch(
                session_id=session_id,
                messages=[{"id": message_id, "role": "user", "content": "hi"}],
                start_sequence=1,
            )
        next_sequence.assert_not_awaited()
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_skips_a_real_duplicate_id(
    database: None,
):
    """The in-memory id check misses when the session it reads is stale (the
    lock degraded, or a writer on another pod); the insert must still dedupe."""
    from backend.copilot import model as copilot_model

    user_id = f"duplicate-id-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await _new_session(user_id)
        stale = await copilot_model.get_chat_session(session_id, user_id)
        assert stale is not None
        message_id = str(uuid4())
        await _insert(session_id, message_id, sequence=0)

        with (
            patch.object(
                copilot_model, "get_chat_session", new=AsyncMock(return_value=stale)
            ),
            patch.object(
                copilot_model,
                "_get_session_from_db",
                new=AsyncMock(return_value=stale),
            ),
        ):
            result = await append_and_save_message(
                session_id,
                ChatMessage(id=message_id, role="user", content="hi again"),
                dedupe_content=False,
            )

        assert result is None
        rows = await PrismaChatMessage.prisma().find_many(
            where={"sessionId": session_id}
        )
        assert [(r.id, r.content) for r in rows] == [(message_id, "hi")]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_enqueue_turn_skips_a_real_duplicate_id(database: None):
    user_id = f"duplicate-id-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await _new_session(user_id)
        message_id = str(uuid4())

        first = await turn_queue.enqueue_turn(
            user_id=user_id, session_id=session_id, message="hi", message_id=message_id
        )
        second = await turn_queue.enqueue_turn(
            user_id=user_id, session_id=session_id, message="hi", message_id=message_id
        )

        assert first is not None
        assert second is None
        assert (
            await PrismaChatMessage.prisma().count(where={"sessionId": session_id}) == 1
        )
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_session_note_skips_a_real_duplicate_id_without_retrying(
    database: None,
):
    """Past the ``find_unique`` pre-check (a concurrent double fire), the
    insert itself must dedupe."""
    user_id = f"duplicate-id-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await _new_session(user_id)
        message_id = str(uuid4())
        await _insert(session_id, message_id, sequence=0)

        add_message = AsyncMock(wraps=copilot_db.add_chat_message)
        with (
            patch.object(
                type(PrismaChatMessage.prisma()),
                "find_unique",
                new=AsyncMock(return_value=None),
            ),
            patch("backend.copilot.db.add_chat_message", new=add_message),
        ):
            posted = await copilot_db.append_session_message(
                session_id=session_id,
                user_id=user_id,
                content="note",
                message_id=message_id,
            )

        assert posted is False
        add_message.assert_awaited_once()
    finally:
        await _cleanup(user_id)
