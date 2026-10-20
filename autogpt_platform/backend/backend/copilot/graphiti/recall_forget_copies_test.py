"""Unit tests for the last step of a hard forget in ``recall_forget.retract``:
blanking the dream's copies of what it reached in Postgres
(``dream/forget_copies.py``), against a mock driver and the conftest's
``dream_copies`` stand-in for the blanking itself.

What the blanking does to a pass's row and narrative is
``dream/forget_copies_test.py``.
"""

from unittest.mock import AsyncMock

import pytest

from .memory_model import MemoryForgetFailureCode
from .recall_forget_fake import CLEANUP, NOTHING_DERIVED, SCOPE, SOFT, forget, scripted

_DREAM_WRITE = "dream_p1_consolidate_001"


def _hard_forget_driver() -> AsyncMock:
    """A hard forget of ``u1``: two episodes cite it, one of them a dream
    write the cascade erases, and nothing is derived from it."""
    redacted = [{"uuid": "ep1", "via": ["u1"]}, {"uuid": "ep2", "via": ["u1"]}]
    return scripted(
        *SOFT[:4],
        redacted,  # redact
        [],  # the cascade: no earlier try,
        redacted,  # the episodes citing the fact,
        [{"name": _DREAM_WRITE}],  # the dream's among them erased,
        [],  # nothing derived
        [],
        [{"uuid": "ep1", "content": "Alice works on Atlas"}],  # citing
        [{"uuid": "ep1"}],  # tombstone
        [{"uuid": "u1", "deleted_entities": ["alice"]}],  # delete
    )


@pytest.mark.asyncio
async def test_a_hard_forget_blanks_the_dreams_copies_of_what_it_reached(
    dream_copies: AsyncMock,
) -> None:
    result = await forget(_hard_forget_driver(), ["u1"], hard=True)

    assert (result.deleted, result.failures) == (["u1"], [])
    assert result.dream_writes == [_DREAM_WRITE]
    dream_copies.assert_awaited_once()
    assert dream_copies.await_args is not None
    (scope,) = dream_copies.await_args.args
    kwargs = dream_copies.await_args.kwargs
    assert scope == SCOPE
    assert set(kwargs["reached"]) == {"u1", "ep1", "ep2"}
    assert list(kwargs["dream_writes"]) == [_DREAM_WRITE]


@pytest.mark.asyncio
async def test_a_soft_forget_leaves_them(dream_copies: AsyncMock) -> None:
    await forget(scripted(*SOFT, *NOTHING_DERIVED), ["u1"])

    dream_copies.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_forget_that_found_nothing_leaves_them(
    dream_copies: AsyncMock,
) -> None:
    result = await forget(scripted([], [], [], [], [], []), ["u1"], hard=True)

    assert [f.code for f in result.failures] == [MemoryForgetFailureCode.NO_MATCH]
    dream_copies.assert_not_awaited()


@pytest.mark.asyncio
async def test_failing_to_blank_them_is_a_cleanup_error_on_each_fact(
    dream_copies: AsyncMock,
) -> None:
    """The fact is still purged and hidden; forgetting it again retries."""
    dream_copies.side_effect = ConnectionError("database down")

    result = await forget(_hard_forget_driver(), ["u1"], hard=True)

    assert result.deleted == ["u1"]
    assert [(f.uuid, f.code) for f in result.failures] == [("u1", CLEANUP)]
    assert "ConnectionError" in result.failures[0].reason
    assert "database down" not in result.failures[0].reason


@pytest.mark.asyncio
async def test_a_pass_still_running_is_no_failure(dream_copies: AsyncMock) -> None:
    """The pass is marked, and blanked in the statement that closes it
    (``dream/forget_copies.py``): nothing is left for the user to repeat."""
    dream_copies.return_value = ["p-running"]

    result = await forget(_hard_forget_driver(), ["u1"], hard=True)

    assert (result.deleted, result.failures) == (["u1"], [])
