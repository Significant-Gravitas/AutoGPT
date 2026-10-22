"""The scripted driver the forget's unit tests run ``recall_forget.retract``
against, with the in-memory write lock (``conftest.lock_redis``): it
answers in order, the graph's two reads of dream markers apart. Shared by
``recall_forget_test.py``, ``recall_forget_writes_test.py`` and
``recall_forget_cascade_test.py``; not collected by pytest.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from . import recall_forget, recall_reconcile
from .memory_model import ForgetResult, MemoryForgetFailureCode
from .scope import MemoryScope, write_lock_key

SCOPE = MemoryScope.for_user("user-abc")
LOCK = write_lock_key(SCOPE.group_id)
CLEANUP = MemoryForgetFailureCode.CLEANUP_ERROR
# The graph's two reads of dream markers, answered apart from the rest.
MARKER_READS = (recall_reconcile.MARKERS_QUERY, recall_reconcile.IN_FLIGHT_QUERY)


def scripted(*results, markers: object = None, in_flight: object = 0) -> AsyncMock:
    """A driver whose queries return (or raise) ``results`` in order. The
    two reads of the graph's dream markers answer apart: the one every
    forget makes first ``markers`` (none by default), the one of the dream
    writes still in flight ``in_flight`` (none)."""
    driver = AsyncMock()
    answers = iter(results)

    async def answer(query: str, **params):
        if query == recall_reconcile.MARKERS_QUERY:
            value = [] if markers is None else markers
        elif query == recall_reconcile.IN_FLIGHT_QUERY:
            value = (
                in_flight
                if isinstance(in_flight, Exception)
                else [{"count": in_flight}]
            )
        else:
            value = next(answers)
        if isinstance(value, Exception):
            raise value
        return value, [], None

    driver.execute_query.side_effect = answer
    return driver


async def forget(driver: AsyncMock, uuids: list[str], **kwargs) -> ForgetResult:
    with patch.object(recall_forget, "open_driver", MagicMock(return_value=driver)):
        return await recall_forget.retract(SCOPE, uuids, **kwargs)


def own_calls(driver: AsyncMock) -> list:
    """The forget's own queries: every one but its reads of dream markers."""
    return [
        call
        for call in driver.execute_query.await_args_list
        if call.args[0] not in MARKER_READS
    ]


def own_queries(driver: AsyncMock) -> list[str]:
    return [call.args[0] for call in own_calls(driver)]


def own_call(driver: AsyncMock, index: int) -> tuple[str, dict]:
    """The ``index``-th of the forget's own queries."""
    call = own_calls(driver)[index]
    return call.args[0], call.kwargs


def in_flight_reads(driver: AsyncMock) -> list[dict]:
    return [
        call.kwargs
        for call in driver.execute_query.await_args_list
        if call.args[0] == recall_reconcile.IN_FLIGHT_QUERY
    ]


# lookup, retract, scrub facts, entity keys (none), redact
SOFT = (
    [{"uuid": "u1"}],
    [{"uuid": "u1"}],
    [{"ends": []}],
    [],
    [{"uuid": "ep1", "via": ["u1"]}],
)
# The cascade finding nothing derived: no earlier try, the episodes citing
# the forgotten fact, no derived fact, no derived dream episode.
NOTHING_DERIVED = ([], [{"uuid": "ep1", "via": ["u1"]}], [], [])
