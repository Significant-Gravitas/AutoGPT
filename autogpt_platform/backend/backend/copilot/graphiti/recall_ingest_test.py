"""Unit tests for ``recall_ingest``: how the forget repair carries a plan
out after ``add_episode``, against a mock driver and the in-memory forget
stash.

What it decides is pinned in ``recall_ingest_plan_test.py``, whose builders
these tests share; the live runs, through the production worker, are
``recall_ingest_integration_test.py``, ``recall_repair_integration_test.py``
and ``recall_inflight_integration_test.py``.
"""

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import recall_ingest
from .memory_model import ForgetResult
from .recall_hide import Hiding
from .recall_ingest import carry_out
from .recall_ingest_plan import Plan, Reapply
from .recall_ingest_plan_test import _RUN, _SENTENCE, _edge, _record, _result, _state
from .recall_restore import spec_for


class TestCarryOut:
    @pytest.mark.asyncio
    async def test_every_step_runs_and_forgotten_edges_leave_the_result(self) -> None:
        spec = spec_for(_record(), None)
        assert spec is not None
        covered = spec.model_copy(update={"uuid": "f2"})
        plan = Plan(
            restores=[spec],
            merged=[spec],
            covered=[covered],
            unlink=["f1"],
            reapply=[Reapply(uuid="l1", hard=True, reason="user_signal")],
            scrub=["bob"],
            forgotten={"f1", "l1"},
        )
        client = MagicMock()
        client.driver.execute_query = AsyncMock()
        result = _result(_edge("f1"), _edge("l1"), _edge("live"), cites=["f1"])
        new = _edge("new", fact="Alice leads Atlas")
        mocks = _patches(AsyncMock(return_value=[new]))

        with _patched(mocks):
            await carry_out(client, _RUN, plan, result)

        repoint = client.driver.execute_query.await_args_list[-1]
        assert repoint.kwargs == {"uuid": "ep-new", "unlink": ["f1"], "added": ["new"]}
        mocks["restore"].assert_awaited_once_with(client.driver, "user_test", spec)
        hiding = mocks["hide"].await_args_list[-1].args[2]
        assert hiding == Hiding(
            uuids=["f2"], recovered=[["f2", _SENTENCE, "MemoryFact"]]
        )
        mocks["scrub_entities"].assert_awaited_once_with(client.driver, [], ["bob"])
        mocks["forget_edges"].assert_awaited_once_with(
            client.driver, "user_test", ["l1"], hard=True, reason="user_signal"
        )
        assert [edge.uuid for edge in result.edges] == ["live", "new"]

    @pytest.mark.asyncio
    async def test_a_failing_restate_leaves_the_other_steps_done(self) -> None:
        spec = spec_for(_record(), None)
        assert spec is not None
        plan = Plan(restores=[spec], merged=[spec], unlink=["f1"])
        client = MagicMock()
        client.driver.execute_query = AsyncMock()
        mocks = _patches(AsyncMock(side_effect=RuntimeError("model down")))

        with _patched(mocks):
            await carry_out(client, _RUN, plan, _result(_edge("f1"), cites=["f1"]))

        repoint = client.driver.execute_query.await_args_list[-1]
        assert repoint.kwargs["added"] == []
        mocks["restore"].assert_awaited_once()


class TestKeepForgotten:
    @pytest.mark.asyncio
    async def test_nothing_forgotten_reads_nothing(self) -> None:
        client = MagicMock()
        client.driver.execute_query = AsyncMock()

        await recall_ingest.keep_forgotten(client, _RUN, {}, _result(cites=[]))

        client.driver.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_read_leaves_the_episode_as_graphiti_wrote_it(self) -> None:
        client = MagicMock()
        client.driver.execute_query = AsyncMock(side_effect=RuntimeError("down"))
        result = _result(_edge("f1"), cites=["f1"])
        carry = AsyncMock()

        with patch.object(recall_ingest, "carry_out", carry):
            await recall_ingest.keep_forgotten(client, _RUN, {"f1": _state()}, result)

        carry.assert_not_awaited()


def _patches(restate: AsyncMock) -> dict[str, AsyncMock]:
    return {
        "restate": restate,
        "restore": AsyncMock(return_value=True),
        "hide": AsyncMock(return_value=True),
        "scrub_entities": AsyncMock(),
        "forget_edges": AsyncMock(return_value=ForgetResult()),
    }


def _patched(mocks: dict[str, AsyncMock]) -> ExitStack:
    stack = ExitStack()
    for name, mock in mocks.items():
        stack.enter_context(patch.object(recall_ingest, name, mock))
    return stack
