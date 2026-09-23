"""Input-moderation behaviour when the graph can't be resolved.

`moderate_graph_execution_inputs()` needs the graph's nodes to collect the
inputs it moderates. The read bypasses the caller's access check — the run is
already authorized by then — so an empty lookup means the version is gone, not
that the runner may not read it. Whether that kills the run or lets it through
unmoderated is the `automod_fail_open` operator setting, not an accident of
control flow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import ModerationError

from .manager import AutoModManager


@pytest.mark.asyncio
async def test_missing_graph_fails_closed_by_default() -> None:
    """A graph that no longer exists means moderation could not run. Reporting
    that as "passed" would let the execution through unvetted."""
    manager, db_client = _manager_with_graph(None, fail_open=False)

    result = await manager.moderate_graph_execution_inputs(
        db_client=db_client, graph_exec=_graph_exec()
    )

    assert isinstance(result, ModerationError)


@pytest.mark.asyncio
async def test_missing_graph_passes_when_fail_open_is_set() -> None:
    """`automod_fail_open` is the operator's explicit choice to keep executions
    running when moderation is unavailable."""
    manager, db_client = _manager_with_graph(None, fail_open=True)

    result = await manager.moderate_graph_execution_inputs(
        db_client=db_client, graph_exec=_graph_exec()
    )

    assert result is None


@pytest.mark.asyncio
async def test_graph_without_nodes_passes() -> None:
    """An empty graph is not a failure to moderate -- there is genuinely
    nothing to moderate -- so it must not take the fail-closed branch."""
    manager, db_client = _manager_with_graph(MagicMock(nodes=[]), fail_open=False)

    result = await manager.moderate_graph_execution_inputs(
        db_client=db_client, graph_exec=_graph_exec()
    )

    assert result is None


@pytest.mark.asyncio
async def test_marketplace_sub_agent_run_is_moderated_not_denied() -> None:
    """SECRT-1125 lets a user execute a marketplace graph they cannot read, as
    a sub-agent. Resolving the graph through their read gate would return None
    for exactly those runs and fail them closed on an unrelated error, so the
    read is bypassed: the run was already authorized by add_graph_execution().
    """
    graph = MagicMock(nodes=[])
    manager, db_client = _manager_with_graph(graph, fail_open=False)

    result = await manager.moderate_graph_execution_inputs(
        db_client=db_client, graph_exec=_graph_exec()
    )

    assert result is None, "an authorized sub-agent run must not be failed here"
    assert db_client.get_graph.await_args.kwargs["skip_access_check"] is True
    # Still the runner's own execution, and still version-specific.
    assert db_client.get_graph.await_args.kwargs["user_id"] == "runner-user-id"
    assert db_client.get_graph.await_args.kwargs["version"] == 3


def _graph_exec() -> MagicMock:
    return MagicMock(
        user_id="runner-user-id",
        graph_id="graph-id",
        graph_version=3,
        graph_exec_id="exec-id",
        nodes_input_masks=None,
    )


def _manager_with_graph(graph, *, fail_open: bool) -> tuple[AutoModManager, MagicMock]:
    with patch.object(AutoModManager, "_load_config") as load_config:
        load_config.return_value = MagicMock(enabled=True, fail_open=fail_open)
        manager = AutoModManager()

    db_client = MagicMock()
    db_client.get_graph = AsyncMock(return_value=graph)
    return manager, db_client


@pytest.fixture(autouse=True)
def _automod_flag_enabled():
    with patch(
        "backend.executor.automod.manager.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    ):
        yield
