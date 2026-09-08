"""Cross-tool expert isolation: workflow discovery, block credentials, MCP
credentials, and run visibility all follow the session's expert."""

from unittest.mock import AsyncMock, MagicMock, patch

from backend.copilot.model import ChatSession
from backend.copilot.tools.agent_output import AgentOutputTool, _run_visible
from backend.copilot.tools.expert_scope import ExpertWorkflowScope
from backend.copilot.tools.find_library_agent import FindLibraryAgentTool
from backend.copilot.tools.helpers import resolve_block_credentials
from backend.copilot.tools.models import AgentInfo, AgentsFoundResponse


def _session(expert_id: str | None) -> ChatSession:
    return ChatSession.new("user-1", dry_run=False, expert_id=expert_id)


def _agent(agent_id: str, graph_id: str) -> AgentInfo:
    return AgentInfo(
        id=agent_id, name=agent_id, description="", source="library", graph_id=graph_id
    )


async def test_find_library_agent_lists_only_installed_workflows_for_experts():
    found = AgentsFoundResponse(
        message="Found",
        agents=[_agent("lib-in", "graph-in"), _agent("lib-out", "graph-out")],
        count=2,
        session_id="s",
    )
    scope = ExpertWorkflowScope(expert_id="expert-a", graph_ids=["graph-in"])
    with (
        patch.object(
            FindLibraryAgentTool, "_search", new=AsyncMock(return_value=found)
        ),
        patch(
            "backend.copilot.tools.find_library_agent.session_workflow_scope",
            new=AsyncMock(return_value=scope),
        ),
    ):
        result = await FindLibraryAgentTool()._execute(
            "user-1", _session("expert-a"), query="anything"
        )
    assert isinstance(result, AgentsFoundResponse)
    assert [a.id for a in result.agents] == ["lib-in"]
    assert result.count == 1


async def test_find_library_agent_is_unfiltered_for_personal_autopilot():
    found = AgentsFoundResponse(
        message="Found", agents=[_agent("a", "g1"), _agent("b", "g2")], count=2
    )
    with patch.object(
        FindLibraryAgentTool, "_search", new=AsyncMock(return_value=found)
    ):
        result = await FindLibraryAgentTool()._execute("user-1", _session(None))
    assert isinstance(result, AgentsFoundResponse) and result.count == 2


async def test_block_credentials_are_matched_within_the_expert_grant():
    block = MagicMock()
    with (
        patch(
            "backend.copilot.tools.helpers._resolve_discriminated_credentials",
            return_value={"credentials": MagicMock()},
        ),
        patch(
            "backend.copilot.tools.helpers.match_credentials_to_requirements",
            new=AsyncMock(return_value=({}, [])),
        ) as match,
    ):
        await resolve_block_credentials("user-1", block, {}, "expert-a")
    assert match.await_args.args[2] == "expert-a"


def test_run_visibility_follows_the_session_expert():
    assert _run_visible(MagicMock(expert_id="expert-a"), None)
    assert _run_visible(MagicMock(expert_id=None), None)
    assert _run_visible(MagicMock(expert_id="expert-a"), "expert-a")
    assert not _run_visible(MagicMock(expert_id="expert-b"), "expert-a")
    assert not _run_visible(MagicMock(expert_id=None), "expert-a")


async def test_expert_cannot_read_another_experts_run_by_id():
    exec_db = MagicMock()
    exec_db.get_graph_execution = AsyncMock(
        return_value=MagicMock(expert_id="expert-b")
    )
    with patch("backend.copilot.tools.agent_output.execution_db", return_value=exec_db):
        execution, _, error = await AgentOutputTool()._get_execution(
            user_id="user-1",
            graph_id="g",
            execution_id="run-1",
            time_start=None,
            time_end=None,
            expert_id="expert-a",
        )
    assert execution is None and error is not None


async def test_expert_run_listing_is_filtered_in_the_query():
    exec_db = MagicMock()
    exec_db.get_graph_executions = AsyncMock(return_value=[])
    with patch("backend.copilot.tools.agent_output.execution_db", return_value=exec_db):
        await AgentOutputTool()._get_execution(
            user_id="user-1",
            graph_id="g",
            execution_id=None,
            time_start=None,
            time_end=None,
            expert_id="expert-a",
        )
    assert exec_db.get_graph_executions.await_args.kwargs["expert_id"] == "expert-a"


async def test_run_agent_refuses_uninstalled_workflow_for_expert():
    from backend.copilot.tools.run_agent import RunAgentTool

    lib_agent = MagicMock(id="lib-1", graph_id="graph-out", graph_version=1)
    graph = MagicMock(id="graph-out", name="Outside", version=1)
    lib_db = MagicMock()
    lib_db.get_library_agent = AsyncMock(return_value=lib_agent)
    graph_db = MagicMock()
    graph_db.get_graph = AsyncMock(return_value=graph)
    experts = MagicMock()
    experts.get_expert = AsyncMock(
        return_value=MagicMock(
            workflows=[MagicMock(library_agent_id="lib-in", graph_id="graph-in")]
        )
    )
    add_exec = AsyncMock()
    with (
        patch("backend.copilot.tools.run_agent.library_db", return_value=lib_db),
        patch("backend.copilot.tools.run_agent.graph_db", return_value=graph_db),
        patch("backend.copilot.tools.expert_scope.experts_db", return_value=experts),
        patch(
            "backend.copilot.tools.run_agent.execution_utils.add_graph_execution",
            new=add_exec,
        ),
    ):
        result = await RunAgentTool()._execute(
            user_id="user-1", session=_session("expert-a"), library_agent_id="lib-1"
        )
    assert result.error == "workflow_not_installed"
    add_exec.assert_not_awaited()
