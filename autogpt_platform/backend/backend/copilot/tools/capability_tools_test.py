"""find/describe/run/resume_capability over the real registry with the
execution paths mocked at their boundaries."""

import ast
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.capabilities.mcp_review import COPILOT_MCP_NODE_PREFIX
from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.capabilities.sources import EAGER_CORE
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.context import set_execution_context
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.prompting import SHARED_TOOL_NOTES
from backend.copilot.tools import (
    DEFERRED_TOOL_NAMES,
    TOOL_REGISTRY,
    get_available_tools,
)

from ._test_data import make_session
from .describe_capability import DescribeCapabilityTool
from .find_capability import FindCapabilityTool
from .models import (
    BlockDetails,
    BlockDetailsResponse,
    BlockOutputResponse,
    CapabilityDetailsResponse,
    CapabilityListResponse,
    ErrorResponse,
    MCPToolOutputResponse,
    NoResultsResponse,
    ReviewRequiredResponse,
)
from .resume_capability import ResumeCapabilityTool
from .run_capability import RunCapabilityTool
from .session_registry import session_registry
from .skills import ParsedSkill, ReadSkillResponse

USER = "user-cap-tools"
LINEAR_STATE = ConnectionState(providers=frozenset({"linear"}))
TRIAGE = ParsedSkill(
    name="triage-and-prioritize",
    description="Triage a support ticket and set its priority.",
    body="1. read the ticket\n2. set the priority",
    triggers=("triage ticket",),
)


@pytest.fixture(autouse=True)
def _clean_context():
    session = make_session(USER)
    set_execution_context(USER, session)
    yield
    set_execution_context(None, None)


@pytest.fixture(autouse=True)
def skills():
    """The session's skill list is per user and read through Redis and the
    workspace; stub it empty so no test here reaches either.  A test that
    wants skills sets ``skills.return_value``."""
    with (
        patch(
            "backend.copilot.tools.session_registry.is_skills_feature_enabled",
            AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.session_registry.list_all_skills",
            AsyncMock(return_value=[]),
        ) as listed,
    ):
        yield listed


# ---------------------------------------------------------------- registry


def test_eager_and_deferred_split_the_registry():
    eager = set(TOOL_REGISTRY) & EAGER_CORE
    assert eager.isdisjoint(DEFERRED_TOOL_NAMES)
    assert eager | DEFERRED_TOOL_NAMES == set(TOOL_REGISTRY)
    shown = {t["function"]["name"] for t in get_available_tools()}
    assert shown <= eager
    assert {
        "find_capability",
        "describe_capability",
        "run_capability",
        "resume_capability",
    } <= shown
    assert len(get_available_tools(include_deferred=True)) > len(shown)


def test_enter_building_mode_stays_eager_because_the_refusal_names_it():
    """The building gate's refusal says to call ``enter_agent_building_mode``;
    deferred, that instruction cannot be followed, because naming a deferred
    tool directly is refused."""
    assert "enter_agent_building_mode" in EAGER_CORE
    assert "enter_agent_building_mode" not in DEFERRED_TOOL_NAMES


def test_memory_search_stays_eager_because_the_prompt_demands_it():
    """The memory supplement orders a search before answering from a past
    conversation; deferred, the model reported having no such tool and
    answered from injected context, because naming a deferred tool is
    refused."""
    assert "memory_search" in EAGER_CORE
    assert "memory_search" not in DEFERRED_TOOL_NAMES


def test_start_desktop_stays_eager_because_the_prompt_names_it():
    """``expert_context`` tells the model "Use start_desktop"; a deferred
    tool called by name is refused, so the instruction only works eager."""
    assert "start_desktop" in EAGER_CORE
    assert "start_desktop" not in DEFERRED_TOOL_NAMES


def test_prompt_names_only_registry_tools():
    for legacy in (
        "find_block",
        "run_block",
        "run_mcp_tool",
        "get_mcp_guide",
        "continue_run_block",
    ):
        assert legacy not in SHARED_TOOL_NOTES, legacy
    assert (
        "find_capability" in SHARED_TOOL_NOTES
        and "resume_capability" in SHARED_TOOL_NOTES
    )


# A deferred tool called by name is refused, so model-facing text must name it
# by capability id (``tool:<name>``), which ``SHARED_TOOL_NOTES`` teaches the
# model to pass to ``run_capability``.
_COPILOT_DIR = Path(__file__).resolve().parent.parent
# Not a mention: ``tool:<name>`` ids, dotted module paths, ``<name>.py`` files,
# and the ``[<name>]`` / ``copilot:<name>`` labels that usage tracking records.
_BARE_DEFERRED_NAME = re.compile(
    r"(?<![\w.\[])(?<!tool:)(?<!copilot:)("
    + "|".join(sorted(DEFERRED_TOOL_NAMES, key=len, reverse=True))
    + r")(?!\w|\.py)"
)
# Files left out of the literal scan, each with the reason it is safe.
_UNSCANNED_MODULES = {
    # Registry keys and permission tables: tool names as data, not prose.
    "__init__.py",
    # Response-model ``Field(description=...)`` text feeds the OpenAPI schema
    # and the generated frontend client; the model never reads it.
    "models.py",
}
# Modules whose string literals reach the model: prompt builders, injected
# context blocks, and the tools' descriptions and result messages.
_MODEL_FACING_MODULES = sorted(
    path
    for path in [
        _COPILOT_DIR / "prompting.py",
        _COPILOT_DIR / "service.py",
        _COPILOT_DIR / "expert_context.py",
        _COPILOT_DIR / "builder_context.py",
        *(_COPILOT_DIR / "tools").rglob("*.py"),
    ]
    if "test" not in path.name and path.name not in _UNSCANNED_MODULES
)
_NAME = "(?:" + "|".join(sorted(DEFERRED_TOOL_NAMES, key=len, reverse=True)) + ")"
_NAMES = rf"`*{_NAME}`*(?:(?: ?/ ?| or | and )`*{_NAME}`*)*"
# Mentions that are not call instructions, so they keep the bare name.  Each
# pattern is matched against whitespace-normalised text and blanked out before
# the scan; everything else naming a deferred tool must use ``tool:<name>``.
_ALLOWED_MENTIONS = re.compile(
    "|".join(
        [
            # Where a value came from: that tool has already run.
            rf"\b(?:from|returned by|proposed by|sent with|ids for|in the original) (?:an earlier )?{_NAMES}",
            rf"`(?:channel_id|ref_id)` {_NAME} returned",
            # Ordering relative to a call the model has made or will make.
            rf"\b(?i:before|after every) {_NAMES}|\bafter {_NAME} returns",
            # Prohibitions: telling the model NOT to call the tool.
            rf"\b(?i:do not) (?:call|make a follow-up) {_NAMES}",
            rf"no reason to reach for {_NAMES}",
            # A tool describing its own or a sibling's behaviour.
            rf"{_NAME} (?:creates|applies) exactly",
            rf"{_NAME} is for reaching",
            rf"{_NAMES} accepts the same node/link payload that {_NAMES} would",
            rf"{_NAMES} will reject",
            rf"not supported by {_NAME}",
            rf"Apply a {_NAMES} proposal",
            rf"runs the {_NAME} similarity",
            rf"set on the trigger node via {_NAMES}",
            # Feature-flag notices naming the tools the flag gates.
            rf"flag to use ``{_NAME}``(?: / ``{_NAME}``)*",
        ]
    )
)


def _unread_literals(tree: ast.AST) -> set[int]:
    """Docstrings, bare string statements and logger arguments: strings the
    model never sees."""
    unread: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            unread.add(id(node.value))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ):
            unread.update(id(sub) for sub in ast.walk(node))
    return unread


def _bare_mentions(source: str, text: str, line: int | None = None) -> list[str]:
    where = source if line is None else f"{source}:{line}"
    instructions = _ALLOWED_MENTIONS.sub(" ", " ".join(text.split()))
    return [
        f"{where}: {match.group(1)}"
        for match in _BARE_DEFERRED_NAME.finditer(instructions)
    ]


def _bare_mentions_in_module(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    unread = _unread_literals(tree)
    return [
        mention
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in unread
        # A literal that is exactly a tool name is an identifier
        # (``name = "edit_agent"``), not prose.
        and node.value not in DEFERRED_TOOL_NAMES
        for mention in _bare_mentions(path.name, node.value, node.lineno)
    ]


def test_model_facing_text_names_deferred_tools_by_capability_id():
    assert 'run_capability(id="tool:<name>"' in SHARED_TOOL_NOTES
    offenders = [
        mention
        for path in _MODEL_FACING_MODULES
        for mention in _bare_mentions_in_module(path)
    ]
    guide = _COPILOT_DIR / "sdk" / "agent_generation_guide.md"
    offenders += _bare_mentions(guide.name, guide.read_text(encoding="utf-8"))
    # Schemas as the model receives them, which also covers text a tool
    # interpolates into its description at runtime.
    for name, tool in TOOL_REGISTRY.items():
        offenders += _bare_mentions(
            f"{name} schema", f"{tool.description} {json.dumps(tool.parameters)}"
        )
    assert not offenders, (
        "A deferred tool is refused when called by name. Write `tool:<name>` "
        "(its run_capability id) instead of the bare name in:\n" + "\n".join(offenders)
    )


# The mirror of the rule above: an eager tool is IN the model's tool list, so a
# ``tool:`` id sends it through ``run_capability`` for nothing. Nothing else
# catches this — the scan above only knows the names that are deferred today,
# so a promotion into ``EAGER_CORE`` leaves the old ids behind silently.
_EAGER_ID = re.compile(
    r"tool:(" + "|".join(sorted(EAGER_CORE, key=len, reverse=True)) + r")\b"
)


def _eager_ids_in_module(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    unread = _unread_literals(tree)
    return [
        f"{path.name}:{node.lineno}: tool:{match.group(1)}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in unread
        for match in _EAGER_ID.finditer(node.value)
    ]


def test_model_facing_text_names_eager_tools_by_their_bare_name():
    offenders = [
        mention
        for path in _MODEL_FACING_MODULES
        for mention in _eager_ids_in_module(path)
    ]
    guide = _COPILOT_DIR / "sdk" / "agent_generation_guide.md"
    offenders += [
        f"{guide.name}: tool:{match.group(1)}"
        for match in _EAGER_ID.finditer(guide.read_text(encoding="utf-8"))
    ]
    for name, tool in TOOL_REGISTRY.items():
        offenders += [
            f"{name} schema: tool:{match.group(1)}"
            for match in _EAGER_ID.finditer(
                f"{tool.description} {json.dumps(tool.parameters)}"
            )
        ]
    assert not offenders, (
        "An eager tool is in the model's tool list and is called by name. Drop "
        "the `tool:` prefix in:\n" + "\n".join(offenders)
    )


# ------------------------------------------------------------ find_capability


async def test_find_capability_ranks_connected_service_first():
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=LINEAR_STATE),
    ):
        result = await FindCapabilityTool()._execute(
            USER, make_session(USER), query="linear issue"
        )
    assert isinstance(result, CapabilityListResponse)
    assert result.service == "linear"
    assert result.capabilities[0]["name"].startswith("Linear")
    assert result.capabilities[0]["connected"] is True
    assert all(c.get("class") != "primitive" for c in result.capabilities)


async def test_find_capability_no_results_points_to_open_world():
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=ConnectionState()),
    ):
        result = await FindCapabilityTool()._execute(
            USER, make_session(USER), query="zzqx"
        )
    assert isinstance(result, NoResultsResponse)
    assert any("MCP server" in s for s in result.suggestions)


async def test_find_capability_returns_the_session_owner_s_skill(skills):
    skills.return_value = [TRIAGE]
    session = make_session(USER, expert_id="expert-1")
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=ConnectionState()),
    ):
        result = await FindCapabilityTool()._execute(
            USER, session, query="triage a support ticket"
        )
    assert isinstance(result, CapabilityListResponse)
    assert result.capabilities[0]["id"] == "skill:triage-and-prioritize"
    assert result.capabilities[0]["kind"] == "skill"
    assert "kind=skill" in result.message
    # The owner's folder, never another expert's.
    skills.assert_awaited_with(USER, "expert-1")


async def test_the_layered_index_is_reused_while_the_skills_are_the_same(skills):
    """Layering rebuilds BM25 over the whole corpus, so the result is kept for
    the skill-cache window; a changed skill, or no skills, gets its own."""
    skills.return_value = [TRIAGE]
    session = make_session(USER)
    first = await session_registry(USER, session)
    second = await session_registry(USER, session)
    assert first is second and first is not get_registry()
    skills.return_value = [
        ParsedSkill(name=TRIAGE.name, description="Rewritten.", body="", triggers=())
    ]
    assert await session_registry(USER, session) is not first
    skills.return_value = []
    assert await session_registry(USER, session) is get_registry()


async def test_find_capability_kind_skill_lists_skills_only(skills):
    skills.return_value = [TRIAGE]
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=ConnectionState()),
    ):
        result = await FindCapabilityTool()._execute(
            USER, make_session(USER), query="ticket", kind="skill"
        )
    assert isinstance(result, CapabilityListResponse)
    assert [c["kind"] for c in result.capabilities] == ["skill"]


async def test_find_capability_survives_a_skill_listing_failure(skills):
    skills.side_effect = RuntimeError("redis down")
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=LINEAR_STATE),
    ):
        result = await FindCapabilityTool()._execute(
            USER, make_session(USER), query="linear issue"
        )
    assert isinstance(result, CapabilityListResponse)
    assert result.capabilities[0]["name"].startswith("Linear")


# -------------------------------------------------------- describe_capability


async def test_describe_skill_points_at_run_capability(skills):
    skills.return_value = [TRIAGE]
    result = await DescribeCapabilityTool()._execute(
        USER, make_session(USER), id="skill:triage-and-prioritize"
    )
    assert isinstance(result, CapabilityDetailsResponse)
    assert "run_capability(id='skill:triage-and-prioritize'" in result.message
    assert "Triggers: triage ticket." in result.message
    assert result.capability["kind"] == "skill"
    assert result.parameters == {"type": "object", "properties": {}}


async def test_describe_unknown_skill_names_the_id_shape(skills):
    result = await DescribeCapabilityTool()._execute(
        USER, make_session(USER), id="skill:nope"
    )
    assert isinstance(result, ErrorResponse) and "skill:<name>" in result.message


async def test_a_skill_id_never_resolves_to_the_platform_tool_of_that_name(skills):
    """``resolve_entry`` falls back to matching a bare name, which would let
    ``skill:web_search`` describe the platform's ``web_search`` tool."""
    result = await DescribeCapabilityTool()._execute(
        USER, make_session(USER), id="skill:web_search"
    )
    assert isinstance(result, ErrorResponse)


async def test_describe_tool_returns_parameters():
    result = await DescribeCapabilityTool()._execute(
        USER, make_session(USER), id="tool:list_schedules"
    )
    assert isinstance(result, CapabilityDetailsResponse)
    assert result.capability["id"] == "tool:list_schedules"
    assert result.parameters == TOOL_REGISTRY["list_schedules"].parameters


async def test_describe_block_collapses_large_enums_unless_expanded():
    details = BlockDetailsResponse(
        message="Block 'X' details.",
        block=BlockDetails(
            id="b",
            name="X",
            description="",
            inputs={"properties": {"model": {"enum": list(range(40))}}},
        ),
    )
    with patch(
        "backend.copilot.tools.describe_capability.RunBlockTool._execute",
        AsyncMock(return_value=details.model_copy(deep=True)),
    ):
        collapsed = await DescribeCapabilityTool()._execute(
            USER, make_session(USER), id="SendWebRequestBlock"
        )
    assert isinstance(collapsed, BlockDetailsResponse)
    assert collapsed.block.inputs["properties"]["model"]["enum_count"] == 40
    assert "run_capability" in collapsed.message
    with patch(
        "backend.copilot.tools.describe_capability.RunBlockTool._execute",
        AsyncMock(return_value=details.model_copy(deep=True)),
    ):
        full = await DescribeCapabilityTool()._execute(
            USER, make_session(USER), id="SendWebRequestBlock", expand=True
        )
    assert isinstance(full, BlockDetailsResponse)
    assert len(full.block.inputs["properties"]["model"]["enum"]) == 40


async def test_describe_block_honours_the_run_block_gate():
    """A withheld block's schema is part of what was withheld.

    Describing an MCP server is not even a local lookup -- it connects to the
    server to list its tools -- so both kinds answer to their gate here the
    way they do in run_capability.
    """
    session = make_session(USER)
    set_execution_context(
        USER, session, permissions=CopilotPermissions(tools=["run_block"])
    )
    result = await DescribeCapabilityTool()._execute(
        USER, session, id="SendWebRequestBlock"
    )
    assert isinstance(result, ErrorResponse) and result.error == "tool_disabled"


async def test_describe_mcp_honours_the_run_mcp_gate():
    session = make_session(USER)
    set_execution_context(
        USER, session, permissions=CopilotPermissions(tools=["run_mcp_tool"])
    )
    with patch(
        "backend.copilot.tools.describe_capability._describe_mcp", AsyncMock()
    ) as describe:
        result = await DescribeCapabilityTool()._execute(
            USER, session, id="https://mcp.linear.app/mcp"
        )
    assert isinstance(result, ErrorResponse) and result.error == "tool_disabled"
    describe.assert_not_awaited()


async def test_describe_unknown_id():
    result = await DescribeCapabilityTool()._execute(
        USER, make_session(USER), id="tool:nope"
    )
    assert isinstance(result, ErrorResponse) and "find_capability" in result.message


# ------------------------------------------------------------ run_capability


def _stub_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = "stub"
    tool.parameters = {"type": "object", "properties": {"x": {"type": "string"}}}
    tool._execute = AsyncMock(return_value=ErrorResponse(message="ran", session_id="s"))
    return tool


async def test_run_tool_describes_without_running():
    """Running a platform tool never reaches the dispatcher — the engines
    resolve the dispatch into a call to the tool itself (see
    ``capabilities/dispatch_test.py``). What is left here is the description
    ``validate_only`` asks for, which must not run anything."""
    stub = _stub_tool("list_schedules")
    with patch(
        "backend.copilot.tools.run_capability.configured_tool", return_value=stub
    ):
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="tool:list_schedules",
            input={"x": "1"},
            validate_only=True,
        )
    assert isinstance(result, CapabilityDetailsResponse)
    assert result.parameters == stub.parameters
    stub._execute.assert_not_awaited()


async def test_run_tool_respects_turn_hidden_tools():
    session = make_session(USER)
    set_execution_context(USER, session, hidden_tools=frozenset({"list_schedules"}))
    stub = _stub_tool("list_schedules")
    with patch(
        "backend.copilot.tools.run_capability.configured_tool", return_value=stub
    ):
        result = await RunCapabilityTool()._execute(
            USER, session, id="tool:list_schedules", input={}, validate_only=True
        )
    assert isinstance(result, ErrorResponse) and result.error == "tool_disabled"
    stub._execute.assert_not_awaited()


async def test_run_skill_validate_only_describes_without_loading(skills):
    skills.return_value = [TRIAGE]
    with patch("backend.copilot.tools.run_capability.ReadSkillTool") as reader:
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="skill:triage-and-prioritize",
            input={},
            validate_only=True,
        )
    assert isinstance(result, CapabilityDetailsResponse)
    assert result.capability["id"] == "skill:triage-and-prioritize"
    reader.assert_not_called()


async def test_run_skill_loads_it_through_the_skill_reader(skills):
    """Off the engine path (``dispatch.py`` resolves the call there), the
    dispatcher itself loads the skill by name."""
    skills.return_value = [TRIAGE]
    session = make_session(USER)
    loaded = ReadSkillResponse(
        name=TRIAGE.name,
        description=TRIAGE.description,
        body=TRIAGE.body,
        message="Loaded",
        session_id=session.session_id,
    )
    with patch("backend.copilot.tools.run_capability.ReadSkillTool") as reader:
        reader.return_value._execute = AsyncMock(return_value=loaded)
        result = await RunCapabilityTool()._execute(
            USER, session, id="skill:triage-and-prioritize", input={}
        )
    assert result is loaded
    reader.return_value._execute.assert_awaited_once_with(
        USER, session, name="triage-and-prioritize"
    )


async def test_run_skill_honours_the_read_skill_gate(skills):
    skills.return_value = [TRIAGE]
    session = make_session(USER)
    set_execution_context(USER, session, hidden_tools=frozenset({"read_skill"}))
    with patch("backend.copilot.tools.run_capability.ReadSkillTool") as reader:
        result = await RunCapabilityTool()._execute(
            USER, session, id="skill:triage-and-prioritize", input={}
        )
    assert isinstance(result, ErrorResponse) and result.error == "tool_disabled"
    reader.assert_not_called()
    described = await DescribeCapabilityTool()._execute(
        USER, session, id="skill:triage-and-prioritize"
    )
    assert isinstance(described, ErrorResponse) and described.error == "tool_disabled"


async def test_run_block_honours_the_run_block_gate():
    session = make_session(USER)
    set_execution_context(
        USER, session, permissions=CopilotPermissions(tools=["run_block"])
    )
    with patch(
        "backend.copilot.tools.run_capability.RunBlockTool._execute", AsyncMock()
    ) as run:
        result = await RunCapabilityTool()._execute(
            USER, session, id="SendWebRequestBlock", input={}
        )
    assert isinstance(result, ErrorResponse) and result.error == "tool_disabled"
    run.assert_not_awaited()


async def test_run_block_forwards_to_run_block_tool():
    output = BlockOutputResponse(
        message="ok", block_id="b", block_name="SendWebRequestBlock", outputs={}
    )
    with patch(
        "backend.copilot.tools.run_capability.RunBlockTool._execute",
        AsyncMock(return_value=output),
    ) as run:
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="SendWebRequestBlock",
            input={"url": "https://x"},
            validate_only=True,
        )
    assert result is output
    kwargs = run.await_args.kwargs
    assert (
        kwargs["input_data"] == {"url": "https://x"} and kwargs["validate_only"] is True
    )
    assert kwargs["block_id"]


async def test_run_mcp_catalog_write_runs_without_review():
    out = MCPToolOutputResponse(
        message="done", server_url="u", tool_name="create_issue"
    )
    with (
        patch(
            "backend.copilot.tools.run_capability.RunMCPToolTool._execute",
            AsyncMock(return_value=out),
        ) as run,
        patch(
            "backend.copilot.tools.run_capability.open_mcp_review", AsyncMock()
        ) as review,
    ):
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="mcp:mcp.linear.app",
            input={"tool": "create_issue", "arguments": {"a": 1}},
        )
    assert result is out
    review.assert_not_awaited()
    assert run.await_args.kwargs["tool_name"] == "create_issue"
    assert run.await_args.kwargs["tool_arguments"] == {"a": 1}


async def test_run_mcp_open_world_write_pauses_for_review():
    with (
        patch(
            "backend.copilot.tools.run_capability.RunMCPToolTool._execute", AsyncMock()
        ) as run,
        patch(
            "backend.copilot.tools.run_capability.open_mcp_review",
            AsyncMock(return_value="copilot-mcp-x:1"),
        ),
    ):
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="https://mcp.example.com/mcp",
            input={"tool": "delete_thing", "arguments": {}},
        )
    assert isinstance(result, ReviewRequiredResponse)
    assert (
        result.review_id == "copilot-mcp-x:1" and "resume_capability" in result.message
    )
    run.assert_not_awaited()


async def test_run_mcp_open_world_read_runs():
    out = MCPToolOutputResponse(message="done", server_url="u", tool_name="list_things")
    with patch(
        "backend.copilot.tools.run_capability.RunMCPToolTool._execute",
        AsyncMock(return_value=out),
    ):
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="https://mcp.example.com/mcp",
            input={"tool": "list_things"},
        )
    assert result is out


async def test_run_mcp_validate_only_describes_input_shape():
    result = await RunCapabilityTool()._execute(
        USER, make_session(USER), id="mcp:mcp.linear.app", input={}, validate_only=True
    )
    assert isinstance(result, CapabilityDetailsResponse)
    assert set(result.parameters["properties"]) == {"tool", "arguments", "connect"}


async def test_run_unknown_id():
    result = await RunCapabilityTool()._execute(
        USER, make_session(USER), id="block:nope", input={}
    )
    assert isinstance(result, ErrorResponse) and "find_capability" in result.message


# --------------------------------------------------------- resume_capability


def _review(
    review_id: str, status: ReviewStatus, payload: dict[str, Any], session_id: str
) -> MagicMock:
    review = MagicMock()
    review.node_exec_id = review_id
    review.status = status
    review.payload = payload
    review.graph_exec_id = f"{COPILOT_SESSION_PREFIX}{session_id}"
    return review


async def test_resume_block_review_delegates_to_continue_run_block():
    with patch(
        "backend.copilot.tools.resume_capability.ContinueRunBlockTool._execute",
        AsyncMock(return_value=ErrorResponse(message="cont")),
    ) as cont:
        result = await ResumeCapabilityTool()._execute(
            USER, make_session(USER), review_id="copilot-node-b:1"
        )
    assert result.message == "cont"
    assert cont.await_args.kwargs == {"review_id": "copilot-node-b:1"}


async def test_resume_mcp_review_replays_the_approved_call():
    session = make_session(USER)
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:ab12"
    arguments = {"id": 1, "force": False}
    review = _review(
        review_id,
        ReviewStatus.APPROVED,
        {
            "server_url": "https://mcp.example.com/mcp",
            "tool": "delete_thing",
            "arguments": arguments,
        },
        session.session_id,
    )
    db = MagicMock()
    db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    db.delete_review_by_node_exec_id = AsyncMock()
    out = MCPToolOutputResponse(
        message="done", server_url="u", tool_name="delete_thing"
    )
    with (
        patch("backend.copilot.tools.resume_capability.review_db", return_value=db),
        patch(
            "backend.copilot.tools.resume_capability.RunMCPToolTool._execute",
            AsyncMock(return_value=out),
        ) as run,
    ):
        result = await ResumeCapabilityTool()._execute(
            USER, session, review_id=review_id
        )
    assert result is out
    assert run.await_args.kwargs["tool_arguments"] == arguments
    db.delete_review_by_node_exec_id.assert_awaited_once_with(review_id, USER)


async def test_resume_mcp_review_reopens_when_overrides_change_a_write():
    """An approval covers the call the user saw, not a family of them.

    Replaying it with different arguments is how a prompt injection turns
    "delete this" into "delete that", so the gate runs again on the merged
    arguments and the call waits for a fresh approval.
    """
    session = make_session(USER)
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:ab12"
    review = _review(
        review_id,
        ReviewStatus.APPROVED,
        {
            "server_url": "https://mcp.example.com/mcp",
            "tool": "delete_thing",
            "arguments": {"id": 1, "force": False},
        },
        session.session_id,
    )
    db = MagicMock()
    db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    db.delete_review_by_node_exec_id = AsyncMock()
    with (
        patch("backend.copilot.tools.resume_capability.review_db", return_value=db),
        patch(
            "backend.copilot.tools.resume_capability.open_mcp_review",
            AsyncMock(return_value="copilot-mcp-mcp.example.com:ef56"),
        ) as opened,
        patch(
            "backend.copilot.tools.resume_capability.RunMCPToolTool._execute",
            AsyncMock(),
        ) as run,
    ):
        result = await ResumeCapabilityTool()._execute(
            USER, session, review_id=review_id, input_overrides={"force": True}
        )
    assert result.type == "review_required"
    assert result.review_id == "copilot-mcp-mcp.example.com:ef56"
    assert opened.await_args.kwargs["payload"].arguments == {"id": 1, "force": True}
    run.assert_not_awaited()
    db.delete_review_by_node_exec_id.assert_not_awaited()


async def test_resume_mcp_review_reopens_even_when_the_call_would_not_be_gated():
    """The review's existence is the proof this call needed approving.

    Re-deriving that on resume reads the world as it is now: a read-shaped
    tool name, or a server added to the catalog between turns, both answer
    "no review needed" and would replay the changed arguments unapproved.
    """
    session = make_session(USER)
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:ab12"
    review = _review(
        review_id,
        ReviewStatus.APPROVED,
        {
            "server_url": "https://mcp.example.com/mcp",
            "tool": "get_thing",  # a read: needs_review() would say no
            "arguments": {"id": 1},
        },
        session.session_id,
    )
    db = MagicMock()
    db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    db.delete_review_by_node_exec_id = AsyncMock()
    with (
        patch("backend.copilot.tools.resume_capability.review_db", return_value=db),
        patch(
            "backend.copilot.tools.resume_capability.open_mcp_review",
            AsyncMock(return_value="copilot-mcp-mcp.example.com:ef56"),
        ),
        patch(
            "backend.copilot.tools.resume_capability.RunMCPToolTool._execute",
            AsyncMock(),
        ) as run,
    ):
        result = await ResumeCapabilityTool()._execute(
            USER, session, review_id=review_id, input_overrides={"id": 2}
        )
    assert result.type == "review_required"
    run.assert_not_awaited()
    db.delete_review_by_node_exec_id.assert_not_awaited()


async def test_resume_honours_this_turn_s_gates():
    """An approval is not a standing exemption from later permissions.

    Permissions are rebuilt per turn, and resuming runs the capability, so a
    turn that may not reach MCP servers may not reach them through a review
    approved while it still could.
    """
    session = make_session(USER)
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:ab12"
    db = MagicMock()
    db.get_reviews_by_node_exec_ids = AsyncMock()
    set_execution_context(
        USER, session, permissions=CopilotPermissions(tools=["run_mcp_tool"])
    )
    with patch("backend.copilot.tools.resume_capability.review_db", return_value=db):
        result = await ResumeCapabilityTool()._execute(
            USER, session, review_id=review_id
        )
    assert isinstance(result, ErrorResponse) and result.error == "tool_disabled"
    db.get_reviews_by_node_exec_ids.assert_not_awaited()


async def test_resume_mcp_review_waits_for_approval():
    session = make_session(USER)
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:cd34"
    review = _review(review_id, ReviewStatus.WAITING, {}, session.session_id)
    db = MagicMock()
    db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    with patch("backend.copilot.tools.resume_capability.review_db", return_value=db):
        result = await ResumeCapabilityTool()._execute(
            USER, session, review_id=review_id
        )
    assert isinstance(result, ErrorResponse) and "not been approved" in result.message


async def test_validating_a_dispatch_never_satisfies_a_gate():
    """``validate_only`` describes the call without running it, so nothing
    about it may look like the tool having run."""
    session = make_session(USER)
    stub = _stub_tool("enter_agent_building_mode")
    with patch(
        "backend.copilot.tools.run_capability.configured_tool", return_value=stub
    ):
        await RunCapabilityTool()._execute(
            USER,
            session,
            id="tool:enter_agent_building_mode",
            input={},
            validate_only=True,
        )
    assert session.has_tool_been_called("enter_agent_building_mode") is False
    stub._execute.assert_not_awaited()
