"""One expert turn the way the baseline engine sends it: production's system
prompt, first-turn context blocks and tool schemas, the routed model, the
OpenAI-compat transport.

Tool calls get a stub result and the loop continues, so the model reaches
its written answer instead of narrating a tool call as text — which is what
a tool-less request produced on five of six failure prompts.
"""

import json
from typing import Any, cast

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field

from backend.api.features.experts.models import Expert
from backend.copilot.baseline.service import (
    _build_cached_system_message,
    _fresh_anthropic_caching_headers,
    _mark_tools_with_cache_control,
    _supports_prompt_cache_markers,
)
from backend.copilot.config import ChatConfig
from backend.copilot.tools import expert_tool_disabled_groups, get_available_tools
from backend.copilot.tools.list_agent_triggers import AgentTriggerListResponse
from backend.copilot.tools.manage_presets import PresetListResponse
from backend.copilot.tools.manage_schedules import ScheduleListResponse
from backend.copilot.tools.models import (
    AgentInfo,
    AgentOutputResponse,
    AgentsFoundResponse,
    ErrorResponse,
    ExecutionStartedResponse,
    MemorySearchResponse,
    NoResultsResponse,
    SubSessionStatusResponse,
    TeamExpertInfo,
    TeamRosterResponse,
    ToolResponseBase,
)
from backend.copilot.tools.workspace_files import WorkspaceFileListResponse
from backend.util.llm.conversions import extract_openrouter_cost

from .assembly import chat_system_prompt
from .models import Usage
from .scorer import to_usage

GENERATION_MAX_TOKENS = 2000
GENERATION_TIMEOUT_SECONDS = 180.0
# Production's loop is unbounded. A turn still calling tools at the cap has
# no finished answer to judge, so the runner records it as an error rather
# than scoring the narration it left behind. Forcing text with
# tool_choice=none is not an option: it invalidates the prompt cache and
# Claude answers it with an empty message.
MAX_TOOL_ROUNDS = 10
# Ends the turn in production (the user answers on a card), so it ends the
# loop here with the questions rendered as the turn's visible text.
TERMINAL_TOOL = "ask_question"
LIBRARY_SEARCH_TOOLS = ("find_library_agent", "find_agent")
DELEGATION_TOOLS = ("delegate_to_expert", "run_sub_session")
HANDOFF_TOOL = "handoff_to_expert"
WEB_TOOLS = ("web_search", "web_fetch")
STUB_EXECUTION_ID = "style-eval-execution"
STUB_GRAPH_ID = "style-eval-graph"
STUB_SUB_SESSION_ID = "style-eval-sub-session"


class StubWorkflow(BaseModel):
    name: str
    graph_id: str
    library_agent_id: str | None


class Turn(BaseModel):
    text: str
    truncated: bool
    usage: Usage
    tool_calls: list[str]
    rounds: int = 0
    finish_reasons: list[str] = Field(default_factory=list)
    hit_round_cap: bool = False


def chat_client(config: ChatConfig) -> openai.AsyncOpenAI:
    """The baseline engine's main client: OpenRouter, or api.anthropic.com's
    OpenAI-compat endpoint when only an Anthropic key is configured."""
    api_key, base_url = config.main_client_credentials
    return openai.AsyncOpenAI(api_key=api_key or "", base_url=base_url)


def expert_tools(expert: Expert | None) -> list[ChatCompletionToolParam]:
    """Production's tool surface for the session (hire-experts on, memory on):
    an expert loses the staffing tools, plain AutoPilot the expert-only ones."""
    disabled = expert_tool_disabled_groups(
        experts_enabled=True, expert_id=expert.id if expert else None
    )
    return get_available_tools(disabled_groups=disabled)


async def generate_turn(
    client: openai.AsyncOpenAI,
    config: ChatConfig,
    *,
    model: str,
    expert: Expert | None,
    roster: list[Expert],
    user_message: str,
) -> Turn:
    system: dict[str, Any] = {"role": "system", "content": chat_system_prompt(expert)}
    tools: list[dict[str, Any]] = [dict(t) for t in expert_tools(expert)]
    headers: dict[str, str] | None = None
    if _supports_prompt_cache_markers(model):
        system = _build_cached_system_message(system)
        tools = _mark_tools_with_cache_control(tools)
        headers = _fresh_anthropic_caching_headers()
    messages: list[dict[str, Any]] = [system, {"role": "user", "content": user_message}]
    extra_body = (
        {"usage": {"include": True}}
        if config.baseline_provider == "openrouter"
        else None
    )
    turn = Turn(
        text="", truncated=False, usage=Usage(model=model, cost_usd=0.0), tool_calls=[]
    )
    texts: list[str] = []
    for _ in range(MAX_TOOL_ROUNDS):
        completion = await client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=cast(list[ChatCompletionToolParam], tools),
            max_tokens=GENERATION_MAX_TOKENS,
            timeout=GENERATION_TIMEOUT_SECONDS,
            extra_body=extra_body,
            extra_headers=headers,
        )
        turn.usage = add_usage(turn.usage, usage_of(model, completion))
        turn.rounds += 1
        choice = completion.choices[0]
        turn.finish_reasons.append(choice.finish_reason)
        turn.truncated = turn.truncated or choice.finish_reason == "length"
        if choice.message.content:
            texts.append(choice.message.content)
        tool_calls = [
            tc for tc in choice.message.tool_calls or [] if tc.type == "function"
        ]
        if not tool_calls:
            break
        turn.tool_calls += [tc.function.name for tc in tool_calls]
        asked = [tc for tc in tool_calls if tc.function.name == TERMINAL_TOOL]
        if asked:
            texts += [question_text(tc.function.arguments) for tc in asked]
            break
        messages.append(
            {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in tool_calls],
            }
        )
        messages += [
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": stub_tool_result(
                    tc.function.name, tc.function.arguments, expert, roster
                ),
            }
            for tc in tool_calls
        ]
    else:
        turn.hit_round_cap = True
    turn.text = "\n\n".join(texts)
    return turn


def stub_tool_result(
    name: str, arguments: str, expert: Expert | None, roster: list[Expert]
) -> str:
    """Production's own response model for every tool, filled the way a fresh
    hire's account answers: its preloads in the library, no memories, nothing
    run or scheduled yet, a manual run that queues, a teammate still working.
    An ad-hoc "no results" blob instead kept the model retrying the same tools
    until the round cap, on 6 of 90 prompts."""
    args = _arguments(arguments)
    workflow = _workflow(expert, args)
    if name in LIBRARY_SEARCH_TOOLS:
        return _dump(_library_result(expert))
    if name == "run_agent":
        return _dump(
            ExecutionStartedResponse(
                message=f"Started {workflow.name}. The run is queued.",
                execution_id=STUB_EXECUTION_ID,
                graph_id=workflow.graph_id,
                graph_name=workflow.name,
                library_agent_id=workflow.library_agent_id,
            )
        )
    if name == "view_agent_output":
        return _dump(
            AgentOutputResponse(
                message=f"No finished runs of {workflow.name} to show.",
                agent_name=workflow.name,
                agent_id=workflow.graph_id,
                library_agent_id=workflow.library_agent_id,
                execution=None,
                total_executions=0,
            )
        )
    if name == "memory_search":
        return _dump(
            MemorySearchResponse(
                message="No memories found matching your query.",
                facts=[],
                recent_episodes=[],
            )
        )
    if name in DELEGATION_TOOLS:
        return _dump(
            SubSessionStatusResponse(
                message="Delegated. The teammate is working on it.",
                status="running",
                sub_session_id=STUB_SUB_SESSION_ID,
            )
        )
    if name == HANDOFF_TOOL:
        return _dump(
            SubSessionStatusResponse(
                message="Handed off. The receiving expert reports to the user.",
                status="transferred",
                sub_session_id=STUB_SUB_SESSION_ID,
            )
        )
    if name == "list_team":
        return _dump(
            TeamRosterResponse(
                message="; ".join(
                    f"{e.name} — {e.role} (expert_id: {e.id})" for e in roster
                ),
                experts=[
                    TeamExpertInfo(id=e.id, name=e.name, role=e.role) for e in roster
                ],
            )
        )
    if name == "list_schedules":
        return _dump(ScheduleListResponse(message="No schedules yet.", schedules=[]))
    if name == "list_presets":
        return _dump(
            PresetListResponse(
                message="No presets yet.",
                presets=[],
                total_count=0,
                page=1,
                page_size=20,
            )
        )
    if name == "list_agent_triggers":
        return _dump(
            AgentTriggerListResponse(message="No triggers configured.", triggers=[])
        )
    if name == "list_workspace_files":
        return _dump(
            WorkspaceFileListResponse(
                message="The workspace is empty.", files=[], total_count=0
            )
        )
    if name in WEB_TOOLS:
        return _dump(
            ErrorResponse(
                message="Web access is off in this session.", error="unavailable"
            )
        )
    return _dump(NoResultsResponse(message=f"Nothing to return from {name}."))


def _library_result(expert: Expert | None) -> ToolResponseBase:
    """A library search from a fresh hire: exactly its installed workflows.
    An empty library made the model doubt the user's own run report."""
    workflows = expert.workflows if expert else []
    if not workflows:
        return NoResultsResponse(message="No agents in your library match that.")
    return AgentsFoundResponse(
        message=f"Found {len(workflows)} agents in your library.",
        agents=[
            AgentInfo(
                id=w.library_agent_id or w.id,
                name=w.name or "",
                description=w.description or "",
                source="library",
                in_library=True,
                graph_id=w.graph_id,
            )
            for w in workflows
        ],
        count=len(workflows),
    )


def _workflow(expert: Expert | None, args: dict[str, Any]) -> StubWorkflow:
    """The workflow the call names, so the stub echoes back what was asked
    for; the expert's first preload otherwise."""
    workflows = list(expert.workflows) if expert else []
    wanted = {
        str(args.get(key)) for key in ("library_agent_id", "graph_id", "agent_id")
    }
    named = [
        w for w in workflows if w.library_agent_id in wanted or w.graph_id in wanted
    ]
    for workflow in named + workflows:
        return StubWorkflow(
            name=workflow.name or "the agent",
            graph_id=workflow.graph_id or STUB_GRAPH_ID,
            library_agent_id=workflow.library_agent_id,
        )
    return StubWorkflow(
        name=str(args.get("agent_name") or "the agent"),
        graph_id=STUB_GRAPH_ID,
        library_agent_id=None,
    )


def _arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dump(response: ToolResponseBase) -> str:
    """What ``execute_tool`` hands the loop: the response model as JSON."""
    return response.model_dump_json()


def question_text(arguments: str) -> str:
    """What the user sees of an ``ask_question`` call: the questions, with
    their options where given."""
    try:
        payload = json.loads(arguments)
    except ValueError:
        return arguments
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list):
        return arguments
    lines = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        options = item.get("options")
        suffix = f" ({' / '.join(map(str, options))})" if options else ""
        lines.append(f"{item.get('question', '')}{suffix}")
    return "\n".join(lines)


def usage_of(model: str, completion: ChatCompletion) -> Usage:
    """OpenAI-compat usage: ``prompt_tokens`` includes the cached prefix.
    OpenRouter also reports cache writes; the Anthropic compat endpoint
    reports neither, so there the whole prompt is priced as fresh input."""
    reported = completion.usage
    if reported is None:
        return Usage(model=model)
    details = reported.prompt_tokens_details
    cached = (details.cached_tokens or 0) if details else 0
    extras = (details.model_extra or {}) if details else {}
    written = extras.get("cache_write_tokens") or 0
    return to_usage(
        model,
        input_tokens=reported.prompt_tokens - cached - written,
        output_tokens=reported.completion_tokens,
        cache_read_tokens=cached,
        cache_creation_tokens=written,
        cost_usd=extract_openrouter_cost(completion),
    )


def add_usage(a: Usage, b: Usage) -> Usage:
    both_known = a.cost_usd is not None and b.cost_usd is not None
    return Usage(
        model=a.model,
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_creation_tokens=a.cache_creation_tokens + b.cache_creation_tokens,
        cost_usd=(a.cost_usd or 0) + (b.cost_usd or 0) if both_known else None,
    )
