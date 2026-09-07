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
from backend.util.llm.conversions import extract_openrouter_cost

from .assembly import chat_system_prompt
from .models import Usage
from .scorer import to_usage

GENERATION_MAX_TOKENS = 2000
GENERATION_TIMEOUT_SECONDS = 180.0
# Production's loop is unbounded. A turn still calling tools at the cap is
# reported as such and judged on whatever text it produced. Forcing text
# with tool_choice=none is not an option: it invalidates the prompt cache
# and Claude answers it with an empty message.
MAX_TOOL_ROUNDS = 8
# Ends the turn in production (the user answers on a card), so it ends the
# loop here with the questions rendered as the turn's visible text.
TERMINAL_TOOL = "ask_question"
LIBRARY_SEARCH_TOOLS = ("find_library_agent", "find_agent")


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
                "content": stub_tool_result(tc.function.name, expert),
            }
            for tc in tool_calls
        ]
    else:
        turn.hit_round_cap = True
    turn.text = "\n\n".join(texts)
    return turn


def stub_tool_result(name: str, expert: Expert | None) -> str:
    """What a fresh hire's session gets back: its installed workflows from a
    library search, nothing from memory, no runs or schedules elsewhere. An
    "unavailable" stub made the model narrate the harness; an empty library
    made it doubt the user's own run report."""
    if name in LIBRARY_SEARCH_TOOLS and expert and expert.workflows:
        agents = [
            {
                "id": w.library_agent_id,
                "name": w.name,
                "description": w.description,
                "source": "library",
                "in_library": True,
                "graph_id": w.graph_id,
            }
            for w in expert.workflows
        ]
        return json.dumps(
            {
                "type": "agents_found",
                "message": f"Found {len(agents)} agents in your library.",
                "agents": agents,
                "count": len(agents),
            }
        )
    if name == "memory_search":
        return json.dumps(
            {
                "type": "memory_search",
                "message": "No memories found for this query.",
                "results": [],
            }
        )
    return json.dumps({"success": True, "message": "No results.", "results": []})


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
