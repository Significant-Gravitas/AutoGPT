"""Run the 50 task-rubric messages through Otto and score the verdicts.

    poetry run task-rubric-eval                     # every example
    poetry run task-rubric-eval --ids 1 2 34        # a few
    poetry run task-rubric-eval --model <slug>      # a proposed model
    poetry run task-rubric-eval --subscription      # through `claude login`

Each message is one plain-Otto first turn in the account ``fixture.py``
describes: production's system prompt with the Tasks v1 rubric section, the
first-turn context production prepends, the real tool schemas, and tool
results from that account (the style eval's stubs for everything else). A
question gets the user's answer and the turn goes on; it ends at the first
call that commits to an outcome (see ``verdict.py``). A split plan the real
tool would refuse is refused here too, so it doesn't count as a decision.

Two scores: the right outcome, and the right outcome without a question the
example didn't need.

By default the turn goes over the baseline engine's API client, which costs
money. With ``--subscription`` (on by default when
``CHAT_USE_CLAUDE_CODE_SUBSCRIPTION`` is set) it runs through the Claude Agent
SDK on the host's Claude Code login instead, with the same tools served from
an in-process MCP server.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import tempfile
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import openai
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultError,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.eval.style.assembly import chat_system_prompt, resolve_chat_model
from backend.copilot.eval.style.generation import (
    GENERATION_MAX_TOKENS,
    GENERATION_TIMEOUT_SECONDS,
    RUN_TOOL,
    called_name,
    chat_client,
    expert_tools,
    stub_tool_result,
)
from backend.copilot.sdk.subscription import host_login_env, validate_subscription
from backend.copilot.tasks_rubric import TASK_RUBRIC
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.tasks_helpers import TaskToolError
from backend.copilot.tools.tasks_plan_tools import check_plan_shape

from . import fixture
from .examples import RubricExample, Verdict, load_examples
from .verdict import (
    ASK_TOOL,
    ToolCall,
    TurnVerdict,
    asked_needlessly,
    call_verdict,
    passes,
    turn_verdict,
)

logger = logging.getLogger(__name__)

MAX_ROUNDS = 8
# A second question means Otto still wouldn't act after an answer.
MAX_QUESTIONS = 2
MCP_SERVER = "copilot"
MCP_PREFIX = f"mcp__{MCP_SERVER}__"
REFUSED = " (refused)"
Classifier = Callable[[RubricExample], Awaitable[list[ToolCall]]]


class ExampleResult(BaseModel):
    id: int
    input: str
    expected: Verdict
    outcome: Verdict | None = None
    asked_first: bool = False
    calls: list[str] = []
    questions: list[str] = []
    deciding_args: dict[str, Any] | None = None
    error: str | None = None

    def verdict(self) -> TurnVerdict | None:
        if self.outcome is None:
            return None
        return TurnVerdict(outcome=self.outcome, asked_first=self.asked_first)

    @property
    def passed(self) -> bool:
        got = self.verdict()
        return got is not None and passes(self.expected, got)

    @property
    def asked_needlessly(self) -> bool:
        got = self.verdict()
        return got is not None and asked_needlessly(self.expected, got)


def system_prompt() -> str:
    return chat_system_prompt(None) + TASK_RUBRIC


def parse_call(name: str, arguments: str) -> ToolCall:
    """The call as the verdict reads it: the real tool behind
    ``run_capability``, marked refused when the real tool would refuse it."""
    try:
        args = json.loads(arguments)
    except ValueError:
        args = {}
    args = args if isinstance(args, dict) else {}
    if name == RUN_TOOL:
        nested = args.get("input")
        args = nested if isinstance(nested, dict) else {}
    target = called_name(name, arguments)
    if _refusal(target, args):
        target += REFUSED
    return ToolCall(name=target, args=args)


def _refusal(name: str, args: dict[str, Any]) -> str | None:
    if name != "propose_plan":
        return None
    steps = args.get("steps") or []
    titles = [str(s.get("title", "")) for s in steps if isinstance(s, dict)]
    try:
        check_plan_shape(titles, bool(args.get("split")))
    except TaskToolError as e:
        return str(e)
    return None


def tool_result(example: RubricExample, call: ToolCall, name: str, arguments: str):
    """What the tool hands back in this account."""
    if call.name == ASK_TOOL:
        return f"The user answered: {fixture.answer(example.id)}"
    if call.name.endswith(REFUSED):
        message = _refusal(call.name.removesuffix(REFUSED), call.args) or ""
        return ErrorResponse(message=message, error="refused").model_dump_json()
    return fixture.account_result(call.name, call.args) or stub_tool_result(
        name, arguments, None, fixture.TEAM
    )


def turn_over(calls: list[ToolCall]) -> bool:
    return any(call_verdict(c) for c in calls) or (
        sum(c.name == ASK_TOOL for c in calls) >= MAX_QUESTIONS
    )


async def classify(
    client: openai.AsyncOpenAI, model: str, example: RubricExample
) -> list[ToolCall]:
    """The tool calls one turn makes, up to the first that decides."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": fixture.first_message(example.id, example.input)},
    ]
    tools = [dict(t) for t in expert_tools(None)]
    calls: list[ToolCall] = []
    for _ in range(MAX_ROUNDS):
        completion = await client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=cast(list[ChatCompletionToolParam], tools),
            max_tokens=GENERATION_MAX_TOKENS,
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
        choice = completion.choices[0]
        raw = [tc for tc in choice.message.tool_calls or [] if tc.type == "function"]
        if not raw:
            break
        made = [parse_call(tc.function.name, tc.function.arguments) for tc in raw]
        calls += made
        if turn_over(calls):
            break
        messages.append(
            {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in raw],
            }
        )
        messages += [
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result(
                    example, call, tc.function.name, tc.function.arguments
                ),
            }
            for tc, call in zip(raw, made)
        ]
    return calls


def _stub_handler(
    example: RubricExample, name: str
) -> Callable[[dict[str, Any]], Awaitable[dict]]:
    async def handle(args: dict[str, Any]) -> dict:
        arguments = json.dumps(args)
        text = tool_result(example, parse_call(name, arguments), name, arguments)
        return {"content": [{"type": "text", "text": text}]}

    return handle


def subscription_options(
    model: str, workdir: str, example: RubricExample
) -> ClaudeAgentOptions:
    """The SDK turn production runs on a Claude Code login, minus the
    workspace: no built-in tools, only the platform tools, answered from the
    fixture account."""
    config = ChatConfig()
    schemas = [t["function"] for t in expert_tools(None)]
    server = create_sdk_mcp_server(
        name=MCP_SERVER,
        version="1.0.0",
        tools=[
            tool(s["name"], s.get("description", ""), s.get("parameters", {}))(
                _stub_handler(example, s["name"])
            )
            for s in schemas
        ],
    )
    return ClaudeAgentOptions(
        system_prompt=system_prompt(),
        mcp_servers={MCP_SERVER: server},
        tools=[],
        allowed_tools=[MCP_PREFIX + s["name"] for s in schemas],
        setting_sources=[],
        extra_args={"strict-mcp-config": None},
        model=model,
        max_turns=MAX_ROUNDS,
        cwd=workdir,
        env={
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
            **host_login_env(),
        },
        effort=config.claude_agent_thinking_effort or "high",
        thinking={"type": "adaptive"},
    )


async def classify_subscription(model: str, workdir: str, example: RubricExample):
    options = subscription_options(model, workdir, example)
    prompt = fixture.first_message(example.id, example.input)
    calls: list[ToolCall] = []
    try:
        async with contextlib.aclosing(
            cast(AsyncGenerator[Any, None], query(prompt=prompt, options=options))
        ) as stream:
            async for event in stream:
                if not isinstance(event, AssistantMessage):
                    continue
                calls += [
                    parse_call(
                        block.name.removeprefix(MCP_PREFIX), json.dumps(block.input)
                    )
                    for block in event.content
                    if isinstance(block, ToolUseBlock)
                ]
                if turn_over(calls):
                    break
    except ResultError as exc:
        # Running out of rounds ends the API loop silently; the CLI raises.
        if "maximum number of turns" not in str(exc):
            raise
    return calls


async def score(classifier: Classifier, example: RubricExample) -> ExampleResult:
    result = ExampleResult(id=example.id, input=example.input, expected=example.verdict)
    for attempt in range(2):
        try:
            calls = await classifier(example)
        except (openai.OpenAIError, ClaudeSDKError) as exc:
            logger.warning(f"[task-rubric] #{example.id} attempt {attempt}: {exc}")
            result.error = f"{type(exc).__name__}: {exc}"
            continue
        got = turn_verdict(calls)
        deciding = next((c for c in calls if call_verdict(c)), None)
        return result.model_copy(
            update={
                "outcome": got.outcome,
                "asked_first": got.asked_first,
                "calls": [c.name for c in calls],
                "questions": [_question_text(c) for c in calls if c.name == ASK_TOOL],
                "deciding_args": deciding.args if deciding else None,
                "error": None,
            }
        )
    return result


def _question_text(call: ToolCall) -> str:
    questions = call.args.get("questions")
    if not isinstance(questions, list):
        return json.dumps(call.args)
    return " / ".join(
        str(q.get("question", "")) for q in questions if isinstance(q, dict)
    )


async def run(
    ids: list[int] | None,
    model: str | None,
    concurrency: int,
    out: Path,
    subscription: bool,
):
    config = ChatConfig()
    subscription = subscription or config.use_claude_code_subscription
    slug = model or (await resolve_chat_model(config)).transport_slug
    examples = [e for e in load_examples() if not ids or e.id in ids]
    gate = asyncio.Semaphore(concurrency)
    with tempfile.TemporaryDirectory(prefix="task-rubric-") as workdir:
        if subscription:
            validate_subscription()

            async def classifier(example: RubricExample) -> list[ToolCall]:
                return await classify_subscription(slug, workdir, example)

        else:
            client = chat_client(config)

            async def classifier(example: RubricExample) -> list[ToolCall]:
                return await classify(client, slug, example)

        async def one(example: RubricExample) -> ExampleResult:
            async with gate:
                return await score(classifier, example)

        results = await asyncio.gather(*(one(e) for e in examples))
    out.write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
    label = f"{slug} (Claude Code subscription)" if subscription else slug
    print_summary(label, results, out)


def print_summary(model: str, results: list[ExampleResult], out: Path) -> None:
    total = len(results)
    passed = [r for r in results if r.passed]
    clean = [r for r in passed if not r.asked_needlessly]
    print(f"\n{model}")
    print(f"  right outcome:                    {len(passed)}/{total}")
    print(f"  right outcome, no needless ask:   {len(clean)}/{total}")
    print(
        "  asked a question it didn't need:  "
        f"{sum(r.asked_needlessly for r in results)}/{total}\n"
    )
    totals = Counter(r.expected for r in results)
    hits = Counter(r.expected for r in passed)
    for verdict in Verdict:
        if totals[verdict]:
            print(f"  {verdict.value:<22} {hits[verdict]}/{totals[verdict]}")
    misses = [r for r in results if not r.passed]
    if misses:
        print("\nMisses:")
    for r in misses:
        got = r.outcome.value if r.outcome else f"error ({r.error})"
        asked = " after asking" if r.asked_first else ""
        print(f"  #{r.id} {r.input!r}: expected {r.expected.value}, got {got}{asked}")
        print(f"      calls: {', '.join(r.calls) or 'none'}")
        for question in r.questions:
            print(f"      asked: {question}")
    needless = [r for r in passed if r.asked_needlessly]
    if needless:
        print("\nRight outcome, but asked first:")
    for r in needless:
        print(f"  #{r.id} {r.input!r}: {' | '.join(r.questions)}")
    print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ids", nargs="*", type=int, help="example ids; default all")
    parser.add_argument("--model", help="override the routed chat model")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("task_rubric_results.json"))
    parser.add_argument(
        "--subscription",
        action="store_true",
        help="run turns through the host's Claude Code login instead of an API key",
    )
    args = parser.parse_args()
    asyncio.run(
        run(args.ids or None, args.model, args.concurrency, args.out, args.subscription)
    )


if __name__ == "__main__":
    main()
