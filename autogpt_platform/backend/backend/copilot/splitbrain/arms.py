"""The two architectures under test, on one shared model budget.

``run_baseline`` is today's AutoPilot shape: one transcript that reasons and
acts, so every block schema it reads stays in front of it for the rest of the
run. ``run_split`` is the proposal: a reasoner that holds the goal and can only
``dispatch``, and an executor that holds the tools and can only ``report``.

Both arms get the same model, the same work tools, the same task and the same
ceiling on model calls, so the only thing that differs is where the tokens land.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import anthropic

from .metrics import Meter
from .protocol import INTENT_TOOL_SCHEMA, REPORT_TOOL_SCHEMA
from .tasks import TaskSpec
from .world import WORK_TOOL_SCHEMAS, World

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"
MAX_TOKENS = 8000
# Shared ceiling. Neither arm can win by being allowed more thinking.
DEFAULT_CALL_BUDGET = 45

ExecutorMode = Literal["persistent", "per_intent"]

_BUILDER_RULES = """\
You build AutoGPT agent graphs.

A graph is nodes and links. A node is {"id", "block_id", "input_default": {}}.
A link is {"id", "source_id", "sink_id", "source_name", "sink_name"}, where
source_name is an output field of the source block and sink_name is an input
field of the sink block. Ids you invent must be UUIDs.

Work in this order: find_block to get ids, get_block_schema for the exact field
names and types of every block you will wire, write_graph, then validate_graph.
Fix what validate_graph reports and write again. You are not done until
validate_graph returns valid=true.

Do not guess field names. A link whose source_name is not a real output of that
block fails validation."""

_FINISH_TOOL: dict[str, Any] = {
    "name": "finish",
    "description": "Declare the task complete. Only call this after validate_graph returned valid=true.",
    "input_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}


@dataclass
class RunResult:
    arm: str
    task: str
    success: bool
    score: dict[str, Any]
    model_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    peak_context: int
    reasoner_peak: int
    by_transcript: dict[str, dict[str, int]]
    tool_calls: list[str]
    intents: int
    wall_seconds: float
    stop_reason: str
    # Work tools the TOP-LEVEL transcript ran itself. Zero by construction in
    # the structural split; the whole question in the prompted arm.
    leaked_tool_calls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_baseline(
    client: anthropic.Anthropic,
    task: TaskSpec,
    call_budget: int = DEFAULT_CALL_BUDGET,
) -> RunResult:
    """One transcript, all tools — the shape AutoPilot has today."""
    world = World()
    meter = Meter()
    started = time.monotonic()
    tools = [*WORK_TOOL_SCHEMAS, _FINISH_TOOL]
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.brief}]
    stop_reason = "budget_exhausted"

    while meter.model_calls < call_budget:
        response = _call_model(client, _BUILDER_RULES, messages, tools, meter, "main")
        messages.append({"role": "assistant", "content": response.content})
        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses:
            stop_reason = "model_stopped_without_tool_call"
            break
        if any(u.name == "finish" for u in uses):
            stop_reason = "finished"
            break
        messages.append({"role": "user", "content": _run_work_tools(world, uses)})

    return _assemble("baseline", task, world, meter, started, stop_reason, intents=0)


def run_split(
    client: anthropic.Anthropic,
    task: TaskSpec,
    call_budget: int = DEFAULT_CALL_BUDGET,
    executor_mode: ExecutorMode = "persistent",
) -> RunResult:
    """Two transcripts, structurally separated: the reasoner has no work tools.

    ``persistent`` keeps one executor transcript for the whole run, so it
    accumulates like the baseline does. ``per_intent`` starts the executor fresh
    on every intent — the graph lives in the tool-backed store, so nothing is
    lost by dropping the transcript, and that is the variant where context
    actually stops growing.
    """
    return _two_transcript_run(
        client,
        task,
        call_budget,
        executor_mode,
        arm=f"split-{executor_mode}",
        reasoner_tools=[INTENT_TOOL_SCHEMA, _FINISH_TOOL],
        reasoner_rules=_reasoner_rules(task),
    )


def run_prompted_delegation(
    client: anthropic.Anthropic,
    task: TaskSpec,
    call_budget: int = DEFAULT_CALL_BUDGET,
    executor_mode: ExecutorMode = "per_intent",
) -> RunResult:
    """Today's alternative: one transcript told to delegate, but not stopped.

    Same executor and same protocol as the structural split; the difference is
    that the top-level transcript still holds every work tool and only an
    instruction keeps it from using them. What this arm measures is whether the
    instruction holds — every work tool it calls itself lands in the context the
    instruction is trying to protect.
    """
    return _two_transcript_run(
        client,
        task,
        call_budget,
        executor_mode,
        arm="prompted-delegation",
        reasoner_tools=[INTENT_TOOL_SCHEMA, *WORK_TOOL_SCHEMAS, _FINISH_TOOL],
        reasoner_rules=_prompted_delegation_rules(task),
    )


def _two_transcript_run(
    client: anthropic.Anthropic,
    task: TaskSpec,
    call_budget: int,
    executor_mode: ExecutorMode,
    *,
    arm: str,
    reasoner_tools: list[dict[str, Any]],
    reasoner_rules: str,
) -> RunResult:
    world = World()
    meter = Meter()
    started = time.monotonic()
    reasoner: list[dict[str, Any]] = [{"role": "user", "content": task.brief}]
    executor: list[dict[str, Any]] = []
    intents = 0
    leaked: list[str] = []
    stop_reason = "budget_exhausted"

    while meter.model_calls < call_budget:
        response = _call_model(
            client, reasoner_rules, reasoner, reasoner_tools, meter, "reasoner"
        )
        reasoner.append({"role": "assistant", "content": response.content})
        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses:
            stop_reason = "model_stopped_without_tool_call"
            break
        if any(u.name == "finish" for u in uses):
            stop_reason = "finished"
            break

        results = []
        for use in uses:
            if use.name == "Agent":
                intents += 1
                if executor_mode == "per_intent":
                    executor = []
                results.append(
                    _tool_result(
                        use.id,
                        _run_executor(
                            client, world, meter, executor, use.input, call_budget
                        ),
                    )
                )
            else:
                # A work tool called from the top-level transcript: the leak the
                # instruction is supposed to prevent, and it lands in context.
                leaked.append(use.name)
                results.extend(_run_work_tools(world, [use]))
        reasoner.append({"role": "user", "content": results})

    result = _assemble(arm, task, world, meter, started, stop_reason, intents)
    result.leaked_tool_calls = leaked
    return result


def _run_executor(
    client: anthropic.Anthropic,
    world: World,
    meter: Meter,
    executor: list[dict[str, Any]],
    intent: dict[str, Any],
    call_budget: int,
) -> dict[str, Any]:
    """Run one intent to a report. Everything it reads stays in its own list."""
    max_steps = max(1, min(int(intent.get("max_steps") or 10), 15))
    executor.append({"role": "user", "content": _format_intent(intent)})
    tools = [*WORK_TOOL_SCHEMAS, REPORT_TOOL_SCHEMA]
    steps = 0

    while True:
        out_of_room = steps >= max_steps or meter.model_calls >= call_budget - 1
        response = _call_model(
            client,
            _EXECUTOR_RULES,
            executor,
            tools,
            meter,
            "executor",
            tool_choice={"type": "tool", "name": "report"} if out_of_room else None,
        )
        executor.append({"role": "assistant", "content": response.content})
        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses:
            return {
                "status": "partial",
                "summary": _text_of(response) or "executor ended without a report",
                "problems": ["executor stopped without calling report"],
                "steps_used": steps,
            }
        # Every tool_use needs its tool_result appended before returning, or a
        # persistent executor transcript starts the next intent malformed.
        results: list[dict[str, Any]] = []
        report: dict[str, Any] | None = None
        for use in uses:
            if use.name == "report":
                report = {**use.input, "steps_used": steps}
                results.append(_tool_result(use.id, {"received": True}))
            else:
                results.extend(_run_work_tools(world, [use]))
                steps += 1
        executor.append({"role": "user", "content": results})
        if report is not None:
            return report


def _call_model(
    client: anthropic.Anthropic,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    meter: Meter,
    transcript: str,
    tool_choice: dict[str, Any] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
        "tools": tools,
        # Off on both arms: the variable under test is where context lands, and
        # thinking blocks would add spend that differs by arm for other reasons.
        "thinking": {"type": "disabled"},
    }
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    response = client.messages.create(**kwargs)
    meter.record(transcript, response.usage.input_tokens, response.usage.output_tokens)
    return response


def _run_work_tools(world: World, uses: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "tool_result",
            "tool_use_id": use.id,
            "content": world.call(use.name, use.input),
        }
        for use in uses
    ]


def _tool_result(use_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": use_id,
        "content": json.dumps(payload, default=str),
    }


def _format_intent(intent: dict[str, Any]) -> str:
    # Accepts the CLI Agent tool's {description, prompt} as well as the rig's
    # own {goal, acceptance}, so both tiers drive the same executor.
    lines = [f"INTENT: {intent.get('goal') or intent.get('prompt', '')}"]
    lines.append(
        f"DONE WHEN: {intent.get('acceptance') or intent.get('description', '')}"
    )
    if intent.get("context"):
        lines.append(f"CONTEXT FROM THE REASONER: {intent['context']}")
    lines.append(
        f"Budget: {intent.get('max_steps', 10)} tool calls. Call report when done "
        "or when you need a decision."
    )
    return "\n".join(lines)


def _text_of(response: Any) -> str:
    return " ".join(b.text for b in response.content if b.type == "text").strip()


def _assemble(
    arm: str,
    task: TaskSpec,
    world: World,
    meter: Meter,
    started: float,
    stop_reason: str,
    intents: int,
) -> RunResult:
    score = task.score(world)
    return RunResult(
        arm=arm,
        task=task.key,
        success=bool(score["success"]),
        score=score,
        model_calls=meter.model_calls,
        input_tokens=meter.input_tokens,
        output_tokens=meter.output_tokens,
        cost_usd=meter.cost_usd,
        peak_context=meter.peak_context(),
        reasoner_peak=meter.peak_context("reasoner") or meter.peak_context("main"),
        by_transcript=meter.by_transcript(),
        tool_calls=world.tool_calls,
        intents=intents,
        wall_seconds=round(time.monotonic() - started, 1),
        stop_reason=stop_reason,
    )


def _reasoner_rules(task: TaskSpec) -> str:
    return f"""\
You are the REASONER half of a two-part agent. You hold the goal and make every
decision. You have no tools for doing the work — you cannot search blocks, read
schemas or write a graph.

Your executor can. It has the full toolset and its own context. You direct it
with `dispatch`, one unit of work at a time, and it reports back what happened.
You see its report and nothing else: not its tool calls, not the schemas it read,
not the raw errors. Ask for what you need to decide.

Because you never see the details, write intents that are self-contained: say
what to achieve and what "done" looks like, and put any decision you have already
made into `context`. If a report comes back `blocked` or `partial`, decide what
to change and dispatch again.

{_BUILDER_RULES}

Call `finish` only once a report tells you validate_graph returned valid=true."""


def _prompted_delegation_rules(task: TaskSpec) -> str:
    return f"""\
You are the top-level agent. Your context is the scarcest resource you have: it
holds the goal, the plan and every decision you have made, and it has to stay
readable for the whole task.

So DELEGATE ZEALOUSLY. You have an executor with the same tools you do and its
own separate context. Every unit of concrete work — searching for blocks, reading
schemas, writing the graph, running validation, fixing errors — goes to it via
`dispatch`. It reports back a summary; the bulk of what it read never enters your
context.

You do still hold the work tools, for the rare case where delegating is plainly
wasteful. Using them is a real cost: the entire result lands in your context and
stays there. Reading even one block schema yourself can cost you thousands of
tokens you will carry for the rest of the task. Prefer `dispatch` in essentially
every case, and reach for a tool yourself only when you have a specific reason.

{_BUILDER_RULES}

Call `finish` only once you know validate_graph returned valid=true."""


_EXECUTOR_RULES = f"""\
You are the EXECUTOR half of a two-part agent. You do the work; the reasoner
holds the goal and decides. You get one intent at a time and you have the tools.

Do the intent with as few tool calls as you can, then call `report`. The reasoner
sees only your report — it cannot see your tool calls, the schemas you read or the
errors you got. If something blocks you or needs a decision, put it in `problems`
with enough detail to decide on, and report `blocked` rather than guessing at the
goal. Put ids and counts in `artifacts`, never payloads.

{_BUILDER_RULES}"""
