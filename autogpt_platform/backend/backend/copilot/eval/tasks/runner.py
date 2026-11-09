"""Run one case: Otto's real prompt and tools, against the simulated world.

The system prompt and tool list come from production (``eval/style``
assembly: plain Otto with the SDK supplement, delegation and memory, plus
the case's approval mode). The turn follows production's rules for closing
(``turn_end``): a valid question ends the turn and refuses every later call;
after a started task only more tasks and a closing line may follow. Which
questions may be asked is decided by the production rules
(``tools/question_rules``). Model calls go through ``llm.Backend``. A simulated user answers
questions from the case's answer sheet, in the persona's voice.
"""

import json
import time
from typing import Any

from backend.copilot.autopilot_mode import AutopilotMode
from backend.copilot.eval.style.assembly import chat_system_prompt
from backend.copilot.eval.style.generation import expert_tools
from backend.copilot.prompting import approval_mode_supplement
from backend.copilot.service import format_user_context_prefix
from backend.copilot.tools import question_rules
from backend.copilot.tools.ask_question import parse_questions
from backend.copilot.turn_end import ENDED_REFUSAL, WRAP_UP_REFUSAL
from backend.copilot.untrusted_text import flag_instructions

from .llm import Backend, Call, Reply
from .models import Case, Question, Run, ToolCall, TurnRecord, World
from .simulator import Simulator

MAX_ROUNDS = 10
MAX_ANSWERS_PER_TURN = 2
TODAY = "Monday 2026-11-09"


class Models:
    """The models a run uses: Otto's, and the simulated user's."""

    def __init__(self, backend: Backend, otto: str, user: str) -> None:
        self.backend, self.otto, self.user = backend, otto, user


async def run_case(case: Case, world: World, models: Models) -> Run:
    mode: AutopilotMode = case.approval_mode or world.approval_mode
    sim = Simulator(world, mode)
    system = chat_system_prompt(None) + approval_mode_supplement(mode, True)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    tools = [dict(t) for t in expert_tools(None)]
    run = Run(case_id=case.id)
    try:
        for index, scripted in enumerate(case.turns):
            text = (context_prefix(world, mode) if index == 0 else "") + scripted.user
            record = await _turn(
                text, scripted.user, messages, tools, sim, mode, run, models
            )
            answers = 0
            while record.ended_by == "question" and answers < MAX_ANSWERS_PER_TURN:
                answers += 1
                reply = await simulated_answer(case, world, record.questions, models)
                record = await _turn(
                    reply, reply, messages, tools, sim, mode, run, models
                )
    except Exception as e:  # a broken case is scored, not fatal to the run
        run.error = f"{type(e).__name__}: {e}"[:500]
    run.writes = sim.writes
    return run


async def _turn(
    text: str,
    shown: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    sim: Simulator,
    mode: AutopilotMode,
    run: Run,
    models: Models,
) -> TurnRecord:
    started = time.monotonic()
    record = TurnRecord(user=shown)
    run.turns.append(record)
    messages.append({"role": "user", "content": text})
    replies: list[str] = []
    for _ in range(MAX_ROUNDS):
        record.rounds += 1
        message = await _complete(models, messages, tools, run)
        if message.content:
            replies.append(message.content)
        calls = message.tool_calls
        messages.append(_assistant(message.content, calls))
        if not calls:
            break
        for call in calls:
            result = execute_call(call.name, call.arguments, sim, mode, run, record)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
        if record.ended_by == "question":
            break
    else:
        record.ended_by = "round_cap"
    record.reply = "\n\n".join(replies)
    record.seconds = round(time.monotonic() - started, 2)
    return record


def execute_call(
    name: str,
    raw_args: str,
    sim: Simulator,
    mode: AutopilotMode,
    run: Run,
    record: TurnRecord,
) -> str:
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        args = {}
    if record.ended_by == "question":
        record.tool_calls.append(
            ToolCall(name=name, arguments=args, refused=ENDED_REFUSAL)
        )
        return ENDED_REFUSAL
    if record.ended_by == "task_started" and name != "start_task":
        record.tool_calls.append(
            ToolCall(name=name, arguments=args, refused=WRAP_UP_REFUSAL)
        )
        return WRAP_UP_REFUSAL
    if name == "ask_question":
        return _ask(args, mode, run, record)
    call, result = sim.call(name, args)
    result = flag_instructions(result)  # as BaseTool.execute does in production
    call.result_preview = result[:300]
    record.tool_calls.append(call)
    if call.name == "start_task":
        record.ended_by = "task_started"
    return result


def _ask(
    args: dict[str, Any], mode: AutopilotMode, run: Run, record: TurnRecord
) -> str:
    call = ToolCall(name="ask_question", arguments=args)
    record.tool_calls.append(call)
    questions = parse_questions(args.get("questions") or [])
    if not questions:
        call.refused = "ask_question requires at least one valid question"
        return call.refused
    in_a_row = _questions_in_a_row(run.turns[:-1])
    if refused := question_rules.refusal(questions, mode, in_a_row):
        call.refused = refused
        return refused
    record.questions = [
        Question(question=q.question, reason=q.reason, default_guess=q.default_guess)
        for q in questions
    ]
    record.ended_by = "question"
    return json.dumps({"type": "clarification_needed", "message": "Shown to the user."})


def _questions_in_a_row(turns: list[TurnRecord]) -> int:
    count = 0
    for turn in reversed(turns):
        if turn.ended_by != "question":
            break
        count += 1
    return count


async def _complete(
    models: Models,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    run: Run,
) -> Reply:
    reply = await models.backend.chat(models.otto, messages, tools)
    run.prompt_tokens += reply.prompt_tokens
    run.completion_tokens += reply.completion_tokens
    return reply


def _assistant(content: str, calls: list[Call]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in calls
        ]
    return message


def context_prefix(world: World, mode: AutopilotMode) -> str:
    """What production prepends to the first message: the user's business,
    what Otto remembers, the team and the accounts, in a ``<user_context>``."""
    lines = [
        f"Today: {TODAY}. Approval mode: {mode}.",
        f"User: {json.dumps(world.user)}",
        f"Company: {json.dumps(world.company)}",
        f"Business: {world.business_understanding}",
        f"Team: {json.dumps(world.team)}",
        f"Hired experts: {json.dumps(world.experts)}",
        "Connected accounts: "
        + ", ".join(f"{a.provider} ({a.account})" for a in world.connected_accounts),
        "Not connected: " + ", ".join(world.not_connected),
        "Open tasks: " + json.dumps([t.model_dump() for t in world.tasks]),
        "Remembered: " + " | ".join(world.memories),
    ]
    return format_user_context_prefix("\n".join(lines))


async def simulated_answer(
    case: Case, world: World, questions: list[Question], models: Models
) -> str:
    """The user's reply to Otto's questions, from the answer sheet, in the
    persona's own voice."""
    sheet = "\n".join(f"- {a.topic}: {a.answer}" for a in case.answer_sheet) or "(none)"
    asked = "\n".join(f"- {q.question}" for q in questions)
    prompt = (
        f"You are {world.user.get('name', 'the user')}, "
        f"{world.user.get('role', '')} at {world.company.get('name', 'a company')}. "
        f"You write like this: {world.user.get('writing_style', 'short and casual')}.\n"
        f"Your assistant asked:\n{asked}\n\nWhat you know:\n{sheet}\n\n"
        f"Answer only from what you know. For anything not covered, reply "
        f"'{case.default_answer}'. One or two lines, in your own voice, no preamble."
    )
    answer = await models.backend.text(
        models.user, "You play a busy user replying to their assistant.", prompt, False
    )
    return (answer or case.default_answer).strip()
