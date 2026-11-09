"""Run one case on Claude with Otto's tools as real tools, like production.

Production's SDK engine is the Claude Code CLI with Otto's tools served over
MCP and a PostToolUse hook that stops the turn after a question. This does
the same with ``claude -p``: ``bridge.py`` serves the real tool schemas and
forwards each call here, where the simulated world answers it and the turn
rules apply (``runner.execute_call``); ``hook.py`` asks here whether to stop.
Later turns resume the same Claude session, as production does.
"""

import asyncio
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from backend.copilot.autopilot_mode import AutopilotMode
from backend.copilot.eval.style.assembly import chat_system_prompt
from backend.copilot.eval.style.generation import expert_tools
from backend.copilot.prompting import approval_mode_supplement

from ..models import Case, Run, TurnRecord, World
from ..runner import (
    MAX_ANSWERS_PER_TURN,
    Models,
    context_prefix,
    execute_call,
    simulated_answer,
)
from ..simulator import Simulator

HERE = Path(__file__).parent
MAX_MODEL_TURNS = 12
TURN_TIMEOUT = 900


class _Session:
    """One case's live state: the world, the run, and the turn in progress."""

    def __init__(self, case: Case, world: World) -> None:
        self.mode: AutopilotMode = case.approval_mode or world.approval_mode
        self.sim = Simulator(world, self.mode)
        self.run = Run(case_id=case.id)
        self.record: TurnRecord | None = None
        self.session_id: str | None = None

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        message = json.loads(await reader.readline())
        writer.write(json.dumps(self._answer(message)).encode() + b"\n")
        await writer.drain()
        writer.close()

    def _answer(self, message: dict[str, Any]) -> dict[str, Any]:
        record = self.record
        if record is None:
            return {"result": "No turn is running.", "output": {}}
        if message["kind"] == "call":
            arguments = json.dumps(message.get("arguments") or {})
            result = execute_call(
                message["name"], arguments, self.sim, self.mode, self.run, record
            )
            return {"result": result}
        # post_tool: production's hook returns continue=false after a question
        return {"output": {"continue": False} if record.ended_by == "question" else {}}


async def run_case_native(
    case: Case, world: World, models: Models, binary: str = "claude"
) -> Run:
    session = _Session(case, world)
    workdir = Path(tempfile.mkdtemp(prefix=f"otto-eval-{case.id}-")).resolve()
    server = await asyncio.start_server(session.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    args = _cli_args(workdir, port, session.mode, binary, models.otto)
    try:
        for index, scripted in enumerate(case.turns):
            prefix = context_prefix(world, session.mode) if index == 0 else ""
            record = await _turn(
                session, args, workdir, prefix + scripted.user, scripted.user
            )
            answers = 0
            while record.ended_by == "question" and answers < MAX_ANSWERS_PER_TURN:
                answers += 1
                reply = await simulated_answer(case, world, record.questions, models)
                record = await _turn(session, args, workdir, reply, reply)
    except Exception as e:  # a broken case is scored, not fatal to the run
        session.run.error = f"{type(e).__name__}: {e}"[:500]
    finally:
        server.close()
        shutil.rmtree(workdir, ignore_errors=True)
        _forget_claude_project(workdir)
    session.run.writes = session.sim.writes
    return session.run


def _cli_args(
    workdir: Path, port: int, mode: AutopilotMode, binary: str, model: str
) -> list[str]:
    tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "parameters": t["function"].get("parameters", {}),
        }
        for t in expert_tools(None)
    ]
    (workdir / "tools.json").write_text(json.dumps(tools))
    env = {"OTTO_EVAL_TOOLS": str(workdir / "tools.json"), "OTTO_EVAL_PORT": str(port)}
    server = {"command": sys.executable, "args": [str(HERE / "bridge.py")], "env": env}
    (workdir / "mcp.json").write_text(json.dumps({"mcpServers": {"otto": server}}))
    hook = f"OTTO_EVAL_PORT={port} {sys.executable} {HERE / 'hook.py'}"
    post_tool = {
        "matcher": "mcp__otto__.*",
        "hooks": [{"type": "command", "command": hook}],
    }
    (workdir / "settings.json").write_text(
        json.dumps({"hooks": {"PostToolUse": [post_tool]}})
    )
    system = chat_system_prompt(None) + approval_mode_supplement(mode, True)
    return [
        binary, "-p", "--output-format", "stream-json", "--verbose",
        "--model", model, "--system-prompt", system,
        "--tools", "", "--mcp-config", str(workdir / "mcp.json"), "--strict-mcp-config",
        "--allowedTools", "mcp__otto", "--setting-sources", "",
        "--settings", str(workdir / "settings.json"),
        "--max-turns", str(MAX_MODEL_TURNS),
    ]  # fmt: skip


async def _turn(
    session: _Session, args: list[str], workdir: Path, text: str, shown: str
) -> TurnRecord:
    record = TurnRecord(user=shown)
    session.run.turns.append(record)
    session.record = record
    started = time.monotonic()
    resume = ["--resume", session.session_id] if session.session_id else []
    process = await asyncio.create_subprocess_exec(
        *args,
        *resume,
        cwd=workdir,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            process.communicate(text.encode()), TURN_TIMEOUT
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError("claude turn timed out")
    finally:
        session.record = None
    texts, result = read_stream(out.decode(errors="replace"))
    if result is None:
        raise RuntimeError(f"claude failed: {err.decode(errors='replace')[-300:]}")
    session.session_id = result.get("session_id") or session.session_id
    usage = result.get("usage") or {}
    session.run.prompt_tokens += int(usage.get("input_tokens") or 0) + int(
        usage.get("cache_read_input_tokens") or 0
    )
    session.run.completion_tokens += int(usage.get("output_tokens") or 0)
    record.reply = "\n\n".join(t for t in texts if t.strip())
    record.rounds = int(result.get("num_turns") or 1)
    if record.ended_by == "reply" and result.get("subtype") == "error_max_turns":
        record.ended_by = "round_cap"
    record.seconds = round(time.monotonic() - started, 2)
    return record


def read_stream(stdout: str) -> tuple[list[str], dict[str, Any] | None]:
    """The assistant's text blocks and the final result event of a
    ``claude -p --output-format stream-json`` run."""
    texts: list[str] = []
    result: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            texts += [
                block.get("text", "")
                for block in event.get("message", {}).get("content", [])
                if block.get("type") == "text"
            ]
        elif event.get("type") == "result":
            result = event
    return texts, result


def _forget_claude_project(workdir: Path) -> None:
    """Remove the session transcripts Claude Code kept for this run's folder."""
    name = re.sub(r"[^A-Za-z0-9]", "-", str(workdir))
    shutil.rmtree(Path.home() / ".claude" / "projects" / name, ignore_errors=True)
