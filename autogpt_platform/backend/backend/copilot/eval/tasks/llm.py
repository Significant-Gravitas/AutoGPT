"""Where the eval's model calls go: the OpenAI API, or the Cursor CLI.

Otto, the simulated user and the reviewer all call a ``Backend``. The OpenAI
backend uses native tool calling. The Cursor backend (``cursor-agent -p``,
on the user's Cursor subscription: Auto, Grok, Claude, ...) has no tool
calling of its own, so the tool schemas are written into the prompt and the
model answers with one JSON object naming its tool calls. Same prompt, same
tools, a different calling mechanism: scores from the two are not directly
comparable.
"""

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, Protocol, cast

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel, Field

_ATTEMPTS = 7  # a rate-limited burst backs off for up to a minute
SANDBOX = Path("/tmp/cursor-eval-sandbox")


class Call(BaseModel):
    id: str
    name: str
    arguments: str = "{}"


class Reply(BaseModel):
    content: str = ""
    tool_calls: list[Call] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(Protocol):
    async def chat(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply: ...

    async def text(self, model: str, system: str, user: str, json_out: bool) -> str: ...


class OpenAIBackend:
    def __init__(self, client: openai.AsyncOpenAI) -> None:
        self.client = client

    async def chat(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        completion = await _retry(
            lambda: self.client.chat.completions.create(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                tools=cast(list[ChatCompletionToolParam], tools),
                max_completion_tokens=6000,
                timeout=180,
            )
        )
        message = completion.choices[0].message
        usage = completion.usage
        return Reply(
            content=message.content or "",
            tool_calls=[
                Call(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in message.tool_calls or []
                if tc.type == "function"
            ],
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    async def text(self, model: str, system: str, user: str, json_out: bool) -> str:
        completion = await _retry(
            lambda: self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=(
                    {"type": "json_object"} if json_out else {"type": "text"}
                ),
                max_completion_tokens=3000,
                timeout=180,
            )
        )
        return completion.choices[0].message.content or ""


class CursorBackend:
    """``cursor-agent -p --mode ask`` (read-only Q&A) in an empty, sandboxed
    folder, so the agent's own tools have nothing to act on."""

    def __init__(self, binary: str = "cursor-agent", timeout: float = 300) -> None:
        self.binary, self.timeout = binary, timeout
        SANDBOX.mkdir(parents=True, exist_ok=True)

    async def chat(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        raw = await self._run(model, render_chat_prompt(messages, tools))
        return parse_reply(raw)

    async def text(self, model: str, system: str, user: str, json_out: bool) -> str:
        suffix = "\n\nAnswer with the JSON object only." if json_out else ""
        return await self._run(model, f"{system}\n\n{user}{suffix}")

    async def _run(self, model: str, prompt: str) -> str:
        for attempt in range(4):
            process = await asyncio.create_subprocess_exec(
                self.binary,
                "-p",
                "--mode",
                "ask",
                "--trust",
                "--sandbox",
                "enabled",
                "--output-format",
                "json",
                "--model",
                model,
                prompt,
                cwd=SANDBOX,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(process.communicate(), self.timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
                continue
            text = cli_result(out.decode(errors="replace"))
            if process.returncode == 0 and text is not None:
                return text
            if attempt == 3:
                detail = (err or out).decode(errors="replace")[-300:]
                raise RuntimeError(f"cursor-agent failed: {detail}")
            await asyncio.sleep(5 * (attempt + 1))
        raise RuntimeError("cursor-agent timed out")


class ClaudeBackend:
    """``claude -p`` (Claude Code on the user's Claude subscription) with all
    of its own tools, settings and MCP servers off. Otto's system prompt
    replaces Claude Code's (``--system-prompt``), so the model sees what
    production sends; the tools travel in the prompt, as for Cursor."""

    def __init__(self, binary: str = "claude", timeout: float = 600) -> None:
        self.binary, self.timeout = binary, timeout
        SANDBOX.mkdir(parents=True, exist_ok=True)

    async def chat(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        raw = await self._run(model, system, render_chat_prompt(rest, tools))
        return parse_reply(raw)

    async def text(self, model: str, system: str, user: str, json_out: bool) -> str:
        suffix = "\n\nAnswer with the JSON object only." if json_out else ""
        if not user.strip():  # claude -p needs a prompt on stdin
            system, user = "You are a helpful assistant.", system
        return await self._run(model, system, user + suffix)

    async def _run(self, model: str, system: str, prompt: str) -> str:
        args = [
            self.binary, "-p", "--model", model, "--output-format", "json",
            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--no-session-persistence", "--system-prompt", system,
        ]  # fmt: skip
        for attempt in range(4):
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=SANDBOX,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(
                    process.communicate(prompt.encode()), self.timeout
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                continue
            text = cli_result(out.decode(errors="replace"))
            if process.returncode == 0 and text is not None:
                return text
            if attempt == 3:
                detail = (err or out).decode(errors="replace")[-300:]
                raise RuntimeError(f"claude failed: {detail}")
            await asyncio.sleep(10 * (attempt + 1))
        raise RuntimeError("claude timed out")


def cli_result(stdout: str) -> str | None:
    """The answer text from ``cursor-agent --output-format json``."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("result"), str):
            return None if data.get("is_error") else data["result"]
    return None


_PROTOCOL = """# How to answer in this environment
You cannot call tools directly here. Answer with ONE JSON object and nothing
else:
{"text": "<what you say to the user now; may be empty>",
 "tool_calls": [{"name": "<tool name>", "arguments": {<arguments>}}]}
Use "tool_calls": [] when you are done for this turn. Tool results come back
in the next message. Never write tool calls inside "text", and never use
XML or <invoke> function-call syntax: only this JSON object. Saying you will
do something does not do it: put the calls in the same answer."""


def render_chat_prompt(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> str:
    """The whole conversation as one prompt: system prompt, tool list, the
    turns so far, and the JSON answer protocol."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    tool_lines = [
        json.dumps(
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "parameters": t["function"].get("parameters", {}),
            },
            separators=(",", ":"),
        )
        for t in tools
    ]
    turns = [_render_turn(m) for m in messages if m["role"] != "system"]
    head = f"{system}\n\n" if system else ""
    return (
        f"{head}# Tools\n"
        + "\n".join(tool_lines)
        + "\n\n# Conversation so far\n"
        + "\n\n".join(turns)
        + f"\n\n{_PROTOCOL}"
    )


def _render_turn(message: dict[str, Any]) -> str:
    role = message["role"]
    if role == "tool":
        return f"[tool result {message.get('tool_call_id', '')}]\n{message['content']}"
    if role == "assistant":
        calls = [
            {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in message.get("tool_calls") or []
        ]
        body = json.dumps({"text": message.get("content") or "", "tool_calls": calls})
        return f"[you]\n{body}"
    return f"[user]\n{message['content']}"


def parse_reply(raw: str) -> Reply:
    """Read the model's JSON answer. Claude may still write calls in its own
    ``<invoke name=...>`` form; those count as calls too. Plain text with
    neither is a reply without tool calls."""
    invoked = _invoked_calls(raw)
    text = _INVOKE.sub("", raw)
    text = re.sub(r"</?function_calls>", "", text).strip()
    data = first_json_object(text)
    if data is None:
        return Reply(content=text, tool_calls=invoked)
    calls = [
        _call(str(c.get("name", "")), c.get("arguments") or {})
        for c in data.get("tool_calls") or []
        if isinstance(c, dict) and c.get("name")
    ]
    return Reply(content=str(data.get("text") or ""), tool_calls=calls + invoked)


_INVOKE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.DOTALL)
_PARAMETER = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.DOTALL)


def _invoked_calls(raw: str) -> list[Call]:
    return [
        _call(name, {key: _value(value) for key, value in _PARAMETER.findall(body)})
        for name, body in _INVOKE.findall(raw)
    ]


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()


def _call(name: str, arguments: Any) -> Call:
    return Call(
        id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=json.dumps(arguments)
    )


def first_json_object(raw: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start = raw.find("{")
    if start != -1:
        candidates.append(raw[start : raw.rfind("}") + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


async def _retry(call: Any) -> Any:
    for attempt in range(_ATTEMPTS):
        try:
            return await call()
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ):
            if attempt == _ATTEMPTS - 1:
                raise
            await asyncio.sleep(min(60, 4 * 2**attempt))
