import logging

import click

# Default max length for rendered message content before truncation (override
# per-call with ``full=True`` / the ``--full`` flag).
_TRUNCATE_LENGTH = 500


@click.group()
def chat():
    """
    Group for CoPilot chat inspection commands
    """
    # Suppress INFO/DEBUG logs (Prisma connection chatter, service-client
    # setup) so they don't interleave with the rendered transcript. Use
    # logging.disable so it can't be undone by configure_logging() running
    # later during connect().
    logging.disable(logging.INFO)


@chat.command(name="view")
@click.argument("session_id")
@click.argument("seq_range", required=False, default=None, metavar="[RANGE]")
@click.option("--full", "-f", is_flag=True, help="Disable truncation of long content.")
@click.option(
    "--json", "as_json", is_flag=True, help="Dump the raw session+messages as JSON."
)
@click.option(
    "--limit", "-n", type=int, default=None, help="Show only the last N messages."
)
def chat_view(
    session_id: str,
    seq_range: str | None,
    full: bool,
    as_json: bool,
    limit: int | None,
):
    """
    Print a readable transcript of a CoPilot ChatSession by ID.

    Optionally pass a message RANGE to show only those sequences: 'N' (one),
    'N-M' (inclusive), 'N-' (from N), '-M' (up to M). Combine with --full to
    read them untruncated, e.g. `chat view <id> 6-10 --full`.
    """
    import asyncio

    import prisma.models

    from backend.data.db import connect, disconnect

    # Parse up front so a bad value fails fast with a clean message.
    bounds = _parse_seq_range(seq_range) if seq_range else None

    async def run():
        await connect()
        try:
            session = await prisma.models.ChatSession.prisma().find_unique(
                where={"id": session_id}
            )
            if session is None:
                print(f"No chat session found with id {session_id!r}")
                return

            messages = await prisma.models.ChatMessage.prisma().find_many(
                where={"sessionId": session_id},
                order={"sequence": "asc"},
            )
            if bounds is not None:
                lo, hi = bounds
                messages = [
                    m
                    for m in messages
                    if (lo is None or m.sequence >= lo)
                    and (hi is None or m.sequence <= hi)
                ]
            if limit is not None and limit > 0:
                messages = messages[-limit:]

            if as_json:
                print(_render_session_json(session, messages))
            else:
                print(_render_session(session, messages, full=full))
        finally:
            await disconnect()

    asyncio.run(run())


@chat.command(name="list")
@click.option("--user", "user_id", default=None, help="Filter by user ID.")
@click.option(
    "--limit", "-n", type=int, default=20, help="Max number of sessions to list."
)
def chat_list(user_id: str | None, limit: int):
    """
    List recent CoPilot chat sessions (id, title, updatedAt, message count).
    """
    import asyncio

    import prisma.models
    import prisma.types

    from backend.data.db import connect, disconnect

    async def run():
        await connect()
        try:
            where: prisma.types.ChatSessionWhereInput = {}
            if user_id:
                where["userId"] = user_id
            sessions = await prisma.models.ChatSession.prisma().find_many(
                where=where,
                order={"updatedAt": "desc"},
                take=limit,
            )
            if not sessions:
                print("No chat sessions found")
                return

            for session in sessions:
                count = await prisma.models.ChatMessage.prisma().count(
                    where={"sessionId": session.id}
                )
                title = session.title or "(untitled)"
                print(
                    f"{session.id}  "
                    f"{session.updatedAt:%Y-%m-%d %H:%M}  "
                    f"{count:>4} msgs  "
                    f"{title}"
                )
        finally:
            await disconnect()

    asyncio.run(run())


def _parse_seq_range(text: str) -> tuple[int | None, int | None]:
    """Parse a sequence selector into inclusive ``(lo, hi)`` bounds (``None`` =
    open-ended): ``'N'`` -> ``(N, N)``, ``'N-M'`` -> ``(N, M)``, ``'N-'`` ->
    ``(N, None)``, ``'-M'`` -> ``(None, M)``."""
    raw = text.strip()
    try:
        if "-" not in raw:
            n = int(raw)
            return (n, n)
        lo_str, hi_str = raw.split("-", 1)
        lo = int(lo_str) if lo_str.strip() else None
        hi = int(hi_str) if hi_str.strip() else None
    except ValueError as e:
        raise click.BadParameter(
            f"Invalid --range {text!r}; use forms like '7', '6-10', '6-', '-10'."
        ) from e
    if lo is not None and hi is not None and lo > hi:
        raise click.BadParameter(f"Invalid --range {text!r}: start is after end.")
    return (lo, hi)


def _render_session(session, messages, *, full: bool) -> str:
    import prisma.models

    s: prisma.models.ChatSession = session
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"ChatSession {s.id}")
    lines.append(f"  title:    {s.title or '(untitled)'}")
    lines.append(f"  status:   {s.chatStatus}")
    lines.append(f"  user:     {s.userId}")
    lines.append(f"  created:  {s.createdAt:%Y-%m-%d %H:%M:%S}")
    lines.append(f"  updated:  {s.updatedAt:%Y-%m-%d %H:%M:%S}")
    lines.append(
        f"  tokens:   prompt={s.totalPromptTokens} "
        f"completion={s.totalCompletionTokens}"
    )
    lines.append(f"  messages: {len(messages)}")
    lines.append("=" * 80)

    for msg in messages:
        lines.append("")
        lines.extend(_render_message(msg, full=full))

    return "\n".join(lines)


def _render_message(msg, *, full: bool) -> list[str]:
    import prisma.models

    m: prisma.models.ChatMessage = msg
    role = m.role
    header = f"[{m.sequence}] {role.upper()}"
    if m.name:
        header += f" ({m.name})"
    if m.durationMs is not None:
        header += f"  {m.durationMs}ms"

    if role == "reasoning":
        return [header] + [f"  | {line}" for line in _reasoning_lines(m, full)]
    if role == "assistant":
        return [header] + _assistant_lines(m, full)
    if role == "tool":
        return [header] + _tool_lines(m, full)
    if role == "user":
        return [header] + _user_lines(m, full)

    body = _truncate(m.content or "", full)
    return [header] + _indent(body)


def _reasoning_lines(msg, full: bool) -> list[str]:
    return (_truncate(msg.content or "", full) or "(empty)").splitlines() or ["(empty)"]


def _assistant_lines(msg, full: bool) -> list[str]:
    import json

    lines: list[str] = []
    if msg.content:
        lines.extend(_indent(_truncate(msg.content, full)))

    tool_calls = msg.toolCalls
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            name = fn.get("name", "?")
            raw_args = fn.get("arguments", "")
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (ValueError, TypeError):
                parsed = raw_args
            if isinstance(parsed, dict):
                args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in parsed.items())
            else:
                args_str = str(parsed)
            lines.append(f"  -> {name}({_truncate(args_str, full)})")

    if not lines:
        lines.append("  (no content)")
    return lines


def _tool_lines(msg, full: bool) -> list[str]:
    import json

    content = msg.content or ""
    try:
        parsed = json.loads(content)
    except ValueError:
        return _indent(_truncate(content, full))

    if isinstance(parsed, dict):
        type_field = parsed.get("type", "result")
        summary = {k: v for k, v in parsed.items() if k != "type"}
        summary_str = json.dumps(summary, separators=(", ", ": "))
        return [f"  <{type_field}> {_truncate(summary_str, full)}"]

    return _indent(_truncate(json.dumps(parsed), full))


def _user_lines(msg, full: bool) -> list[str]:
    content = _strip_context_blocks(msg.content or "")
    return _indent(_truncate(content, full) or "(empty)")


def _strip_context_blocks(content: str) -> str:
    import re

    # The first user message wraps the real request in large preamble blocks
    # like <available_skills>...</available_skills>, <user_context>...,
    # <session_context>... — collapse those so only the actual request shows.
    for tag in ("available_skills", "user_context", "session_context"):
        content = re.sub(
            rf"<{tag}>.*?</{tag}>\s*",
            f"[{tag} omitted]\n",
            content,
            flags=re.DOTALL,
        )
    return content.strip()


def _truncate(text: str, full: bool, length: int = _TRUNCATE_LENGTH) -> str:
    text = text.strip()
    if full or len(text) <= length:
        return text
    return text[:length] + " ... [truncated]"


def _indent(text: str, prefix: str = "  ") -> list[str]:
    if not text:
        return [f"{prefix}(empty)"]
    return [f"{prefix}{line}" for line in text.splitlines()]


def _render_session_json(session, messages) -> str:
    import json

    payload = {
        "session": json.loads(session.model_dump_json()),
        "messages": [json.loads(m.model_dump_json()) for m in messages],
    }
    return json.dumps(payload, indent=2, default=str)


if __name__ == "__main__":
    chat()
