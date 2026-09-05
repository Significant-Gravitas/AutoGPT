"""Base classes and shared utilities for chat tools."""

import json
import logging
from collections import deque
from typing import Any

from openai.types.chat import ChatCompletionToolParam

from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.data.activity_event import ActivityEventDraft
from backend.data.db_accessors import activity_event_db, workspace_db
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.truncate import truncate
from backend.util.workspace import WorkspaceManager

from .models import ErrorResponse, NeedLoginResponse, ToolResponseBase

logger = logging.getLogger(__name__)

# Persist full tool output to workspace when it exceeds this threshold.
# Must be below _MAX_TOOL_OUTPUT_SIZE (100K) in response_model.py so we
# capture the data before model_post_init middle-out truncation discards it.
_LARGE_OUTPUT_THRESHOLD = 80_000

# Character budget for the middle-out preview.  The total preview + wrapper
# must stay below BOTH:
#   - _MAX_TOOL_OUTPUT_SIZE (100K) in response_model.py (our own truncation)
#   - Claude SDK's ~100 KB tool-result spill-to-disk threshold
# to avoid double truncation/spilling.  95K + ~300 wrapper = ~95.3K, under both.
_PREVIEW_CHARS = 95_000

# Threshold and budget used when AUTOPILOT_DELEGATION is on.  The largest block
# schema in the registry is ~34K chars, so the 80K trigger above has never once
# fired on the reads that fill AutoPilot's transcript.  The budget is derived
# from the trigger so a digest can never be larger than the output it replaces.
_DIGEST_THRESHOLD = 8_000
_DIGEST_PREVIEW_CHARS = _DIGEST_THRESHOLD // 4
_OUTLINE_SCALAR_CHARS = 120


# Fields whose values are binary/base64 data — truncating them produces
# garbage, so we replace them with a human-readable size summary instead.
_BINARY_FIELD_NAMES = {"content_base64"}


def _summarize_binary_fields(raw_json: str) -> str:
    """Replace known binary fields with a size summary so truncate() doesn't
    produce garbled base64 in the middle-out preview."""
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return raw_json

    if not isinstance(data, dict):
        return raw_json

    changed = False
    for key in _BINARY_FIELD_NAMES:
        if key in data and isinstance(data[key], str) and len(data[key]) > 1_000:
            byte_size = len(data[key]) * 3 // 4  # approximate decoded size
            data[key] = f"<binary, ~{byte_size:,} bytes>"
            changed = True

    return json.dumps(data, ensure_ascii=False) if changed else raw_json


async def _persist_and_summarize(
    raw_output: str,
    user_id: str,
    session_id: str,
    tool_call_id: str,
    digest: bool = False,
) -> str:
    """Persist full output to workspace and return a preview with retrieval
    instructions — a structural outline when *digest*, else a middle-out slice.

    On failure, returns the original ``raw_output`` unchanged so that the
    existing ``model_post_init`` middle-out truncation handles it as before.
    """
    file_path = f"tool-outputs/{tool_call_id}.json"

    # The outline quotes offsets into the persisted file, so the file has to be
    # the text the index was built from, not the original serialisation.
    body, outline = raw_output, ""
    if digest:
        indexed = _index_json(raw_output)
        if indexed is not None:
            body, data, offsets = indexed
            outline = _outline(data, offsets, _DIGEST_PREVIEW_CHARS)

    try:
        workspace = await workspace_db().get_or_create_workspace(user_id)
        manager = WorkspaceManager(user_id, workspace.id, session_id)
        await manager.write_file(
            content=body.encode("utf-8"),
            filename=f"{tool_call_id}.json",
            path=file_path,
            mime_type="application/json",
            overwrite=True,
        )
    except Exception:
        logger.warning(
            "Failed to persist large tool output for %s",
            tool_call_id,
            exc_info=True,
        )
        return raw_output  # fall back to normal truncation

    total = len(body)
    if outline:
        shape, preview = "outline", outline
        # A flat length= larger than the file makes one "narrow" read pull
        # everything back, which is how a digest ends up costing more.
        retrieval = (
            f"\nThe preview above is a structural outline of the {total:,}-char "
            f"output, not a literal prefix of it. Each `@offset+length` is that "
            f"node's exact window in the file: "
            f'read_workspace_file(path="{file_path}", offset=<offset>, '
            f"length=<length>) returns it and nothing else."
        )
    else:
        shape, preview = "excerpt", truncate(
            _summarize_binary_fields(body), _PREVIEW_CHARS
        )
        retrieval = (
            f"\nFull output ({total:,} chars) saved to workspace. "
            f"Use read_workspace_file("
            f'path="{file_path}", offset=<char_offset>, length=50000) '
            f"to read any section. "
        )
    retrieval += (
        f"\nTo process the file in the sandbox/working dir, use "
        f"read_workspace_file("
        f'path="{file_path}", save_to_path="<working_dir>/{tool_call_id}.json") '
        f"first, then use bash_exec to work with the local copy."
    )
    return (
        f"<tool-output-truncated total_chars={total} "
        f'workspace_path="{file_path}" format="{shape}">\n'
        f"{preview}\n"
        f"{retrieval}\n"
        f"</tool-output-truncated>"
    )


def _index_json(raw_output: str) -> tuple[str, Any, dict[str, tuple[int, int]]] | None:
    """``(text, data, path -> (offset, length))``, or ``None`` for non-JSON.

    Serialising it ourselves is what makes the offsets exact: they are recorded
    as the text is built, so no re-parsing can drift from what was persisted.
    """
    try:
        data = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, (dict, list)):
        return None
    chunks: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    _write_indexed(data, "$", chunks, offsets, 0)
    return "".join(chunks), data, offsets


def _write_indexed(
    node: Any,
    path: str,
    chunks: list[str],
    offsets: dict[str, tuple[int, int]],
    pos: int,
) -> int:
    start = pos
    if isinstance(node, (dict, list)):
        pairs = node.items() if isinstance(node, dict) else enumerate(node)
        opening, closing = ("{", "}") if isinstance(node, dict) else ("[", "]")
        chunks.append(opening)
        pos += 1
        for index, (key, value) in enumerate(pairs):
            if index:
                chunks.append(",")
                pos += 1
            if isinstance(node, dict):
                label = json.dumps(key, ensure_ascii=False) + ":"
                chunks.append(label)
                pos += len(label)
                child = f"{path}.{key}"
            else:
                child = f"{path}[{key}]"
            pos = _write_indexed(value, child, chunks, offsets, pos)
        chunks.append(closing)
        pos += 1
    else:
        text = json.dumps(node, ensure_ascii=False)
        chunks.append(text)
        pos += len(text)
    offsets[path] = (start, pos - start)
    return pos


def _outline(data: Any, offsets: dict[str, tuple[int, int]], budget: int) -> str:
    """Structural outline of *data* within *budget* characters.

    A middle-out slice of a schema shows one arbitrary window; an outline names
    every property first and carries each node's window in the file, so the
    model can both judge whether it needs more and fetch only that.
    """
    # Widen the per-scalar cap when the structure alone leaves most of the
    # budget unspent, so a payload that is one long string still says something.
    tight = _render_outline(data, offsets, budget, _OUTLINE_SCALAR_CHARS)
    if len(tight) >= budget // 2:
        return tight
    wide = _render_outline(data, offsets, budget, budget // 2)
    return max((tight, wide), key=lambda text: (text.count("\n"), len(text)))


def _render_outline(
    root: Any, offsets: dict[str, tuple[int, int]], budget: int, scalar_chars: int
) -> str:
    """Breadth-first so shallow facts outlive the budget: one line per
    container, its scalar children inline and its container children named
    with the window that holds them."""
    lines: list[str] = []
    used = 0
    queue: deque[tuple[str, Any]] = deque([("$", root)])
    while queue:
        path, node = queue.popleft()
        is_map = isinstance(node, dict)
        parts: list[str] = []
        for key, value in node.items() if is_map else enumerate(node):
            child = f"{path}.{key}" if is_map else f"{path}[{key}]"
            if isinstance(value, (dict, list)):
                braces = "{}" if isinstance(value, dict) else "[]"
                unit = "keys" if isinstance(value, dict) else "items"
                start, length = offsets.get(child, (0, 0))
                parts.append(
                    f"{key}={braces[0]}…{len(value)} {unit} "
                    f"@{start}+{length}{braces[1]}"
                )
                queue.append((child, value))
            else:
                parts.append(f"{key}={_scalar(value, scalar_chars)}")
        braces = "{}" if is_map else "[]"
        line = f"{path}: {braces[0]}{', '.join(parts)}{braces[1]}"
        if len(line) + 1 > budget - used:
            marker = f"… {len(queue) + 1} more nodes not shown; read their windows"
            while lines and used + len(marker) > budget:
                used -= len(lines.pop()) + 1
            room = budget - used - len(marker) - 1
            if room > _OUTLINE_SCALAR_CHARS:
                lines.append(line[: room - 1] + "…")
            lines.append(marker)
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _scalar(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _record_activity(
    tool: "BaseTool",
    user_id: str,
    session: ChatSession,
    result: "ToolResponseBase",
    kwargs: dict[str, Any],
) -> None:
    """Persist the tool call's activity event, if the tool reports one.

    Best-effort by contract: the audit log must never break the tool call
    that produced it, so every failure is swallowed after a warning.
    """
    try:
        draft = tool.activity_event(session=session, result=result, **kwargs)
        if draft is None:
            return
        draft.session_id = draft.session_id or session.session_id
        draft.expert_id = draft.expert_id or session.expert_id
        draft.organization_id = draft.organization_id or session.organization_id
        await activity_event_db().create_activity_event(user_id=user_id, draft=draft)
    except Exception:
        logger.warning(
            "Failed to record activity event for tool %s", tool.name, exc_info=True
        )


class BaseTool:
    """Base class for all chat tools."""

    @property
    def name(self) -> str:
        """Tool name for OpenAI function calling."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Tool description for OpenAI."""
        raise NotImplementedError

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters schema for OpenAI."""
        raise NotImplementedError

    @property
    def requires_auth(self) -> bool:
        """Whether this tool requires authentication."""
        return False

    @property
    def is_available(self) -> bool:
        """Whether this tool is available in the current environment.

        Override to check required env vars, binaries, or other dependencies.
        Unavailable tools are excluded from the LLM tool list so the model is
        never offered an option that will immediately fail.
        """
        return True

    def activity_event(
        self,
        session: ChatSession,
        result: "ToolResponseBase",
        **kwargs,
    ) -> ActivityEventDraft | None:
        """Describe the durable work this call performed.

        Side-effecting tools override this to report what they did (file
        written, schedule created, integration used) for the activity log.
        Read-only tools keep the default None. Called only on successful
        ``_execute`` returns; overrides narrow on their success response
        type, which skips error responses for free.
        """
        return None

    def as_openai_tool(self) -> ChatCompletionToolParam:
        """Convert to OpenAI tool format."""
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        )

    async def execute(
        self,
        user_id: str | None,
        session: ChatSession,
        tool_call_id: str,
        **kwargs,
    ) -> StreamToolOutputAvailable:
        """Execute the tool with authentication check.

        Args:
            user_id: User ID (None for anonymous users)
            session_id: Chat session ID
            **kwargs: Tool-specific parameters

        Returns:
            Pydantic response object

        """
        if self.requires_auth and not user_id:
            logger.warning(
                "Attempted tool call for %s but user not authenticated",
                self.name,
            )
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=NeedLoginResponse(
                    message=f"Please sign in to use {self.name}",
                    session_id=session.session_id,
                ).model_dump_json(),
                success=False,
            )

        try:
            result = await self._execute(user_id, session, **kwargs)
            if user_id:
                await _record_activity(self, user_id, session, result, kwargs)
            raw_output = result.model_dump_json(exclude_none=True)

            # Consult the flag only once the output could plausibly be digested,
            # so the common small-output path stays a pure local check.
            digest = (
                len(raw_output) > _DIGEST_THRESHOLD
                and user_id is not None
                and await is_feature_enabled(
                    Flag.AUTOPILOT_DELEGATION, user_id, default=False
                )
            )
            threshold = _DIGEST_THRESHOLD if digest else _LARGE_OUTPUT_THRESHOLD
            if len(raw_output) > threshold and user_id and session.session_id:
                raw_output = await _persist_and_summarize(
                    raw_output, user_id, session.session_id, tool_call_id, digest
                )

            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=raw_output,
            )
        except Exception as e:
            logger.warning("Error in %s", self.name, exc_info=True)
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=ErrorResponse(
                    message=f"An error occurred while executing {self.name}",
                    error=str(e),
                    session_id=session.session_id,
                ).model_dump_json(),
                success=False,
            )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Internal execution logic to be implemented by subclasses.

        Args:
            user_id: User ID (authenticated or anonymous)
            session_id: Chat session ID
            **kwargs: Tool-specific parameters

        Returns:
            Pydantic response object

        """
        raise NotImplementedError
