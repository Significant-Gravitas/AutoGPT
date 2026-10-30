"""Base classes and shared utilities for chat tools."""

import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Literal, Sequence

from openai.types.chat import ChatCompletionToolParam
from prisma.enums import APIKeyPermission

from backend.copilot.ai_spans import mark_tool_failed, mark_tool_raised, tool_span
from backend.copilot.context import args_already_bound, get_current_envelope
from backend.copilot.google_data_redaction import (
    GOOGLE_DATA_METADATA_KEY,
    is_google_call,
    remember_cut_google_output,
    remember_google_output,
    tracing_google_data,
)
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tool_image import HeldImage, parse_held_image
from backend.copilot.trial_reads import trial_reads_fresh
from backend.copilot.turn_tracing import tool_transaction
from backend.data.activity_event import ActivityEventDraft
from backend.data.db_accessors import activity_event_db, workspace_db
from backend.data.workspace_internal import TOOL_OUTPUT_PURPOSE
from backend.util import tracing
from backend.util.truncate import truncate
from backend.util.workspace import WorkspaceManager

from .capability_gates import gate_denied, gate_denied_error
from .models import (
    ApprovalRequiredResponse,
    ErrorResponse,
    NeedLoginResponse,
    ResponseType,
    ToolResponseBase,
)

if TYPE_CHECKING:
    from backend.copilot.gate import Decision
    from backend.copilot.gate.headline import Headline
    from backend.copilot.gate.subject import Subject

logger = logging.getLogger(__name__)

# Outputs that stop a call to wait on the user (an approval card, a sign-in),
# which is not the tool failing.
_AWAITING_THE_USER = frozenset(
    {ResponseType.APPROVAL_REQUIRED.value, ResponseType.NEED_LOGIN.value}
)

# The Claude CLI saves any tool result longer than 50,000 characters (it
# counts characters, not tokens) to a ``tool-results/`` folder on the executor
# host and shows the model a 2 KB preview.  bash_exec runs in the sandbox and
# can't open that folder, so anything that long has to reach the workspace
# first, where ``read_workspace_file(save_to_path=...)`` can copy it over.
_CLI_SPILL_CHARS = 50_000

# Persist full tool output to workspace when it exceeds this threshold.
# Must be below _CLI_SPILL_CHARS (see above) and _MAX_TOOL_OUTPUT_SIZE (100K)
# in response_model.py, so we keep the data before either discards it.
_LARGE_OUTPUT_THRESHOLD = 45_000

# Character budget for the middle-out preview.  Preview + wrapper (under 1K)
# must stay below _CLI_SPILL_CHARS, or the CLI spills the preview itself.
_PREVIEW_CHARS = 40_000

# Threshold and budget for the digest, which fires when the tool opts in.  The
# budget is derived from the trigger so a digest can never be larger than the
# output it replaces.
_DIGEST_THRESHOLD = 8_000
_DIGEST_PREVIEW_CHARS = _DIGEST_THRESHOLD // 4
_OUTLINE_SCALAR_CHARS = 120


# Fields whose values are binary/base64 data — truncating them produces
# garbage, so we replace them with a human-readable size summary instead.
# Top-level only: the outline reads offsets built from the unsummarised text,
# so a nested field added here would make the two drift.
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
    google_data: bool = False,
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
            body, _, offsets = indexed
            # Summarised for display only: the file keeps the real bytes, so a
            # window read still yields them.
            summarized = json.loads(_summarize_binary_fields(body))
            outline = _outline(summarized, offsets, _DIGEST_PREVIEW_CHARS)

    try:
        workspace = await workspace_db().get_or_create_workspace(user_id)
        manager = WorkspaceManager(user_id, workspace.id, session_id)
        await manager.write_file(
            content=body.encode("utf-8"),
            filename=f"{tool_call_id}.json",
            path=file_path,
            mime_type="application/json",
            metadata={
                "purpose": TOOL_OUTPUT_PURPOSE,
                **({GOOGLE_DATA_METADATA_KEY: True} if google_data else {}),
            },
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
    sandbox = (
        f"To process the file in the sandbox/working dir, use "
        f"read_workspace_file("
        f'path="{file_path}", save_to_path="<working_dir>/{tool_call_id}.json") '
        f"first, then use bash_exec to work with the local copy."
    )
    if outline:
        fmt, preview = ' format="outline"', outline
        # A flat length= larger than the file makes one "narrow" read pull
        # everything back, which is how a digest ends up costing more.
        retrieval = (
            f"\nThe preview above is a structural outline of the {total:,}-char "
            f"output, not a literal prefix of it. Each `@offset+length` is that "
            f"node's exact window in the file: "
            f'read_workspace_file(path="{file_path}", offset=<offset>, '
            f"length=<length>) starts that window. Reads may return less than "
            f"requested; continue from the returned end offset with the "
            f"remaining length until the node's window is complete."
            f"\n{sandbox}"
        )
    else:
        fmt, preview = "", truncate(_summarize_binary_fields(body), _PREVIEW_CHARS)
        retrieval = (
            f"\nFull output ({total:,} chars) saved to workspace. "
            f"Use read_workspace_file("
            f'path="{file_path}", offset=<char_offset>, length=50000) '
            f"to read any section. If a read returns less than requested, "
            f"continue from the returned end offset with the remaining length. "
            f"{sandbox}"
        )
    return (
        f"<tool-output-truncated total_chars={total} "
        f'workspace_path="{file_path}"{fmt}>\n'
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
    # read_workspace_file base64-encodes its slice, so a window past three
    # quarters of the file costs more to read back than the whole output did.
    widest = offsets.get("$", (0, 0))[1] * 3 // 4
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
                window = f" @{start}+{length}" if length < widest else ""
                parts.append(
                    f"{key}={braces[0]}…{len(value)} {unit}{window}{braces[1]}"
                )
                queue.append((child, value))
            else:
                # `message` is the tool's own instruction to the model and the
                # 120-char cut lands mid-sentence; half the budget is as much
                # as it can have without crowding out the field names.
                cap = budget // 2 if key == "message" else scalar_chars
                parts.append(f"{key}={_scalar(value, cap)}")
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


def _scalar(value: Any, limit: int | None) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if limit is None or len(text) <= limit else text[: limit - 1] + "…"


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


# Passed to ``_execute`` of a tool with a gate subject when the user approved
# this exact call; stripped from the model's own arguments so it cannot be forged.
GATE_APPROVED = "_gate_approved"
# Passed beside it when an active gate reached any verdict on the call: a card,
# a rule, the mode or the supervisor. Stripped the same way.
GATE_DECIDED = "_gate_decided"
# Passed when ``bind_args`` already resolved the call's arguments, so the tool
# runs them as reviewed rather than resolving them a second time. Stripped the
# same way.
ARGS_BOUND = "_args_bound"


def parameters_without(parameters: dict[str, Any], *names: str) -> dict[str, Any]:
    """A copy of a tool's parameter schema without the named arguments."""
    return {
        **parameters,
        "properties": {
            name: spec
            for name, spec in parameters.get("properties", {}).items()
            if name not in names
        },
        "required": [
            name for name in parameters.get("required", []) if name not in names
        ],
    }


def _awaits_the_user(output: StreamToolOutputAvailable) -> bool:
    raw = output.output
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return False
    return isinstance(raw, dict) and raw.get("type") in _AWAITING_THE_USER


class BaseTool:
    """Base class for all chat tools."""

    # Opt-in for the digest: an outline is only readable back in windows, so a
    # tool whose bulk is one long text (a guide, a docs page) must not set it.
    digest_large_output: bool = False
    # True where ``gate_subject`` is implemented; ``_execute`` then accepts
    # GATE_APPROVED.
    has_gate_subject: bool = False

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
    def model_parameters(self) -> dict[str, Any]:
        """``parameters`` as the model is sent them: without the card's hints."""
        return without_card_hints(self.parameters)

    @property
    def search_keywords(self) -> tuple[str, ...]:
        """Extra words ``find_capability`` indexes for this tool.

        Index-only: never sent to the model, so they cost no schema budget.
        """
        return ()

    @property
    def is_available(self) -> bool:
        """Whether this tool is available in the current environment.

        Override to check required env vars, binaries, or other dependencies.
        Unavailable tools are excluded from the LLM tool list so the model is
        never offered an option that will immediately fail.
        """
        return True

    @property
    def requires_auth(self) -> bool:
        """Whether this tool requires an authenticated end user."""
        return False

    @property
    def allow_external_use(
        self,
    ) -> tuple[Literal[False], None] | tuple[Literal[True], Sequence[APIKeyPermission]]:
        """
        Whether this tool may be used through our external MCP server.
        Returns `True` and the API-key permissions it requires if so.

        Every tool must either override this or be listed with a reason in
        ``EXTERNAL_USE_EXCLUSIONS`` (api/external/v2/mcp_server.py);
        mcp_server_test.py fails otherwise.
        """
        return False, None

    @property
    def external_parameters(self) -> dict[str, Any]:
        """The parameters an external caller may pass; a subset of ``parameters``.

        The MCP server advertises this schema and refuses a call that passes
        anything outside it. Override to keep an argument off the external
        surface when it reaches something the v2 API has no permission for.
        """
        return self.parameters

    def external_permissions(self, args: dict[str, Any]) -> list[APIKeyPermission]:
        """Permissions an external call needs on top of ``allow_external_use``'s.

        For a tool whose arguments decide what it does: a branch that writes
        what the tool's own permissions don't cover names that permission
        here, so choosing an argument can't reach past the caller's scopes.
        """
        return []

    @property
    def spends_platform_money(self) -> bool:
        """Whether a call bills the platform's own account, such as a paid model.

        The MCP server checks the caller's paywall and usage allowance before
        running one, as a chat turn does before it spends the same budget.
        """
        return False

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

    async def gate_subject(
        self, user_id: str, session: ChatSession, args: dict[str, Any]
    ) -> "Subject | None":
        """What this call acts on, where the tool's name alone does not say.

        None leaves the decision to the tool's own effect; a tool that runs a
        block or a workflow names it, so the gate decides on its effect.
        """
        return None

    async def bind_args(
        self, user_id: str | None, session: ChatSession, args: dict[str, Any]
    ) -> dict[str, Any]:
        """The call's arguments as they will run. A tool that resolves part of
        its input at run time (a file reference) resolves it here, once, so the
        approval binds what runs and a changed file asks again."""
        return args

    async def gate_context(self, args: dict[str, Any]) -> dict[str, str | None] | None:
        """The content of files this call runs, by path, for the supervisor; None
        for one that could not be read, which holds the call."""
        return None

    def as_openai_tool(self) -> ChatCompletionToolParam:
        """Convert to OpenAI tool format."""
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": self.name,
                "description": self.description,
                "parameters": self.model_parameters,
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

        Each call is its own Sentry ``copilot.tool`` transaction, named after
        the tool, in the trace of the turn that made it, and a PostHog
        ``$ai_span`` in the trace of the reply (``copilot.ai_spans``): its
        name, latency and whether it failed, never its input or output.

        Args:
            user_id: User ID (None for anonymous users)
            session_id: Chat session ID
            **kwargs: Tool-specific parameters

        Returns:
            Pydantic response object

        """
        with (
            tool_transaction(self.name, tool_call_id, session.session_id) as span,
            trial_reads_fresh(),
            tool_span(
                self.name,
                user_id=user_id,
                session_id=session.session_id,
                tool_call_id=tool_call_id,
            ) as outcome,
        ):
            output = await self._execute_checked(
                user_id, session, tool_call_id, **kwargs
            )
            span.set_tag("copilot.tool.success", "true" if output.success else "false")
            if not output.success and not _awaits_the_user(output):
                outcome.is_error = True
            return output

    async def _execute_checked(
        self,
        user_id: str | None,
        session: ChatSession,
        tool_call_id: str,
        **kwargs,
    ) -> StreamToolOutputAvailable:
        kwargs.pop(GATE_APPROVED, None)
        kwargs.pop(GATE_DECIDED, None)
        kwargs.pop(ARGS_BOUND, None)
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

        # Hiding a tool from the schema is presentation; this is the boundary.
        # A spawned turn's envelope can only ever be narrower than its spawner's.
        envelope = get_current_envelope()
        if envelope is not None and not envelope.permits(self.name):
            logger.warning(
                "Refusing tool outside turn envelope: tool=%s session=%s depth=%s",
                self.name,
                session.session_id,
                envelope.depth,
            )
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=ErrorResponse(
                    message=(
                        f"{self.name} is not available in this delegated turn. "
                        "If the task needs it, say so in your report instead."
                    ),
                    session_id=session.session_id,
                ).model_dump_json(),
                success=False,
            )

        # A deferred tool arrives here resolved out of a ``run_capability``
        # dispatch, so gating it by name has to happen on the way in or it
        # does not happen at all.
        if gate_denied(self.name):
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=gate_denied_error(
                    self.name, session.session_id
                ).model_dump_json(),
                success=False,
            )

        # A released read is answered from its row, never re-run, so the bytes
        # the user approved are the bytes the model gets.
        released = await self._released_read(user_id, session, tool_call_id, kwargs)
        if released is not None:
            return released

        # Auto-mode gate. Sits here because both engines funnel every registry
        # tool through this method — baseline via ``execute_tool``, SDK via
        # ``_execute_tool_sync`` — so there is one place to add, not two.
        # Must stay AFTER the envelope and name gates: a call the envelope refuses can
        # never run, so approving it would spend a user's decision on nothing.
        # The gate's model calls quote the call's arguments and result in
        # prompts of their own, so for a Google call their traces are redacted.
        google_call = is_google_call(kwargs)
        # What the call will run, resolved once: the gate reviews, the card
        # shows and the approval binds these values, and the tool runs them.
        # An approved held call replays the arguments it was resolved to.
        replayed = self.has_gate_subject and args_already_bound()
        bound = (
            await self.bind_args(user_id, session, kwargs)
            if self.has_gate_subject and not replayed
            else kwargs
        )
        args_bound = replayed or bound is not kwargs
        with tracing_google_data(google_call):
            gated, decision = await self._gate(
                user_id, session, tool_call_id, bound, args_bound
            )
        if gated is not None:
            return gated
        run_kwargs = {**bound, ARGS_BOUND: True} if args_bound else kwargs
        if decision is not None and self.has_gate_subject:
            if decision.approved:
                run_kwargs = {**run_kwargs, GATE_APPROVED: True}
            if decision.decided:
                run_kwargs = {**run_kwargs, GATE_DECIDED: True}

        # After the gates, so a refused call never looks to a turn-scoped gate
        # like the tool having run, and before the await, because the gates ask
        # whether it was dispatched rather than whether it succeeded.
        session.announce_inflight_tool_call(self.name, kwargs)

        try:
            result = await self._execute(user_id, session, **run_kwargs)
            if result.type == ResponseType.ERROR:
                mark_tool_failed()
            if user_id:
                await _record_activity(self, user_id, session, result, kwargs)
            raw_output = full = result.model_dump_json(exclude_none=True)

            digest = (
                self.digest_large_output
                and len(raw_output) > _DIGEST_THRESHOLD
                and user_id is not None
            )
            threshold = _DIGEST_THRESHOLD if digest else _LARGE_OUTPUT_THRESHOLD
            # An External API client can't follow a preview to the copy it
            # points at, so it gets the output itself, cut in the middle past
            # _MAX_TOOL_OUTPUT_SIZE, and nothing is written to its workspace.
            if (
                len(raw_output) > threshold
                and user_id
                and session.session_id
                and not session.external_caller
            ):
                raw_output = await _persist_and_summarize(
                    raw_output,
                    user_id,
                    session.session_id,
                    tool_call_id,
                    digest,
                    google_data=result.holds_google_data,
                )
            output = StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=raw_output,
                # An MCP client reads a refusal from the call's status; chat's
                # engines hand it to the model as the tool's result.
                success=not (
                    session.external_caller and result.type == ResponseType.ERROR
                ),
            ).with_image(result.image)
            # Remembered as the model gets it, previews and caps included, so
            # the traces can redact it even where it no longer parses.
            if result.holds_google_data and isinstance(output.output, str):
                remember_google_output(output.output)
        except Exception as e:
            logger.warning("Error in %s", self.name, exc_info=True)
            tracing.set_status("internal_error")
            mark_tool_raised(e)
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=ErrorResponse(
                    message=f"An error occurred while executing {self.name}",
                    # AutoPilot's model reads the detail to recover; an
                    # External API client gets no internal detail.
                    error=None if session.external_caller else str(e),
                    session_id=session.session_id,
                ).model_dump_json(),
                success=False,
            )
        with tracing_google_data(google_call or result.holds_google_data):
            return await self._screen_read(
                user_id, session, tool_call_id, kwargs, output, result.outside, full
            )

    async def _gate(
        self,
        user_id: str | None,
        session: ChatSession,
        tool_call_id: str,
        kwargs: dict[str, Any],
        args_bound: bool = False,
    ) -> tuple[StreamToolOutputAvailable | None, "Decision | None"]:
        """A refusal to return instead of running (or None to proceed), and
        the gate's decision on a call that proceeds.

        A gate that crashes must not become a gate that passes, so an
        unexpected failure here refuses the call rather than falling through.
        """
        from backend.copilot.gate import check_action

        async def subject_of() -> "Subject | None":
            return await self.gate_subject(user_id or "", session, kwargs)

        try:
            decision = await check_action(
                self.name,
                kwargs,
                user_id,
                session,
                tool_call_id,
                subject_of=subject_of if self.has_gate_subject else None,
                context_of=lambda: self.gate_context(kwargs),
                args_bound=args_bound,
            )
        except Exception:
            logger.warning(f"Action gate failed for {self.name}", exc_info=True)
            return (
                self._refusal(
                    tool_call_id,
                    session,
                    "This action could not be checked against your approval "
                    "settings, so nothing ran. Tell the user and stop.",
                    args=kwargs,
                ),
                None,
            )

        if decision.allowed:
            return None, decision
        return (
            self._refusal(
                tool_call_id,
                session,
                decision.reason,
                review_id=decision.review_id,
                args=kwargs,
                headline=decision.headline,
            ),
            None,
        )

    async def _released_read(
        self,
        user_id: str | None,
        session: ChatSession,
        tool_call_id: str,
        kwargs: dict[str, Any],
    ) -> StreamToolOutputAvailable | None:
        from backend.copilot.gate.reads import release_held_read

        try:
            release = await release_held_read(self.name, kwargs, user_id, session)
        except Exception:
            logger.warning(f"Held-read lookup failed for {self.name}", exc_info=True)
            return self._refusal(
                tool_call_id,
                session,
                "This read could not be checked against your approvals, so "
                "nothing ran. Tell the user and stop.",
            )
        if release is None:
            return None
        image_result = parse_held_image(release.output) if release.released else None
        return StreamToolOutputAvailable(
            toolCallId=tool_call_id,
            toolName=self.name,
            output=image_result.text if image_result else release.output,
            success=release.success,
        ).with_image(image_result.image if image_result else None)

    async def _screen_read(
        self,
        user_id: str | None,
        session: ChatSession,
        tool_call_id: str,
        kwargs: dict[str, Any],
        result: StreamToolOutputAvailable,
        outside: tuple[Any, ...] | None,
        full: str,
    ) -> StreamToolOutputAvailable:
        """Judge the output as the model will receive it, after every cap: the
        parts ``outside`` declares, or all of it when nothing was declared."""
        from backend.copilot.gate.reads import model_view, readable_parts, screen_read

        seen = (
            result.output
            if isinstance(result.output, str)
            else json.dumps(result.output)
        )
        view = model_view.get()
        if view is not None:
            whole, seen = seen, view(seen, result.success)
            remember_cut_google_output(whole, seen)
        if result.image is not None:
            whole, seen = (
                seen,
                HeldImage(text=seen, image=result.image).model_dump_json(),
            )
            remember_cut_google_output(whole, seen)
        text, images = readable_parts(seen)
        stub = await screen_read(
            self.name,
            kwargs,
            user_id,
            session,
            output=seen,
            success=result.success,
            text=text,
            images=images,
            tool_call_id=tool_call_id,
            outside=outside,
            full=full,
        )
        if stub is None:
            return result
        return StreamToolOutputAvailable(
            toolCallId=tool_call_id, toolName=self.name, output=stub, success=False
        )

    def _refusal(
        self,
        tool_call_id: str,
        session: ChatSession,
        reason: str,
        review_id: str | None = None,
        args: dict[str, Any] | None = None,
        headline: "Headline | None" = None,
    ) -> StreamToolOutputAvailable:
        from backend.copilot.gate import refusal_message
        from backend.copilot.gate.headline import headline_for

        headline = headline or headline_for(self.name, args or {})
        return StreamToolOutputAvailable(
            toolCallId=tool_call_id,
            toolName=self.name,
            output=ApprovalRequiredResponse(
                message=refusal_message(reason, review_id),
                session_id=session.session_id,
                tool_name=self.name,
                reason=reason,
                review_id=review_id,
                ask=headline.ask,
                object=headline.object,
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


# Read by the approval card (gate/review.py, gate/references.py), never by the
# model: a label, how to show the value, and what platform thing an id names.
CARD_HINTS = frozenset({"title", "format", "entity"})


def without_card_hints(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not properties:
        return schema
    return {
        **schema,
        "properties": {
            key: {k: v for k, v in spec.items() if k not in CARD_HINTS}
            for key, spec in properties.items()
        },
    }
