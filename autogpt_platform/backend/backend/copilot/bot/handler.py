"""Platform-agnostic message handler.

Receives a MessageContext from any adapter and drives the full AutoPilot
interaction: link resolution, thread routing, batched streaming with a
persistent typing indicator.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from backend.data.redis_client import get_redis_async
from backend.data.sharing.workspace_refs import (
    WorkspaceArtifactLink,
    extract_artifact_links,
)
from backend.util.exceptions import (
    DuplicateChatMessageError,
    LinkAlreadyExistsError,
    NotFoundError,
)
from backend.util.settings import Settings

from . import sessions, threads
from .adapters.base import (
    FileAttachment,
    MessageContext,
    MessageHistoryEntry,
    PlatformAdapter,
)
from .bot_backend import BotBackend, BotStreamError
from .config import SESSION_TTL
from .text import format_batch, split_at_boundary

logger = logging.getLogger(__name__)

THREAD_NAME_MAX_LENGTH = 100
THREAD_NAME_PREFIX = "AutoPilot: "
TITLE_RENAME_ATTEMPTS = 5
TITLE_RENAME_INTERVAL_SECONDS = 1.0


@dataclass
class TargetState:
    """Per-target streaming state.

    A "target" is wherever the bot replies — a thread ID, a DM channel ID.
    `pending` holds messages that arrived while a stream was running; they
    get drained as a single batched follow-up turn when the stream ends.
    """

    processing: bool = False
    pending: list[tuple[str, str, str]] = field(default_factory=list)
    # Each entry: (username, user_id, text)


class MessageHandler:
    def __init__(self, api: BotBackend):
        self._api = api
        self._targets: dict[str, TargetState] = {}
        # Strong-ref set so the GC doesn't drop fire-and-forget rename tasks.
        self._rename_tasks: set[asyncio.Task[None]] = set()

    async def handle(self, ctx: MessageContext, adapter: PlatformAdapter) -> None:
        if not ctx.text.strip():
            if ctx.channel_type == "channel":
                await adapter.send_reply(
                    ctx.channel_id,
                    "You mentioned me but didn't say anything. How can I help?",
                    ctx.message_id,
                )
            return

        # In a thread we only auto-reply when we own it (= we created it in
        # response to an @mention in a channel). For any other existing
        # thread we'd been added to, require an explicit @mention each turn
        # so we don't hijack ongoing team conversations.
        include_thread_history = False
        if ctx.channel_type == "thread":
            is_subscribed = await threads.is_subscribed(ctx.platform, ctx.channel_id)
            if not is_subscribed:
                if not ctx.bot_mentioned:
                    return
                # First time we're @-ed into this thread — pull the recent
                # thread history into the prompt so AutoPilot has context,
                # but DON'T subscribe. Future messages here need another @.
                include_thread_history = True

        if not await self._ensure_linked(ctx, adapter):
            return

        target_id = await self._resolve_target(ctx, adapter)
        if not target_id:
            return  # Thread not subscribed, ignore silently

        self._api.track_event(
            platform=ctx.platform,
            event_type="message_received",
            server_id=ctx.server_id,
            channel_type=ctx.channel_type,
            char_count=len(ctx.text),
        )

        message_text = self._message_text(ctx, include_thread_history)
        await self._enqueue_and_process(ctx, adapter, target_id, message_text)

    # -- Target resolution --

    async def _resolve_target(
        self, ctx: MessageContext, adapter: PlatformAdapter
    ) -> str | None:
        if ctx.channel_type == "dm":
            return ctx.channel_id

        if ctx.channel_type == "thread":
            return ctx.channel_id

        # channel_type == "channel" — create a thread and subscribe
        thread_name = build_thread_name(ctx.text, ctx.username)
        thread_id = await adapter.create_thread(
            ctx.channel_id, ctx.message_id, thread_name
        )
        if not thread_id:
            logger.warning("Thread creation failed, falling back to channel reply")
            return ctx.channel_id
        await threads.subscribe(ctx.platform, thread_id)
        return thread_id

    # -- Batched streaming --

    def _message_text(self, ctx: MessageContext, include_thread_history: bool) -> str:
        # Referenced conversations (links/@-mentions the user pasted) are always
        # surfaced — that's the whole point of fetching them. Thread history is
        # only included on the first @-into a thread we don't own; a subscribed
        # thread's prior turns already live in the session.
        thread_history = ctx.thread_history if include_thread_history else ()
        if not thread_history and not ctx.referenced_conversations:
            return ctx.text

        platform_display = ctx.platform.capitalize()
        lines: list[str] = []
        if ctx.referenced_conversations:
            # The user pointed at other channels/threads; their content is
            # already fetched and inlined below. Lead with a firm instruction so
            # the model answers from it instead of fixating on the link and
            # claiming it can't access the platform.
            lines.append(
                f"[The {platform_display} channel(s) the user referenced have "
                f"already been read for you — their full content is included "
                f"below under each #name. Answer directly from it. Do NOT say you "
                f"can't open links or access {platform_display}; you already have "
                f"the content.]"
            )
            for convo in ctx.referenced_conversations:
                lines.append(f"\n[Content of #{convo.title}]")
                for entry in convo.messages:
                    lines.append(self._format_history_entry(entry, platform_display))
        if thread_history:
            lines.append("\n[Recent thread context before this message]")
            for entry in thread_history:
                lines.append(self._format_history_entry(entry, platform_display))
        lines.append(f"\n[Current message]\n{ctx.text}")
        return "\n".join(lines)

    @staticmethod
    def _format_history_entry(entry: MessageHistoryEntry, platform_display: str) -> str:
        user = (
            f"{entry.username} ({platform_display} user ID: {entry.user_id})"
            if entry.user_id
            else entry.username
        )
        return f"\n[From {user}]\n{entry.text}"

    async def _enqueue_and_process(
        self,
        ctx: MessageContext,
        adapter: PlatformAdapter,
        target_id: str,
        message_text: str | None = None,
    ) -> None:
        state = self._targets.setdefault(target_id, TargetState())
        state.pending.append((ctx.username, ctx.user_id, message_text or ctx.text))

        if state.processing:
            # Another invocation is streaming for this target — it will pick
            # up the message we just appended when its current stream ends.
            return

        state.processing = True
        try:
            while state.pending:
                batch = list(state.pending)
                state.pending.clear()
                await self._stream_batch(batch, ctx, adapter, target_id)
        finally:
            state.processing = False
            # Drop the empty state so the dict doesn't grow unbounded across
            # the bot's lifetime.
            if not state.pending:
                self._targets.pop(target_id, None)

    async def _send_text_and_artifacts(
        self,
        adapter: PlatformAdapter,
        target_id: str,
        text: str,
        ctx: MessageContext,
        session_id: str | None,
    ) -> bool:
        """Send a finished chunk of text, then any workspace artifacts it
        referenced. Returns True if anything was sent to the channel.

        Each artifact gets its own platform message — files attach inline
        when small enough, otherwise we drop a link button pointing at the
        chat on the platform so the user can grab it from there.
        """
        stripped, artifacts = extract_artifact_links(text)
        sent_any = False
        if stripped:
            await adapter.send_message(
                target_id, stripped, mentionable_users=ctx.mentionable_users
            )
            sent_any = True
        for artifact in artifacts:
            sent = await self._deliver_artifact(
                adapter, target_id, artifact, session_id
            )
            sent_any = sent_any or sent
        return sent_any

    async def _deliver_artifact(
        self,
        adapter: PlatformAdapter,
        target_id: str,
        artifact: WorkspaceArtifactLink,
        session_id: str | None,
    ) -> bool:
        """Attach the file inline when possible; otherwise drop a link to
        the chat on the platform. Returns whether anything was sent."""
        if session_id is None:
            # Can't fetch or build a link button without a session id. Surface
            # the artifact name as plain text so the user knows something was
            # produced, even if they can't grab it from here.
            logger.warning(
                "Workspace artifact %s referenced before session id known",
                artifact.file_id,
            )
            await self._send_artifact_note(adapter, target_id, artifact)
            return True
        if await self._attach_artifact(adapter, target_id, artifact, session_id):
            return True
        await self._send_artifact_fallback(adapter, target_id, artifact, session_id)
        return True

    async def _attach_artifact(
        self,
        adapter: PlatformAdapter,
        target_id: str,
        artifact: WorkspaceArtifactLink,
        session_id: str,
    ) -> bool:
        """Fetch and inline-attach the file. Returns False when it's missing,
        unowned, too large, or the fetch errored."""
        try:
            fetched = await self._api.fetch_workspace_artifact(
                session_id=session_id,
                file_id=artifact.file_id,
                max_bytes=adapter.max_attachment_bytes,
            )
        except Exception:
            logger.exception("Failed to fetch workspace artifact %s", artifact.file_id)
            return False
        if fetched is None:
            return False
        await adapter.send_file(
            target_id,
            text="",
            file=FileAttachment(
                filename=artifact.display_name or fetched.filename,
                mime_type=fetched.mime_type,
                content=fetched.content,
            ),
        )
        return True

    async def _send_artifact_fallback(
        self,
        adapter: PlatformAdapter,
        target_id: str,
        artifact: WorkspaceArtifactLink,
        session_id: str,
    ) -> None:
        """Link the user to the chat when the file can't be attached here —
        covers missing, unavailable, errored and too-large cases alike."""
        session_url = _copilot_session_url(session_id)
        if session_url is None:
            logger.warning(
                "No base URL configured; can't render fallback link for %s",
                artifact.file_id,
            )
            await self._send_artifact_note(adapter, target_id, artifact)
            return
        await adapter.send_link(
            target_id,
            f"Open AutoGPT to download `{artifact.display_name}`.",
            link_label="Open in AutoGPT",
            link_url=session_url,
        )

    async def _send_artifact_note(
        self,
        adapter: PlatformAdapter,
        target_id: str,
        artifact: WorkspaceArtifactLink,
    ) -> None:
        await adapter.send_message(
            target_id,
            f"_(produced `{artifact.display_name}` — open the chat to download)_",
        )

    async def _stream_batch(
        self,
        batch: list[tuple[str, str, str]],
        ctx: MessageContext,
        adapter: PlatformAdapter,
        target_id: str,
    ) -> None:
        prefixed = format_batch(batch, ctx.platform)

        redis = await get_redis_async()
        cache_key = sessions.session_cache_key(ctx.platform, target_id)
        cached_session_id = await redis.get(cache_key)
        active_session_id = (
            cached_session_id.decode()
            if isinstance(cached_session_id, bytes)
            else cached_session_id
        )

        async def _on_session_id(sid: str) -> None:
            nonlocal active_session_id
            active_session_id = sid
            try:
                await redis.set(cache_key, sid, ex=SESSION_TTL)
            except Exception:
                logger.warning("Failed to cache session id for target %s", target_id)

        flush_at = adapter.chunk_flush_at
        buffer = ""
        sent_any_content = False
        setup_prompt_sent = False

        async def _on_setup_required(
            session_id: str,
            setup_output: dict[str, Any],
            _tool_name: str | None,
        ) -> None:
            nonlocal active_session_id, buffer, sent_any_content, setup_prompt_sent
            if setup_prompt_sent:
                return
            setup_prompt_sent = True
            # This callback carries the authoritative session id — adopt it so
            # buffered workspace artifacts resolve instead of falling back to
            # the "no session" plain-text note.
            active_session_id = session_id
            # Drain any pending text first so the link button doesn't render
            # ahead of the message it belongs to.
            if buffer.strip():
                if await self._send_text_and_artifacts(
                    adapter, target_id, buffer, ctx, session_id
                ):
                    sent_any_content = True
                buffer = ""
            sent_any_content = True
            session_url = _copilot_session_url(session_id)
            message = _setup_required_message(setup_output)
            if session_url is None:
                # No base URL configured — fall back to plain text since
                # Discord rejects relative URLs on link buttons.
                logger.warning(
                    "No frontend/platform base URL configured; "
                    "sending setup-required prompt without a button"
                )
                await adapter.send_message(
                    target_id, message, mentionable_users=ctx.mentionable_users
                )
                return
            await adapter.send_link(
                target_id,
                message,
                link_label="Open AutoGPT",
                link_url=session_url,
            )

        async def _on_setup_dropped(
            session_id: str,
            _tool_name: str | None,
        ) -> None:
            nonlocal active_session_id, buffer, sent_any_content
            active_session_id = session_id
            # Drain pending text first so the notice doesn't jump ahead of the
            # message it follows.
            if buffer.strip():
                if await self._send_text_and_artifacts(
                    adapter, target_id, buffer, ctx, session_id
                ):
                    sent_any_content = True
                buffer = ""
            sent_any_content = True
            await adapter.send_message(
                target_id,
                _setup_dropped_message(),
                mentionable_users=ctx.mentionable_users,
            )

        started_at = time.monotonic()
        reply_chars = 0
        typing_task = asyncio.create_task(_keep_typing(adapter, target_id))
        try:
            async for chunk in self._api.stream_chat(
                platform=ctx.platform,
                platform_user_id=ctx.user_id,
                message=prefixed,
                session_id=active_session_id,
                platform_server_id=ctx.server_id,
                on_session_id=_on_session_id,
                on_setup_required=_on_setup_required,
                on_setup_dropped=_on_setup_dropped,
            ):
                buffer += chunk
                reply_chars += len(chunk)
                if len(buffer) >= flush_at:
                    post, buffer = split_at_boundary(buffer, flush_at)
                    if post and post.strip():
                        if await self._send_text_and_artifacts(
                            adapter, target_id, post, ctx, active_session_id
                        ):
                            sent_any_content = True
        except DuplicateChatMessageError:
            # Another in-flight turn is already processing this exact message —
            # stay quiet so the user doesn't get a double response.
            logger.info("Duplicate message dropped for target %s", target_id)
            return
        except BotStreamError as exc:
            # Stream couldn't complete (timeout, subscribe fail, backend stream
            # error). Track the specific kind, surface a generic message, and
            # do NOT fire reply_sent below.
            logger.warning(
                "Stream failed for target %s: %s (%s)",
                target_id,
                exc,
                exc.error_kind,
            )
            self._track_stream_error(ctx, exc.error_kind)
            await adapter.send_message(
                target_id,
                "AutoPilot ran into an error. Try again in a moment.",
            )
            return
        except NotFoundError:
            logger.exception("Chat turn rejected")
            self._track_stream_error(ctx, "chat_turn_rejected")
            await adapter.send_message(
                target_id, "AutoPilot ran into an error. Try again later."
            )
            return
        except Exception:
            logger.exception(
                "Unexpected error during streaming for target %s", target_id
            )
            self._track_stream_error(ctx, "stream_exception")
            await adapter.send_message(
                target_id,
                "Something went wrong. Try again in a moment.",
            )
            return
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            await adapter.stop_typing(target_id)

        if buffer.strip():
            if await self._send_text_and_artifacts(
                adapter, target_id, buffer, ctx, active_session_id
            ):
                sent_any_content = True

        if not sent_any_content:
            await adapter.send_message(
                target_id,
                "AutoPilot didn't produce a response. Try rephrasing your question.",
            )
            self._track_stream_error(ctx, "empty_reply")
            return

        self._api.track_event(
            platform=ctx.platform,
            event_type="reply_sent",
            server_id=ctx.server_id,
            channel_type=ctx.channel_type,
            char_count=reply_chars,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )

        if (
            ctx.channel_type == "channel"
            and target_id != ctx.channel_id
            and active_session_id
        ):
            # Fire-and-forget so the rename poll doesn't stall follow-up turns.
            task = asyncio.create_task(
                self._rename_thread_from_session_title(
                    adapter, target_id, active_session_id
                )
            )
            self._rename_tasks.add(task)
            task.add_done_callback(self._rename_tasks.discard)

    async def _rename_thread_from_session_title(
        self,
        adapter: PlatformAdapter,
        thread_id: str,
        session_id: str,
    ) -> None:
        for attempt in range(TITLE_RENAME_ATTEMPTS):
            try:
                title = await self._api.get_session_title(session_id)
            except Exception:
                logger.warning(
                    "Failed to fetch generated title for %s (attempt %d/%d)",
                    session_id,
                    attempt + 1,
                    TITLE_RENAME_ATTEMPTS,
                    exc_info=True,
                )
                title = None
            if title:
                await adapter.rename_thread(thread_id, clamp_thread_name(title))
                return
            if attempt < TITLE_RENAME_ATTEMPTS - 1:
                await asyncio.sleep(TITLE_RENAME_INTERVAL_SECONDS)

    def _track_stream_error(self, ctx: MessageContext, error_kind: str) -> None:
        self._api.track_event(
            platform=ctx.platform,
            event_type="stream_error",
            server_id=ctx.server_id,
            channel_type=ctx.channel_type,
            error_kind=error_kind,
        )

    # -- Linking --

    async def _ensure_linked(
        self, ctx: MessageContext, adapter: PlatformAdapter
    ) -> bool:
        try:
            if ctx.is_dm:
                result = await self._api.resolve_user(ctx.platform, ctx.user_id)
                if not result.linked:
                    await self._prompt_user_link(ctx, adapter)
                    return False
            else:
                if not ctx.server_id:
                    logger.error("Non-DM message missing server_id: %r", ctx)
                    return False
                result = await self._api.resolve_server(ctx.platform, ctx.server_id)
                if not result.linked:
                    await adapter.send_message(
                        ctx.channel_id,
                        "This server isn't linked to an AutoGPT account yet. "
                        "Ask a server admin to run `/setup` first.",
                    )
                    return False
        except ValueError:
            # ValueError-based domain exceptions (NotFoundError etc.) arrive
            # over RPC with this base type.
            logger.exception("Failed to check link status")
            await adapter.send_message(
                ctx.channel_id, "Something went wrong. Try again later."
            )
            return False
        except Exception:
            logger.exception("Unexpected error while checking link status")
            await adapter.send_message(
                ctx.channel_id,
                "Something went wrong. Try again in a moment.",
            )
            return False
        return True

    async def _prompt_user_link(
        self, ctx: MessageContext, adapter: PlatformAdapter
    ) -> None:
        try:
            result = await self._api.create_user_link_token(
                platform=ctx.platform,
                platform_user_id=ctx.user_id,
                platform_username=ctx.username,
            )
            platform_display = ctx.platform.capitalize()
            await adapter.send_link(
                ctx.channel_id,
                f"Your {platform_display} DMs aren't linked to an AutoGPT "
                "account yet. Click below to connect — once linked, you can "
                "chat with AutoPilot right here.",
                link_label="Link Account",
                link_url=result.link_url,
            )
        except LinkAlreadyExistsError:
            # Race: user got linked between resolve_user and create. Re-check
            # — if still not linked, the backend returned a stale error and
            # we shouldn't spam the user.
            re_check = await self._api.resolve_user(ctx.platform, ctx.user_id)
            if re_check.linked:
                return
            logger.exception(
                "create_user_link_token raised 'already exists' "
                "but user isn't actually linked"
            )
        except Exception:
            logger.exception("Failed to create user link token")
            await adapter.send_message(
                ctx.channel_id,
                "Something went wrong setting up the link. Try again later.",
            )


async def _keep_typing(adapter: PlatformAdapter, target_id: str) -> None:
    """Re-fire the typing indicator every 8s so it doesn't expire mid-stream."""
    try:
        while True:
            await adapter.start_typing(target_id)
            await asyncio.sleep(8)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Typing loop error", exc_info=True)


def build_thread_name(text: str, username: str) -> str:
    """Build a Discord-safe thread name from the user's first prompt."""
    cleaned = " ".join(text.split())
    if not cleaned:
        cleaned = f"{username} with AutoPilot"
    return clamp_thread_name(f"{THREAD_NAME_PREFIX}{cleaned}")


def clamp_thread_name(name: str) -> str:
    cleaned = " ".join(name.split()) or "AutoPilot Chat"
    if len(cleaned) > THREAD_NAME_MAX_LENGTH:
        return cleaned[: THREAD_NAME_MAX_LENGTH - 3].rstrip() + "..."
    return cleaned


def _copilot_session_url(session_id: str) -> str | None:
    """Absolute URL to the live copilot session, or None if no base URL set."""
    config = Settings().config
    base_url = (config.frontend_base_url or config.platform_base_url).rstrip("/")
    if not base_url:
        return None
    return f"{base_url}/copilot?sessionId={quote(session_id, safe='')}"


def _setup_dropped_message() -> str:
    return (
        "⚠️ AutoPilot tried to send you a sign-in link, but the data arrived "
        "corrupted so the button couldn't be shown. Please ask it to try that "
        "step again."
    )


def _setup_required_message(setup_output: dict[str, Any]) -> str:
    message = str(setup_output.get("message") or "").strip()
    if not message:
        message = (
            "AutoPilot needs you to sign in or authorize an integration "
            "before it can continue."
        )

    return (
        f"{message}\n\n"
        "Click the button below to open your AutoGPT chat and finish setup "
        "there. Reply here when you're done."
    )
