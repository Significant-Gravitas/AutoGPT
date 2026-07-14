"""Streaming one batched turn to a platform target.

Owns everything between "a batch of messages is ready" and "the reply is fully
delivered": the AutoGPT stream, chunked sends with a persistent typing
indicator, workspace-artifact delivery, setup-required prompts, error surfaces,
and the post-turn thread rename. The orchestration (batching, linking, target
resolution) stays in ``handler``.
"""

import asyncio
import logging
import time
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from backend.data.redis_client import get_redis_async
from backend.data.sharing.workspace_refs import (
    WorkspaceArtifactLink,
    extract_artifact_links,
)
from backend.platform_linking.models import TurnDenial
from backend.util.exceptions import DuplicateChatMessageError, NotFoundError
from backend.util.settings import Settings

from . import sessions
from .adapters.base import (
    FileAttachment,
    MessageContext,
    PlatformAdapter,
    StreamDraftOutcome,
)
from .bot_backend import BotBackend, BotStreamError, ChatTurnDeniedError
from .config import SESSION_TTL
from .prompt import clamp_thread_name
from .text import format_batch, split_at_boundary

logger = logging.getLogger(__name__)

TITLE_RENAME_ATTEMPTS = 5
TITLE_RENAME_INTERVAL_SECONDS = 1.0

# Cadence for live draft previews on platforms that support them — throttled
# so a fast stream doesn't turn every chunk into an API call.
DRAFT_UPDATE_INTERVAL_SECONDS = 1.0


class DraftStreamer:
    """Throttled live preview of the in-progress reply.

    Purely additive on top of the buffered sends: drafts are ephemeral
    previews (Telegram ``sendMessageDraft``), and the finished reply still
    arrives via the normal send path, which supersedes them. Any failure
    silently disables previews for the rest of the turn — content delivery
    is never at stake.
    """

    def __init__(self, adapter: PlatformAdapter, target_id: str):
        self._adapter = adapter
        self._target_id = target_id
        self._enabled = adapter.supports_stream_drafts
        # Telegram requires a nonzero draft id; same id per turn = animated
        # in-place updates.
        self._draft_id = (uuid4().int & 0x7FFFFFFF) or 1
        self._last_sent_at = 0.0
        self._last_text = ""

    async def update(self, text: str) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if now - self._last_sent_at < DRAFT_UPDATE_INTERVAL_SECONDS:
            return
        preview = text.strip()
        if not preview or preview == self._last_text:
            return
        try:
            outcome = await self._adapter.send_stream_draft(
                self._target_id, self._draft_id, preview
            )
        except Exception:
            logger.debug("Stream draft update failed", exc_info=True)
            self._enabled = False
            return
        if outcome is StreamDraftOutcome.STOPPED:
            self._enabled = False
            return
        # SKIPPED = transient no-op (preview momentarily too long): leave the
        # throttle untouched so the next chunk retries instead of eating a
        # full interval of silence on a draft the user never saw.
        if outcome is StreamDraftOutcome.SHOWN:
            self._last_sent_at = now
            self._last_text = preview


async def send_denial(
    api: BotBackend,
    ctx: MessageContext,
    adapter: PlatformAdapter,
    target_id: str,
    denial: TurnDenial,
) -> None:
    """Render a turn denial: message plus CTA button when configured."""
    logger.info("Turn denied (%s) for target %s", denial.reason, target_id)
    api.track_event(
        platform=ctx.platform,
        event_type="turn_denied",
        server_id=ctx.server_id,
        channel_type=ctx.channel_type,
        error_kind=denial.reason,
    )
    if denial.button_url and denial.button_label:
        await adapter.send_link(
            target_id,
            denial.message,
            link_label=denial.button_label,
            link_url=denial.button_url,
        )
    else:
        await adapter.send_message(
            target_id, denial.message, mentionable_users=ctx.mentionable_users
        )


class TurnStreamer:
    def __init__(self, api: BotBackend):
        self._api = api
        # Strong-ref set so the GC doesn't drop fire-and-forget rename tasks.
        self._rename_tasks: set[asyncio.Task[None]] = set()

    async def stream_batch(
        self,
        batch: list[tuple[str, str, str]],
        ctx: MessageContext,
        adapter: PlatformAdapter,
        target_id: str,
        file_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        prefixed = format_batch(batch, ctx.platform)

        redis = await get_redis_async()
        cache_key = sessions.session_cache_key(ctx.platform, target_id)
        if session_id is not None:
            # Attachments were already uploaded to this session — use it
            # directly so the turn can't diverge from where the files went.
            active_session_id: str | None = session_id
        else:
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
            session_url = copilot_session_url(session_id)
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
        draft = DraftStreamer(adapter, target_id)
        typing_task = asyncio.create_task(_keep_typing(adapter, target_id))
        try:
            async for chunk in self._api.stream_chat(
                platform=ctx.platform,
                platform_user_id=ctx.user_id,
                message=prefixed,
                session_id=active_session_id,
                platform_server_id=ctx.server_id,
                file_ids=file_ids,
                on_session_id=_on_session_id,
                on_setup_required=_on_setup_required,
                on_setup_dropped=_on_setup_dropped,
            ):
                buffer += chunk
                reply_chars += len(chunk)
                await draft.update(buffer)
                if len(buffer) >= flush_at:
                    post, buffer = split_at_boundary(buffer, flush_at)
                    if post and post.strip():
                        if await self._send_text_and_artifacts(
                            adapter, target_id, post, ctx, active_session_id
                        ):
                            sent_any_content = True
        except ChatTurnDeniedError as exc:
            # Refused before running (paywall / rate limit). Show the message
            # and, when present, a CTA button (Subscribe / Upgrade).
            await send_denial(self._api, ctx, adapter, target_id, exc.denial)
            return
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
                "AutoGPT ran into an error. Try again in a moment.",
            )
            return
        except NotFoundError:
            logger.exception("Chat turn rejected")
            self._track_stream_error(ctx, "chat_turn_rejected")
            await adapter.send_message(
                target_id, "AutoGPT ran into an error. Try again later."
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
                "AutoGPT didn't produce a response. Try rephrasing your question.",
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

    # -- Artifact delivery --

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
        session_url = copilot_session_url(session_id)
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

    # -- Post-turn --

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
                await adapter.rename_thread(
                    thread_id, clamp_thread_name(title, adapter.max_thread_name_length)
                )
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


async def _keep_typing(adapter: PlatformAdapter, target_id: str) -> None:
    """Re-fire the typing indicator so it doesn't expire mid-stream.

    Cadence is the adapter's ``typing_refresh_interval`` — platform indicators
    auto-expire at different rates.
    """
    try:
        while True:
            await adapter.start_typing(target_id)
            await asyncio.sleep(adapter.typing_refresh_interval)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Typing loop error", exc_info=True)


def copilot_session_url(session_id: str) -> str | None:
    """Absolute URL to the live copilot session, or None if no base URL set."""
    config = Settings().config
    base_url = (config.frontend_base_url or config.platform_base_url).rstrip("/")
    if not base_url:
        return None
    return f"{base_url}/copilot?sessionId={quote(session_id, safe='')}"


def _setup_dropped_message() -> str:
    return (
        "⚠️ AutoGPT tried to send you a sign-in link, but the data arrived "
        "corrupted so the button couldn't be shown. Please ask it to try that "
        "step again."
    )


def _setup_required_message(setup_output: dict[str, Any]) -> str:
    message = str(setup_output.get("message") or "").strip()
    if not message:
        message = (
            "AutoGPT needs you to sign in or authorize an integration "
            "before it can continue."
        )

    return (
        f"{message}\n\n"
        "Click the button below to open your AutoGPT chat and finish setup "
        "there. Reply here when you're done."
    )
