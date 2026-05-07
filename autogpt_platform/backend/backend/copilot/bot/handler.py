"""Platform-agnostic message handler.

Receives a MessageContext from any adapter and drives the full AutoPilot
interaction: link resolution, thread routing, batched streaming with a
persistent typing indicator.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from backend.data.redis_client import get_redis_async
from backend.util.exceptions import (
    DuplicateChatMessageError,
    LinkAlreadyExistsError,
    NotFoundError,
)
from backend.util.settings import Settings

from . import threads
from .adapters.base import MessageContext, PlatformAdapter
from .bot_backend import BotBackend
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

        should_subscribe_thread = False
        if ctx.channel_type == "thread":
            is_subscribed = await threads.is_subscribed(ctx.platform, ctx.channel_id)
            if not is_subscribed:
                if not ctx.bot_mentioned:
                    return
                should_subscribe_thread = True

        if not await self._ensure_linked(ctx, adapter):
            return

        target_id = await self._resolve_target(ctx, adapter)
        if not target_id:
            return  # Thread not subscribed, ignore silently

        if should_subscribe_thread:
            await threads.subscribe(ctx.platform, ctx.channel_id)

        message_text = self._message_text(ctx) if should_subscribe_thread else ctx.text
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

    def _message_text(self, ctx: MessageContext) -> str:
        if not ctx.thread_history:
            return ctx.text

        platform_display = ctx.platform.capitalize()
        lines = ["[Recent thread context before this message]"]
        for entry in ctx.thread_history:
            user = (
                f"{entry.username} ({platform_display} user ID: {entry.user_id})"
                if entry.user_id
                else entry.username
            )
            lines.append(f"\n[From {user}]\n{entry.text}")
        lines.append(f"\n[Current message]\n{ctx.text}")
        return "\n".join(lines)

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

    async def _stream_batch(
        self,
        batch: list[tuple[str, str, str]],
        ctx: MessageContext,
        adapter: PlatformAdapter,
        target_id: str,
    ) -> None:
        prefixed = format_batch(batch, ctx.platform)

        redis = await get_redis_async()
        cache_key = f"copilot-bot:session:{ctx.platform}:{target_id}"
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
            nonlocal buffer, sent_any_content, setup_prompt_sent
            if setup_prompt_sent:
                return
            setup_prompt_sent = True
            # Drain any pending text first so the link button doesn't render
            # ahead of the message it belongs to.
            if buffer.strip():
                await adapter.send_message(
                    target_id, buffer, mentionable_users=ctx.mentionable_users
                )
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
            ):
                buffer += chunk
                if len(buffer) >= flush_at:
                    post, buffer = split_at_boundary(buffer, flush_at)
                    if post:
                        await adapter.send_message(
                            target_id,
                            post,
                            mentionable_users=ctx.mentionable_users,
                        )
                        if post.strip():
                            sent_any_content = True
        except DuplicateChatMessageError:
            # Another in-flight turn is already processing this exact message —
            # stay quiet so the user doesn't get a double response.
            logger.info("Duplicate message dropped for target %s", target_id)
            return
        except NotFoundError:
            logger.exception("Chat turn rejected")
            await adapter.send_message(
                target_id, "AutoPilot ran into an error. Try again later."
            )
            return
        except Exception:
            logger.exception(
                "Unexpected error during streaming for target %s", target_id
            )
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
            await adapter.send_message(
                target_id, buffer, mentionable_users=ctx.mentionable_users
            )
            sent_any_content = True

        if not sent_any_content:
            await adapter.send_message(
                target_id,
                "AutoPilot didn't produce a response. Try rephrasing your question.",
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
