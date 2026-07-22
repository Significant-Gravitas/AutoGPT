"""Platform-agnostic message handler.

Receives a MessageContext from any adapter and orchestrates the turn: link
resolution, thread routing, attachment ingestion, then batched streaming.
The streaming machinery itself lives in ``turn_stream``; prompt assembly in
``prompt``; attachment upload + failure notes in ``attachments``.
"""

import asyncio
import logging

from pydantic import BaseModel, Field

from backend.platform_linking.models import TurnDenial
from backend.util.exceptions import LinkAlreadyExistsError

from . import sessions, threads
from .adapters.base import MessageContext, PlatformAdapter
from .attachments import (
    format_attachment_problems,
    model_attachment_note,
    upload_attachments,
)
from .bot_backend import BotBackend
from .prompt import build_message_text, build_thread_name
from .turn_stream import TurnStreamer, send_denial

logger = logging.getLogger(__name__)

# Cap on file IDs carried into a single turn — must stay <= the
# ``BotChatRequest.file_ids`` ``max_length``. Batching several file-heavy
# messages can accumulate more than that; cap so the turn runs (with a log)
# instead of failing validation.
MAX_TURN_FILE_IDS = 20


class TargetState(BaseModel):
    """Per-target streaming state.

    A "target" is wherever the bot replies — a thread ID, a DM channel ID.
    `pending` holds messages that arrived while a stream was running; they
    get drained as a single batched follow-up turn when the stream ends.
    """

    processing: bool = False
    # Each entry: (username, user_id, text)
    pending: list[tuple[str, str, str]] = Field(default_factory=list)
    # Workspace file IDs uploaded for messages in `pending`, drained together
    # with them so a batched turn carries every attached file.
    pending_file_ids: list[str] = Field(default_factory=list)
    # Session the pending attachments were uploaded to, carried straight to the
    # turn so it uses the same session (not a separate Redis read that could
    # diverge). None for text-only batches, which resolve the session normally.
    session_id: str | None = None


class MessageHandler:
    def __init__(self, api: BotBackend):
        self._api = api
        self._streamer = TurnStreamer(api)
        self._targets: dict[str, TargetState] = {}
        # Per-target lock serialising session resolution, so two attachment
        # messages racing on a fresh target converge on ONE session instead of
        # each creating its own and splitting the files across them.
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def handle(self, ctx: MessageContext, adapter: PlatformAdapter) -> None:
        if not ctx.text.strip() and not ctx.attachments:
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
                # thread history into the prompt so AutoGPT has context,
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

        # Attachments must be written into the turn's session folder so
        # AutoGPT can read them (same as a web upload), and the turn must run
        # in that same session — resolve it up front and thread it through.
        session_id: str | None = None
        file_ids: list[str]
        upload_problems: list[tuple[str, str]]
        if ctx.attachments:
            session_id, denial = await self._resolve_session_for_attachments(
                ctx, target_id
            )
            if denial is not None:
                # The turn gate refused the user before anything was uploaded —
                # render the denial and stop; no file is scanned or stored and
                # no turn runs.
                await send_denial(self._api, ctx, adapter, target_id, denial)
                return
            if session_id is None:
                # Without a session the files can't be made readable to
                # AutoGPT, so report them as failed rather than uploading
                # them somewhere it can't see.
                file_ids = []
                upload_problems = [
                    (a.filename, "couldn't be uploaded") for a in ctx.attachments
                ]
            else:
                file_ids, upload_problems = await upload_attachments(
                    self._api, ctx, session_id
                )
        else:
            file_ids, upload_problems = await upload_attachments(self._api, ctx)

        # Files that didn't make it: adapter-stage (too large / failed download)
        # plus upload-stage (virus / quota). Tell the user AND, below, the model
        # — so neither assumes a dropped file was actually read.
        problems = list(ctx.skipped_attachments) + upload_problems
        if problems:
            await adapter.send_message(target_id, format_attachment_problems(problems))

        # Decide on NEW input only — typed text or a usable upload. Thread
        # history (folded into the prompt below) is context, not a reason to
        # respond, so don't let it keep an otherwise-empty turn alive.
        if not ctx.text.strip() and not file_ids:
            # The user already got the note. Don't enqueue a blank turn, and
            # unsubscribe a thread we just created for a channel message so it
            # doesn't linger orphaned-but-subscribed for 7 days.
            if ctx.channel_type == "channel" and target_id != ctx.channel_id:
                await threads.unsubscribe(ctx.platform, target_id)
            return

        # A file-only message gets a nudge so AutoGPT looks at the uploads
        # instead of seeing an empty "[Current message]".
        current_text = ctx.text if ctx.text.strip() else "(see the attached file(s))"
        message_text = self._message_text(ctx, include_thread_history, current_text)
        if problems:
            message_text += "\n\n" + model_attachment_note(problems)
        await self._enqueue_and_process(
            ctx, adapter, target_id, message_text, file_ids, session_id
        )

    # -- Session + target resolution --

    def _session_lock(self, target_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(target_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[target_id] = lock
        return lock

    async def _resolve_session_for_attachments(
        self, ctx: MessageContext, target_id: str
    ) -> tuple[str | None, TurnDenial | None]:
        """Resolve (or create) the session this message's attachments upload
        into, so they land where the turn will read them (``/sessions/<id>/``).

        Serialised per target so concurrent attachment messages converge on one
        session instead of splitting the files. Returns ``(session_id, None)``
        on success, ``(None, denial)`` when the turn gate refused the user
        (the caller renders the denial and skips the upload + turn), and
        ``(None, None)`` on failure — the caller then reports the files as
        un-attachable rather than uploading them somewhere AutoGPT can't see.
        """
        async with self._session_lock(target_id):
            try:
                result = await self._api.ensure_session(
                    platform=ctx.platform,
                    platform_user_id=ctx.user_id,
                    platform_server_id=ctx.server_id,
                    session_id=await sessions.get_session(ctx.platform, target_id),
                )
                if result.denial is not None:
                    return None, result.denial
                if result.session_id:
                    await sessions.set_session(
                        ctx.platform, target_id, result.session_id
                    )
                return result.session_id, None
            except Exception:
                logger.exception(
                    "Failed to resolve session for uploads (user %s)", ctx.user_id
                )
                return None, None

    async def _resolve_target(
        self, ctx: MessageContext, adapter: PlatformAdapter
    ) -> str | None:
        if ctx.channel_type == "dm":
            return ctx.channel_id

        if ctx.channel_type == "thread":
            return ctx.channel_id

        # channel_type == "channel" — create a thread and subscribe
        thread_name = build_thread_name(
            ctx.text, ctx.username, adapter.max_thread_name_length
        )
        thread_id = await adapter.create_thread(
            ctx.channel_id, ctx.message_id, thread_name
        )
        if not thread_id:
            logger.warning("Thread creation failed, falling back to channel reply")
            return ctx.channel_id
        await threads.subscribe(ctx.platform, thread_id)
        return thread_id

    # -- Batching --

    def _message_text(
        self,
        ctx: MessageContext,
        include_thread_history: bool,
        current_text: str | None = None,
    ) -> str:
        return build_message_text(ctx, include_thread_history, current_text)

    async def _stream_batch(
        self,
        batch: list[tuple[str, str, str]],
        ctx: MessageContext,
        adapter: PlatformAdapter,
        target_id: str,
        file_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        await self._streamer.stream_batch(
            batch, ctx, adapter, target_id, file_ids=file_ids, session_id=session_id
        )

    async def _enqueue_and_process(
        self,
        ctx: MessageContext,
        adapter: PlatformAdapter,
        target_id: str,
        message_text: str | None = None,
        file_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        state = self._targets.setdefault(target_id, TargetState())
        state.pending.append((ctx.username, ctx.user_id, message_text or ctx.text))
        if file_ids:
            state.pending_file_ids.extend(file_ids)
        # First attachment message in a batch pins the session; later ones
        # resolved the same session (serialised by _session_lock), so keep it.
        if session_id and state.session_id is None:
            state.session_id = session_id

        if state.processing:
            # Another invocation is streaming for this target — it will pick
            # up the message we just appended when its current stream ends.
            return

        state.processing = True
        try:
            while state.pending:
                batch = list(state.pending)
                batch_file_ids = list(state.pending_file_ids)
                batch_session_id = state.session_id
                state.pending.clear()
                state.pending_file_ids.clear()
                state.session_id = None
                if len(batch_file_ids) > MAX_TURN_FILE_IDS:
                    logger.warning(
                        "Dropping %d batched file(s) over the per-turn cap of %d",
                        len(batch_file_ids) - MAX_TURN_FILE_IDS,
                        MAX_TURN_FILE_IDS,
                    )
                    batch_file_ids = batch_file_ids[:MAX_TURN_FILE_IDS]
                await self._stream_batch(
                    batch,
                    ctx,
                    adapter,
                    target_id,
                    file_ids=batch_file_ids,
                    session_id=batch_session_id,
                )
        finally:
            state.processing = False
            # Drop the empty state so the dict doesn't grow unbounded across
            # the bot's lifetime.
            if not state.pending:
                self._targets.pop(target_id, None)

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
                "chat with AutoGPT right here.",
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
