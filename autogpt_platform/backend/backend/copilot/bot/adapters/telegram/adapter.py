"""Telegram adapter — webhook-based (Bot API), single bot for all chats.

Mounts one inbound updates route on the main backend API. Telegram signs
nothing — instead the webhook is registered with a secret token that Telegram
echoes back in the ``X-Telegram-Bot-Api-Secret-Token`` header on every POST;
verification is a constant-time compare of that header.

Chat model mapping: a private chat is a DM (auto-converse); a group or
supergroup message engages the bot only when it @mentions the bot or replies
to one of the bot's messages — the bot never subscribes itself to a
human-owned group conversation. Forum-topic placement rides along in the
encoded target (``chat_id|thread_id``) so replies land in the right topic.
"""

import asyncio
import hmac
import html
import json
import logging
import re
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from backend.copilot.bot.adapters.base import (
    ChannelInfo,
    ChannelType,
    FileAttachment,
    MessageCallback,
    MessageContext,
    PostedRef,
    StreamDraftOutcome,
    WebhookAdapter,
    read_verified_webhook_body,
    unauthorized_webhook_response,
)
from backend.copilot.bot.adapters.shared import InboundFile, collect_attachments
from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.config import MAX_INBOUND_ATTACHMENTS
from backend.copilot.bot.text import iter_chunks, resolve_mentions

from . import commands, config
from .api_client import TelegramClient
from .targets import decode_target as _decode_target
from .targets import encode_target as _encode_target
from .text import to_html

logger = logging.getLogger(__name__)

UPDATES_PATH = "/api/copilot-webhooks/telegram/updates"

# Telegram chat IDs are integers of any length; groups/supergroups are
# negative (supergroups start with -100). Used to tell a raw proactive chat
# ref from a name (names are never purely numeric).
_CHAT_ID_RE = re.compile(r"^-?\d+$")

# Message fields that carry a downloadable file, with a fallback filename for
# the kinds Telegram doesn't name.
_FILE_FIELDS = (
    ("document", None),
    ("video", "video.mp4"),
    ("audio", "audio.mp3"),
    ("voice", "voice.ogg"),
)


class TelegramAdapter(WebhookAdapter):
    def __init__(self, api: BotBackend):
        self._api = api
        self._client = TelegramClient(config.get_bot_token())
        self._on_message_callback: Optional[MessageCallback] = None
        # getMe identity, resolved once (truthy-only cache).
        self._bot_id: str = ""
        self._bot_username: str = ""
        # Strong-ref set so the GC doesn't drop fire-and-forget update tasks.
        self._update_tasks: set[asyncio.Task[None]] = set()

    @property
    def platform_name(self) -> str:
        return "telegram"

    @property
    def max_message_length(self) -> int:
        return config.MAX_MESSAGE_LENGTH

    @property
    def chunk_flush_at(self) -> int:
        return config.CHUNK_FLUSH_AT

    @property
    def max_attachment_bytes(self) -> int:
        return config.MAX_ATTACHMENT_BYTES

    @property
    def max_thread_name_length(self) -> int:
        return config.MAX_THREAD_NAME_LENGTH

    @property
    def typing_refresh_interval(self) -> float:
        return config.TYPING_REFRESH_SECONDS

    def looks_like_channel_id(self, ref: str) -> bool:
        return bool(_CHAT_ID_RE.match(ref))

    def localize_markup(self, text: str) -> str:
        return to_html(text)

    def on_message(self, callback: MessageCallback) -> None:
        self._on_message_callback = callback

    def register_routes(self, app: FastAPI) -> None:
        app.add_api_route(UPDATES_PATH, self._handle_update_request, methods=["POST"])
        # Publish the command menu on startup so BotFather needs no manual
        # /setcommands step and the menu can't drift from the code.
        app.add_event_handler("startup", self._register_command_menu)

    async def _register_command_menu(self) -> None:
        try:
            await self._client.call("setMyCommands", commands=commands.COMMAND_MENU)
        except Exception:
            logger.warning("Failed to register Telegram command menu", exc_info=True)

    # -- Inbound --

    async def _handle_update_request(self, request: Request) -> Response:
        raw = await read_verified_webhook_body(request, _verify_secret)
        if raw is None:
            return unauthorized_webhook_response()
        update = json.loads(raw)
        # Fire-and-forget so we ACK immediately — Telegram retries slow
        # webhooks, which would double-process the turn.
        task = asyncio.create_task(self._dispatch_update(update))
        self._update_tasks.add(task)
        task.add_done_callback(self._update_tasks.discard)
        return PlainTextResponse("ok")

    async def _dispatch_update(self, update: dict[str, Any]) -> None:
        membership = update.get("my_chat_member")
        if membership:
            self._track_membership_change(membership)
            return
        message = update.get("message")
        if not message:
            return  # Edits, reactions, other member updates — not conversation input.
        sender = message.get("from") or {}
        if sender.get("is_bot"):
            return  # Skip bot messages (including our own) to avoid loops.
        _bot_id, bot_username = await self._bot_identity()
        command = commands.parse_command(message, bot_username)
        if command is not None:
            # This runs as a fire-and-forget task — an unhandled error would
            # only surface as asyncio's deferred "exception never retrieved".
            try:
                await commands.handle(self._api, self._client, message, command)
            except Exception:
                logger.exception("Telegram command handler failed")
            return
        if self._on_message_callback is None:
            return
        ctx = await self._build_context(message)
        if ctx is None:
            return
        await self._ack_reaction(message)
        try:
            await self._on_message_callback(ctx, self)
        except Exception:
            logger.exception("Telegram update handler failed")

    async def _ack_reaction(self, message: dict[str, Any]) -> None:
        """Best-effort 👀 on the triggering message — instant feedback that
        the bot picked it up, ahead of the (slower) typing indicator."""
        try:
            await self._client.call(
                "setMessageReaction",
                chat_id=str((message.get("chat") or {}).get("id", "")),
                message_id=message.get("message_id"),
                reaction=[{"type": "emoji", "emoji": "👀"}],
            )
        except Exception:
            logger.debug("Telegram reaction ack failed", exc_info=True)

    def _track_membership_change(self, membership: dict[str, Any]) -> None:
        """Keep the admin server roster current: the bot being added to /
        removed from a group arrives as a my_chat_member update."""
        chat = membership.get("chat") or {}
        if chat.get("type") not in ("group", "supergroup"):
            return
        status = (membership.get("new_chat_member") or {}).get("status")
        chat_id = str(chat.get("id", ""))
        if not chat_id or not status:
            return
        if status in ("member", "administrator"):
            self._api.track_guild_joined("telegram", chat_id, chat.get("title"))
        elif status in ("left", "kicked"):
            self._api.track_guild_left("telegram", chat_id)

    async def _build_context(self, message: dict[str, Any]) -> Optional[MessageContext]:
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        chat_id = str(chat.get("id", ""))
        message_id = message.get("message_id")
        if not chat_id or message_id is None:
            return None
        is_private = chat.get("type") == "private"
        text = message.get("text") or message.get("caption") or ""

        if is_private:
            channel_type: ChannelType = "dm"
            bot_mentioned = True
        else:
            # In groups the bot engages only when addressed: an @mention in
            # the text or a direct reply to one of its messages.
            bot_id, bot_username = await self._bot_identity()
            mentioned = bool(
                bot_username
                and re.search(rf"@{re.escape(bot_username)}\b", text, re.IGNORECASE)
            )
            reply_to = (message.get("reply_to_message") or {}).get("from") or {}
            replied_to_bot = bool(bot_id) and str(reply_to.get("id", "")) == bot_id
            if not mentioned and not replied_to_bot:
                return None
            channel_type = "channel"
            bot_mentioned = True
            if bot_username:
                text = re.sub(
                    rf"@{re.escape(bot_username)}\b", "", text, flags=re.IGNORECASE
                ).strip()

        target_id = _encode_target(chat_id, message.get("message_thread_id"))
        attachments, skipped = await self._extract_attachments(message)
        return MessageContext(
            platform="telegram",
            channel_type=channel_type,
            # DMs bill to the user (no server); group chats are the "server".
            server_id=None if is_private else chat_id,
            channel_id=target_id,
            message_id=str(message_id),
            user_id=str(sender.get("id", "")),
            username=sender.get("username") or sender.get("first_name") or "unknown",
            text=text,
            bot_mentioned=bot_mentioned,
            mentionable_users=_collect_mentionable_users(message),
            attachments=attachments,
            skipped_attachments=skipped,
        )

    async def _extract_attachments(
        self, message: dict[str, Any]
    ) -> tuple[tuple, tuple]:
        inbound: list[InboundFile] = []
        for field, fallback_name in _FILE_FIELDS:
            f = message.get(field)
            if not f:
                continue
            size = int(f.get("file_size") or 0)
            if size <= 0:
                # The size cap is enforced against the declared size before
                # fetching — a file without one can't be admitted safely.
                logger.info("Skipping %s attachment without a declared size", field)
                continue
            inbound.append(
                InboundFile(
                    filename=f.get("file_name") or fallback_name or field,
                    size=size,
                    mime_type=f.get("mime_type"),
                    fetch=lambda f=f: self._client.download_file(f["file_id"]),
                )
            )
        photos = [
            p for p in message.get("photo") or [] if int(p.get("file_size") or 0) > 0
        ]
        if photos:
            # Telegram sends one photo in several resolutions — take the largest.
            best = max(photos, key=lambda p: int(p["file_size"]))
            inbound.append(
                InboundFile(
                    filename="photo.jpg",
                    size=int(best["file_size"]),
                    mime_type="image/jpeg",
                    fetch=lambda b=best: self._client.download_file(b["file_id"]),
                )
            )
        return await collect_attachments(
            inbound,
            max_count=MAX_INBOUND_ATTACHMENTS,
            max_bytes=self.max_attachment_bytes,
        )

    # -- Outbound --

    def _render(self, text: str, mentionable_users: tuple[tuple[str, str], ...]) -> str:
        # Localize (which HTML-escapes) FIRST, then inject mention anchors —
        # the anchors are HTML and must survive escaping. The allowlist IS the
        # ping safety: non-allowlisted names stay plain text.
        rendered, _pinged = resolve_mentions(
            self.localize_markup(text),
            mentionable_users,
            lambda name, uid: f'<a href="tg://user?id={uid}">@{html.escape(name)}</a>',
        )
        return rendered

    async def send_message(
        self,
        channel_id: str,
        text: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        chat_id, thread_id = _decode_target(channel_id)
        await self._client.call(
            "sendMessage",
            chat_id=chat_id,
            text=self._render(text, mentionable_users),
            parse_mode="HTML",
            message_thread_id=thread_id,
        )

    async def send_reply(
        self,
        channel_id: str,
        text: str,
        reply_to_message_id: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        chat_id, thread_id = _decode_target(channel_id)
        await self._client.call(
            "sendMessage",
            chat_id=chat_id,
            text=self._render(text, mentionable_users),
            parse_mode="HTML",
            message_thread_id=thread_id,
            reply_parameters={"message_id": int(reply_to_message_id)},
        )

    async def send_link(
        self, channel_id: str, text: str, link_label: str, link_url: str
    ) -> None:
        chat_id, thread_id = _decode_target(channel_id)
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": self.localize_markup(text),
            "parse_mode": "HTML",
            "message_thread_id": thread_id,
        }
        # Prefer a login_url button: Telegram attaches a signed identity to
        # the opened URL, so our /link page can verify WHO tapped it (seamless
        # linking). Requires the domain to be registered with @BotFather
        # (/setdomain) and HTTPS — fall back to a plain URL button otherwise
        # so local/dev setups keep working.
        if link_url.startswith("https://"):
            try:
                await self._client.call(
                    "sendMessage",
                    **params,
                    reply_markup={
                        "inline_keyboard": [
                            [{"text": link_label, "login_url": {"url": link_url}}]
                        ]
                    },
                )
                return
            except Exception:
                logger.info(
                    "login_url button rejected (domain not registered with "
                    "BotFather?); falling back to a plain URL button"
                )
        try:
            await self._client.call(
                "sendMessage",
                **params,
                reply_markup={
                    "inline_keyboard": [[{"text": link_label, "url": link_url}]]
                },
            )
        except Exception:
            # Telegram validates button URLs and rejects e.g. localhost (local
            # dev). The URL is still fine as plain text — degrade so the flow
            # keeps working everywhere.
            logger.info("URL button rejected; sending the link as plain text")
            # Escaped because parse_mode stays HTML — a bare & in the URL
            # would read as a malformed entity and kill the fallback too.
            params["text"] += html.escape(f"\n\n{link_label}: {link_url}")
            await self._client.call("sendMessage", **params)

    async def send_file(self, channel_id: str, text: str, file: FileAttachment) -> None:
        chat_id, thread_id = _decode_target(channel_id)
        # Images render inline as photos; everything else arrives as a document.
        send = (
            self._client.send_photo
            if (file.mime_type or "").startswith("image/")
            else self._client.send_document
        )
        await send(
            chat_id=chat_id,
            content=file.content,
            filename=file.filename or "file",
            caption=self.localize_markup(text) if text else None,
            message_thread_id=thread_id,
        )

    async def send_ephemeral(self, channel_id: str, user_id: str, text: str) -> None:
        # Telegram has no ephemeral messages — send normally.
        await self.send_message(channel_id, text)

    @property
    def supports_stream_drafts(self) -> bool:
        return True

    async def send_stream_draft(
        self, channel_id: str, draft_id: int, text: str
    ) -> StreamDraftOutcome:
        chat_id, thread_id = _decode_target(channel_id)
        if chat_id.startswith("-"):
            # sendMessageDraft only works in private chats — groups and
            # supergroups (negative chat ids) keep typing + buffered sends.
            return StreamDraftOutcome.STOPPED
        # localize_markup only rewrites complete constructs (closed code
        # fences, **pairs**, [label](url)), so rendering a partial stream
        # never produces unbalanced HTML — incomplete markdown just shows
        # literally until the closing token streams in.
        rendered = self.localize_markup(text)
        if len(rendered) > config.MAX_MESSAGE_LENGTH:
            # Over the draft cap (HTML escaping can expand a near-flush
            # buffer past it) — skip this one without an API call; the next
            # chunk flush shrinks the buffer and drafting resumes.
            return StreamDraftOutcome.SKIPPED
        try:
            await self._client.call(
                "sendMessageDraft",
                chat_id=int(chat_id),
                draft_id=draft_id,
                text=rendered,
                parse_mode="HTML",
                message_thread_id=thread_id,
            )
        except Exception:
            logger.debug("Telegram sendMessageDraft failed", exc_info=True)
            return StreamDraftOutcome.STOPPED
        return StreamDraftOutcome.SHOWN

    async def start_typing(self, channel_id: str) -> None:
        chat_id, thread_id = _decode_target(channel_id)
        try:
            await self._client.call(
                "sendChatAction",
                chat_id=chat_id,
                action="typing",
                message_thread_id=thread_id,
            )
        except Exception:
            logger.debug("Telegram sendChatAction failed", exc_info=True)

    async def create_thread(
        self, channel_id: str, message_id: str, name: str
    ) -> Optional[str]:
        # Group conversations stay in the chat (replying into the same forum
        # topic when there is one) — creating/subscribing threads in a
        # human-owned group would risk hijacking it. The handler falls back to
        # in-channel replies, and users continue by replying to the bot.
        return None

    # -- Proactive output --

    async def list_text_channels(
        self, server_ids: tuple[str, ...]
    ) -> list[ChannelInfo]:
        # The Bot API can't enumerate a bot's chats; proactive posts must
        # target a raw chat id.
        return []

    async def get_channel_server_id(self, channel_id: str) -> Optional[str]:
        chat_id, _ = _decode_target(channel_id)
        try:
            chat = await self._client.call("getChat", chat_id=chat_id)
        except Exception:
            return None
        if chat.get("type") in ("group", "supergroup"):
            return str(chat.get("id"))
        return None

    async def post_channel_message(
        self, channel_id: str, text: str
    ) -> Optional[PostedRef]:
        chat_id, thread_id = _decode_target(channel_id)
        first_id: Optional[str] = None
        for chunk in iter_chunks(self.localize_markup(text), config.CHUNK_FLUSH_AT):
            result = await self._client.call(
                "sendMessage",
                chat_id=chat_id,
                text=chunk,
                parse_mode="HTML",
                message_thread_id=thread_id,
            )
            if first_id is None:
                first_id = str(result.get("message_id", ""))
        if first_id is None:
            return None
        return PostedRef(id=first_id, url=None)

    async def create_channel_thread(
        self, channel_id: str, name: str, text: str
    ) -> Optional[PostedRef]:
        # No named threads to create — post with the name as a bold header;
        # the returned ref keeps subsequent sends in the same chat/topic.
        posted = await self.post_channel_message(channel_id, f"**{name}**\n\n{text}")
        if posted is None:
            return None
        return PostedRef(id=channel_id, url=posted.url)

    # -- Helpers --

    async def _bot_identity(self) -> tuple[str, str]:
        """The bot's own (id, username) from getMe, cached truthy-only so a
        transient failure doesn't permanently break mention detection."""
        if self._bot_id and self._bot_username:
            return self._bot_id, self._bot_username
        try:
            me = await self._client.call("getMe")
        except Exception:
            logger.warning("Failed to fetch Telegram identity (getMe)", exc_info=True)
            return self._bot_id, self._bot_username or config.get_bot_username()
        self._bot_id = str(me.get("id", "")) or self._bot_id
        self._bot_username = me.get("username", "") or self._bot_username
        return self._bot_id, self._bot_username


def _verify_secret(request: Request, _body: bytes) -> bool:
    expected = config.get_webhook_secret()
    provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    return bool(expected) and hmac.compare_digest(provided, expected)


def _collect_mentionable_users(message: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    # text_mention entities carry full user objects (users without a public
    # @username); those are the only inbound mentions with a numeric id we
    # can ping safely on the way back out.
    pairs: list[tuple[str, str]] = []
    for entity in message.get("entities") or []:
        user = entity.get("user")
        if entity.get("type") == "text_mention" and user and not user.get("is_bot"):
            pair = (user.get("first_name") or "user", str(user.get("id")))
            if pair not in pairs:
                pairs.append(pair)
    return tuple(pairs)
