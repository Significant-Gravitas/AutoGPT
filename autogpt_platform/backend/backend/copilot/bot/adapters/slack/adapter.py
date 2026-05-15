"""Slack adapter — webhook-based.

Mounts inbound Events API + slash command routes on the shared FastAPI app.
All platform-agnostic logic lives in the core handler.
"""

import asyncio
import json
import logging
import re
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from slack_sdk.web.async_client import AsyncWebClient

from backend.copilot.bot.bot_backend import BotBackend

from ..base import ChannelType, MessageCallback, MessageContext, WebhookAdapter
from . import commands, config, signing
from .text import to_mrkdwn

logger = logging.getLogger(__name__)

EVENTS_PATH = "/api/copilot-webhooks/slack/events"
COMMANDS_PATH = "/api/copilot-webhooks/slack/commands"

# Matches both `<@U123>` and `<@U123|displayname>` mention forms.
_USER_MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)(?:\|[^>]+)?>")


class SlackAdapter(WebhookAdapter):
    def __init__(self, api: BotBackend):
        self._api = api
        self._client = AsyncWebClient(token=config.get_bot_token())
        self._on_message_callback: Optional[MessageCallback] = None
        self._bot_user_id: Optional[str] = None
        self._user_name_cache: dict[str, str] = {}
        # Strong-ref set so the GC doesn't drop fire-and-forget event tasks.
        self._event_tasks: set[asyncio.Task[None]] = set()

    @property
    def platform_name(self) -> str:
        return "slack"

    @property
    def max_message_length(self) -> int:
        return config.MAX_MESSAGE_LENGTH

    @property
    def chunk_flush_at(self) -> int:
        return config.CHUNK_FLUSH_AT

    def on_message(self, callback: MessageCallback) -> None:
        self._on_message_callback = callback

    def register_routes(self, app: FastAPI) -> None:
        app.add_api_route(EVENTS_PATH, self._handle_event_request, methods=["POST"])
        app.add_api_route(COMMANDS_PATH, self._handle_command_request, methods=["POST"])

    # -- Inbound --

    async def _handle_event_request(self, request: Request) -> Response:
        raw = await request.body()
        if not _verify_signature(request, raw):
            return PlainTextResponse("invalid signature", status_code=401)
        payload = json.loads(raw)
        if payload.get("type") == "url_verification":
            return JSONResponse({"challenge": payload.get("challenge", "")})
        if payload.get("type") == "event_callback":
            event = payload.get("event") or {}
            # Fire-and-forget so we ACK within Slack's 3s window.
            task = asyncio.create_task(self._dispatch_event(event))
            self._event_tasks.add(task)
            task.add_done_callback(self._event_tasks.discard)
        return PlainTextResponse("ok")

    async def _handle_command_request(self, request: Request) -> Response:
        raw = await request.body()
        if not _verify_signature(request, raw):
            return PlainTextResponse("invalid signature", status_code=401)
        form_data = await request.form()
        # Slack slash-command form data is always string-valued; drop anything
        # else (UploadFile etc.) defensively before passing on.
        form: dict[str, str] = {
            k: v for k, v in form_data.items() if isinstance(v, str)
        }
        return await commands.handle(self._api, form)

    async def _dispatch_event(self, event: dict[str, Any]) -> None:
        if self._on_message_callback is None:
            return
        # Skip bot messages (including our own) to avoid loops.
        if event.get("subtype") == "bot_message" or event.get("bot_id"):
            return
        event_type = event.get("type")
        channel_type = event.get("channel_type")
        ctx: Optional[MessageContext] = None
        if event_type == "app_mention":
            ctx = await self._build_mention_context(event)
        elif event_type == "message" and channel_type == "im":
            ctx = await self._build_dm_context(event)
        elif (
            event_type == "message"
            and channel_type == "channel"
            and event.get("thread_ts")
        ):
            # Reply in a channel thread without an @mention — the handler
            # will check thread-subscription state and ignore if not ours.
            ctx = await self._build_thread_reply_context(event)
        if ctx is None:
            return
        try:
            await self._on_message_callback(ctx, self)
        except Exception:
            logger.exception("Slack event handler failed")

    async def _build_mention_context(
        self, event: dict[str, Any]
    ) -> Optional[MessageContext]:
        channel = event.get("channel")
        ts = event.get("ts")
        user = event.get("user")
        text = event.get("text") or ""
        if not channel or not ts or not user:
            return None
        thread_ts = event.get("thread_ts")
        channel_type: ChannelType
        if thread_ts and thread_ts != ts:
            channel_type = "thread"
            target_channel_id = _encode_target(channel, thread_ts)
        else:
            channel_type = "channel"
            target_channel_id = channel
        return MessageContext(
            platform="slack",
            channel_type=channel_type,
            server_id=event.get("team") or None,
            channel_id=target_channel_id,
            message_id=ts,
            user_id=user,
            username=await self._user_display_name(user),
            text=await self._strip_mentions(text),
            bot_mentioned=True,
            mentionable_users=await self._collect_mentionable_users(text),
        )

    async def _build_thread_reply_context(
        self, event: dict[str, Any]
    ) -> Optional[MessageContext]:
        channel = event.get("channel")
        ts = event.get("ts")
        user = event.get("user")
        thread_ts = event.get("thread_ts")
        text = event.get("text") or ""
        if not channel or not ts or not user or not thread_ts:
            return None
        return MessageContext(
            platform="slack",
            channel_type="thread",
            server_id=event.get("team") or None,
            channel_id=_encode_target(channel, thread_ts),
            message_id=ts,
            user_id=user,
            username=await self._user_display_name(user),
            text=await self._strip_mentions(text),
            bot_mentioned=False,
            mentionable_users=await self._collect_mentionable_users(text),
        )

    async def _build_dm_context(
        self, event: dict[str, Any]
    ) -> Optional[MessageContext]:
        channel = event.get("channel")
        ts = event.get("ts")
        user = event.get("user")
        text = event.get("text") or ""
        if not channel or not ts or not user:
            return None
        return MessageContext(
            platform="slack",
            channel_type="dm",
            server_id=None,
            channel_id=channel,
            message_id=ts,
            user_id=user,
            username=await self._user_display_name(user),
            text=await self._strip_mentions(text),
            bot_mentioned=True,
            mentionable_users=await self._collect_mentionable_users(text),
        )

    # -- Outbound --

    async def send_message(
        self,
        channel_id: str,
        text: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        channel, thread_ts = _decode_target(channel_id)
        await self._client.chat_postMessage(
            channel=channel,
            text=to_mrkdwn(text, mentionable_users),
            thread_ts=thread_ts,
        )

    async def send_reply(
        self,
        channel_id: str,
        text: str,
        reply_to_message_id: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        channel, thread_ts = _decode_target(channel_id)
        await self._client.chat_postMessage(
            channel=channel,
            text=to_mrkdwn(text, mentionable_users),
            thread_ts=thread_ts or reply_to_message_id,
        )

    async def send_link(
        self, channel_id: str, text: str, link_label: str, link_url: str
    ) -> None:
        channel, thread_ts = _decode_target(channel_id)
        await self._client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=thread_ts,
            blocks=_link_blocks(text, link_label, link_url),
        )

    async def send_ephemeral(self, channel_id: str, user_id: str, text: str) -> None:
        channel, _ = _decode_target(channel_id)
        await self._client.chat_postEphemeral(channel=channel, user=user_id, text=text)

    async def start_typing(self, channel_id: str) -> None:
        pass  # Slack bot apps don't expose a typing indicator API.

    async def stop_typing(self, channel_id: str) -> None:
        pass

    async def create_thread(
        self, channel_id: str, message_id: str, name: str
    ) -> Optional[str]:
        # Slack threads are implicit — pack (channel, parent_ts) into a single
        # target_id the handler can pass through to outbound send_* calls.
        return _encode_target(channel_id, message_id)

    async def rename_thread(self, thread_id: str, name: str) -> bool:
        # Slack threads can't be renamed.
        return False

    # -- Helpers --

    async def _user_display_name(self, user_id: str) -> str:
        if user_id in self._user_name_cache:
            return self._user_name_cache[user_id]
        try:
            resp = await self._client.users_info(user=user_id)
            user = resp.get("user") or {}
            profile = user.get("profile") or {}
            name = (
                profile.get("display_name")
                or user.get("real_name")
                or user.get("name")
                or user_id
            )
        except Exception:
            logger.warning("Failed to fetch Slack user %s", user_id, exc_info=True)
            name = user_id
        self._user_name_cache[user_id] = name
        return name

    async def _strip_mentions(self, text: str) -> str:
        """Drop the bot's own mention; rewrite others as `@displayname`."""
        bot_id = await self._bot_user_id_cached()
        user_ids = {m.group(1) for m in _USER_MENTION_RE.finditer(text)}
        names: dict[str, str] = {}
        for uid in user_ids:
            names[uid] = await self._user_display_name(uid)

        def _replace(match: re.Match[str]) -> str:
            uid = match.group(1)
            if bot_id and uid == bot_id:
                return ""
            return f"@{names.get(uid, uid)}"

        return _USER_MENTION_RE.sub(_replace, text).strip()

    async def _collect_mentionable_users(
        self, text: str
    ) -> tuple[tuple[str, str], ...]:
        bot_id = await self._bot_user_id_cached()
        pairs: list[tuple[str, str]] = []
        for match in _USER_MENTION_RE.finditer(text):
            uid = match.group(1)
            if bot_id and uid == bot_id:
                continue
            name = await self._user_display_name(uid)
            pair = (name, uid)
            if pair not in pairs:
                pairs.append(pair)
        return tuple(pairs)

    async def _bot_user_id_cached(self) -> str:
        cached = self._bot_user_id
        if cached is not None:
            return cached
        try:
            resp = await self._client.auth_test()
            user_id = resp.get("user_id") or ""
        except Exception:
            logger.warning("Failed to fetch Slack bot user_id", exc_info=True)
            user_id = ""
        self._bot_user_id = user_id
        return user_id


def _verify_signature(request: Request, body: bytes) -> bool:
    return signing.verify(
        body=body,
        timestamp=request.headers.get("X-Slack-Request-Timestamp", ""),
        signature=request.headers.get("X-Slack-Signature", ""),
    )


def _encode_target(channel_id: str, thread_ts: str) -> str:
    return f"{channel_id}|{thread_ts}"


def _decode_target(target_id: str) -> tuple[str, Optional[str]]:
    if "|" in target_id:
        channel, thread_ts = target_id.split("|", 1)
        return channel, thread_ts
    return target_id, None


def _link_blocks(text: str, link_label: str, link_url: str) -> list[dict[str, Any]]:
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": link_label[:75]},
                    "url": link_url,
                }
            ],
        },
    ]
