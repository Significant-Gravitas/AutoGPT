"""Slack adapter — webhook-based (Events API).

Mounts inbound Events API + slash-command routes on the main backend API. All
platform-agnostic logic (attachment policy, mention allowlist, chunk splitting,
history budgeting) comes from the shared layer; only Slack API calls and Slack's
event/ID grammar live here.
"""

import asyncio
import json
import logging
import re
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from slack_sdk.web.async_client import AsyncWebClient

from backend.copilot.bot.adapters.base import (
    ChannelInfo,
    ChannelType,
    FileAttachment,
    MessageCallback,
    MessageContext,
    PostedRef,
    WebhookAdapter,
)
from backend.copilot.bot.adapters.shared import InboundFile, collect_attachments
from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.config import MAX_INBOUND_ATTACHMENTS
from backend.copilot.bot.text import iter_chunks, resolve_mentions

from . import commands, config, signing
from .text import to_mrkdwn

logger = logging.getLogger(__name__)

EVENTS_PATH = "/api/copilot-webhooks/slack/events"
COMMANDS_PATH = "/api/copilot-webhooks/slack/commands"

# Matches both `<@U123>` and `<@U123|displayname>` mention forms.
_USER_MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)(?:\|[^>]+)?>")

# Slack channel/group/DM IDs — C/G/D + uppercase alphanumerics. Channel *names*
# are lowercase-with-hyphens, so they never match; used by the proactive-post
# resolver to tell an ID from a name.
_CHANNEL_ID_RE = re.compile(r"^[CGD][A-Z0-9]{7,}$")


class SlackAdapter(WebhookAdapter):
    def __init__(self, api: BotBackend):
        self._api = api
        self._client = AsyncWebClient(token=config.get_bot_token())
        self._on_message_callback: Optional[MessageCallback] = None
        self._bot_user_id: Optional[str] = None
        self._team_id: Optional[str] = None
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
        return bool(_CHANNEL_ID_RE.match(ref))

    def localize_markup(self, text: str) -> str:
        return to_mrkdwn(text)

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
            # The workspace (team) ID lives at the top level of the Events API
            # payload — it isn't reliably inside the event object — so thread it
            # through for server_id resolution.
            team_id = payload.get("team_id")
            # Fire-and-forget so we ACK within Slack's 3s window.
            task = asyncio.create_task(self._dispatch_event(event, team_id))
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

    async def _dispatch_event(
        self, event: dict[str, Any], team_id: Optional[str] = None
    ) -> None:
        if self._on_message_callback is None:
            return
        # Skip bot messages (including our own) to avoid reply loops.
        if event.get("subtype") == "bot_message" or event.get("bot_id"):
            return
        event_type = event.get("type")
        channel_type = event.get("channel_type")
        ctx: Optional[MessageContext] = None
        if event_type == "app_mention":
            ctx = await self._build_context(event, team_id, bot_mentioned=True)
        elif event_type == "message" and channel_type == "im":
            ctx = await self._build_context(
                event, team_id, bot_mentioned=True, is_dm=True
            )
        elif (
            event_type == "message"
            and channel_type == "channel"
            and event.get("thread_ts")
        ):
            # Reply in a channel thread without an @mention — the handler checks
            # thread-subscription state and ignores it if it isn't ours.
            ctx = await self._build_context(event, team_id, bot_mentioned=False)
        if ctx is None:
            return
        try:
            await self._on_message_callback(ctx, self)
        except Exception:
            logger.exception("Slack event handler failed")

    async def _build_context(
        self,
        event: dict[str, Any],
        team_id: Optional[str] = None,
        *,
        bot_mentioned: bool,
        is_dm: bool = False,
    ) -> Optional[MessageContext]:
        channel = event.get("channel")
        ts = event.get("ts")
        user = event.get("user")
        text = event.get("text") or ""
        if not channel or not ts or not user:
            return None

        thread_ts = event.get("thread_ts")
        channel_type: ChannelType
        target_channel_id: str
        if is_dm:
            channel_type = "dm"
            target_channel_id = channel
        elif thread_ts:
            channel_type = "thread"
            target_channel_id = _encode_target(channel, thread_ts)
        else:
            channel_type = "channel"
            target_channel_id = channel

        attachments, skipped = await self._extract_attachments(event)
        return MessageContext(
            platform="slack",
            channel_type=channel_type,
            # DMs bill to the user (no server); channel/thread need the workspace.
            server_id=None if is_dm else (event.get("team") or team_id),
            channel_id=target_channel_id,
            message_id=ts,
            user_id=user,
            username=await self._user_display_name(user),
            text=await self._strip_mentions(text),
            bot_mentioned=bot_mentioned,
            mentionable_users=await self._collect_mentionable_users(text),
            attachments=attachments,
            skipped_attachments=skipped,
        )

    async def _extract_attachments(self, event: dict[str, Any]) -> tuple[tuple, tuple]:
        files = event.get("files") or []
        inbound = [
            InboundFile(
                filename=f.get("name"),
                size=int(f.get("size") or 0),
                mime_type=f.get("mimetype"),
                fetch=lambda f=f: self._download_slack_file(f),
            )
            for f in files
        ]
        return await collect_attachments(
            inbound,
            max_count=MAX_INBOUND_ATTACHMENTS,
            max_bytes=self.max_attachment_bytes,
        )

    async def _download_slack_file(self, file_obj: dict[str, Any]) -> bytes:
        # Slack file URLs are private — download needs the bot token as a
        # bearer credential (files:read scope), unlike Discord's public CDN.
        url = file_obj.get("url_private_download") or file_obj.get("url_private") or ""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {config.get_bot_token()}"}
            )
            resp.raise_for_status()
            return resp.content

    # -- Outbound --

    def _render(self, text: str, mentionable_users: tuple[tuple[str, str], ...]) -> str:
        # Allowlisted @-mentions → <@Uid> (the allowlist IS Slack's ping
        # safety — non-allowlisted names stay plain and never ping), then
        # localize CommonMark → mrkdwn.
        rendered, _pinged = resolve_mentions(
            text, mentionable_users, lambda _name, uid: f"<@{uid}>"
        )
        return self.localize_markup(rendered)

    async def send_message(
        self,
        channel_id: str,
        text: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        channel, thread_ts = _decode_target(channel_id)
        await self._client.chat_postMessage(
            channel=channel,
            text=self._render(text, mentionable_users),
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
            text=self._render(text, mentionable_users),
            thread_ts=thread_ts or reply_to_message_id,
        )

    async def send_link(
        self, channel_id: str, text: str, link_label: str, link_url: str
    ) -> None:
        channel, thread_ts = _decode_target(channel_id)
        rendered = self.localize_markup(text)
        await self._client.chat_postMessage(
            channel=channel,
            text=rendered,
            thread_ts=thread_ts,
            blocks=_link_blocks(rendered, link_label, link_url),
        )

    async def send_file(self, channel_id: str, text: str, file: FileAttachment) -> None:
        channel, thread_ts = _decode_target(channel_id)
        await self._client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            content=file.content,
            filename=file.filename or "file",
            initial_comment=self.localize_markup(text) if text else None,
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
        # target_id the handler passes back to the outbound send_* calls.
        return _encode_target(channel_id, message_id)

    async def rename_thread(self, thread_id: str, name: str) -> bool:
        return False  # Slack threads have no name.

    # -- Proactive output --

    async def list_text_channels(
        self, server_ids: tuple[str, ...]
    ) -> list[ChannelInfo]:
        team_id = await self._team_id_cached()
        if not team_id or team_id not in server_ids:
            return []
        channels: list[ChannelInfo] = []
        cursor: Optional[str] = None
        while True:
            resp = await self._client.conversations_list(
                types="public_channel",
                exclude_archived=True,
                limit=200,
                cursor=cursor,
            )
            for c in resp.get("channels") or []:
                # Only channels the bot is in can be posted to without a join.
                if c.get("is_member"):
                    channels.append(
                        ChannelInfo(
                            id=c["id"], name=c.get("name", ""), server_id=team_id
                        )
                    )
            cursor = (resp.get("response_metadata") or {}).get("next_cursor")
            if not cursor:
                break
        return channels

    async def get_channel_server_id(self, channel_id: str) -> Optional[str]:
        team_id = await self._team_id_cached()
        if not team_id:
            return None
        try:
            await self._client.conversations_info(channel=channel_id)
        except Exception:
            return None
        return team_id

    async def post_channel_message(
        self, channel_id: str, text: str
    ) -> Optional[PostedRef]:
        channel, thread_ts = _decode_target(channel_id)
        first_ts = await self._post_chunked(channel, text, thread_ts=thread_ts)
        if first_ts is None:
            return None
        return PostedRef(id=first_ts, url=await self._permalink(channel, first_ts))

    async def create_channel_thread(
        self, channel_id: str, name: str, text: str
    ) -> Optional[PostedRef]:
        # Slack threads are implicit + unnamed: post text as the root message;
        # the returned ref threads subsequent sends off it.
        channel, _ = _decode_target(channel_id)
        root_ts = await self._post_chunked(channel, text)
        if root_ts is None:
            return None
        return PostedRef(
            id=_encode_target(channel, root_ts),
            url=await self._permalink(channel, root_ts),
        )

    async def _post_chunked(
        self, channel: str, text: str, thread_ts: Optional[str] = None
    ) -> Optional[str]:
        """Post ``text`` chunked under the message cap; return the first ts.

        Later chunks thread off the first so a long post stays one conversation.
        """
        first_ts = thread_ts
        for chunk in iter_chunks(self.localize_markup(text), config.CHUNK_FLUSH_AT):
            resp = await self._client.chat_postMessage(
                channel=channel, text=chunk, thread_ts=first_ts
            )
            if first_ts is None:
                first_ts = resp.get("ts")
        return first_ts

    async def _permalink(self, channel: str, ts: str) -> Optional[str]:
        try:
            resp = await self._client.chat_getPermalink(channel=channel, message_ts=ts)
            return resp.get("permalink")
        except Exception:
            return None

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
        names: dict[str, str] = {}
        for match in _USER_MENTION_RE.finditer(text):
            uid = match.group(1)
            if uid not in names:
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
            pair = (await self._user_display_name(uid), uid)
            if pair not in pairs:
                pairs.append(pair)
        return tuple(pairs)

    async def _ensure_identity(self) -> None:
        """Resolve the bot's own user_id + team_id from auth_test, once.

        Only a truthy result is cached — on a transient failure *or* a falsy
        response the fields stay ``None`` so the next event retries. Otherwise a
        single blip would permanently break self-mention stripping and channel/
        team lookups.
        """
        if self._bot_user_id and self._team_id:
            return
        try:
            resp = await self._client.auth_test()
        except Exception:
            logger.warning("Failed to fetch Slack identity (auth_test)", exc_info=True)
            return
        self._bot_user_id = self._bot_user_id or resp.get("user_id") or None
        self._team_id = self._team_id or resp.get("team_id") or None

    async def _bot_user_id_cached(self) -> str:
        await self._ensure_identity()
        return self._bot_user_id or ""

    async def _team_id_cached(self) -> str:
        await self._ensure_identity()
        return self._team_id or ""


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
