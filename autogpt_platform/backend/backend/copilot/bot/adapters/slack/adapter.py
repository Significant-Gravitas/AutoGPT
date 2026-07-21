"""Slack adapter — webhook-based (Events API), multi-workspace.

Mounts inbound Events API + slash-command + OAuth-install routes on the main
backend API. Each installed workspace has its own bot token (obtained via the
"Add to Slack" OAuth flow, stored encrypted per team); every outbound call
resolves the right token from the workspace (team) the message came from. All
platform-agnostic logic (attachment policy, mention allowlist, chunk splitting,
history budgeting) comes from the shared layer; only Slack API calls and Slack's
event/ID grammar live here.

The opaque ``channel_id`` the handler passes back into ``send_*`` therefore
carries the team: it is ``team|channel|thread_ts`` so a send can pick the
workspace's client without any extra lookup.
"""

import asyncio
import json
import logging
import re
import time
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
    read_verified_webhook_body,
    unauthorized_webhook_response,
)
from backend.copilot.bot.adapters.shared import InboundFile, collect_attachments
from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.config import MAX_INBOUND_ATTACHMENTS
from backend.copilot.bot.text import iter_chunks, resolve_mentions

# Accessor, not direct Prisma: the outbound half also runs in the Prisma-less
# copilot-bot bridge pod, which must go through the DatabaseManager.
from backend.data.db_accessors import bot_installs_db
from backend.platform_linking.models import Platform

from . import commands, config, oauth, signing
from .text import to_mrkdwn

logger = logging.getLogger(__name__)

EVENTS_PATH = "/api/copilot-webhooks/slack/events"
COMMANDS_PATH = "/api/copilot-webhooks/slack/commands"

# Slack lifecycle events that end a workspace's install — revoke its token.
_UNINSTALL_EVENTS = {"app_uninstalled", "tokens_revoked"}

# Matches both `<@U123>` and `<@U123|displayname>` mention forms.
_USER_MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)(?:\|[^>]+)?>")

# Slack channel/group/DM IDs — C/G/D + uppercase alphanumerics. Channel *names*
# are lowercase-with-hyphens, so they never match; used by the proactive-post
# resolver to tell an ID from a name.
_CHANNEL_ID_RE = re.compile(r"^[CGD][A-Z0-9]{7,}$")

# Cached per-workspace clients expire so a token replaced by a re-install (scope
# change / rotation) stops being used within this window on EVERY replica — the
# OAuth callback only evicts the replica that handled it.
_CLIENT_CACHE_TTL_SECONDS = 15 * 60


class SlackAdapter(WebhookAdapter):
    def __init__(self, api: BotBackend):
        self._api = api
        self._on_message_callback: Optional[MessageCallback] = None
        # Per-workspace caches, keyed by team_id. A workspace's client + bot user
        # id are resolved once from its stored install token and reused.
        self._clients: dict[str, AsyncWebClient] = {}
        self._client_cached_at: dict[str, float] = {}
        self._bot_user_ids: dict[str, str] = {}
        # User display names are workspace-scoped: keyed by (team_id, user_id).
        self._user_name_cache: dict[tuple[str, str], str] = {}
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
        # Multi-workspace "Add to Slack" install + OAuth callback (no-op unless
        # the app's client id/secret are configured). A (re)install replaces the
        # workspace's token, so it must drop this replica's cached client.
        oauth.register_routes(app, on_installed=self._evict)

    # -- Per-workspace client resolution --

    async def _client_for(self, team_id: str) -> Optional[AsyncWebClient]:
        """Return a cached Slack client for ``team_id``, building it from the
        workspace's stored install token (or the static back-compat token).

        An empty ``team_id`` (e.g. a raw channel ref in a proactive post that
        carries no workspace) can't be looked up, so it falls straight through to
        the static single-workspace token. Returns ``None`` only when there is no
        usable token at all — the caller then can't respond, which is the correct
        outcome for an uninstalled workspace."""
        client = self._clients.get(team_id)
        if client is not None:
            cached_at = self._client_cached_at.get(team_id)
            if (
                cached_at is None
                or time.monotonic() - cached_at < _CLIENT_CACHE_TTL_SECONDS
            ):
                return client
            self._evict(team_id)
        install = (
            await bot_installs_db().get_bot_install(Platform.SLACK, team_id)
            if team_id
            else None
        )
        if install is None:
            # A workspace that explicitly uninstalled / revoked us gets NOTHING —
            # the static token belongs to the app's own workspace and must never
            # be used on a revoked one's behalf. Fallback is only for
            # never-installed refs (single-workspace mode, raw proactive refs).
            if team_id and await bot_installs_db().is_install_revoked(
                Platform.SLACK, team_id
            ):
                return None
            token = config.get_bot_token()
        else:
            token = install.bot_token
        if not token:
            return None
        if install and install.bot_user_id:
            self._bot_user_ids[team_id] = install.bot_user_id
        client = AsyncWebClient(token=token)
        self._clients[team_id] = client
        self._client_cached_at[team_id] = time.monotonic()
        return client

    def _evict(self, team_id: str) -> None:
        self._clients.pop(team_id, None)
        self._client_cached_at.pop(team_id, None)
        self._bot_user_ids.pop(team_id, None)

    # -- Inbound --

    async def _handle_event_request(self, request: Request) -> Response:
        raw = await read_verified_webhook_body(request, _verify_signature)
        if raw is None:
            return unauthorized_webhook_response()
        payload = json.loads(raw)
        if payload.get("type") == "url_verification":
            return JSONResponse({"challenge": payload.get("challenge", "")})
        if payload.get("type") == "event_callback":
            event = payload.get("event") or {}
            # The workspace (team) ID lives at the top level of the Events API
            # payload — it isn't reliably inside the event object — so thread it
            # through for token + server_id resolution.
            team_id = payload.get("team_id")
            # Fire-and-forget so we ACK within Slack's 3s window.
            task = asyncio.create_task(self._dispatch_event(event, team_id))
            self._event_tasks.add(task)
            task.add_done_callback(self._event_tasks.discard)
        return PlainTextResponse("ok")

    async def _handle_command_request(self, request: Request) -> Response:
        if await read_verified_webhook_body(request, _verify_signature) is None:
            return unauthorized_webhook_response()
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
        # Workspace removed the app / revoked its token → drop its stored install
        # and cached client so we stop trying to use a dead token.
        if event.get("type") in _UNINSTALL_EVENTS:
            if team_id:
                self._evict(team_id)
                await bot_installs_db().revoke_bot_install(Platform.SLACK, team_id)
            return
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
        # team is required to resolve the workspace's token for the reply.
        team = event.get("team") or team_id
        if not channel or not ts or not user or not team:
            return None

        thread_ts = event.get("thread_ts")
        channel_type: ChannelType
        if is_dm:
            channel_type = "dm"
        elif thread_ts:
            channel_type = "thread"
        else:
            channel_type = "channel"
        target_channel_id = _encode_target(team, channel, thread_ts)

        attachments, skipped = await self._extract_attachments(team, event)
        return MessageContext(
            platform="slack",
            channel_type=channel_type,
            # DMs bill to the user (no server); channel/thread need the workspace.
            server_id=None if is_dm else team,
            channel_id=target_channel_id,
            message_id=ts,
            user_id=user,
            username=await self._user_display_name(team, user),
            text=await self._strip_mentions(team, text),
            bot_mentioned=bot_mentioned,
            mentionable_users=await self._collect_mentionable_users(team, text),
            attachments=attachments,
            skipped_attachments=skipped,
        )

    async def _extract_attachments(
        self, team_id: str, event: dict[str, Any]
    ) -> tuple[tuple, tuple]:
        files = event.get("files") or []
        inbound = [
            InboundFile(
                filename=f.get("name"),
                size=int(f.get("size") or 0),
                mime_type=f.get("mimetype"),
                fetch=lambda f=f: self._download_slack_file(team_id, f),
            )
            for f in files
        ]
        return await collect_attachments(
            inbound,
            max_count=MAX_INBOUND_ATTACHMENTS,
            max_bytes=self.max_attachment_bytes,
        )

    async def _download_slack_file(
        self, team_id: str, file_obj: dict[str, Any]
    ) -> bytes:
        # Slack file URLs are private — download needs the workspace's bot token
        # as a bearer credential (files:read scope), unlike Discord's public CDN.
        url = file_obj.get("url_private_download") or file_obj.get("url_private") or ""
        token = await self._token_for(team_id)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
            resp.raise_for_status()
            return resp.content

    async def _token_for(self, team_id: str) -> str:
        client = await self._client_for(team_id)
        return client.token or "" if client else ""

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
        team, channel, thread_ts = _decode_target(channel_id)
        client = await self._client_for(team)
        if client is None:
            return
        await client.chat_postMessage(
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
        team, channel, thread_ts = _decode_target(channel_id)
        client = await self._client_for(team)
        if client is None:
            return
        await client.chat_postMessage(
            channel=channel,
            text=self._render(text, mentionable_users),
            thread_ts=thread_ts or reply_to_message_id,
        )

    async def send_link(
        self, channel_id: str, text: str, link_label: str, link_url: str
    ) -> None:
        team, channel, thread_ts = _decode_target(channel_id)
        client = await self._client_for(team)
        if client is None:
            return
        rendered = self.localize_markup(text)
        await client.chat_postMessage(
            channel=channel,
            text=rendered,
            thread_ts=thread_ts,
            blocks=_link_blocks(rendered, link_label, link_url),
        )

    async def send_file(self, channel_id: str, text: str, file: FileAttachment) -> None:
        team, channel, thread_ts = _decode_target(channel_id)
        client = await self._client_for(team)
        if client is None:
            return
        await client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            content=file.content,
            filename=file.filename or "file",
            initial_comment=self.localize_markup(text) if text else None,
        )

    async def send_ephemeral(self, channel_id: str, user_id: str, text: str) -> None:
        team, channel, _ = _decode_target(channel_id)
        client = await self._client_for(team)
        if client is None:
            return
        await client.chat_postEphemeral(channel=channel, user=user_id, text=text)

    async def create_thread(
        self, channel_id: str, message_id: str, name: str
    ) -> Optional[str]:
        # Slack threads are implicit — re-pack (team, channel, parent_ts) into a
        # single target_id the handler passes back to the outbound send_* calls.
        team, channel, _ = _decode_target(channel_id)
        return _encode_target(team, channel, message_id)

    # -- Proactive output --

    async def list_text_channels(
        self, server_ids: tuple[str, ...]
    ) -> list[ChannelInfo]:
        channels: list[ChannelInfo] = []
        for team_id in server_ids:
            client = await self._client_for(team_id)
            if client is None:
                continue
            cursor: Optional[str] = None
            while True:
                resp = await client.conversations_list(
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
                                id=_encode_target(team_id, c["id"]),
                                name=c.get("name", ""),
                                server_id=team_id,
                            )
                        )
                cursor = (resp.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
        return channels

    async def get_channel_server_id(self, channel_id: str) -> Optional[str]:
        team, channel, _ = _decode_target(channel_id)
        client = await self._client_for(team)
        if client is None:
            return None
        try:
            await client.conversations_info(channel=channel)
        except Exception:
            return None
        if team:
            return team
        # Static single-workspace fallback: the ref carried no team, so resolve
        # the bot's own workspace from auth_test.
        try:
            resp = await client.auth_test()
            return resp.get("team_id") or None
        except Exception:
            return None

    async def post_channel_message(
        self, channel_id: str, text: str
    ) -> Optional[PostedRef]:
        team, channel, thread_ts = _decode_target(channel_id)
        first_ts = await self._post_chunked(team, channel, text, thread_ts=thread_ts)
        if first_ts is None:
            return None
        return PostedRef(
            id=first_ts, url=await self._permalink(team, channel, first_ts)
        )

    async def create_channel_thread(
        self, channel_id: str, name: str, text: str
    ) -> Optional[PostedRef]:
        # Slack threads are implicit + unnamed: post text as the root message;
        # the returned ref threads subsequent sends off it.
        team, channel, _ = _decode_target(channel_id)
        root_ts = await self._post_chunked(team, channel, text)
        if root_ts is None:
            return None
        return PostedRef(
            id=_encode_target(team, channel, root_ts),
            url=await self._permalink(team, channel, root_ts),
        )

    async def _post_chunked(
        self, team_id: str, channel: str, text: str, thread_ts: Optional[str] = None
    ) -> Optional[str]:
        """Post ``text`` chunked under the message cap; return the first ts.

        Later chunks thread off the first so a long post stays one conversation.
        """
        client = await self._client_for(team_id)
        if client is None:
            return None
        first_ts = thread_ts
        for chunk in iter_chunks(self.localize_markup(text), config.CHUNK_FLUSH_AT):
            resp = await client.chat_postMessage(
                channel=channel, text=chunk, thread_ts=first_ts
            )
            if first_ts is None:
                first_ts = resp.get("ts")
        return first_ts

    async def _permalink(self, team_id: str, channel: str, ts: str) -> Optional[str]:
        client = await self._client_for(team_id)
        if client is None:
            return None
        try:
            resp = await client.chat_getPermalink(channel=channel, message_ts=ts)
            return resp.get("permalink")
        except Exception:
            return None

    # -- Helpers --

    async def _user_display_name(self, team_id: str, user_id: str) -> str:
        cache_key = (team_id, user_id)
        if cache_key in self._user_name_cache:
            return self._user_name_cache[cache_key]
        client = await self._client_for(team_id)
        if client is not None:
            try:
                resp = await client.users_info(user=user_id)
                user = resp.get("user") or {}
                profile = user.get("profile") or {}
                name = (
                    profile.get("display_name")
                    or user.get("real_name")
                    or user.get("name")
                    or user_id
                )
                # Only cache a real resolution — a transient API failure must not
                # poison the cache with the raw id (mirrors _bot_user_id_for).
                self._user_name_cache[cache_key] = name
                return name
            except Exception:
                logger.warning("Failed to fetch Slack user %s", user_id, exc_info=True)
        return user_id

    async def _strip_mentions(self, team_id: str, text: str) -> str:
        """Drop the bot's own mention; rewrite others as `@displayname`."""
        bot_id = await self._bot_user_id_for(team_id)
        names: dict[str, str] = {}
        for match in _USER_MENTION_RE.finditer(text):
            uid = match.group(1)
            if uid not in names:
                names[uid] = await self._user_display_name(team_id, uid)

        def _replace(match: re.Match[str]) -> str:
            uid = match.group(1)
            if bot_id and uid == bot_id:
                return ""
            return f"@{names.get(uid, uid)}"

        return _USER_MENTION_RE.sub(_replace, text).strip()

    async def _collect_mentionable_users(
        self, team_id: str, text: str
    ) -> tuple[tuple[str, str], ...]:
        bot_id = await self._bot_user_id_for(team_id)
        pairs: list[tuple[str, str]] = []
        for match in _USER_MENTION_RE.finditer(text):
            uid = match.group(1)
            if bot_id and uid == bot_id:
                continue
            pair = (await self._user_display_name(team_id, uid), uid)
            if pair not in pairs:
                pairs.append(pair)
        return tuple(pairs)

    async def _bot_user_id_for(self, team_id: str) -> str:
        """The bot's own user id in ``team_id`` — needed to strip its self-mention.

        Prefer the id captured at install time; fall back to auth_test on the
        workspace's client. Only a truthy result is cached, so a transient blip
        doesn't permanently break self-mention stripping."""
        if team_id in self._bot_user_ids:
            return self._bot_user_ids[team_id]
        client = await self._client_for(team_id)
        if team_id in self._bot_user_ids:  # populated by _client_for from install
            return self._bot_user_ids[team_id]
        if client is None:
            return ""
        try:
            resp = await client.auth_test()
        except Exception:
            logger.warning("Failed to fetch Slack identity (auth_test)", exc_info=True)
            return ""
        uid = resp.get("user_id") or ""
        if uid:
            self._bot_user_ids[team_id] = uid
        return uid


def _verify_signature(request: Request, body: bytes) -> bool:
    return signing.verify(
        body=body,
        timestamp=request.headers.get("X-Slack-Request-Timestamp", ""),
        signature=request.headers.get("X-Slack-Signature", ""),
    )


def _encode_target(
    team_id: str, channel_id: str, thread_ts: Optional[str] = None
) -> str:
    return f"{team_id}|{channel_id}|{thread_ts or ''}"


def _decode_target(target_id: str) -> tuple[str, str, Optional[str]]:
    parts = target_id.split("|")
    if len(parts) == 3:
        team, channel, thread_ts = parts
        return team, channel, (thread_ts or None)
    if len(parts) == 2:
        # Back-compat with the single-workspace 'channel|thread_ts' form.
        channel, thread_ts = parts
        return "", channel, (thread_ts or None)
    return "", target_id, None


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
