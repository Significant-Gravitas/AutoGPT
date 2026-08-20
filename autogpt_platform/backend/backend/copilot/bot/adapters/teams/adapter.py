"""Microsoft Teams adapter — webhook-based, via the Azure Bot Connector.

Teams never talks to us directly: every inbound message arrives as a Bot
Connector "activity" POSTed to one endpoint, and every reply is a POST back to
the ``serviceUrl`` that activity carried. Authentication is therefore a bearer
JWT rather than a body HMAC (see ``auth.py``), which is why this adapter
verifies inbound requests itself instead of using the shared
``read_verified_webhook_body`` helper — that helper's verifier is synchronous
and Teams key validation needs async I/O. The ordering guarantee it exists to
provide (verify before parse) is preserved here explicitly.

Chat model mapping: a ``personal`` conversation is a DM (auto-converse); a
``channel`` conversation engages only when the bot is @mentioned, and replies
land in a Teams reply-chain rooted at that message. Group chats are not
handled in v1 — Teams gives them no team identity, which the core handler
requires for any non-DM turn.
"""

import asyncio
import base64
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

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
from backend.data.redis_client import get_redis_async

from . import auth, commands, config
from .api_client import TeamsApiError, TeamsClient
from .text import mention_entities, mention_token, to_teams_markdown

logger = logging.getLogger(__name__)

MESSAGES_PATH = "/api/copilot-webhooks/teams/messages"

# Conversations we keep a learned serviceUrl for. Evicting one is cheap:
# the next reply falls back to the default host until it is relearned.
_MAX_REMEMBERED_SERVICE_URLS = 10_000

# Inbound rejections answer with fixed text: everything specific about why is
# logged instead, so an unauthenticated caller learns nothing from probing.
_REJECTED = "rejected"
_MALFORMED = "invalid activity payload"

# Teams routes every tenant in the public cloud through this host. Real traffic
# always carries its own ``serviceUrl``; this is only the seed for proactive
# sends that have no inbound activity to learn it from.
DEFAULT_SERVICE_URL = "https://smba.trafficmanager.net/teams/"

# Conversation ids are Teams' own opaque identifiers: channels/group threads are
# ``19:…@thread.tacv2`` (optionally with a ``;messageid=`` reply-chain suffix),
# 1:1 chats are ``a:…``. Channel *names* never take these shapes, which is what
# ``looks_like_channel_id`` relies on.
_CONVERSATION_ID_PREFIXES = ("19:", "a:")

# Attachment content types Teams uses for user-uploaded files.
_FILE_DOWNLOAD_CONTENT_TYPE = "application/vnd.microsoft.teams.file.download.info"


class TeamsAdapter(WebhookAdapter):
    def __init__(self, api: BotBackend):
        self._api = api
        self._client = TeamsClient()
        self._validator = auth.ConnectorTokenValidator()
        self._on_message_callback: Optional[MessageCallback] = None
        # serviceUrl is per-tenant and can change, so it is deliberately NOT
        # baked into the conversation token the core handler round-trips (that
        # token is also a Redis key suffix — folding a mutable value into it
        # would silently orphan live sessions). Learned from inbound traffic
        # instead, with the public-cloud host as the fallback. Bounded: a
        # long-lived process would otherwise hold one entry per conversation
        # forever, and evicting only costs the default host on the next reply.
        self._service_urls: OrderedDict[str, str] = OrderedDict()
        # Strong-ref set so the GC doesn't drop fire-and-forget activity tasks.
        self._activity_tasks: set[asyncio.Task[None]] = set()

    @property
    def platform_name(self) -> str:
        return "teams"

    @property
    def max_message_length(self) -> int:
        return config.MAX_MESSAGE_LENGTH

    @property
    def chunk_flush_at(self) -> int:
        return config.CHUNK_FLUSH_AT

    @property
    def max_attachment_bytes(self) -> int:
        # Outbound cap: only a data-URI inline fits in one activity, so this
        # is the inline ceiling — larger artifacts take the link fallback.
        return config.INLINE_IMAGE_BYTES

    @property
    def max_thread_name_length(self) -> int:
        return config.MAX_THREAD_NAME_LENGTH

    @property
    def typing_refresh_interval(self) -> float:
        return config.TYPING_REFRESH_SECONDS

    def looks_like_channel_id(self, ref: str) -> bool:
        return ref.startswith(_CONVERSATION_ID_PREFIXES)

    def localize_markup(self, text: str) -> str:
        return to_teams_markdown(text)

    def on_message(self, callback: MessageCallback) -> None:
        self._on_message_callback = callback

    def register_routes(self, app: FastAPI) -> None:
        app.add_api_route(
            MESSAGES_PATH, self._handle_messages_request, methods=["POST"]
        )

    # -- Inbound --

    async def _handle_messages_request(self, request: Request) -> Response:
        """Authenticate, then ACK immediately and process off-request.

        The Connector retries deliveries it considers slow, so the real work
        must never happen inside the request; redeliveries that arrive anyway
        are dropped by the activity-id dedupe in ``_dispatch_activity``.
        """
        raw = await request.body()
        claims: dict[str, Any] = {}
        unverified = config.allow_unverified_requests()
        if unverified:
            logger.warning(
                "Accepting an unauthenticated Teams activity — Agents "
                "Playground mode (local dev only)"
            )
        else:
            try:
                claims = await self._validator.validate(
                    request.headers.get("Authorization")
                )
            except auth.TeamsAuthError as e:
                # The reason stays in the log. The caller is unauthenticated
                # here, and the validator's message can carry text from the JWT
                # parser — the status code is all a legitimate sender needs.
                logger.warning(f"Rejected Teams activity: {e}")
                return PlainTextResponse(_REJECTED, status_code=e.status_code)

        try:
            activity = _parse_activity(raw)
        except ValueError:
            logger.warning("Rejected malformed Teams activity", exc_info=True)
            return PlainTextResponse(_MALFORMED, status_code=400)

        # The token authenticates the sender but does NOT sign the body, so it
        # must be bound to this payload before the serviceUrl inside it is
        # trusted as a reply destination.
        if not unverified:
            try:
                auth.verify_service_url(claims, activity.get("serviceUrl", ""))
            except auth.TeamsAuthError as e:
                logger.warning(f"Rejected Teams activity: {e}")
                return PlainTextResponse(_REJECTED, status_code=e.status_code)

        self._remember_service_url(activity)
        task = asyncio.create_task(self._dispatch_activity(activity))
        self._activity_tasks.add(task)
        task.add_done_callback(self._on_activity_task_done)
        return JSONResponse({}, status_code=200)

    async def _dispatch_activity(self, activity: dict[str, Any]) -> None:
        if await self._is_duplicate_activity(activity):
            return
        activity_type = activity.get("type")
        if activity_type in ("installationUpdate", "conversationUpdate"):
            self._track_team_membership(activity)
            return
        if activity_type != "message":
            return  # reactions, invokes — not conversation input.
        # Configured id only — on an activity we sent, ``recipient`` is the
        # other party and must not count as "us".
        if _is_own_id((activity.get("from") or {}).get("id"), _configured_bot_ids()):
            return  # Our own echo.

        command = commands.parse_command(_activity_text(activity))
        if command is not None:
            try:
                await commands.handle(self._api, self, activity, command)
            except Exception:
                logger.exception("Teams command handler failed")
            return

        if self._on_message_callback is None:
            return
        ctx = await self._build_context(activity)
        if ctx is None:
            return
        try:
            await self._on_message_callback(ctx, self)
        except Exception:
            logger.exception("Teams activity handler failed")

    async def _is_duplicate_activity(self, activity: dict[str, Any]) -> bool:
        """Drop redeliveries — first delivery of an activity id wins.

        The Connector retries deliveries it considers slow, and the inbound
        token does not sign the body, so a captured request can also be
        replayed wholesale for the token's lifetime. Both arrive as an
        already-seen activity id. Fails open: losing Redis briefly must not
        take the bot down with it.
        """
        activity_id = activity.get("id")
        if not activity_id:
            return False
        conversation_id = (activity.get("conversation") or {}).get("id") or ""
        key = f"copilot:bot:teams:activity:{conversation_id}:{activity_id}"
        try:
            redis = await get_redis_async()
            claimed = await redis.set(
                key, "1", nx=True, ex=config.ACTIVITY_DEDUPE_TTL_SECONDS
            )
        except Exception:
            logger.warning("Teams activity dedupe unavailable; processing anyway")
            return False
        if claimed:
            return False
        logger.info(f"Ignoring duplicate Teams activity {activity_id!r}")
        return True

    def _track_team_membership(self, activity: dict[str, Any]) -> None:
        """Keep the server roster in step with installs and removals.

        Without this the roster has no row for the team, so the admin
        analytics fall back to showing its raw id instead of its name.
        """
        team = (activity.get("channelData") or {}).get("team") or {}
        team_id = team.get("id")
        if not team_id:
            return  # Personal install — no team to put on the roster.
        own = _bot_identities(activity)
        action = (activity.get("action") or "").lower()
        if action.startswith("remove") or _lists_bot(
            activity.get("membersRemoved"), own
        ):
            self._api.track_guild_left(self.platform_name, team_id)
        elif action.startswith("add") or _lists_bot(activity.get("membersAdded"), own):
            self._api.track_guild_joined(self.platform_name, team_id, team.get("name"))

    @property
    def client(self) -> TeamsClient:
        """The Connector client, for handlers that need their own reads."""
        return self._client

    def _on_activity_task_done(self, task: "asyncio.Task[None]") -> None:
        """Dispatch runs after the 200 ACK, so a failure has nowhere to surface.

        Without retrieving the exception here it is only ever reported as an
        unretrieved-task warning at collection time, detached from the activity
        that caused it.
        """
        self._activity_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("Teams activity dispatch failed", exc_info=error)

    def _remember_service_url(self, activity: dict[str, Any]) -> None:
        conversation_id = (activity.get("conversation") or {}).get("id")
        service_url = activity.get("serviceUrl")
        if conversation_id and service_url:
            # Cache the dialable form; the token binding was checked above
            # against the URL as it arrived. Also key by the channel base id,
            # so ids minted by ``create_thread`` (base + ";messageid=") and
            # sibling reply chains resolve to the learned regional host rather
            # than the geo-routed default.
            url = auth.rewrite_loopback_for_container(service_url)
            self._remember(conversation_id, url)
            base = _base_conversation_id(conversation_id)
            if base != conversation_id:
                self._remember(base, url)

    def _remember(self, conversation_id: str, url: str) -> None:
        self._service_urls[conversation_id] = url
        self._service_urls.move_to_end(conversation_id)
        while len(self._service_urls) > _MAX_REMEMBERED_SERVICE_URLS:
            self._service_urls.popitem(last=False)

    def _service_url_for(self, conversation_id: str) -> str:
        url = self._service_urls.get(conversation_id)
        if url:
            return url
        base = _base_conversation_id(conversation_id)
        return self._service_urls.get(base, DEFAULT_SERVICE_URL)

    # -- Outbound --

    async def send_message(
        self,
        channel_id: str,
        text: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        await self._send_chunked(channel_id, text, mentionable_users)

    async def send_reply(
        self,
        channel_id: str,
        text: str,
        reply_to_message_id: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        # Teams reply-chains are addressed by conversation id, not per-activity,
        # so a reply into an existing chain is an ordinary send.
        await self._send_chunked(channel_id, text, mentionable_users)

    async def send_ephemeral(self, channel_id: str, user_id: str, text: str) -> None:
        # Teams has no ephemeral message; deliver normally (Discord/Telegram
        # precedent) rather than silently dropping it.
        await self._send_chunked(channel_id, text, ())

    async def send_link(
        self, channel_id: str, text: str, link_label: str, link_url: str
    ) -> None:
        activity = {
            "type": "message",
            "text": self.localize_markup(text),
            "textFormat": "markdown",
            "attachments": [_link_card(link_label, link_url)],
        }
        await self._post(channel_id, activity)

    async def send_file(self, channel_id: str, text: str, file: FileAttachment) -> None:
        """Inline a small image; degrade anything else to a note.

        Teams' bot file API is a three-legged consent handshake (card -> user
        click -> upload session) that cannot complete inside a single awaited
        send, and it does not exist at all in channels. Data-URI inlining is
        the only single-shot delivery, and the whole activity is capped at
        ~28KB — ``max_attachment_bytes`` advertises that cap so callers route
        larger files to their link fallback instead of through here.
        """
        if (
            file.mime_type.startswith("image/")
            and len(file.content) <= config.INLINE_IMAGE_BYTES
        ):
            activity = {
                "type": "message",
                "text": self.localize_markup(text) if text else "",
                "textFormat": "markdown",
                "attachments": [
                    {
                        "contentType": file.mime_type,
                        "contentUrl": _data_uri(file),
                        "name": file.filename,
                    }
                ],
            }
            try:
                await self._post(channel_id, activity)
                return
            except TeamsApiError:
                logger.warning(f"Teams rejected inline image {file.filename!r}")
        note = f"{text}\n\n_{file.filename} can't be attached in Teams._".strip()
        await self._send_chunked(channel_id, note, ())

    async def start_typing(self, channel_id: str) -> None:
        try:
            await self._post(channel_id, {"type": "typing"})
        except (TeamsApiError, httpx.HTTPError):
            # Runs in a keep-alive loop; never let it break the turn.
            logger.debug("Teams typing indicator failed", exc_info=True)

    async def create_thread(
        self, channel_id: str, message_id: str, name: str
    ) -> Optional[str]:
        """Root a reply-chain at ``message_id``.

        Teams threads are unnamed, so ``name`` is unused. Only a channel
        conversation can carry a chain, and one that already has a
        ``;messageid=`` suffix IS a chain — re-wrapping it would address a
        conversation that does not exist.
        """
        if not channel_id.startswith("19:") or ";messageid=" in channel_id:
            return None
        return f"{channel_id};messageid={message_id}"

    # -- Proactive output --

    async def list_text_channels(
        self, server_ids: tuple[str, ...]
    ) -> list[ChannelInfo]:
        # Enumerating channels needs a Graph permission the bot does not hold;
        # proactive posts target a conversation id directly.
        return []

    async def get_channel_server_id(self, channel_id: str) -> Optional[str]:
        # Without channel enumeration there is no channel -> team mapping to
        # authorize against, so raw-id posting stays unauthorized by design.
        return None

    async def post_channel_message(
        self, channel_id: str, text: str
    ) -> Optional[PostedRef]:
        first_id = await self._send_chunked(channel_id, text, ())
        return PostedRef(id=first_id) if first_id else None

    async def create_channel_thread(
        self, channel_id: str, name: str, text: str
    ) -> Optional[PostedRef]:
        # Teams threads carry no title; surface it as a heading instead, the
        # same degradation Telegram uses for its unnamed topics.
        body = f"**{name}**\n\n{text}" if name else text
        return await self.post_channel_message(channel_id, body)

    async def open_dm_channel(self, platform_user_id: str) -> Optional[str]:
        """Create (or fetch) the bot's 1:1 conversation with a user.

        Teams returns the existing conversation for a repeat call, so this is
        effectively idempotent. Only reachable for users who have the app
        installed personally.
        """
        payload = {
            # Participant form: Teams spells bot ids "28:<app-id>" on the wire.
            "bot": {"id": f"28:{config.get_app_id()}"},
            "members": [{"id": platform_user_id}],
            "channelData": {"tenant": {"id": config.get_tenant_id()}},
            "isGroup": False,
        }
        try:
            return await self._client.create_conversation(DEFAULT_SERVICE_URL, payload)
        except TeamsApiError:
            logger.warning(f"Cannot open Teams DM with user {platform_user_id}")
            return None

    # -- Send plumbing --

    async def _send_chunked(
        self,
        channel_id: str,
        text: str,
        mentionable_users: tuple[tuple[str, str], ...],
    ) -> Optional[str]:
        """Post ``text`` in message-sized chunks; return the first activity id.

        Chunks are awaited in sequence: Teams does not guarantee ordering for
        messages posted in quick succession.
        """
        first_id: Optional[str] = None
        for chunk in iter_chunks(self.localize_markup(text), config.CHUNK_FLUSH_AT):
            rendered, pinged = resolve_mentions(chunk, mentionable_users, mention_token)
            activity: dict[str, Any] = {
                "type": "message",
                "text": rendered,
                "textFormat": "markdown",
            }
            entities = mention_entities(pinged, mentionable_users)
            if entities:
                activity["entities"] = entities
            activity_id = await self._post(channel_id, activity)
            first_id = first_id or activity_id
        return first_id

    async def _post(self, channel_id: str, activity: dict[str, Any]) -> Optional[str]:
        return await self._client.send_activity(
            self._service_url_for(channel_id), channel_id, activity
        )

    async def _build_context(
        self, activity: dict[str, Any]
    ) -> Optional[MessageContext]:
        """Map a Teams message activity onto the platform-neutral context."""
        conversation = activity.get("conversation") or {}
        conversation_id = conversation.get("id")
        sender = activity.get("from") or {}
        if not conversation_id or not sender.get("id"):
            return None

        conversation_type = conversation.get("conversationType")
        if conversation_type == "personal":
            channel_type: ChannelType = "dm"
            server_id = None
        elif conversation_type == "channel":
            channel_type = _classify_channel_message(
                conversation_id, activity.get("id")
            )
            server_id = ((activity.get("channelData") or {}).get("team") or {}).get(
                "id"
            )
            if not server_id:
                logger.info(
                    f"Ignoring Teams channel activity in {conversation_id!r}: "
                    f"no team id in channelData"
                )
                return None
        else:
            # groupChat has no team identity for the core handler to bill
            # against, and a non-DM turn without one is dropped downstream.
            logger.info(
                f"Ignoring Teams {conversation_type!r} activity in "
                f"{conversation_id!r}: only personal chats and channels are "
                f"supported"
            )
            return None

        attachments, skipped = await collect_attachments(
            _inbound_files(activity, self._client),
            max_count=MAX_INBOUND_ATTACHMENTS,
            max_bytes=config.MAX_ATTACHMENT_BYTES,
        )

        return MessageContext(
            platform="teams",
            channel_type=channel_type,
            server_id=server_id,
            channel_id=conversation_id,
            message_id=activity.get("id", ""),
            user_id=sender["id"],
            username=sender.get("name") or "",
            text=_activity_text(activity),
            bot_mentioned=_mentions_bot(activity),
            attachments=attachments,
            skipped_attachments=skipped,
            mentionable_users=_mentionable_users(activity),
        )


def _base_conversation_id(conversation_id: str) -> str:
    """``19:room@thread.tacv2;messageid=17`` -> ``19:room@thread.tacv2``."""
    return conversation_id.split(";messageid=")[0]


def _classify_channel_message(
    conversation_id: str, activity_id: str | None
) -> ChannelType:
    """Top-level channel post vs reply inside an existing chain.

    Real Teams suffixes every channel conversation id with the chain root's
    ``;messageid=`` — including a brand-new top-level post, whose root is the
    post itself. Suffix-presence alone would classify all production traffic
    as thread replies, so the root is compared to the activity's own id.
    Simulators (the Agents Playground) omit the suffix entirely.
    """
    _, sep, root = conversation_id.partition(";messageid=")
    if not sep:
        return "channel"
    return "channel" if root == (activity_id or "") else "thread"


def _activity_text(activity: dict[str, Any]) -> str:
    """The message text with the bot's own @mention removed."""
    text = activity.get("text") or ""
    own = _bot_identities(activity)
    for entity in activity.get("entities") or []:
        if entity.get("type") != "mention":
            continue
        if not _is_own_id((entity.get("mentioned") or {}).get("id"), own):
            continue
        text = text.replace(entity.get("text") or "", "")
    return text.strip()


def _mentions_bot(activity: dict[str, Any]) -> bool:
    own = _bot_identities(activity)
    return any(
        entity.get("type") == "mention"
        and _is_own_id((entity.get("mentioned") or {}).get("id"), own)
        for entity in activity.get("entities") or []
    )


def _mentionable_users(
    activity: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Users the bot may ping back — those @mentioned in this message.

    Teams offers no cheap roster read, so the allowlist is exactly who the
    author already addressed, which is the conservative reading of the shared
    mention-safety contract.
    """
    users: list[tuple[str, str]] = []
    own = _bot_identities(activity)
    for entity in activity.get("entities") or []:
        if entity.get("type") != "mention":
            continue
        mentioned = entity.get("mentioned") or {}
        user_id, name = mentioned.get("id"), mentioned.get("name")
        if user_id and name and not _is_own_id(user_id, own):
            users.append((name, user_id))
    return tuple(users)


def _inbound_files(activity: dict[str, Any], client: TeamsClient) -> list[InboundFile]:
    """Teams file attachments, normalized for ``collect_attachments``.

    Two shapes arrive: file *uploads* (personal chats only) carry a pre-signed
    ``downloadUrl``; *pasted* images carry an ``image/*`` contentType and a
    ``contentUrl`` that needs the bot's own Connector bearer to fetch. The
    full list is passed through — ``collect_attachments`` owns the count cap
    and its user-facing skip notes.
    """
    files: list[InboundFile] = []
    for attachment in activity.get("attachments") or []:
        content_type = attachment.get("contentType") or ""
        if content_type == _FILE_DOWNLOAD_CONTENT_TYPE:
            content = attachment.get("content") or {}
            download_url = content.get("downloadUrl")
            if not download_url:
                continue
            if not auth.is_fetchable_attachment_url(download_url):
                # Unsigned body: without this the URL is a server-side request
                # generator aimed at whatever the sender names.
                logger.warning("Skipping Teams attachment with an unfetchable URL")
                continue
            files.append(
                InboundFile(
                    filename=attachment.get("name"),
                    # Advisory only — the fetch below is what actually
                    # bounds the download.
                    size=_declared_size(content.get("fileSize")),
                    mime_type=content.get("fileType"),
                    fetch=_bounded_fetch(download_url),
                )
            )
        elif content_type.startswith("image/"):
            content_url = attachment.get("contentUrl")
            if not content_url:
                continue
            # This fetch carries the bot's Connector bearer, so it answers to
            # the same allowlist as every other token-bearing call rather than
            # the looser attachment check — the body naming it is unsigned.
            if not auth.is_allowed_service_url(content_url):
                logger.warning(
                    "Refusing to send the Connector bearer to a non-Connector "
                    "attachment host"
                )
                continue
            files.append(
                InboundFile(
                    filename=attachment.get("name") or "pasted-image",
                    size=0,
                    mime_type=content_type,
                    fetch=_bounded_fetch(content_url, headers=client.bearer_headers),
                )
            )
    return files


def _declared_size(value: Any) -> int:
    """The sender's claimed size, which the unsigned body may spell anything.

    Only advisory: :func:`_bounded_fetch` enforces the real cap. Raising here
    would abort the turn, and the dedupe claim stops a redelivery recovering it.
    """
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _bounded_fetch(
    download_url: str,
    headers: Callable[[], Awaitable[dict[str, str]]] | None = None,
) -> Callable[[], Awaitable[bytes]]:
    """Download at most ``MAX_ATTACHMENT_BYTES``.

    The shared size gate cannot protect us here — Teams declares no size on a
    file attachment — so the bound is enforced while streaming instead.
    ``headers`` supplies auth for URLs that need the bot's bearer (pasted
    images); pre-signed upload URLs must be fetched bare.
    """

    async def fetch() -> bytes:
        # Checked here rather than at parse time: this is as close to the
        # connect as we can get without a custom transport, and the name may
        # not have resolved the same way when the activity arrived.
        await auth.ensure_attachment_host_is_external(download_url)
        request_headers = await headers() if headers else {}
        # follow_redirects stays off (httpx's default, pinned here because it
        # is load-bearing): a redirect would land us at an unchecked host.
        async with (
            httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client,
            client.stream("GET", download_url, headers=request_headers) as response,
        ):
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > config.MAX_ATTACHMENT_BYTES:
                    raise ValueError("attachment exceeds the size limit")
                chunks.append(chunk)
        return b"".join(chunks)

    return fetch


def _link_card(link_label: str, link_url: str) -> dict[str, Any]:
    """A minimal Adaptive Card carrying one link button.

    Pinned to schema 1.2 — the highest version every Teams client, including
    mobile, renders reliably.
    """
    return {
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.2",
            "body": [],
            "actions": [
                {"type": "Action.OpenUrl", "title": link_label[:60], "url": link_url}
            ],
        },
    }


def _data_uri(file: FileAttachment) -> str:
    encoded = base64.b64encode(file.content).decode("ascii")
    return f"data:{file.mime_type};base64,{encoded}"


def _parse_activity(raw: bytes) -> dict[str, Any]:
    try:
        activity = json.loads(raw)
    except ValueError:
        raise ValueError("activity body is not valid JSON")
    if not isinstance(activity, dict):
        raise ValueError("activity body is not an object")
    return activity


def _bot_identities(activity: dict[str, Any]) -> set[str]:
    """Every spelling of the bot's own id a mention entity may carry.

    Teams sometimes prefixes a participant id with its type (``28:`` for a
    bot) and sometimes doesn't, so match both. Getting this wrong is silent:
    a missed @mention means the bot never answers in a channel.
    """
    ids = _configured_bot_ids()
    recipient_id = (activity.get("recipient") or {}).get("id")
    if recipient_id:
        ids.add(_strip_participant_prefix(recipient_id))
    return ids


def _configured_bot_ids() -> set[str]:
    """The bot's id from configuration alone, ignoring the activity."""
    app_id = config.get_app_id()
    return {_strip_participant_prefix(app_id)} if app_id else set()


def _is_own_id(candidate: str | None, own: set[str]) -> bool:
    return bool(candidate) and _strip_participant_prefix(str(candidate)) in own


def _lists_bot(members: list[dict[str, Any]] | None, own: set[str]) -> bool:
    """Whether a membersAdded/membersRemoved list names the bot itself."""
    return any(_is_own_id((m or {}).get("id"), own) for m in members or [])


def _strip_participant_prefix(value: str) -> str:
    """``28:<app-id>`` -> ``<app-id>``; anything unprefixed is returned as-is."""
    prefix, sep, tail = value.partition(":")
    return (tail if sep and prefix.isdigit() else value).lower()
