"""AppService exposing bot-facing platform_linking ops over internal RPC."""

import logging

from backend.data.db_accessors import bot_analytics_db, platform_linking_db
from backend.util.service import AppService, AppServiceClient, endpoint_to_async, expose
from backend.util.settings import Settings

from .chat import list_user_chats, start_chat_turn
from .models import (
    BotChatRequest,
    BotEventInput,
    BotGuildInput,
    ChatTurnHandle,
    CreateLinkTokenRequest,
    CreateUserLinkTokenRequest,
    LinkTokenResponse,
    LinkTokenStatusResponse,
    ListUserChatsResponse,
    Platform,
    ResolveResponse,
    WorkspaceArtifact,
)

logger = logging.getLogger(__name__)


class PlatformLinkingManager(AppService):
    @classmethod
    def get_port(cls) -> int:
        return Settings().config.platform_linking_service_port

    @expose
    async def resolve_server_link(
        self, platform: Platform, platform_server_id: str
    ) -> ResolveResponse:
        return await platform_linking_db().resolve_server_link(
            platform.value, platform_server_id
        )

    @expose
    async def resolve_user_link(
        self, platform: Platform, platform_user_id: str
    ) -> ResolveResponse:
        return await platform_linking_db().resolve_user_link(
            platform.value, platform_user_id
        )

    @expose
    async def create_server_link_token(
        self, request: CreateLinkTokenRequest
    ) -> LinkTokenResponse:
        return await platform_linking_db().create_server_link_token(request)

    @expose
    async def create_user_link_token(
        self, request: CreateUserLinkTokenRequest
    ) -> LinkTokenResponse:
        return await platform_linking_db().create_user_link_token(request)

    @expose
    async def get_link_token_status(self, token: str) -> LinkTokenStatusResponse:
        return await platform_linking_db().get_link_token_status(token)

    @expose
    async def list_user_server_ids(self, platform: Platform, user_id: str) -> list[str]:
        """Bot-scoped: the platform server IDs ``user_id`` has linked.

        Deliberately returns only the IDs (not full ``PlatformLinkInfo``) so the
        bot client stays a narrow surface — the user-facing ``list_server_links``
        stays off the bot client. Backs proactive-output authorization.
        """
        links = await platform_linking_db().list_server_links(user_id)
        return [
            link.platform_server_id for link in links if link.platform == platform.value
        ]

    @expose
    async def start_chat_turn(self, request: BotChatRequest) -> ChatTurnHandle:
        return await start_chat_turn(request)

    @expose
    async def refresh_server_link_name(
        self, platform: Platform, platform_server_id: str, server_name: str
    ) -> None:
        await platform_linking_db().refresh_server_link_name(
            platform.value, platform_server_id, server_name
        )

    @expose
    async def list_user_chats(
        self,
        platform: Platform,
        platform_user_id: str,
        limit: int = 25,
        offset: int = 0,
    ) -> ListUserChatsResponse:
        return await list_user_chats(platform, platform_user_id, limit, offset)

    @expose
    async def fetch_workspace_artifact(
        self, session_id: str, file_id: str, max_bytes: int
    ) -> WorkspaceArtifact | None:
        return await platform_linking_db().fetch_workspace_artifact(
            session_id, file_id, max_bytes
        )

    @expose
    async def record_bot_event(self, event: BotEventInput) -> None:
        await bot_analytics_db().record_bot_event(event)

    @expose
    async def record_guild_joined(self, guild: BotGuildInput) -> None:
        await bot_analytics_db().record_guild_joined(guild)

    @expose
    async def mark_guild_left(self, platform: Platform, server_id: str) -> None:
        await bot_analytics_db().mark_guild_left(platform, server_id)

    @expose
    async def sync_guild_presence(
        self, platform: Platform, guilds: list[BotGuildInput]
    ) -> None:
        await bot_analytics_db().sync_guild_presence(platform, guilds)


class PlatformLinkingManagerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return PlatformLinkingManager

    resolve_server_link = endpoint_to_async(PlatformLinkingManager.resolve_server_link)
    resolve_user_link = endpoint_to_async(PlatformLinkingManager.resolve_user_link)
    create_server_link_token = endpoint_to_async(
        PlatformLinkingManager.create_server_link_token
    )
    create_user_link_token = endpoint_to_async(
        PlatformLinkingManager.create_user_link_token
    )
    get_link_token_status = endpoint_to_async(
        PlatformLinkingManager.get_link_token_status
    )
    list_user_server_ids = endpoint_to_async(
        PlatformLinkingManager.list_user_server_ids
    )
    start_chat_turn = endpoint_to_async(PlatformLinkingManager.start_chat_turn)
    refresh_server_link_name = endpoint_to_async(
        PlatformLinkingManager.refresh_server_link_name
    )
    list_user_chats = endpoint_to_async(PlatformLinkingManager.list_user_chats)
    fetch_workspace_artifact = endpoint_to_async(
        PlatformLinkingManager.fetch_workspace_artifact
    )
    record_bot_event = endpoint_to_async(PlatformLinkingManager.record_bot_event)
    record_guild_joined = endpoint_to_async(PlatformLinkingManager.record_guild_joined)
    mark_guild_left = endpoint_to_async(PlatformLinkingManager.mark_guild_left)
    sync_guild_presence = endpoint_to_async(PlatformLinkingManager.sync_guild_presence)
