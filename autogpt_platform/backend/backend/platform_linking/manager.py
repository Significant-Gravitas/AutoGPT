"""AppService exposing bot-facing platform_linking ops over internal RPC."""

import logging

from backend.data.db_accessors import platform_linking_db
from backend.util.service import AppService, AppServiceClient, endpoint_to_async, expose
from backend.util.settings import Settings

from .chat import start_chat_turn
from .models import (
    BotChatRequest,
    ChatTurnHandle,
    CreateLinkTokenRequest,
    CreateUserLinkTokenRequest,
    LinkTokenResponse,
    LinkTokenStatusResponse,
    Platform,
    ResolveResponse,
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
    async def start_chat_turn(self, request: BotChatRequest) -> ChatTurnHandle:
        return await start_chat_turn(request)


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
    start_chat_turn = endpoint_to_async(PlatformLinkingManager.start_chat_turn)
