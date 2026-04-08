"""Tests for platform bot linking API routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.api.features.platform_linking.auth import check_bot_api_key
from backend.api.features.platform_linking.models import (
    BotChatRequest,
    ConfirmLinkResponse,
    CreateLinkTokenRequest,
    DeleteLinkResponse,
    LinkTokenStatusResponse,
    Platform,
    ResolveRequest,
    ResolveResponse,
)


class TestPlatformEnum:
    def test_all_platforms_exist(self):
        assert Platform.DISCORD.value == "DISCORD"
        assert Platform.TELEGRAM.value == "TELEGRAM"
        assert Platform.SLACK.value == "SLACK"
        assert Platform.TEAMS.value == "TEAMS"
        assert Platform.WHATSAPP.value == "WHATSAPP"
        assert Platform.GITHUB.value == "GITHUB"
        assert Platform.LINEAR.value == "LINEAR"


class TestBotApiKeyAuth:
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": ""}, clear=False)
    @patch("backend.api.features.platform_linking.auth.Settings")
    def test_no_key_configured_allows_when_auth_disabled(self, mock_settings_cls):
        mock_settings_cls.return_value.config.enable_auth = False
        check_bot_api_key(None)

    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": ""}, clear=False)
    @patch("backend.api.features.platform_linking.auth.Settings")
    def test_no_key_configured_rejects_when_auth_enabled(self, mock_settings_cls):
        mock_settings_cls.return_value.config.enable_auth = True
        with pytest.raises(HTTPException) as exc_info:
            check_bot_api_key(None)
        assert exc_info.value.status_code == 503

    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "secret123"}, clear=False)
    def test_valid_key(self):
        check_bot_api_key("secret123")

    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "secret123"}, clear=False)
    def test_invalid_key_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            check_bot_api_key("wrong")
        assert exc_info.value.status_code == 401

    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "secret123"}, clear=False)
    def test_missing_key_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            check_bot_api_key(None)
        assert exc_info.value.status_code == 401


class TestCreateLinkTokenRequest:
    def test_valid_request(self):
        req = CreateLinkTokenRequest(
            platform=Platform.DISCORD,
            platform_server_id="1126875755960336515",
            platform_user_id="353922987235213313",
            platform_username="Bently",
            server_name="My Discord Server",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_server_id == "1126875755960336515"
        assert req.platform_user_id == "353922987235213313"
        assert req.server_name == "My Discord Server"

    def test_minimal_request(self):
        req = CreateLinkTokenRequest(
            platform=Platform.TELEGRAM,
            platform_server_id="-100123456789",
            platform_user_id="987654321",
        )
        assert req.server_name is None
        assert req.platform_username is None

    def test_empty_server_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_server_id="",
                platform_user_id="123",
            )

    def test_too_long_server_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_server_id="x" * 256,
                platform_user_id="123",
            )

    def test_invalid_platform_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest.model_validate(
                {
                    "platform": "INVALID",
                    "platform_server_id": "123",
                    "platform_user_id": "456",
                }
            )


class TestResolveRequest:
    def test_valid_request(self):
        req = ResolveRequest(
            platform=Platform.DISCORD,
            platform_server_id="1126875755960336515",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_server_id == "1126875755960336515"

    def test_empty_server_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResolveRequest(
                platform=Platform.SLACK,
                platform_server_id="",
            )


class TestBotChatRequest:
    def test_valid_request(self):
        req = BotChatRequest(
            platform=Platform.DISCORD,
            platform_server_id="1126875755960336515",
            platform_user_id="353922987235213313",
            message="Hello CoPilot!",
        )
        assert req.platform == Platform.DISCORD
        assert req.session_id is None

    def test_with_session_id(self):
        req = BotChatRequest(
            platform=Platform.DISCORD,
            platform_server_id="guild_123",
            platform_user_id="user_456",
            message="follow up",
            session_id="session-uuid-here",
        )
        assert req.session_id == "session-uuid-here"

    def test_empty_message_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BotChatRequest(
                platform=Platform.DISCORD,
                platform_server_id="guild_123",
                platform_user_id="user_456",
                message="",
            )


class TestResponseModels:
    def test_link_token_status_pending(self):
        resp = LinkTokenStatusResponse(status="pending")
        assert resp.status == "pending"

    def test_link_token_status_linked(self):
        resp = LinkTokenStatusResponse(status="linked")
        assert resp.status == "linked"

    def test_link_token_status_expired(self):
        resp = LinkTokenStatusResponse(status="expired")
        assert resp.status == "expired"

    def test_resolve_linked(self):
        resp = ResolveResponse(linked=True)
        assert resp.linked is True

    def test_resolve_not_linked(self):
        resp = ResolveResponse(linked=False)
        assert resp.linked is False

    def test_confirm_link_response(self):
        resp = ConfirmLinkResponse(
            success=True,
            platform="DISCORD",
            platform_server_id="1126875755960336515",
            server_name="My Server",
        )
        assert resp.success is True
        assert resp.server_name == "My Server"

    def test_delete_link_response(self):
        resp = DeleteLinkResponse(success=True)
        assert resp.success is True


class TestResolveEndpoint:
    """Endpoint-level tests using mocked Prisma."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_resolve_linked_server(self):
        from backend.api.features.platform_linking.routes import (
            resolve_platform_server,
        )

        mock_link = MagicMock()
        mock_link.userId = "autogpt-user-123"

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLink"
        ) as mock_model:
            mock_model.prisma.return_value.find_first = AsyncMock(
                return_value=mock_link
            )
            result = await resolve_platform_server(
                ResolveRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="guild_123",
                ),
                x_bot_api_key="testkey",
            )

        assert result.linked is True

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_resolve_unlinked_server(self):
        from backend.api.features.platform_linking.routes import (
            resolve_platform_server,
        )

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLink"
        ) as mock_model:
            mock_model.prisma.return_value.find_first = AsyncMock(return_value=None)
            result = await resolve_platform_server(
                ResolveRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="guild_unknown",
                ),
                x_bot_api_key="testkey",
            )

        assert result.linked is False

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_resolve_rejects_wrong_api_key(self):
        from backend.api.features.platform_linking.routes import (
            resolve_platform_server,
        )

        with pytest.raises(HTTPException) as exc_info:
            await resolve_platform_server(
                ResolveRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="guild_123",
                ),
                x_bot_api_key="wrong_key",
            )
        assert exc_info.value.status_code == 401


class TestCreateLinkTokenEndpoint:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_create_token_for_unlinked_server(self):
        from backend.api.features.platform_linking.routes import create_link_token

        with (
            patch(
                "backend.api.features.platform_linking.routes.PlatformLink"
            ) as mock_link_model,
            patch(
                "backend.api.features.platform_linking.routes.PlatformLinkToken"
            ) as mock_token_model,
        ):
            mock_link_model.prisma.return_value.find_first = AsyncMock(
                return_value=None
            )
            mock_token_model.prisma.return_value.update_many = AsyncMock(
                return_value=0
            )
            mock_token_model.prisma.return_value.create = AsyncMock(
                return_value=MagicMock()
            )

            result = await create_link_token(
                CreateLinkTokenRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="guild_123",
                    platform_user_id="user_456",
                    server_name="Test Server",
                ),
                x_bot_api_key="testkey",
            )

        assert result.token
        assert "guild_123" not in result.token  # token is random
        assert result.link_url.endswith(result.token)

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"PLATFORM_BOT_API_KEY": "testkey"}, clear=False)
    async def test_create_token_409_if_already_linked(self):
        from backend.api.features.platform_linking.routes import create_link_token

        with patch(
            "backend.api.features.platform_linking.routes.PlatformLink"
        ) as mock_link_model:
            mock_link_model.prisma.return_value.find_first = AsyncMock(
                return_value=MagicMock()  # server is already linked
            )

            with pytest.raises(HTTPException) as exc_info:
                await create_link_token(
                    CreateLinkTokenRequest(
                        platform=Platform.DISCORD,
                        platform_server_id="guild_already_linked",
                        platform_user_id="user_456",
                    ),
                    x_bot_api_key="testkey",
                )

        assert exc_info.value.status_code == 409
