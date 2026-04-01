"""Tests for platform bot linking API routes."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from backend.api.features.platform_linking.auth import check_bot_api_key
from backend.api.features.platform_linking.models import (
    ConfirmLinkResponse,
    CreateLinkTokenRequest,
    DeleteLinkResponse,
    LinkTokenStatusResponse,
    Platform,
    ResolveRequest,
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
            platform_user_id="353922987235213313",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_user_id == "353922987235213313"

    def test_empty_platform_user_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_user_id="",
            )

    def test_too_long_platform_user_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_user_id="x" * 256,
            )

    def test_invalid_platform_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateLinkTokenRequest.model_validate(
                {"platform": "INVALID", "platform_user_id": "123"}
            )


class TestResolveRequest:
    def test_valid_request(self):
        req = ResolveRequest(
            platform=Platform.TELEGRAM,
            platform_user_id="123456789",
        )
        assert req.platform == Platform.TELEGRAM

    def test_empty_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResolveRequest(
                platform=Platform.SLACK,
                platform_user_id="",
            )


class TestResponseModels:
    def test_link_token_status_literal(self):
        resp = LinkTokenStatusResponse(status="pending")
        assert resp.status == "pending"

        resp = LinkTokenStatusResponse(status="linked", user_id="abc")
        assert resp.status == "linked"

        resp = LinkTokenStatusResponse(status="expired")
        assert resp.status == "expired"

    def test_confirm_link_response(self):
        resp = ConfirmLinkResponse(
            success=True,
            platform="DISCORD",
            platform_user_id="123",
            platform_username="testuser",
        )
        assert resp.success is True

    def test_delete_link_response(self):
        resp = DeleteLinkResponse(success=True)
        assert resp.success is True
