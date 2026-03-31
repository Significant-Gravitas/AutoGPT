"""Tests for platform bot linking API routes."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from backend.api.features.platform_linking.routes import Platform, _check_bot_api_key


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
    @patch("backend.api.features.platform_linking.routes.BOT_API_KEY", "")
    def test_no_key_configured_allows_in_dev(self):
        # No key configured = dev mode, should not raise
        _check_bot_api_key(None)

    @patch("backend.api.features.platform_linking.routes.BOT_API_KEY", "secret123")
    def test_valid_key(self):
        _check_bot_api_key("secret123")

    @patch("backend.api.features.platform_linking.routes.BOT_API_KEY", "secret123")
    def test_invalid_key_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _check_bot_api_key("wrong")
        assert exc_info.value.status_code == 401

    @patch("backend.api.features.platform_linking.routes.BOT_API_KEY", "secret123")
    def test_missing_key_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _check_bot_api_key(None)
        assert exc_info.value.status_code == 401


class TestCreateLinkTokenRequest:
    """Test request validation."""

    from backend.api.features.platform_linking.routes import CreateLinkTokenRequest

    def test_valid_request(self):
        req = self.CreateLinkTokenRequest(
            platform=Platform.DISCORD,
            platform_user_id="353922987235213313",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_user_id == "353922987235213313"

    def test_empty_platform_user_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self.CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_user_id="",
            )

    def test_too_long_platform_user_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self.CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_user_id="x" * 256,
            )

    def test_invalid_platform_rejected(self):
        from pydantic import ValidationError

        invalid_platform = "INVALID"
        with pytest.raises(ValidationError):
            self.CreateLinkTokenRequest.model_validate(
                {"platform": invalid_platform, "platform_user_id": "123"}
            )


class TestResolveRequest:
    from backend.api.features.platform_linking.routes import ResolveRequest

    def test_valid_request(self):
        req = self.ResolveRequest(
            platform=Platform.TELEGRAM,
            platform_user_id="123456789",
        )
        assert req.platform == Platform.TELEGRAM

    def test_empty_id_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self.ResolveRequest(
                platform=Platform.SLACK,
                platform_user_id="",
            )


class TestResponseModels:
    """Test that response models are properly typed."""

    from backend.api.features.platform_linking.routes import (
        ConfirmLinkResponse,
        DeleteLinkResponse,
        LinkTokenStatusResponse,
    )

    def test_link_token_status_literal(self):
        resp = self.LinkTokenStatusResponse(status="pending")
        assert resp.status == "pending"

        resp = self.LinkTokenStatusResponse(status="linked", user_id="abc")
        assert resp.status == "linked"

        resp = self.LinkTokenStatusResponse(status="expired")
        assert resp.status == "expired"

    def test_confirm_link_response(self):
        resp = self.ConfirmLinkResponse(
            success=True,
            platform="DISCORD",
            platform_user_id="123",
            platform_username="testuser",
        )
        assert resp.success is True

    def test_delete_link_response(self):
        resp = self.DeleteLinkResponse(success=True)
        assert resp.success is True
