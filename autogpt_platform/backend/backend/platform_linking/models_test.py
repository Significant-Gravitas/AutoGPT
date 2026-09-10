"""Schema validation tests for platform_linking Pydantic models."""

import pytest
from pydantic import ValidationError

from backend.util.json import to_dict

from .models import (
    BotChatRequest,
    ConfirmLinkResponse,
    CreateLinkTokenRequest,
    DeleteLinkResponse,
    LinkTokenStatusResponse,
    Platform,
    ResolveResponse,
    ResolveServerRequest,
    WorkspaceArtifact,
    WorkspaceUploadRequest,
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
        with pytest.raises(ValidationError):
            CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_server_id="",
                platform_user_id="123",
            )

    def test_too_long_server_id_rejected(self):
        with pytest.raises(ValidationError):
            CreateLinkTokenRequest(
                platform=Platform.DISCORD,
                platform_server_id="x" * 256,
                platform_user_id="123",
            )

    def test_invalid_platform_rejected(self):
        with pytest.raises(ValidationError):
            CreateLinkTokenRequest.model_validate(
                {
                    "platform": "INVALID",
                    "platform_server_id": "123",
                    "platform_user_id": "456",
                }
            )


class TestResolveServerRequest:
    def test_valid_request(self):
        req = ResolveServerRequest(
            platform=Platform.DISCORD,
            platform_server_id="1126875755960336515",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_server_id == "1126875755960336515"

    def test_empty_server_id_rejected(self):
        with pytest.raises(ValidationError):
            ResolveServerRequest(
                platform=Platform.SLACK,
                platform_server_id="",
            )


class TestBotChatRequest:
    def test_server_context(self):
        req = BotChatRequest(
            platform=Platform.DISCORD,
            platform_server_id="1126875755960336515",
            platform_user_id="353922987235213313",
            message="Hello CoPilot!",
        )
        assert req.platform == Platform.DISCORD
        assert req.platform_server_id == "1126875755960336515"
        assert req.session_id is None

    def test_dm_context_omits_server_id(self):
        req = BotChatRequest(
            platform=Platform.DISCORD,
            platform_user_id="353922987235213313",
            message="Hello in DMs!",
        )
        assert req.platform_server_id is None

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
        with pytest.raises(ValidationError):
            BotChatRequest(
                platform=Platform.DISCORD,
                platform_server_id="guild_123",
                platform_user_id="user_456",
                message="",
            )

    def test_empty_string_server_id_rejected(self):
        with pytest.raises(ValidationError):
            BotChatRequest(
                platform=Platform.DISCORD,
                platform_server_id="",
                platform_user_id="user_456",
                message="hi",
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


class TestBinaryContentOverJsonRpc:
    """Binary file bytes must survive the JSON service RPC.

    Plain ``bytes`` fields are strict-UTF-8-decoded by JSON serialization,
    so a binary artifact (e.g. a JPEG starting 0xff) fails to serialize on
    the ``fetch_workspace_artifact`` / ``upload_workspace_files`` hops.
    """

    # First bytes of a JPEG — not valid UTF-8.
    _BINARY = b"\xff\xd8\xff\xe0" + bytes(range(256))

    def _artifact(self) -> WorkspaceArtifact:
        return WorkspaceArtifact(
            file_id="f1",
            filename="photo.jpg",
            mime_type="image/jpeg",
            size_bytes=len(self._BINARY),
            content=self._BINARY,
        )

    def test_artifact_binary_content_survives_the_wire(self):
        # Server response leg: FastAPI serializes via jsonable_encoder …
        wire = to_dict(self._artifact())
        assert isinstance(wire["content"], str)  # base64, JSON-safe
        # … client leg: TypeAdapter validates the JSON back into the model.
        restored = WorkspaceArtifact.model_validate(wire)
        assert restored.content == self._BINARY

    def test_upload_request_binary_content_survives_the_wire(self):
        request = WorkspaceUploadRequest(
            platform=Platform.TELEGRAM,
            platform_user_id="u1",
            filename="photo.jpg",
            mime_type="image/jpeg",
            content=self._BINARY,
        )
        wire = to_dict(request)
        assert isinstance(wire["content"], str)
        restored = WorkspaceUploadRequest.model_validate(wire)
        assert restored.content == self._BINARY

    def test_raw_bytes_stay_bytes_in_process(self):
        # In-process consumers (adapters attach .content directly) must keep
        # seeing raw bytes — only the JSON wire format is base64.
        artifact = self._artifact()
        assert artifact.content == self._BINARY
        assert artifact.model_dump()["content"] == self._BINARY
