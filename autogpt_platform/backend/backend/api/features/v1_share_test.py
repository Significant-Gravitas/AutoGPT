"""Tests for the public shared file download endpoint."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import Response

from backend.api.features.v1 import v1_router
from backend.data.workspace import WorkspaceFile

app = FastAPI()
app.include_router(v1_router, prefix="/api")

VALID_TOKEN = "550e8400-e29b-41d4-a716-446655440000"
VALID_FILE_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


def _make_workspace_file(**overrides) -> WorkspaceFile:
    defaults = {
        "id": VALID_FILE_ID,
        "workspace_id": "ws-001",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "name": "image.png",
        "path": "/image.png",
        "storage_path": "local://uploads/image.png",
        "mime_type": "image/png",
        "size_bytes": 4,
        "checksum": None,
        "is_deleted": False,
        "deleted_at": None,
        "metadata": {},
    }
    defaults.update(overrides)
    return WorkspaceFile(**defaults)


def _mock_download_response(**kwargs):
    """Return an AsyncMock that resolves to a Response with inline disposition."""

    async def _handler(file, *, inline=False):
        return Response(
            content=b"\x89PNG",
            media_type="image/png",
            headers={
                "Content-Disposition": (
                    'inline; filename="image.png"'
                    if inline
                    else 'attachment; filename="image.png"'
                ),
                "Content-Length": "4",
            },
        )

    return _handler


class TestDownloadSharedFile:
    """Tests for GET /api/public/shared/{token}/files/{id}/download."""

    @pytest.fixture(autouse=True)
    def _client(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_valid_token_and_file_returns_inline_content(self):
        with (
            patch(
                "backend.api.features.v1.execution_db.get_shared_execution_file",
                new_callable=AsyncMock,
                return_value="exec-123",
            ),
            patch(
                "backend.api.features.v1.get_workspace_file_by_id",
                new_callable=AsyncMock,
                return_value=_make_workspace_file(),
            ),
            patch(
                "backend.api.features.v1.create_file_download_response",
                side_effect=_mock_download_response(),
            ),
        ):
            response = self.client.get(
                f"/api/public/shared/{VALID_TOKEN}/files/{VALID_FILE_ID}/download"
            )

        assert response.status_code == 200
        assert response.content == b"\x89PNG"
        assert "inline" in response.headers["Content-Disposition"]

    def test_invalid_token_format_returns_422(self):
        response = self.client.get(
            f"/api/public/shared/not-a-uuid/files/{VALID_FILE_ID}/download"
        )
        assert response.status_code == 422

    def test_token_not_in_allowlist_returns_404(self):
        with patch(
            "backend.api.features.v1.execution_db.get_shared_execution_file",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = self.client.get(
                f"/api/public/shared/{VALID_TOKEN}/files/{VALID_FILE_ID}/download"
            )
        assert response.status_code == 404

    def test_file_missing_from_workspace_returns_404(self):
        with (
            patch(
                "backend.api.features.v1.execution_db.get_shared_execution_file",
                new_callable=AsyncMock,
                return_value="exec-123",
            ),
            patch(
                "backend.api.features.v1.get_workspace_file_by_id",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            response = self.client.get(
                f"/api/public/shared/{VALID_TOKEN}/files/{VALID_FILE_ID}/download"
            )
        assert response.status_code == 404

    def test_uniform_404_prevents_enumeration(self):
        """Both failure modes produce identical 404 — no information leak."""
        with patch(
            "backend.api.features.v1.execution_db.get_shared_execution_file",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp_no_allow = self.client.get(
                f"/api/public/shared/{VALID_TOKEN}/files/{VALID_FILE_ID}/download"
            )

        with (
            patch(
                "backend.api.features.v1.execution_db.get_shared_execution_file",
                new_callable=AsyncMock,
                return_value="exec-123",
            ),
            patch(
                "backend.api.features.v1.get_workspace_file_by_id",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            resp_no_file = self.client.get(
                f"/api/public/shared/{VALID_TOKEN}/files/{VALID_FILE_ID}/download"
            )

        assert resp_no_allow.status_code == 404
        assert resp_no_file.status_code == 404
        assert resp_no_allow.json() == resp_no_file.json()
