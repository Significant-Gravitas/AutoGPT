from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.conductor.get_workspace import ConductorGetWorkspaceBlock
from backend.blocks.conductor.test_fixtures import TEST_CREDENTIALS_INPUT, collect


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preview_fails,sessions_fail", [(True, False), (False, True), (True, True)]
)
async def test_initializing_workspace_keeps_details_when_optional_reads_fail(
    preview_fails: bool, sessions_fail: bool
):
    workspace = {"id": "ws1", "state": "initializing"}
    client = Mock(
        get_workspace=AsyncMock(return_value=workspace),
        workspace_status=AsyncMock(return_value={"status": "initializing"}),
        get_preview=AsyncMock(
            return_value={"preview": {"url": "https://preview.example"}},
            side_effect=ValueError("not ready") if preview_fails else None,
        ),
        workspace_sessions=AsyncMock(
            return_value={"data": [{"id": "s1"}], "hasMore": False},
            side_effect=ValueError("not ready") if sessions_fail else None,
        ),
    )
    with patch(
        "backend.blocks.conductor.get_workspace.ConductorClient", return_value=client
    ):
        result = await collect(
            ConductorGetWorkspaceBlock(),
            {"credentials": TEST_CREDENTIALS_INPUT, "workspace_id": "ws1"},
        )

    assert result["workspace"] == workspace
    assert result["status"] == "initializing"
    assert result["preview_url"] == ("" if preview_fails else "https://preview.example")
    assert result["sessions"] == ([] if sessions_fail else [{"id": "s1"}])
