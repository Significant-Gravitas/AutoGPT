"""Tests for the Menlo copilot tools (control + runtime).

All Menlo SDK access is behind patched manager seams, so these run without the
optional ``menlo`` extra installed.
"""

from datetime import UTC, datetime
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pydantic import BaseModel

from backend.copilot.model import ChatSession
from backend.copilot.tools.menlo import control, runtime
from backend.copilot.tools.menlo.manager import LiveConnection, MenloNotConnectedError
from backend.copilot.tools.menlo.responses import (
    MenloRobotConnectedResponse,
    MenloRobotStateResponse,
    MenloSkillResultResponse,
    MenloSkillsDiscoveredResponse,
    MenloVisionResponse,
)
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workspace_files import WorkspaceWriteResponse


def _session(session_id: str = "sess1") -> ChatSession:
    return ChatSession(
        session_id=session_id,
        user_id="test-user",
        title=None,
        messages=[],
        usage=[],
        credentials={},
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _fake_runtime_session() -> MagicMock:
    session = MagicMock()
    session.discover_skills = AsyncMock()
    session.invoke = AsyncMock()
    session.get_vision = AsyncMock()
    session.state = MagicMock()
    session.state.get = AsyncMock()
    return session


def _conn(session: MagicMock) -> LiveConnection:
    return LiveConnection("rb_1", MagicMock(), session)


class TestConnectTool:
    @pytest.mark.asyncio
    async def test_returns_viewer_url(self):
        conn = LiveConnection("rb_42", MagicMock(), MagicMock())
        with patch.object(
            control, "connect_new_robot", AsyncMock(return_value=conn)
        ), patch.object(
            control, "generate_viewer_url", AsyncMock(return_value="https://v/?key=k")
        ):
            result = await control.MenloConnectRobotTool()._execute(
                "test-user", _session(), model="asimov-v0"
            )
        assert isinstance(result, MenloRobotConnectedResponse)
        assert result.robot_id == "rb_42"
        assert result.viewer_url == "https://v/?key=k"


class TestDisconnectTool:
    @pytest.mark.asyncio
    async def test_reports_deleted_robot(self):
        with patch.object(control, "disconnect_robot", AsyncMock(return_value="rb_7")):
            result = await control.MenloDisconnectRobotTool()._execute(
                "test-user", _session()
            )
        assert result.robot_id == "rb_7"
        assert "rb_7" in result.message


class TestDiscoverSkillsTool:
    @pytest.mark.asyncio
    async def test_maps_skill_descriptors(self):
        skill = MagicMock()
        skill.name = "go_to"
        skill.description = "Navigate the robot."
        skill.input_schema = {"type": "object"}
        skill.annotations = None
        session = _fake_runtime_session()
        session.discover_skills.return_value = [skill]
        with patch.object(
            runtime, "resolve_connection", AsyncMock(return_value=_conn(session))
        ):
            result = await runtime.MenloDiscoverSkillsTool()._execute(
                "test-user", _session()
            )
        assert isinstance(result, MenloSkillsDiscoveredResponse)
        assert result.count == 1
        assert result.skills[0].name == "go_to"

    @pytest.mark.asyncio
    async def test_not_connected_returns_error(self):
        with patch.object(
            runtime,
            "resolve_connection",
            AsyncMock(side_effect=MenloNotConnectedError()),
        ):
            result = await runtime.MenloDiscoverSkillsTool()._execute(
                "test-user", _session()
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "menlo_not_connected"

    @pytest.mark.asyncio
    async def test_runtime_error_gives_viewer_hint(self):
        with patch.object(
            runtime,
            "resolve_connection",
            AsyncMock(side_effect=RuntimeError("no rcw answering")),
        ):
            result = await runtime.MenloDiscoverSkillsTool()._execute(
                "test-user", _session()
            )
        assert isinstance(result, ErrorResponse)
        assert "viewer" in result.message.lower()


class _Err(BaseModel):
    code: str
    message: str


class TestInvokeSkillTool:
    @pytest.mark.asyncio
    async def test_done_status(self):
        result_obj = MagicMock(status="done", error=None, result=None)
        result_obj.meta = MagicMock(action_id="act_1")
        session = _fake_runtime_session()
        session.invoke.return_value = result_obj
        with patch.object(
            runtime, "resolve_connection", AsyncMock(return_value=_conn(session))
        ):
            out = await runtime.MenloInvokeSkillTool()._execute(
                "test-user", _session(), skill="go_to", parameters={"target": {}}
            )
        assert isinstance(out, MenloSkillResultResponse)
        assert out.status == "done"
        assert out.action_id == "act_1"
        session.invoke.assert_awaited_once_with("go_to", {"target": {}}, timeout_s=120)

    @pytest.mark.asyncio
    async def test_failed_status_surfaces_error_message(self):
        result_obj = MagicMock(
            status="failed", error=_Err(code="NAV_STUCK", message="stuck"), result=None
        )
        result_obj.meta = MagicMock(action_id="act_2")
        session = _fake_runtime_session()
        session.invoke.return_value = result_obj
        with patch.object(
            runtime, "resolve_connection", AsyncMock(return_value=_conn(session))
        ):
            out = await runtime.MenloInvokeSkillTool()._execute(
                "test-user", _session(), skill="go_to"
            )
        assert out.status == "failed"
        assert out.error == "stuck"

    @pytest.mark.asyncio
    async def test_missing_skill_returns_error(self):
        out = await runtime.MenloInvokeSkillTool()._execute(
            "test-user", _session(), skill=""
        )
        assert isinstance(out, ErrorResponse)
        assert out.error == "missing_skill"


class TestGetRobotStateTool:
    @pytest.mark.asyncio
    async def test_serializes_pydantic_state(self):
        class _State(BaseModel):
            status: str
            yaw: float

        session = _fake_runtime_session()
        session.state.get.return_value = _State(status="ready", yaw=1.5)
        with patch.object(
            runtime, "resolve_connection", AsyncMock(return_value=_conn(session))
        ):
            out = await runtime.MenloGetRobotStateTool()._execute(
                "test-user", _session(), key="robot_status"
            )
        assert isinstance(out, MenloRobotStateResponse)
        assert out.state == {"status": "ready", "yaw": 1.5}


class TestGetVisionTool:
    @pytest.mark.asyncio
    async def test_saves_frame_to_workspace(self):
        buf = BytesIO()
        Image.new("RGB", (8, 6)).save(buf, format="JPEG")
        jpeg = buf.getvalue()

        session = _fake_runtime_session()
        session.get_vision.return_value = jpeg
        write_resp = WorkspaceWriteResponse(
            message="saved",
            file_id="file_1",
            name="menlo.jpg",
            path="menlo.jpg",
            mime_type="image/jpeg",
            size_bytes=len(jpeg),
            download_url="workspace://file_1",
        )
        with patch.object(
            runtime, "resolve_connection", AsyncMock(return_value=_conn(session))
        ), patch(
            "backend.copilot.tools.workspace_files.WriteWorkspaceFileTool._execute",
            new=AsyncMock(return_value=write_resp),
        ):
            out = await runtime.MenloGetVisionTool()._execute(
                "test-user", _session(), camera="pov"
            )
        assert isinstance(out, MenloVisionResponse)
        assert out.file_id == "file_1"
        assert (out.width, out.height) == (8, 6)
        assert out.camera == "pov"
