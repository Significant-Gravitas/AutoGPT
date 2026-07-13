"""Runtime-plane copilot tools: discover skills, drive the robot, read state, see.

These call the Menlo runtime worker (rcw) over the LiveKit SFU. They only work
once a runtime worker is in the room — for the SimpleSim model that means the
user has opened the viewer URL from ``menlo_connect_robot`` in Chrome. A missing
rcw surfaces as a ``RuntimeError`` / ``TimeoutError`` from the SDK, which we turn
into an actionable message telling the copilot to have the user open the viewer.
"""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from typing import Any

from PIL import Image
from pydantic_core import to_jsonable_python

from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ErrorResponse, ToolResponseBase

from .manager import MenloNotConnectedError, menlo_available, resolve_connection
from .responses import (
    MenloRobotStateResponse,
    MenloSkillInfo,
    MenloSkillResultResponse,
    MenloSkillsDiscoveredResponse,
    MenloVisionResponse,
)

logger = logging.getLogger(__name__)

_NOT_CONNECTED_HINT = "No robot is connected. Call menlo_connect_robot first."
_NO_RUNTIME_HINT = (
    "The robot runtime is not responding yet. Make sure the viewer URL from "
    "menlo_connect_robot is open in a Chrome tab and the 3D scene has loaded, "
    "then retry."
)


class _MenloRuntimeTool(BaseTool):
    """Shared availability + error handling for runtime-plane tools."""

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return menlo_available()

    def _not_connected(self, session_id: str | None) -> ErrorResponse:
        return ErrorResponse(
            message=_NOT_CONNECTED_HINT,
            error="menlo_not_connected",
            session_id=session_id,
        )

    def _no_runtime(self, session_id: str | None, err: Exception) -> ErrorResponse:
        return ErrorResponse(
            message=_NO_RUNTIME_HINT, error=str(err), session_id=session_id
        )


class MenloDiscoverSkillsTool(_MenloRuntimeTool):
    """List the robot's live runtime skills."""

    @property
    def name(self) -> str:
        return "menlo_discover_skills"

    @property
    def description(self) -> str:
        return (
            "List the connected robot's available skills (go_to, set_velocity, "
            "pick_entity, ...). Requires the viewer tab to be open. Returns each "
            "skill's name, description, and input schema for menlo_invoke_skill."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        session_id = session.session_id
        try:
            conn = await resolve_connection(session_id)
            skills = await conn.session.discover_skills()
        except MenloNotConnectedError:
            return self._not_connected(session_id)
        except (RuntimeError, TimeoutError) as e:
            return self._no_runtime(session_id, e)

        infos = [
            MenloSkillInfo(
                name=s.name,
                description=s.description,
                input_schema=s.input_schema or {},
                tags=list((to_jsonable_python(s.annotations) or {}).get("tags") or []),
            )
            for s in skills
        ]
        return MenloSkillsDiscoveredResponse(
            message=f"Discovered {len(infos)} skill(s).",
            count=len(infos),
            skills=infos,
            session_id=session_id,
        )


class MenloInvokeSkillTool(_MenloRuntimeTool):
    """Invoke a robot skill and wait for its terminal result."""

    @property
    def name(self) -> str:
        return "menlo_invoke_skill"

    @property
    def description(self) -> str:
        return (
            "Drive the robot by invoking one skill and waiting for it to finish. "
            "skill = a name from menlo_discover_skills (e.g. 'go_to', 'set_velocity', "
            "'pick_entity'); parameters = that skill's input object. Example: "
            'go_to {"target": {"kind": "entity", "entity_id": "pad_B"}}. '
            "Returns status 'done' or 'failed' — 'failed' is a normal outcome; read "
            "the error. Navigation can take a while, so raise timeout_s for go_to."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "Skill name from menlo_discover_skills.",
                },
                "parameters": {
                    "type": "object",
                    "description": "The skill's input parameters object.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Max seconds to wait for the action (default 120).",
                    "default": 120,
                },
            },
            "required": ["skill"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        skill: str = "",
        parameters: dict[str, Any] | None = None,
        timeout_s: float = 120,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not skill:
            return ErrorResponse(
                message="Provide a skill name (see menlo_discover_skills).",
                error="missing_skill",
                session_id=session_id,
            )
        try:
            conn = await resolve_connection(session_id)
            result = await conn.session.invoke(
                skill, parameters or {}, timeout_s=timeout_s
            )
        except MenloNotConnectedError:
            return self._not_connected(session_id)
        except (RuntimeError, TimeoutError) as e:
            return self._no_runtime(session_id, e)

        error = to_jsonable_python(result.error) if result.error else None
        action_id = result.meta.action_id if result.meta else None
        return MenloSkillResultResponse(
            message=f"{skill} finished with status: {result.status}",
            skill=skill,
            status=str(result.status),
            action_id=action_id,
            error=error if error is None else error.get("message", str(error)),
            result=to_jsonable_python(result.result) if result.result else None,
            session_id=session_id,
        )


class MenloGetRobotStateTool(_MenloRuntimeTool):
    """Read a runtime state key (robot_status / scene_state)."""

    @property
    def name(self) -> str:
        return "menlo_get_robot_state"

    @property
    def description(self) -> str:
        return (
            "Read the robot's runtime state. key='robot_status' for position, yaw, "
            "and status (ready/moving/holding/fallen); key='scene_state' for every "
            "entity (robot, pads, cubes) with positions and colors. Use this to "
            "decide where to go and to verify results (e.g. a delivered cube "
            "becomes invisible)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "State key to read.",
                    "enum": ["robot_status", "scene_state"],
                    "default": "robot_status",
                },
            },
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        key: str = "robot_status",
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id
        try:
            conn = await resolve_connection(session_id)
            state = await conn.session.state.get(key)
        except MenloNotConnectedError:
            return self._not_connected(session_id)
        except (RuntimeError, TimeoutError) as e:
            return self._no_runtime(session_id, e)

        return MenloRobotStateResponse(
            message=f"Read state '{key}'.",
            key=key,
            state=to_jsonable_python(state),
            session_id=session_id,
        )


class MenloGetVisionTool(_MenloRuntimeTool):
    """Capture a robot camera frame into the workspace."""

    @property
    def name(self) -> str:
        return "menlo_get_vision"

    @property
    def description(self) -> str:
        return (
            "Capture the robot's camera and save the JPEG to the workspace. "
            "camera='pov' is the robot's first-person head camera. Returns a "
            "file_id you can pass to read_workspace_file to view the image."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "description": "Camera id (default: pov).",
                    "default": "pov",
                },
            },
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        camera: str = "pov",
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id
        try:
            conn = await resolve_connection(session_id)
            jpeg = await conn.session.get_vision(camera)
        except MenloNotConnectedError:
            return self._not_connected(session_id)
        except (RuntimeError, TimeoutError) as e:
            return self._no_runtime(session_id, e)

        return await self._save_frame(user_id, session, camera, jpeg)

    async def _save_frame(
        self, user_id: str | None, session: ChatSession, camera: str, jpeg: bytes
    ) -> ToolResponseBase:
        # Lazy import to avoid a circular import (workspace_files -> .models -> ...),
        # mirroring agent_browser.py's screenshot handling.
        from backend.copilot.tools.workspace_files import (
            WorkspaceWriteResponse,
            WriteWorkspaceFileTool,
        )

        width, height = Image.open(BytesIO(jpeg)).size
        filename = f"menlo-{camera}-{int(time.time())}.jpg"
        write_resp = await WriteWorkspaceFileTool()._execute(
            user_id=user_id,
            session=session,
            filename=filename,
            content_base64=base64.b64encode(jpeg).decode(),
        )
        if not isinstance(write_resp, WorkspaceWriteResponse):
            return ErrorResponse(
                message="Captured the frame but failed to save it to the workspace.",
                error="workspace_write_failed",
                session_id=session.session_id,
            )
        return MenloVisionResponse(
            message=(
                f"Captured {camera} camera ({width}x{height}). Use read_workspace_file "
                f"with file_id='{write_resp.file_id}' to view it."
            ),
            camera=camera,
            file_id=write_resp.file_id,
            filename=filename,
            width=width,
            height=height,
            session_id=session.session_id,
        )
