"""Connect / disconnect copilot tools for the Menlo robot platform."""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ErrorResponse, ToolResponseBase

from .manager import (
    MenloConfigError,
    connect_new_robot,
    disconnect_robot,
    generate_viewer_url,
    menlo_available,
)
from .responses import MenloRobotConnectedResponse, MenloRobotDisconnectedResponse

logger = logging.getLogger(__name__)


class MenloConnectRobotTool(BaseTool):
    """Create a simulated Menlo robot and open its browser runtime."""

    @property
    def name(self) -> str:
        return "menlo_connect_robot"

    @property
    def description(self) -> str:
        return (
            "Create a simulated Menlo warehouse robot and open its 3D runtime. "
            "Returns a viewer URL — the user MUST open it in Chrome to start the "
            "simulation before any robot skills (go_to, pick_entity, set_velocity, "
            "...) will run. One robot per chat session; call this once, before the "
            "menlo_discover_skills / menlo_invoke_skill tools."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Robot model to create (default: asimov-v0).",
                    "default": "asimov-v0",
                },
            },
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return menlo_available()

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        model: str = "asimov-v0",
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id
        try:
            name = f"copilot-{session_id}-{int(time.time())}"
            conn = await connect_new_robot(session_id, model=model, name=name)
            viewer_url = await generate_viewer_url(conn)
        except MenloConfigError as e:
            return ErrorResponse(
                message="Menlo robot support is not configured on this server.",
                error=str(e),
                session_id=session_id,
            )

        return MenloRobotConnectedResponse(
            message=(
                f"Robot {conn.robot_id} created. Open the viewer URL in Chrome to "
                "start the simulation runtime, then wait a few seconds and call "
                "menlo_discover_skills."
            ),
            robot_id=conn.robot_id,
            model=model,
            viewer_url=viewer_url,
            session_id=session_id,
        )


class MenloDisconnectRobotTool(BaseTool):
    """Disconnect and delete the session's robot."""

    @property
    def name(self) -> str:
        return "menlo_disconnect_robot"

    @property
    def description(self) -> str:
        return (
            "Disconnect and delete the robot for this chat session, freeing platform "
            "resources. Call this when finished driving the robot."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return menlo_available()

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id
        robot_id = await disconnect_robot(session_id, delete_robot=True)
        message = (
            f"Robot {robot_id} disconnected and deleted."
            if robot_id
            else "No robot was connected for this session."
        )
        return MenloRobotDisconnectedResponse(
            message=message,
            robot_id=robot_id,
            session_id=session_id,
        )
