"""Menlo robot platform tools for the copilot.

Lets the copilot drive a simulated Menlo warehouse robot: create/connect a
robot, discover its skills, invoke them, read runtime state, and capture the
robot's camera. Gated on ``MENLO_API_KEY`` (see ``manager.menlo_available``);
requires the optional ``menlo`` install extra.
"""

from .control import MenloConnectRobotTool, MenloDisconnectRobotTool
from .runtime import (
    MenloDiscoverSkillsTool,
    MenloGetRobotStateTool,
    MenloGetVisionTool,
    MenloInvokeSkillTool,
)

__all__ = [
    "MenloConnectRobotTool",
    "MenloDisconnectRobotTool",
    "MenloDiscoverSkillsTool",
    "MenloGetRobotStateTool",
    "MenloGetVisionTool",
    "MenloInvokeSkillTool",
]
