from __future__ import annotations

"""The command system provides a way to extend the functionality of the AI agent."""

from AFAAS.app.core.tools.base import BaseToolsRegistry
from AFAAS.app.core.tools.schema import ToolResult
from AFAAS.app.core.tools.simple import SimpleToolRegistry
from AFAAS.app.core.tools.tools import Tool, ToolOutput

from .builtins import TOOL_CATEGORIES
