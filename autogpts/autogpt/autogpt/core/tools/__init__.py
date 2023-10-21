from __future__ import annotations

"""The command system provides a way to extend the functionality of the AI agent."""

from autogpts.autogpt.autogpt.core.tools.base import BaseToolsRegistry
from autogpts.autogpt.autogpt.core.tools.tools import Tool, ToolOutput
from autogpts.autogpt.autogpt.core.tools.schema import ToolResult
from autogpts.autogpt.autogpt.core.tools.simple import SimpleToolRegistry
from .builtins import TOOL_CATEGORIES
