from __future__ import annotations

"""The command system provides a way to extend the functionality of the AI agent."""

from AFAAS.core.tools.base import BaseToolsRegistry
from AFAAS.core.tools.schema import ToolResult
from AFAAS.core.tools.simple import SimpleToolRegistry
from AFAAS.core.tools.tools import Tool

from .builtins import TOOL_CATEGORIES
