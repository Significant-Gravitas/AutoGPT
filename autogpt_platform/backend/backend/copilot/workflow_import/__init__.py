"""Workflow import module.

Parses workflows from n8n, Make.com, and Zapier into structured descriptions,
then builds CoPilot prompts for the agentic agent-generator to handle conversion.
"""

from .converter import build_copilot_prompt
from .format_detector import SourcePlatform, detect_format
from .models import WorkflowDescription

__all__ = [
    "SourcePlatform",
    "WorkflowDescription",
    "build_copilot_prompt",
    "detect_format",
]
