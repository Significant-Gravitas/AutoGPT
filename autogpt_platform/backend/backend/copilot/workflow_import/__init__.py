"""Workflow import module.

Converts workflows from n8n, Make.com, and Zapier into AutoGPT agent graphs.
"""

from .converter import convert_workflow
from .format_detector import SourcePlatform, detect_format
from .models import WorkflowDescription

__all__ = [
    "SourcePlatform",
    "WorkflowDescription",
    "convert_workflow",
    "detect_format",
]
