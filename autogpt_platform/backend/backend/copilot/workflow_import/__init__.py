"""Competitor workflow import module.

Converts workflows from n8n, Make.com, and Zapier into AutoGPT agent graphs.
"""

from .converter import convert_competitor_workflow
from .format_detector import CompetitorFormat, detect_format
from .models import WorkflowDescription

__all__ = [
    "CompetitorFormat",
    "WorkflowDescription",
    "convert_competitor_workflow",
    "detect_format",
]
