"""
Change Navigator - AI Co-Navigator for executive coaching sessions.

This module implements an AI-powered weekly check-in agent that supports
senior executives enrolled in the 'Change Navigator' coaching program.
It automates journal filling, tracks OKR progress, and surfaces blockers
before each session with the human coach.
"""

from autogpt.change_navigator.agent import ChangeNavigatorAgent
from autogpt.change_navigator.journal import JournalEntry, KeyResult
from autogpt.change_navigator.workflow import CheckInWorkflow

__all__ = ["ChangeNavigatorAgent", "JournalEntry", "KeyResult", "CheckInWorkflow"]
