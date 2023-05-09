"""
This module contains the runner for the v2 agent server and client.
"""
import click

from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.IN_PROGRESS,
    handoff_notes="A basic client and server is being sketched out. It is at the idea stage.",
)
