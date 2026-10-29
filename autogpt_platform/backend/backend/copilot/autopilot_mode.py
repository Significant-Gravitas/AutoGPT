"""Approval modes a chat runs in.

A leaf module so callers that only need the type (the expert models and
``experts_db``) do not import the whole copilot session model with it.
"""

from typing import Literal, get_args

AutopilotMode = Literal["ask_first", "auto", "unsupervised"]
AUTOPILOT_MODES: frozenset[str] = frozenset(get_args(AutopilotMode))
