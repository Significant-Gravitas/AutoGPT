"""
Meeting BaaS integration for AutoGPT Platform.

This integration provides comprehensive access to the Meeting BaaS API,
including:
- Bot management for meeting recordings
- Calendar integration (Google/Microsoft)
- Event management and scheduling
- Webhook triggers for real-time events
"""

# Bot (Recording) Blocks
from .bots import (
    BaasBotDeleteRecordingBlock,
    BaasBotFetchMeetingDataBlock,
    BaasBotFetchScreenshotsBlock,
    BaasBotJoinMeetingBlock,
    BaasBotLeaveMeetingBlock,
    BaasBotRetranscribeBlock,
)

# Calendar Blocks
from .calendars import (
    BaasCalendarConnectBlock,
    BaasCalendarDeleteBlock,
    BaasCalendarListAllBlock,
    BaasCalendarResyncAllBlock,
    BaasCalendarUpdateCredsBlock,
)

# Event Blocks
from .events import (
    BaasEventGetDetailsBlock,
    BaasEventListBlock,
    BaasEventPatchBotBlock,
    BaasEventScheduleBotBlock,
    BaasEventUnscheduleBotBlock,
)

# Webhook Triggers
from .triggers import BaasOnCalendarEventBlock, BaasOnMeetingEventBlock

__all__ = [
    # Bot (Recording) Blocks
    "BaasBotJoinMeetingBlock",
    "BaasBotLeaveMeetingBlock",
    "BaasBotFetchMeetingDataBlock",
    "BaasBotFetchScreenshotsBlock",
    "BaasBotDeleteRecordingBlock",
    "BaasBotRetranscribeBlock",
    # Calendar Blocks
    "BaasCalendarConnectBlock",
    "BaasCalendarListAllBlock",
    "BaasCalendarUpdateCredsBlock",
    "BaasCalendarDeleteBlock",
    "BaasCalendarResyncAllBlock",
    # Event Blocks
    "BaasEventListBlock",
    "BaasEventGetDetailsBlock",
    "BaasEventScheduleBotBlock",
    "BaasEventUnscheduleBotBlock",
    "BaasEventPatchBotBlock",
    # Webhook Triggers
    "BaasOnMeetingEventBlock",
    "BaasOnCalendarEventBlock",
]
