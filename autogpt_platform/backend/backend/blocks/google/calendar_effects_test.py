import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.calendar_availability import (
    GoogleCalendarSuggestMeetingTimesBlock,
)
from backend.blocks.google.calendar_events import (
    GoogleCalendarDeleteEventBlock,
    GoogleCalendarUpdateEventBlock,
)
from backend.blocks.google.calendar_invitations import GoogleCalendarRespondToEventBlock
from backend.blocks.google.calendar_search import (
    GoogleCalendarGetEventBlock,
    GoogleCalendarListCalendarsBlock,
    GoogleCalendarSearchEventsBlock,
)


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleCalendarSearchEventsBlock, BlockEffect.READ),
        (GoogleCalendarGetEventBlock, BlockEffect.READ),
        (GoogleCalendarListCalendarsBlock, BlockEffect.READ),
        (GoogleCalendarSuggestMeetingTimesBlock, BlockEffect.READ),
        (GoogleCalendarUpdateEventBlock, BlockEffect.EXTERNAL),
        (GoogleCalendarDeleteEventBlock, BlockEffect.EXTERNAL),
        (GoogleCalendarRespondToEventBlock, BlockEffect.EXTERNAL),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
