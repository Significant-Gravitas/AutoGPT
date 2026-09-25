import asyncio
from enum import Enum
from typing import Any

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._calendar_api import (
    CALENDAR_EVENTS_SCOPE,
    CALENDAR_ID_DESCRIPTION,
    EVENT_ID_DESCRIPTION,
    TEST_EVENT_RESOURCE,
    CalendarEventDetails,
    build_calendar_service,
    calendar_error,
    get_event,
    require_event_id,
    to_event_details,
)

_TEST_INVITATION = {
    **TEST_EVENT_RESOURCE,
    "organizer": {"email": "alex@example.com"},
    "attendees": [
        {"email": "alex@example.com", "organizer": True, "responseStatus": "accepted"},
        {"email": "me@example.com", "self": True, "responseStatus": "needsAction"},
    ],
}
_TEST_ACCEPTED = {
    **_TEST_INVITATION,
    "attendees": [
        _TEST_INVITATION["attendees"][0],
        {
            "email": "me@example.com",
            "self": True,
            "responseStatus": "accepted",
            "comment": "See you there",
        },
    ],
}


class EventResponse(str, Enum):
    ACCEPTED = "accepted"
    DECLINED = "declined"
    TENTATIVE = "tentative"


class GoogleCalendarRespondToEventBlock(Block):
    """Reply to a Google Calendar invitation."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CALENDAR_EVENTS_SCOPE]
        )
        event_id: str = SchemaField(description=EVENT_ID_DESCRIPTION)
        calendar_id: str = SchemaField(
            description=CALENDAR_ID_DESCRIPTION, default="primary"
        )
        response: EventResponse = SchemaField(description="Your reply")
        comment: str = SchemaField(
            description="Optional note for the organizer", default=""
        )
        notify_organizer: bool = SchemaField(
            description="Email the organizer about your reply", default=True
        )

    class Output(BlockSchemaOutput):
        event: CalendarEventDetails = SchemaField(
            description="The event with your reply"
        )

    def __init__(self):
        super().__init__(
            id="6dbfcfa2-9441-4967-8d8c-8d99ea07cc19",
            description=(
                "Accept, decline or tentatively accept a Google Calendar "
                "invitation, with an optional note to the organizer."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleCalendarRespondToEventBlock.Input,
            output_schema=GoogleCalendarRespondToEventBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "event_id": TEST_EVENT_RESOURCE["id"],
                "response": EventResponse.ACCEPTED,
                "comment": "See you there",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("event", to_event_details(_TEST_ACCEPTED, "primary"))],
            test_mock={
                "_get_event": lambda *args, **kwargs: _TEST_INVITATION,
                "_patch_event": lambda *args, **kwargs: _TEST_ACCEPTED,
            },
            is_irreversible_action=True,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        event_id = require_event_id(input_data.event_id, self.name, self.id)
        service = build_calendar_service(credentials)
        try:
            current = await asyncio.to_thread(
                self._get_event, service, input_data.calendar_id, event_id
            )
            updated = await asyncio.to_thread(
                self._patch_event,
                service,
                input_data.calendar_id,
                event_id,
                self._reply_body(current, input_data),
                "all" if input_data.notify_organizer else "none",
            )
        except HttpError as e:
            raise calendar_error(e, self.name, self.id) from e
        yield "event", to_event_details(updated, input_data.calendar_id)

    def _reply_body(self, current: dict, input_data: Input) -> dict:
        me = next((a for a in current.get("attendees", []) if a.get("self")), None)
        if me is None:
            raise BlockExecutionError(
                message=(
                    "The connected Google account isn't on this event's guest "
                    "list, so there's no invitation to reply to."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        reply: dict[str, Any] = {
            "email": me["email"],
            "responseStatus": input_data.response.value,
        }
        if input_data.comment:
            reply["comment"] = input_data.comment
        # attendeesOmitted: only our own reply changes, the rest of the list stays.
        return {"attendeesOmitted": True, "attendees": [reply]}

    @staticmethod
    def _get_event(service, calendar_id: str, event_id: str) -> dict:
        return get_event(service, calendar_id, event_id)

    @staticmethod
    def _patch_event(
        service, calendar_id: str, event_id: str, body: dict, send_updates: str
    ) -> dict:
        return (
            service.events()
            .patch(
                calendarId=calendar_id,
                eventId=event_id,
                body=body,
                sendUpdates=send_updates,
            )
            .execute()
        )
