"""
Meeting BaaS calendar event blocks.
"""

from typing import Union

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import baas


class BaasEventListBlock(Block):
    """
    Get events for a calendar & date range.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        calendar_id: str = SchemaField(
            description="UUID of the calendar to list events from"
        )
        start_date_gte: str = SchemaField(
            description="ISO date string for start date (greater than or equal)",
            default="",
        )
        start_date_lte: str = SchemaField(
            description="ISO date string for start date (less than or equal)",
            default="",
        )
        cursor: str = SchemaField(
            description="Pagination cursor from previous request", default=""
        )

    class Output(BlockSchema):
        events: list[dict] = SchemaField(description="Array of calendar events")
        next_cursor: str = SchemaField(description="Cursor for next page of results")

    def __init__(self):
        super().__init__(
            id="8e9f0a1b-2c3d-4e5f-6a7b-8c9d0e1f2a3b",
            description="List calendar events with optional date filtering",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params = {"calendar_id": input_data.calendar_id}

        if input_data.start_date_gte:
            params["start_date_gte"] = input_data.start_date_gte
        if input_data.start_date_lte:
            params["start_date_lte"] = input_data.start_date_lte
        if input_data.cursor:
            params["cursor"] = input_data.cursor

        # List events
        response = await Requests().get(
            "https://api.meetingbaas.com/calendar_events",
            headers={"x-meeting-baas-api-key": api_key},
            params=params,
        )

        data = response.json()

        yield "events", data.get("events", [])
        yield "next_cursor", data.get("next", "")


class BaasEventGetDetailsBlock(Block):
    """
    Fetch full object for one event.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        event_id: str = SchemaField(description="UUID of the event to retrieve")

    class Output(BlockSchema):
        event: dict = SchemaField(description="Full event object with all details")

    def __init__(self):
        super().__init__(
            id="9f0a1b2c-3d4e-5f6a-7b8c-9d0e1f2a3b4c",
            description="Get detailed information for a specific calendar event",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Get event details
        response = await Requests().get(
            f"https://api.meetingbaas.com/calendar_events/{input_data.event_id}",
            headers={"x-meeting-baas-api-key": api_key},
        )

        event = response.json()

        yield "event", event


class BaasEventScheduleBotBlock(Block):
    """
    Attach bot config to the event for automatic recording.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        event_id: str = SchemaField(description="UUID of the event to schedule bot for")
        all_occurrences: bool = SchemaField(
            description="Apply to all occurrences of recurring event", default=False
        )
        bot_config: dict = SchemaField(
            description="Bot configuration (same as Bot → Join Meeting)"
        )

    class Output(BlockSchema):
        events: Union[dict, list[dict]] = SchemaField(
            description="Updated event(s) with bot scheduled"
        )

    def __init__(self):
        super().__init__(
            id="0a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
            description="Schedule a recording bot for a calendar event",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params = {"all_occurrences": str(input_data.all_occurrences).lower()}

        # Schedule bot
        response = await Requests().post(
            f"https://api.meetingbaas.com/calendar_events/{input_data.event_id}/bot",
            headers={"x-meeting-baas-api-key": api_key},
            params=params,
            json=input_data.bot_config,
        )

        events = response.json()

        yield "events", events


class BaasEventUnscheduleBotBlock(Block):
    """
    Remove bot from event/series.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        event_id: str = SchemaField(
            description="UUID of the event to unschedule bot from"
        )
        all_occurrences: bool = SchemaField(
            description="Apply to all occurrences of recurring event", default=False
        )

    class Output(BlockSchema):
        events: Union[dict, list[dict]] = SchemaField(
            description="Updated event(s) with bot removed"
        )

    def __init__(self):
        super().__init__(
            id="1b2c3d4e-5f6a-7b8c-9d0e-1f2a3b4c5d6e",
            description="Cancel a scheduled recording for an event",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params = {"all_occurrences": str(input_data.all_occurrences).lower()}

        # Unschedule bot
        response = await Requests().delete(
            f"https://api.meetingbaas.com/calendar_events/{input_data.event_id}/bot",
            headers={"x-meeting-baas-api-key": api_key},
            params=params,
        )

        events = response.json()

        yield "events", events


class BaasEventPatchBotBlock(Block):
    """
    Modify an already-scheduled bot configuration.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        event_id: str = SchemaField(description="UUID of the event with scheduled bot")
        all_occurrences: bool = SchemaField(
            description="Apply to all occurrences of recurring event", default=False
        )
        bot_patch: dict = SchemaField(description="Bot configuration fields to update")

    class Output(BlockSchema):
        events: Union[dict, list[dict]] = SchemaField(
            description="Updated event(s) with modified bot config"
        )

    def __init__(self):
        super().__init__(
            id="2c3d4e5f-6a7b-8c9d-0e1f-2a3b4c5d6e7f",
            description="Update configuration of a scheduled bot",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params = {}
        if input_data.all_occurrences is not None:
            params["all_occurrences"] = str(input_data.all_occurrences).lower()

        # Patch bot
        response = await Requests().patch(
            f"https://api.meetingbaas.com/calendar_events/{input_data.event_id}/bot",
            headers={"x-meeting-baas-api-key": api_key},
            params=params,
            json=input_data.bot_patch,
        )

        events = response.json()

        yield "events", events
