"""
Meeting BaaS calendar blocks.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import MeetingBaasAPI
from ._config import baas


class BaasCalendarConnectBlock(Block):
    """
    One-time integration of a Google or Microsoft calendar.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        oauth_client_id: str = SchemaField(description="OAuth client ID from provider")
        oauth_client_secret: str = SchemaField(description="OAuth client secret")
        oauth_refresh_token: str = SchemaField(
            description="OAuth refresh token with calendar access"
        )
        platform: str = SchemaField(
            description="Calendar platform (Google or Microsoft)"
        )
        calendar_email_or_id: str = SchemaField(
            description="Specific calendar email/ID to connect", default=""
        )

    class Output(BlockSchema):
        calendar_id: str = SchemaField(description="UUID of the connected calendar")
        calendar_obj: dict = SchemaField(description="Full calendar object")

    def __init__(self):
        super().__init__(
            id="a9d926dc-f050-40e3-88a7-75e63cf8b0fa",
            description="Connect a Google or Microsoft calendar for integration",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Connect calendar using API client
        calendar = await api.create_calendar(
            oauth_client_id=input_data.oauth_client_id,
            oauth_client_secret=input_data.oauth_client_secret,
            oauth_refresh_token=input_data.oauth_refresh_token,
            platform=input_data.platform,
            raw_calendar_id=(
                input_data.calendar_email_or_id
                if input_data.calendar_email_or_id
                else None
            ),
        )

        yield "calendar_id", calendar.get("uuid", "")
        yield "calendar_obj", calendar


class BaasCalendarListAllBlock(Block):
    """
    Enumerate connected calendars.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )

    class Output(BlockSchema):
        calendars: list[dict] = SchemaField(
            description="Array of connected calendar objects"
        )

    def __init__(self):
        super().__init__(
            id="80bc1ab0-7f53-483d-8258-3c7fdf3209fe",
            description="List all integrated calendars",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # List calendars
        calendars = await api.list_calendars()

        yield "calendars", calendars


class BaasCalendarUpdateCredsBlock(Block):
    """
    Refresh OAuth or switch provider for an existing calendar.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        calendar_id: str = SchemaField(description="UUID of the calendar to update")
        oauth_client_id: str = SchemaField(
            description="New OAuth client ID", default=""
        )
        oauth_client_secret: str = SchemaField(
            description="New OAuth client secret", default=""
        )
        oauth_refresh_token: str = SchemaField(
            description="New OAuth refresh token", default=""
        )
        platform: str = SchemaField(description="New platform if switching", default="")

    class Output(BlockSchema):
        calendar_obj: dict = SchemaField(description="Updated calendar object")

    def __init__(self):
        super().__init__(
            id="f349c9e4-7417-413b-a9da-a20548944b44",
            description="Update calendar credentials or platform",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Update calendar - API requires all fields for PATCH
        calendar = await api.update_calendar(
            calendar_id=input_data.calendar_id,
            oauth_client_id=input_data.oauth_client_id,
            oauth_client_secret=input_data.oauth_client_secret,
            oauth_refresh_token=input_data.oauth_refresh_token,
            platform=input_data.platform,
        )

        yield "calendar_obj", calendar


class BaasCalendarDeleteBlock(Block):
    """
    Disconnect calendar & unschedule future bots.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        calendar_id: str = SchemaField(description="UUID of the calendar to delete")

    class Output(BlockSchema):
        deleted: bool = SchemaField(
            description="Whether the calendar was successfully deleted"
        )

    def __init__(self):
        super().__init__(
            id="95bfa87e-aaf2-4f5d-b77d-14baae3019e9",
            description="Remove a calendar integration",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Delete calendar
        deleted = await api.delete_calendar(input_data.calendar_id)

        yield "deleted", deleted


class BaasCalendarResyncAllBlock(Block):
    """
    Force full sync now (maintenance).
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )

    class Output(BlockSchema):
        synced_ids: list[str] = SchemaField(
            description="Calendar UUIDs that synced successfully"
        )
        errors: list[list] = SchemaField(
            description="Array of [calendar_id, error_message] tuples"
        )

    def __init__(self):
        super().__init__(
            id="65690c06-c997-4b9a-b48f-8b27dc15f588",
            description="Force immediate re-sync of all connected calendars",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Resync all calendars
        data = await api.resync_all_calendars()

        yield "synced_ids", data.get("synced_calendars", [])
        yield "errors", data.get("errors", [])
