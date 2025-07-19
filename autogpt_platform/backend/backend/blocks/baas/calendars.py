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
    Requests,
    SchemaField,
)

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
            id="3f4a5b6c-7d8e-9f0a-1b2c-3d4e5f6a7b8c",
            description="Connect a Google or Microsoft calendar for integration",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build request body
        body = {
            "oauth_client_id": input_data.oauth_client_id,
            "oauth_client_secret": input_data.oauth_client_secret,
            "oauth_refresh_token": input_data.oauth_refresh_token,
            "platform": input_data.platform,
        }

        if input_data.calendar_email_or_id:
            body["calendar_email"] = input_data.calendar_email_or_id

        # Connect calendar
        response = await Requests().post(
            "https://api.meetingbaas.com/calendars",
            headers={"x-meeting-baas-api-key": api_key},
            json=body,
        )

        calendar = response.json()

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
            id="4a5b6c7d-8e9f-0a1b-2c3d-4e5f6a7b8c9d",
            description="List all integrated calendars",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # List calendars
        response = await Requests().get(
            "https://api.meetingbaas.com/calendars",
            headers={"x-meeting-baas-api-key": api_key},
        )

        calendars = response.json()

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
            id="5b6c7d8e-9f0a-1b2c-3d4e-5f6a7b8c9d0e",
            description="Update calendar credentials or platform",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build request body with only provided fields
        body = {}
        if input_data.oauth_client_id:
            body["oauth_client_id"] = input_data.oauth_client_id
        if input_data.oauth_client_secret:
            body["oauth_client_secret"] = input_data.oauth_client_secret
        if input_data.oauth_refresh_token:
            body["oauth_refresh_token"] = input_data.oauth_refresh_token
        if input_data.platform:
            body["platform"] = input_data.platform

        # Update calendar
        response = await Requests().patch(
            f"https://api.meetingbaas.com/calendars/{input_data.calendar_id}",
            headers={"x-meeting-baas-api-key": api_key},
            json=body,
        )

        calendar = response.json()

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
            id="6c7d8e9f-0a1b-2c3d-4e5f-6a7b8c9d0e1f",
            description="Remove a calendar integration",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Delete calendar
        response = await Requests().delete(
            f"https://api.meetingbaas.com/calendars/{input_data.calendar_id}",
            headers={"x-meeting-baas-api-key": api_key},
        )

        deleted = response.status in [200, 204]

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
            id="7d8e9f0a-1b2c-3d4e-5f6a-7b8c9d0e1f2a",
            description="Force immediate re-sync of all connected calendars",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Resync all calendars
        response = await Requests().post(
            "https://api.meetingbaas.com/internal/calendar/resync_all",
            headers={"x-meeting-baas-api-key": api_key},
        )

        data = response.json()

        yield "synced_ids", data.get("synced_calendars", [])
        yield "errors", data.get("errors", [])
