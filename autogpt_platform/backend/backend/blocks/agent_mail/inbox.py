"""
AgentMail Inbox blocks — create, get, list, update, and delete inboxes.
"""

from typing import Optional

from agentmail import AgentMail

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import agent_mail


def _client(credentials: APIKeyCredentials) -> AgentMail:
    return AgentMail(api_key=credentials.api_key.get_secret_value())


class AgentMailCreateInboxBlock(Block):
    """Creates a new AgentMail inbox with an optional username, domain, and display name."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        username: str = SchemaField(
            description="Username part of the email address (optional, auto-generated if empty)",
            default="",
        )
        domain: str = SchemaField(
            description="Domain for the email address (optional, defaults to agentmail.to)",
            default="",
        )
        display_name: str = SchemaField(
            description="Display name for the inbox",
            default="",
        )

    class Output(BlockSchemaOutput):
        inbox_id: str = SchemaField(description="The ID of the created inbox")
        email_address: str = SchemaField(
            description="The full email address of the inbox"
        )
        result: dict = SchemaField(description="Full inbox object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="7a8ac219-c6ec-4eec-a828-81af283ce04c",
            description="Create a new AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        params: dict = {}
        if input_data.username:
            params["username"] = input_data.username
        if input_data.domain:
            params["domain"] = input_data.domain
        if input_data.display_name:
            params["display_name"] = input_data.display_name

        inbox = client.inboxes.create(**params)
        result = inbox.__dict__ if hasattr(inbox, "__dict__") else {}

        yield "inbox_id", inbox.inbox_id
        yield "email_address", getattr(inbox, "email_address", inbox.inbox_id)
        yield "result", result


class AgentMailGetInboxBlock(Block):
    """Retrieves details of an existing AgentMail inbox."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to retrieve"
        )

    class Output(BlockSchemaOutput):
        inbox_id: str = SchemaField(description="The inbox ID")
        email_address: str = SchemaField(description="The email address of the inbox")
        display_name: str = SchemaField(
            description="The display name of the inbox", default=""
        )
        result: dict = SchemaField(description="Full inbox object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b858f62b-6c12-4736-aaf2-dbc5a9281320",
            description="Get an AgentMail inbox by ID",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        inbox = client.inboxes.get(inbox_id=input_data.inbox_id)
        result = inbox.__dict__ if hasattr(inbox, "__dict__") else {}

        yield "inbox_id", inbox.inbox_id
        yield "email_address", getattr(inbox, "email_address", inbox.inbox_id)
        yield "display_name", getattr(inbox, "display_name", "")
        yield "result", result


class AgentMailListInboxesBlock(Block):
    """Lists all inboxes in your AgentMail organization."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        limit: int = SchemaField(
            description="Maximum number of inboxes to return",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Pagination token from a previous request",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        inboxes: list[dict] = SchemaField(description="List of inbox objects")
        count: int = SchemaField(description="Total number of inboxes returned")
        next_page_token: str = SchemaField(
            description="Token for fetching the next page", default=""
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="cfd84a06-2121-4cef-8d14-8badf52d22f0",
            description="List all AgentMail inboxes",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        params: dict = {"limit": input_data.limit}
        if input_data.page_token:
            params["page_token"] = input_data.page_token

        response = client.inboxes.list(**params)
        inboxes = [
            inbox.__dict__ if hasattr(inbox, "__dict__") else inbox
            for inbox in getattr(response, "inboxes", [])
        ]

        yield "inboxes", inboxes
        yield "count", getattr(response, "count", len(inboxes))
        yield "next_page_token", getattr(response, "next_page_token", "")


class AgentMailUpdateInboxBlock(Block):
    """Updates the display name of an existing AgentMail inbox."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to update"
        )
        display_name: str = SchemaField(
            description="New display name for the inbox"
        )

    class Output(BlockSchemaOutput):
        inbox_id: str = SchemaField(description="The updated inbox ID")
        result: dict = SchemaField(description="Full updated inbox object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="59b49f59-a6d1-4203-94c0-3908adac50b6",
            description="Update an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        inbox = client.inboxes.update(
            inbox_id=input_data.inbox_id,
            display_name=input_data.display_name,
        )
        result = inbox.__dict__ if hasattr(inbox, "__dict__") else {}

        yield "inbox_id", inbox.inbox_id
        yield "result", result


class AgentMailDeleteInboxBlock(Block):
    """Deletes an AgentMail inbox."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to delete"
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the deletion was successful")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="ade970ae-8428-4a7b-9278-b52054dbf535",
            description="Delete an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        client.inboxes.delete(inbox_id=input_data.inbox_id)
        yield "success", True
