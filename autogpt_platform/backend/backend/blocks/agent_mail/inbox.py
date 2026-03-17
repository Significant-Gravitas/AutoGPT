"""
AgentMail Inbox blocks — create, get, list, update, and delete inboxes.

An Inbox is a fully programmable email account for AI agents. Each inbox gets
a unique email address and can send, receive, and manage emails via the
AgentMail API. You can create thousands of inboxes on demand.
"""

from agentmail.inboxes.types import CreateInboxRequest

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

from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, _client, agent_mail


class AgentMailCreateInboxBlock(Block):
    """
    Create a new email inbox for an AI agent via AgentMail.

    Each inbox gets a unique email address (e.g. username@agentmail.to).
    If username and domain are not provided, AgentMail auto-generates them.
    Use custom domains by specifying the domain field.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        username: str = SchemaField(
            description="Local part of the email address (e.g. 'support' for support@domain.com). Leave empty to auto-generate.",
            default="",
            advanced=False,
        )
        domain: str = SchemaField(
            description="Email domain (e.g. 'mydomain.com'). Defaults to agentmail.to if empty.",
            default="",
            advanced=False,
        )
        display_name: str = SchemaField(
            description="Friendly name shown in the 'From' field of sent emails (e.g. 'Support Agent')",
            default="",
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        inbox_id: str = SchemaField(
            description="Unique identifier for the created inbox (also the email address)"
        )
        email_address: str = SchemaField(
            description="Full email address of the inbox (e.g. support@agentmail.to)"
        )
        result: dict = SchemaField(
            description="Complete inbox object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="7a8ac219-c6ec-4eec-a828-81af283ce04c",
            description="Create a new email inbox for an AI agent via AgentMail. Each inbox gets a unique address and can send/receive emails.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("inbox_id", "mock-inbox-id"),
                ("email_address", "mock-inbox-id"),
                ("result", dict),
            ],
            test_mock={
                "create_inbox": lambda *a, **kw: type(
                    "Inbox",
                    (),
                    {
                        "inbox_id": "mock-inbox-id",
                        "model_dump": lambda self: {"inbox_id": "mock-inbox-id"},
                    },
                )(),
            },
        )

    @staticmethod
    async def create_inbox(credentials: APIKeyCredentials, **params):
        client = _client(credentials)
        return await client.inboxes.create(request=CreateInboxRequest(**params))

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {}
            if input_data.username:
                params["username"] = input_data.username
            if input_data.domain:
                params["domain"] = input_data.domain
            if input_data.display_name:
                params["display_name"] = input_data.display_name

            inbox = await self.create_inbox(credentials, **params)
            result = inbox.model_dump()

            yield "inbox_id", inbox.inbox_id
            yield "email_address", inbox.inbox_id
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailGetInboxBlock(Block):
    """
    Retrieve details of an existing AgentMail inbox by its ID or email address.

    Returns the inbox metadata including email address, display name, and
    configuration. Use this to check if an inbox exists or get its properties.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to look up (e.g. 'support@agentmail.to')"
        )

    class Output(BlockSchemaOutput):
        inbox_id: str = SchemaField(description="Unique identifier of the inbox")
        email_address: str = SchemaField(description="Full email address of the inbox")
        display_name: str = SchemaField(
            description="Friendly name shown in the 'From' field", default=""
        )
        result: dict = SchemaField(
            description="Complete inbox object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b858f62b-6c12-4736-aaf2-dbc5a9281320",
            description="Retrieve details of an existing AgentMail inbox including its email address, display name, and configuration.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
            },
            test_output=[
                ("inbox_id", "test-inbox"),
                ("email_address", "test-inbox"),
                ("display_name", ""),
                ("result", dict),
            ],
            test_mock={
                "get_inbox": lambda *a, **kw: type(
                    "Inbox",
                    (),
                    {
                        "inbox_id": "test-inbox",
                        "display_name": "",
                        "model_dump": lambda self: {"inbox_id": "test-inbox"},
                    },
                )(),
            },
        )

    @staticmethod
    async def get_inbox(credentials: APIKeyCredentials, inbox_id: str):
        client = _client(credentials)
        return await client.inboxes.get(inbox_id=inbox_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            inbox = await self.get_inbox(credentials, input_data.inbox_id)
            result = inbox.model_dump()

            yield "inbox_id", inbox.inbox_id
            yield "email_address", inbox.inbox_id
            yield "display_name", inbox.display_name or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailListInboxesBlock(Block):
    """
    List all email inboxes in your AgentMail organization.

    Returns a paginated list of all inboxes with their metadata.
    Use page_token for pagination when you have many inboxes.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        limit: int = SchemaField(
            description="Maximum number of inboxes to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page of results",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        inboxes: list[dict] = SchemaField(
            description="List of inbox objects, each containing inbox_id, email_address, display_name, etc."
        )
        count: int = SchemaField(
            description="Total number of inboxes in your organization"
        )
        next_page_token: str = SchemaField(
            description="Token to pass as page_token to get the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="cfd84a06-2121-4cef-8d14-8badf52d22f0",
            description="List all email inboxes in your AgentMail organization with pagination support.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("inboxes", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_inboxes": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "inboxes": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_inboxes(credentials: APIKeyCredentials, **params):
        client = _client(credentials)
        return await client.inboxes.list(**params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token

            response = await self.list_inboxes(credentials, **params)
            inboxes = [i.model_dump() for i in response.inboxes]

            yield "inboxes", inboxes
            yield "count", (c if (c := response.count) is not None else len(inboxes))
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


class AgentMailUpdateInboxBlock(Block):
    """
    Update the display name of an existing AgentMail inbox.

    Changes the friendly name shown in the 'From' field when emails are sent
    from this inbox. The email address itself cannot be changed.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to update (e.g. 'support@agentmail.to')"
        )
        display_name: str = SchemaField(
            description="New display name for the inbox (e.g. 'Customer Support Bot')"
        )

    class Output(BlockSchemaOutput):
        inbox_id: str = SchemaField(description="The updated inbox ID")
        result: dict = SchemaField(
            description="Complete updated inbox object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="59b49f59-a6d1-4203-94c0-3908adac50b6",
            description="Update the display name of an AgentMail inbox. Changes the 'From' name shown when emails are sent.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "display_name": "Updated",
            },
            test_output=[
                ("inbox_id", "test-inbox"),
                ("result", dict),
            ],
            test_mock={
                "update_inbox": lambda *a, **kw: type(
                    "Inbox",
                    (),
                    {
                        "inbox_id": "test-inbox",
                        "model_dump": lambda self: {"inbox_id": "test-inbox"},
                    },
                )(),
            },
        )

    @staticmethod
    async def update_inbox(credentials: APIKeyCredentials, inbox_id: str, **params):
        client = _client(credentials)
        return await client.inboxes.update(inbox_id=inbox_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            inbox = await self.update_inbox(
                credentials,
                input_data.inbox_id,
                display_name=input_data.display_name,
            )
            result = inbox.model_dump()

            yield "inbox_id", inbox.inbox_id
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailDeleteInboxBlock(Block):
    """
    Permanently delete an AgentMail inbox and all its data.

    This removes the inbox, all its messages, threads, and drafts.
    This action cannot be undone. The email address will no longer
    receive or send emails.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to permanently delete"
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the inbox was successfully deleted"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="ade970ae-8428-4a7b-9278-b52054dbf535",
            description="Permanently delete an AgentMail inbox and all its messages, threads, and drafts. This action cannot be undone.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
            },
            test_output=[("success", True)],
            test_mock={
                "delete_inbox": lambda *a, **kw: None,
            },
        )

    @staticmethod
    async def delete_inbox(credentials: APIKeyCredentials, inbox_id: str):
        client = _client(credentials)
        await client.inboxes.delete(inbox_id=inbox_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            await self.delete_inbox(credentials, input_data.inbox_id)
            yield "success", True
        except Exception as e:
            yield "error", str(e)
