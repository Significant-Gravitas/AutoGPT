"""
AgentMail List blocks — manage allow/block lists for email filtering.

Lists let you control which email addresses and domains your agents can
send to or receive from. There are four list types based on two dimensions:
direction (send/receive) and type (allow/block).

- receive + allow: Only accept emails from these addresses/domains
- receive + block: Reject emails from these addresses/domains
- send + allow: Only send emails to these addresses/domains
- send + block: Prevent sending emails to these addresses/domains
"""

from enum import Enum

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


class ListDirection(str, Enum):
    SEND = "send"
    RECEIVE = "receive"


class ListType(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class AgentMailListEntriesBlock(Block):
    """
    List all entries in an AgentMail allow/block list.

    Retrieves email addresses and domains that are currently allowed
    or blocked for sending or receiving. Use direction and list_type
    to select which of the four lists to query.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        direction: ListDirection = SchemaField(
            description="'send' to filter outgoing emails, 'receive' to filter incoming emails"
        )
        list_type: ListType = SchemaField(
            description="'allow' for whitelist (only permit these), 'block' for blacklist (reject these)"
        )
        limit: int = SchemaField(
            description="Maximum number of entries to return per page",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        entries: list[dict] = SchemaField(
            description="List of entries, each with an email address or domain"
        )
        count: int = SchemaField(description="Number of entries returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="01489100-35da-45aa-8a01-9540ba0e9a21",
            description="List all entries in an AgentMail allow/block list. Choose send/receive direction and allow/block type.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "direction": "receive",
                "list_type": "block",
            },
            test_output=[
                ("entries", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_entries": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "entries": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_entries(
        credentials: APIKeyCredentials, direction: str, list_type: str, **params
    ):
        client = _client(credentials)
        return await client.lists.list(direction, list_type, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token

            response = await self.list_entries(
                credentials,
                input_data.direction.value,
                input_data.list_type.value,
                **params,
            )
            entries = [e.model_dump() for e in response.entries]

            yield "entries", entries
            yield "count", (c if (c := response.count) is not None else len(entries))
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


class AgentMailCreateListEntryBlock(Block):
    """
    Add an email address or domain to an AgentMail allow/block list.

    Entries can be full email addresses (e.g. 'partner@example.com') or
    entire domains (e.g. 'example.com'). For block lists, you can optionally
    provide a reason (e.g. 'spam', 'competitor').
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        direction: ListDirection = SchemaField(
            description="'send' for outgoing email rules, 'receive' for incoming email rules"
        )
        list_type: ListType = SchemaField(
            description="'allow' to whitelist, 'block' to blacklist"
        )
        entry: str = SchemaField(
            description="Email address (user@example.com) or domain (example.com) to add"
        )
        reason: str = SchemaField(
            description="Reason for blocking (only used with block lists, e.g. 'spam', 'competitor')",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        entry: str = SchemaField(
            description="The email address or domain that was added"
        )
        result: dict = SchemaField(description="Complete entry object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b6650a0a-b113-40cf-8243-ff20f684f9b8",
            description="Add an email address or domain to an allow/block list. Block spam senders or whitelist trusted domains.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "direction": "receive",
                "list_type": "block",
                "entry": "spam@example.com",
            },
            test_output=[
                ("entry", "spam@example.com"),
                ("result", dict),
            ],
            test_mock={
                "create_entry": lambda *a, **kw: type(
                    "Entry",
                    (),
                    {
                        "model_dump": lambda self: {"entry": "spam@example.com"},
                    },
                )(),
            },
        )

    @staticmethod
    async def create_entry(
        credentials: APIKeyCredentials, direction: str, list_type: str, **params
    ):
        client = _client(credentials)
        return await client.lists.create(direction, list_type, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"entry": input_data.entry}
            if input_data.reason and input_data.list_type == ListType.BLOCK:
                params["reason"] = input_data.reason

            result = await self.create_entry(
                credentials,
                input_data.direction.value,
                input_data.list_type.value,
                **params,
            )
            result_dict = result.model_dump()

            yield "entry", input_data.entry
            yield "result", result_dict
        except Exception as e:
            yield "error", str(e)


class AgentMailGetListEntryBlock(Block):
    """
    Check if an email address or domain exists in an AgentMail allow/block list.

    Returns the entry details if found. Use this to verify whether a specific
    address or domain is currently allowed or blocked.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        direction: ListDirection = SchemaField(
            description="'send' for outgoing rules, 'receive' for incoming rules"
        )
        list_type: ListType = SchemaField(
            description="'allow' for whitelist, 'block' for blacklist"
        )
        entry: str = SchemaField(description="Email address or domain to look up")

    class Output(BlockSchemaOutput):
        entry: str = SchemaField(
            description="The email address or domain that was found"
        )
        result: dict = SchemaField(description="Complete entry object with metadata")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="fb117058-ab27-40d1-9231-eb1dd526fc7a",
            description="Check if an email address or domain is in an allow/block list. Verify filtering rules.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "direction": "receive",
                "list_type": "block",
                "entry": "spam@example.com",
            },
            test_output=[
                ("entry", "spam@example.com"),
                ("result", dict),
            ],
            test_mock={
                "get_entry": lambda *a, **kw: type(
                    "Entry",
                    (),
                    {
                        "model_dump": lambda self: {"entry": "spam@example.com"},
                    },
                )(),
            },
        )

    @staticmethod
    async def get_entry(
        credentials: APIKeyCredentials, direction: str, list_type: str, entry: str
    ):
        client = _client(credentials)
        return await client.lists.get(direction, list_type, entry=entry)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self.get_entry(
                credentials,
                input_data.direction.value,
                input_data.list_type.value,
                input_data.entry,
            )
            result_dict = result.model_dump()

            yield "entry", input_data.entry
            yield "result", result_dict
        except Exception as e:
            yield "error", str(e)


class AgentMailDeleteListEntryBlock(Block):
    """
    Remove an email address or domain from an AgentMail allow/block list.

    After removal, the address/domain will no longer be filtered by this list.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        direction: ListDirection = SchemaField(
            description="'send' for outgoing rules, 'receive' for incoming rules"
        )
        list_type: ListType = SchemaField(
            description="'allow' for whitelist, 'block' for blacklist"
        )
        entry: str = SchemaField(
            description="Email address or domain to remove from the list"
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the entry was successfully removed"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="2b8d57f1-1c9e-470f-a70b-5991c80fad5f",
            description="Remove an email address or domain from an allow/block list to stop filtering it.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "direction": "receive",
                "list_type": "block",
                "entry": "spam@example.com",
            },
            test_output=[("success", True)],
            test_mock={
                "delete_entry": lambda *a, **kw: None,
            },
        )

    @staticmethod
    async def delete_entry(
        credentials: APIKeyCredentials, direction: str, list_type: str, entry: str
    ):
        client = _client(credentials)
        await client.lists.delete(direction, list_type, entry=entry)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            await self.delete_entry(
                credentials,
                input_data.direction.value,
                input_data.list_type.value,
                input_data.entry,
            )
            yield "success", True
        except Exception as e:
            yield "error", str(e)
