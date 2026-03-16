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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        params: dict = {"limit": input_data.limit}
        if input_data.page_token:
            params["page_token"] = input_data.page_token

        response = client.lists.list(
            input_data.direction.value, input_data.list_type.value, **params
        )
        entries = [
            e.__dict__ if hasattr(e, "__dict__") else e
            for e in getattr(response, "entries", [])
        ]

        yield "entries", entries
        yield "count", (
            c if (c := getattr(response, "count", None)) is not None else len(entries)
        )
        yield "next_page_token", getattr(response, "next_page_token", "") or ""


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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        params: dict = {"entry": input_data.entry}
        if input_data.reason:
            params["reason"] = input_data.reason

        result = client.lists.create(
            input_data.direction.value, input_data.list_type.value, **params
        )
        result_dict = result.__dict__ if hasattr(result, "__dict__") else {}

        yield "entry", input_data.entry
        yield "result", result_dict


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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        result = client.lists.get(
            input_data.direction.value,
            input_data.list_type.value,
            entry=input_data.entry,
        )
        result_dict = result.__dict__ if hasattr(result, "__dict__") else {}

        yield "entry", input_data.entry
        yield "result", result_dict


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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        client.lists.delete(
            input_data.direction.value,
            input_data.list_type.value,
            entry=input_data.entry,
        )
        yield "success", True
