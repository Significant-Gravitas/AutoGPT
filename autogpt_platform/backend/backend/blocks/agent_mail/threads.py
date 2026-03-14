"""
AgentMail Thread blocks — list, get, and delete conversation threads.

A Thread groups related messages into a single conversation. Threads are
created automatically when a new message is sent and grow as replies are added.
Threads can be queried per-inbox or across the entire organization.
"""

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


class AgentMailListInboxThreadsBlock(Block):
    """
    List all conversation threads within a specific AgentMail inbox.

    Returns a paginated list of threads with optional label filtering.
    Use labels to find threads by campaign, status, or custom tags.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to list threads from"
        )
        limit: int = SchemaField(
            description="Maximum number of threads to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Only return threads matching ALL of these labels (e.g. ['q4-campaign', 'follow-up'])",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        threads: list[dict] = SchemaField(
            description="List of thread objects with thread_id, subject, message count, labels, etc."
        )
        count: int = SchemaField(description="Number of threads returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="63dd9e2d-ef81-405c-b034-c031f0437334",
            description="List all conversation threads in an AgentMail inbox. Filter by labels for campaign tracking or status management.",
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
        if input_data.labels:
            params["labels"] = input_data.labels

        response = client.inboxes.threads.list(
            inbox_id=input_data.inbox_id, **params
        )
        threads = [
            t.__dict__ if hasattr(t, "__dict__") else t
            for t in getattr(response, "threads", [])
        ]

        yield "threads", threads
        yield "count", getattr(response, "count", len(threads))
        yield "next_page_token", getattr(response, "next_page_token", "")


class AgentMailGetInboxThreadBlock(Block):
    """
    Retrieve a single conversation thread from an AgentMail inbox.

    Returns the thread with all its messages in chronological order.
    Use this to get the full conversation history for context when
    composing replies.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the thread belongs to"
        )
        thread_id: str = SchemaField(
            description="Thread ID to retrieve"
        )

    class Output(BlockSchemaOutput):
        thread_id: str = SchemaField(description="Unique identifier of the thread")
        messages: list[dict] = SchemaField(
            description="All messages in the thread, in chronological order"
        )
        result: dict = SchemaField(
            description="Complete thread object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="42866290-1479-4153-83e7-550b703e9da2",
            description="Retrieve a conversation thread with all its messages. Use for getting full conversation context before replying.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        thread = client.inboxes.threads.get(
            inbox_id=input_data.inbox_id,
            thread_id=input_data.thread_id,
        )
        messages = [
            m.__dict__ if hasattr(m, "__dict__") else m
            for m in getattr(thread, "messages", [])
        ]
        result = thread.__dict__ if hasattr(thread, "__dict__") else {}

        yield "thread_id", thread.thread_id
        yield "messages", messages
        yield "result", result


class AgentMailDeleteInboxThreadBlock(Block):
    """
    Permanently delete a conversation thread and all its messages from an inbox.

    This removes the thread and every message within it. This action
    cannot be undone.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the thread belongs to"
        )
        thread_id: str = SchemaField(
            description="Thread ID to permanently delete"
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the thread was successfully deleted"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="18cd5f6f-4ff6-45da-8300-25a50ea7fb75",
            description="Permanently delete a conversation thread and all its messages. This action cannot be undone.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        client.inboxes.threads.delete(
            inbox_id=input_data.inbox_id,
            thread_id=input_data.thread_id,
        )
        yield "success", True


class AgentMailListOrgThreadsBlock(Block):
    """
    List conversation threads across ALL inboxes in your organization.

    Unlike per-inbox listing, this returns threads from every inbox.
    Ideal for building supervisor agents that monitor all conversations,
    analytics dashboards, or cross-agent routing workflows.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        limit: int = SchemaField(
            description="Maximum number of threads to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Only return threads matching ALL of these labels",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        threads: list[dict] = SchemaField(
            description="List of thread objects from all inboxes in the organization"
        )
        count: int = SchemaField(description="Number of threads returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="d7a0657b-58ab-48b2-898b-7bd94f44a708",
            description="List threads across ALL inboxes in your organization. Use for supervisor agents, dashboards, or cross-agent monitoring.",
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
        if input_data.labels:
            params["labels"] = input_data.labels

        response = client.threads.list(**params)
        threads = [
            t.__dict__ if hasattr(t, "__dict__") else t
            for t in getattr(response, "threads", [])
        ]

        yield "threads", threads
        yield "count", getattr(response, "count", len(threads))
        yield "next_page_token", getattr(response, "next_page_token", "")


class AgentMailGetOrgThreadBlock(Block):
    """
    Retrieve a single conversation thread by ID from anywhere in the organization.

    Works without needing to know which inbox the thread belongs to.
    Returns the thread with all its messages in chronological order.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        thread_id: str = SchemaField(
            description="Thread ID to retrieve (works across all inboxes)"
        )

    class Output(BlockSchemaOutput):
        thread_id: str = SchemaField(description="Unique identifier of the thread")
        messages: list[dict] = SchemaField(
            description="All messages in the thread, in chronological order"
        )
        result: dict = SchemaField(
            description="Complete thread object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="39aaae31-3eb1-44c6-9e37-5a44a4529649",
            description="Retrieve a conversation thread by ID from anywhere in the organization, without needing the inbox ID.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        thread = client.threads.get(thread_id=input_data.thread_id)
        messages = [
            m.__dict__ if hasattr(m, "__dict__") else m
            for m in getattr(thread, "messages", [])
        ]
        result = thread.__dict__ if hasattr(thread, "__dict__") else {}

        yield "thread_id", thread.thread_id
        yield "messages", messages
        yield "result", result
