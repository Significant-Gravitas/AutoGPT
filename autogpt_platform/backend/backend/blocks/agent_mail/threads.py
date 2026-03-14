"""
AgentMail Thread blocks — list, get, and delete threads (per-inbox and org-wide).
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
    """Lists all threads within a specific AgentMail inbox."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to list threads from"
        )
        limit: int = SchemaField(
            description="Maximum number of threads to return",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Pagination token from a previous request",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Filter threads by labels",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        threads: list[dict] = SchemaField(description="List of thread objects")
        count: int = SchemaField(description="Number of threads returned")
        next_page_token: str = SchemaField(
            description="Token for fetching the next page", default=""
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="63dd9e2d-ef81-405c-b034-c031f0437334",
            description="List threads in an AgentMail inbox",
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
    """Retrieves a single thread from an AgentMail inbox, including its messages."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        thread_id: str = SchemaField(description="The thread ID to retrieve")

    class Output(BlockSchemaOutput):
        thread_id: str = SchemaField(description="The thread ID")
        messages: list[dict] = SchemaField(
            description="List of messages in the thread"
        )
        result: dict = SchemaField(description="Full thread object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="42866290-1479-4153-83e7-550b703e9da2",
            description="Get a thread from an AgentMail inbox",
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
    """Deletes a thread from an AgentMail inbox."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        thread_id: str = SchemaField(description="The thread ID to delete")

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the deletion was successful")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="18cd5f6f-4ff6-45da-8300-25a50ea7fb75",
            description="Delete a thread from an AgentMail inbox",
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
    """Lists threads across all inboxes in the organization (org-wide)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        limit: int = SchemaField(
            description="Maximum number of threads to return",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Pagination token from a previous request",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Filter threads by labels",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        threads: list[dict] = SchemaField(description="List of thread objects")
        count: int = SchemaField(description="Number of threads returned")
        next_page_token: str = SchemaField(
            description="Token for fetching the next page", default=""
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="d7a0657b-58ab-48b2-898b-7bd94f44a708",
            description="List threads across all AgentMail inboxes (org-wide)",
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
    """Retrieves a single thread by ID across the organization (org-wide)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        thread_id: str = SchemaField(description="The thread ID to retrieve")

    class Output(BlockSchemaOutput):
        thread_id: str = SchemaField(description="The thread ID")
        messages: list[dict] = SchemaField(
            description="List of messages in the thread"
        )
        result: dict = SchemaField(description="Full thread object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="39aaae31-3eb1-44c6-9e37-5a44a4529649",
            description="Get a thread by ID across the organization (org-wide)",
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
