"""
AgentMail Thread blocks — list, get, and delete conversation threads.

A Thread groups related messages into a single conversation. Threads are
created automatically when a new message is sent and grow as replies are added.
Threads can be queried per-inbox or across the entire organization.
"""

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
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
            },
            test_output=[
                ("threads", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_threads": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "threads": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_threads(credentials: APIKeyCredentials, inbox_id: str, **params):
        client = _client(credentials)
        return await client.inboxes.threads.list(inbox_id=inbox_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token
            if input_data.labels:
                params["labels"] = input_data.labels

            response = await self.list_threads(
                credentials, input_data.inbox_id, **params
            )
            threads = [t.model_dump() for t in response.threads]

            yield "threads", threads
            yield "count", (c if (c := response.count) is not None else len(threads))
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


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
        thread_id: str = SchemaField(description="Thread ID to retrieve")

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
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "thread_id": "test-thread",
            },
            test_output=[
                ("thread_id", "test-thread"),
                ("messages", []),
                ("result", dict),
            ],
            test_mock={
                "get_thread": lambda *a, **kw: type(
                    "Thread",
                    (),
                    {
                        "thread_id": "test-thread",
                        "messages": [],
                        "model_dump": lambda self: {
                            "thread_id": "test-thread",
                            "messages": [],
                        },
                    },
                )(),
            },
        )

    @staticmethod
    async def get_thread(credentials: APIKeyCredentials, inbox_id: str, thread_id: str):
        client = _client(credentials)
        return await client.inboxes.threads.get(inbox_id=inbox_id, thread_id=thread_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            thread = await self.get_thread(
                credentials, input_data.inbox_id, input_data.thread_id
            )
            messages = [m.model_dump() for m in thread.messages]
            result = thread.model_dump()
            result["messages"] = messages

            yield "thread_id", thread.thread_id
            yield "messages", messages
            yield "result", result
        except Exception as e:
            yield "error", str(e)


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
        thread_id: str = SchemaField(description="Thread ID to permanently delete")

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
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "thread_id": "test-thread",
            },
            test_output=[("success", True)],
            test_mock={
                "delete_thread": lambda *a, **kw: None,
            },
        )

    @staticmethod
    async def delete_thread(
        credentials: APIKeyCredentials, inbox_id: str, thread_id: str
    ):
        client = _client(credentials)
        await client.inboxes.threads.delete(inbox_id=inbox_id, thread_id=thread_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            await self.delete_thread(
                credentials, input_data.inbox_id, input_data.thread_id
            )
            yield "success", True
        except Exception as e:
            yield "error", str(e)


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
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("threads", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_org_threads": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "threads": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_org_threads(credentials: APIKeyCredentials, **params):
        client = _client(credentials)
        return await client.threads.list(**params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token
            if input_data.labels:
                params["labels"] = input_data.labels

            response = await self.list_org_threads(credentials, **params)
            threads = [t.model_dump() for t in response.threads]

            yield "threads", threads
            yield "count", (c if (c := response.count) is not None else len(threads))
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


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
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "thread_id": "test-thread",
            },
            test_output=[
                ("thread_id", "test-thread"),
                ("messages", []),
                ("result", dict),
            ],
            test_mock={
                "get_org_thread": lambda *a, **kw: type(
                    "Thread",
                    (),
                    {
                        "thread_id": "test-thread",
                        "messages": [],
                        "model_dump": lambda self: {
                            "thread_id": "test-thread",
                            "messages": [],
                        },
                    },
                )(),
            },
        )

    @staticmethod
    async def get_org_thread(credentials: APIKeyCredentials, thread_id: str):
        client = _client(credentials)
        return await client.threads.get(thread_id=thread_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            thread = await self.get_org_thread(credentials, input_data.thread_id)
            messages = [m.model_dump() for m in thread.messages]
            result = thread.model_dump()
            result["messages"] = messages

            yield "thread_id", thread.thread_id
            yield "messages", messages
            yield "result", result
        except Exception as e:
            yield "error", str(e)
