"""
AgentMail Draft blocks — create, get, list, update, send, and delete drafts.

A Draft is an unsent message that can be reviewed, edited, and sent later.
Drafts enable human-in-the-loop review, scheduled sending (via send_at),
and complex multi-step email composition workflows.
"""

from typing import Optional

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


class AgentMailCreateDraftBlock(Block):
    """
    Create a draft email in an AgentMail inbox for review or scheduled sending.

    Drafts let agents prepare emails without sending immediately. Use send_at
    to schedule automatic sending at a future time (ISO 8601 format).
    Scheduled drafts are auto-labeled 'scheduled' and can be cancelled by
    deleting the draft.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to create the draft in"
        )
        to: list[str] = SchemaField(
            description="Recipient email addresses (e.g. ['user@example.com'])"
        )
        subject: str = SchemaField(description="Email subject line", default="")
        text: str = SchemaField(description="Plain text body of the draft", default="")
        html: str = SchemaField(
            description="Rich HTML body of the draft", default="", advanced=True
        )
        cc: list[str] = SchemaField(
            description="CC recipient email addresses",
            default_factory=list,
            advanced=True,
        )
        bcc: list[str] = SchemaField(
            description="BCC recipient email addresses",
            default_factory=list,
            advanced=True,
        )
        in_reply_to: str = SchemaField(
            description="Message ID this draft replies to, for threading follow-up drafts",
            default="",
            advanced=True,
        )
        send_at: str = SchemaField(
            description="Schedule automatic sending at this ISO 8601 datetime (e.g. '2025-01-15T09:00:00Z'). Leave empty for manual send.",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        draft_id: str = SchemaField(
            description="Unique identifier of the created draft"
        )
        send_status: str = SchemaField(
            description="'scheduled' if send_at was set, empty otherwise. Values: scheduled, sending, failed.",
            default="",
        )
        result: dict = SchemaField(
            description="Complete draft object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="25ac9086-69fd-48b8-b910-9dbe04b8f3bd",
            description="Create a draft email for review or scheduled sending. Use send_at for automatic future delivery.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "to": ["user@example.com"],
            },
            test_output=[
                ("draft_id", "mock-draft-id"),
                ("send_status", ""),
                ("result", dict),
            ],
            test_mock={
                "create_draft": lambda *a, **kw: type(
                    "Draft",
                    (),
                    {
                        "draft_id": "mock-draft-id",
                        "send_status": "",
                        "model_dump": lambda self: {"draft_id": "mock-draft-id"},
                    },
                )(),
            },
        )

    @staticmethod
    async def create_draft(credentials: APIKeyCredentials, inbox_id: str, **params):
        client = _client(credentials)
        return await client.inboxes.drafts.create(inbox_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"to": input_data.to}
            if input_data.subject:
                params["subject"] = input_data.subject
            if input_data.text:
                params["text"] = input_data.text
            if input_data.html:
                params["html"] = input_data.html
            if input_data.cc:
                params["cc"] = input_data.cc
            if input_data.bcc:
                params["bcc"] = input_data.bcc
            if input_data.in_reply_to:
                params["in_reply_to"] = input_data.in_reply_to
            if input_data.send_at:
                params["send_at"] = input_data.send_at

            draft = await self.create_draft(credentials, input_data.inbox_id, **params)
            result = draft.model_dump()

            yield "draft_id", draft.draft_id
            yield "send_status", draft.send_status or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailGetDraftBlock(Block):
    """
    Retrieve a specific draft from an AgentMail inbox.

    Returns the draft contents including recipients, subject, body, and
    scheduled send status. Use this to review a draft before approving it.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the draft belongs to"
        )
        draft_id: str = SchemaField(description="Draft ID to retrieve")

    class Output(BlockSchemaOutput):
        draft_id: str = SchemaField(description="Unique identifier of the draft")
        subject: str = SchemaField(description="Draft subject line", default="")
        send_status: str = SchemaField(
            description="Scheduled send status: 'scheduled', 'sending', 'failed', or empty",
            default="",
        )
        send_at: str = SchemaField(
            description="Scheduled send time (ISO 8601) if set", default=""
        )
        result: dict = SchemaField(description="Complete draft object with all fields")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="8e57780d-dc25-43d4-a0f4-1f02877b09fb",
            description="Retrieve a draft email to review its contents, recipients, and scheduled send status.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "draft_id": "test-draft",
            },
            test_output=[
                ("draft_id", "test-draft"),
                ("subject", ""),
                ("send_status", ""),
                ("send_at", ""),
                ("result", dict),
            ],
            test_mock={
                "get_draft": lambda *a, **kw: type(
                    "Draft",
                    (),
                    {
                        "draft_id": "test-draft",
                        "subject": "",
                        "send_status": "",
                        "send_at": "",
                        "model_dump": lambda self: {"draft_id": "test-draft"},
                    },
                )(),
            },
        )

    @staticmethod
    async def get_draft(credentials: APIKeyCredentials, inbox_id: str, draft_id: str):
        client = _client(credentials)
        return await client.inboxes.drafts.get(inbox_id=inbox_id, draft_id=draft_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            draft = await self.get_draft(
                credentials, input_data.inbox_id, input_data.draft_id
            )
            result = draft.model_dump()

            yield "draft_id", draft.draft_id
            yield "subject", draft.subject or ""
            yield "send_status", draft.send_status or ""
            yield "send_at", draft.send_at or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailListDraftsBlock(Block):
    """
    List all drafts in an AgentMail inbox with optional label filtering.

    Use labels=['scheduled'] to find all drafts queued for future sending.
    Useful for building approval dashboards or monitoring pending outreach.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to list drafts from"
        )
        limit: int = SchemaField(
            description="Maximum number of drafts to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Filter drafts by labels (e.g. ['scheduled'] for pending sends)",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        drafts: list[dict] = SchemaField(
            description="List of draft objects with subject, recipients, send_status, etc."
        )
        count: int = SchemaField(description="Number of drafts returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="e84883b7-7c39-4c5c-88e8-0a72b078ea63",
            description="List drafts in an AgentMail inbox. Filter by labels=['scheduled'] to find pending sends.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
            },
            test_output=[
                ("drafts", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_drafts": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "drafts": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_drafts(credentials: APIKeyCredentials, inbox_id: str, **params):
        client = _client(credentials)
        return await client.inboxes.drafts.list(inbox_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token
            if input_data.labels:
                params["labels"] = input_data.labels

            response = await self.list_drafts(
                credentials, input_data.inbox_id, **params
            )
            drafts = [d.model_dump() for d in response.drafts]

            yield "drafts", drafts
            yield "count", response.count
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


class AgentMailUpdateDraftBlock(Block):
    """
    Update an existing draft's content, recipients, or scheduled send time.

    Use this to reschedule a draft (change send_at), modify recipients,
    or edit the subject/body before sending. To cancel a scheduled send,
    delete the draft instead.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the draft belongs to"
        )
        draft_id: str = SchemaField(description="Draft ID to update")
        to: Optional[list[str]] = SchemaField(
            description="Updated recipient email addresses (replaces existing list). Omit to keep current value.",
            default=None,
        )
        subject: Optional[str] = SchemaField(
            description="Updated subject line. Omit to keep current value.",
            default=None,
        )
        text: Optional[str] = SchemaField(
            description="Updated plain text body. Omit to keep current value.",
            default=None,
        )
        html: Optional[str] = SchemaField(
            description="Updated HTML body. Omit to keep current value.",
            default=None,
            advanced=True,
        )
        send_at: Optional[str] = SchemaField(
            description="Reschedule: new ISO 8601 send time (e.g. '2025-01-20T14:00:00Z'). Omit to keep current value.",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        draft_id: str = SchemaField(description="The updated draft ID")
        send_status: str = SchemaField(description="Updated send status", default="")
        result: dict = SchemaField(description="Complete updated draft object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="351f6e51-695a-421a-9032-46a587b10336",
            description="Update a draft's content, recipients, or scheduled send time. Use to reschedule or edit before sending.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "draft_id": "test-draft",
            },
            test_output=[
                ("draft_id", "test-draft"),
                ("send_status", ""),
                ("result", dict),
            ],
            test_mock={
                "update_draft": lambda *a, **kw: type(
                    "Draft",
                    (),
                    {
                        "draft_id": "test-draft",
                        "send_status": "",
                        "model_dump": lambda self: {"draft_id": "test-draft"},
                    },
                )(),
            },
        )

    @staticmethod
    async def update_draft(
        credentials: APIKeyCredentials, inbox_id: str, draft_id: str, **params
    ):
        client = _client(credentials)
        return await client.inboxes.drafts.update(
            inbox_id=inbox_id, draft_id=draft_id, **params
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {}
            if input_data.to is not None:
                params["to"] = input_data.to
            if input_data.subject is not None:
                params["subject"] = input_data.subject
            if input_data.text is not None:
                params["text"] = input_data.text
            if input_data.html is not None:
                params["html"] = input_data.html
            if input_data.send_at is not None:
                params["send_at"] = input_data.send_at

            draft = await self.update_draft(
                credentials, input_data.inbox_id, input_data.draft_id, **params
            )
            result = draft.model_dump()

            yield "draft_id", draft.draft_id
            yield "send_status", draft.send_status or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailSendDraftBlock(Block):
    """
    Send a draft immediately, converting it into a delivered message.

    The draft is deleted after successful sending and becomes a regular
    message with a message_id. Use this for human-in-the-loop approval
    workflows: agent creates draft, human reviews, then this block sends it.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the draft belongs to"
        )
        draft_id: str = SchemaField(description="Draft ID to send now")

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(
            description="Message ID of the now-sent email (draft is deleted)"
        )
        thread_id: str = SchemaField(
            description="Thread ID the sent message belongs to"
        )
        result: dict = SchemaField(description="Complete sent message object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="37c39e83-475d-4b3d-843a-d923d001b85a",
            description="Send a draft immediately, converting it into a delivered message. The draft is deleted after sending.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "draft_id": "test-draft",
            },
            test_output=[
                ("message_id", "mock-msg-id"),
                ("thread_id", "mock-thread-id"),
                ("result", dict),
            ],
            test_mock={
                "send_draft": lambda *a, **kw: type(
                    "Msg",
                    (),
                    {
                        "message_id": "mock-msg-id",
                        "thread_id": "mock-thread-id",
                        "model_dump": lambda self: {"message_id": "mock-msg-id"},
                    },
                )(),
            },
        )

    @staticmethod
    async def send_draft(credentials: APIKeyCredentials, inbox_id: str, draft_id: str):
        client = _client(credentials)
        return await client.inboxes.drafts.send(inbox_id=inbox_id, draft_id=draft_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            msg = await self.send_draft(
                credentials, input_data.inbox_id, input_data.draft_id
            )
            result = msg.model_dump()

            yield "message_id", msg.message_id
            yield "thread_id", msg.thread_id or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailDeleteDraftBlock(Block):
    """
    Delete a draft from an AgentMail inbox. Also cancels any scheduled send.

    If the draft was scheduled with send_at, deleting it cancels the
    scheduled delivery. This is the way to cancel a scheduled email.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the draft belongs to"
        )
        draft_id: str = SchemaField(
            description="Draft ID to delete (also cancels scheduled sends)"
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the draft was successfully deleted/cancelled"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="9023eb99-3e2f-4def-808b-d9c584b3d9e7",
            description="Delete a draft or cancel a scheduled email. Removes the draft permanently.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "draft_id": "test-draft",
            },
            test_output=[("success", True)],
            test_mock={
                "delete_draft": lambda *a, **kw: None,
            },
        )

    @staticmethod
    async def delete_draft(
        credentials: APIKeyCredentials, inbox_id: str, draft_id: str
    ):
        client = _client(credentials)
        await client.inboxes.drafts.delete(inbox_id=inbox_id, draft_id=draft_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            await self.delete_draft(
                credentials, input_data.inbox_id, input_data.draft_id
            )
            yield "success", True
        except Exception as e:
            yield "error", str(e)


class AgentMailListOrgDraftsBlock(Block):
    """
    List all drafts across every inbox in your organization.

    Returns drafts from all inboxes in one query. Perfect for building
    a central approval dashboard where a human supervisor can review
    and approve any draft created by any agent.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        limit: int = SchemaField(
            description="Maximum number of drafts to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        drafts: list[dict] = SchemaField(
            description="List of draft objects from all inboxes in the organization"
        )
        count: int = SchemaField(description="Number of drafts returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="ed7558ae-3a07-45f5-af55-a25fe88c9971",
            description="List all drafts across every inbox in your organization. Use for central approval dashboards.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("drafts", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_org_drafts": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "drafts": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_org_drafts(credentials: APIKeyCredentials, **params):
        client = _client(credentials)
        return await client.drafts.list(**params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token

            response = await self.list_org_drafts(credentials, **params)
            drafts = [d.model_dump() for d in response.drafts]

            yield "drafts", drafts
            yield "count", response.count
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)
