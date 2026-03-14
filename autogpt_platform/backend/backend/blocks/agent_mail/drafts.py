"""
AgentMail Draft blocks — create, get, list, update, send, delete drafts
(per-inbox and org-wide).
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


class AgentMailCreateDraftBlock(Block):
    """Creates a new draft email in an AgentMail inbox, optionally scheduled for later."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to create the draft in"
        )
        to: list[str] = SchemaField(description="Recipient email addresses")
        subject: str = SchemaField(
            description="Email subject line", default=""
        )
        text: str = SchemaField(
            description="Plain text body", default=""
        )
        html: str = SchemaField(
            description="HTML body (optional)", default="", advanced=True
        )
        cc: list[str] = SchemaField(
            description="CC recipients", default_factory=list, advanced=True
        )
        bcc: list[str] = SchemaField(
            description="BCC recipients", default_factory=list, advanced=True
        )
        in_reply_to: str = SchemaField(
            description="Message ID this draft is replying to (optional)",
            default="",
            advanced=True,
        )
        send_at: str = SchemaField(
            description="ISO 8601 datetime to schedule sending (optional, e.g. 2025-01-15T09:00:00Z)",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        draft_id: str = SchemaField(description="The ID of the created draft")
        send_status: str = SchemaField(
            description="Send status (scheduled, sending, failed, or empty if not scheduled)",
            default="",
        )
        result: dict = SchemaField(description="Full draft object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="25ac9086-69fd-48b8-b910-9dbe04b8f3bd",
            description="Create a draft email in an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
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

        draft = client.inboxes.drafts.create(input_data.inbox_id, **params)
        result = draft.__dict__ if hasattr(draft, "__dict__") else {}

        yield "draft_id", draft.draft_id
        yield "send_status", getattr(draft, "send_status", "")
        yield "result", result


class AgentMailGetDraftBlock(Block):
    """Retrieves a specific draft from an AgentMail inbox."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        draft_id: str = SchemaField(description="The draft ID to retrieve")

    class Output(BlockSchemaOutput):
        draft_id: str = SchemaField(description="The draft ID")
        subject: str = SchemaField(description="The draft subject", default="")
        send_status: str = SchemaField(description="Send status", default="")
        send_at: str = SchemaField(
            description="Scheduled send time", default=""
        )
        result: dict = SchemaField(description="Full draft object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="8e57780d-dc25-43d4-a0f4-1f02877b09fb",
            description="Get a draft from an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        draft = client.inboxes.drafts.get(
            inbox_id=input_data.inbox_id,
            draft_id=input_data.draft_id,
        )
        result = draft.__dict__ if hasattr(draft, "__dict__") else {}

        yield "draft_id", draft.draft_id
        yield "subject", getattr(draft, "subject", "")
        yield "send_status", getattr(draft, "send_status", "")
        yield "send_at", getattr(draft, "send_at", "")
        yield "result", result


class AgentMailListDraftsBlock(Block):
    """Lists drafts in an AgentMail inbox, with optional label filtering."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        limit: int = SchemaField(
            description="Maximum number of drafts to return",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Pagination token from a previous request",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Filter drafts by labels (e.g. ['scheduled'])",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        drafts: list[dict] = SchemaField(description="List of draft objects")
        count: int = SchemaField(description="Number of drafts returned")
        next_page_token: str = SchemaField(
            description="Token for fetching the next page", default=""
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="e84883b7-7c39-4c5c-88e8-0a72b078ea63",
            description="List drafts in an AgentMail inbox",
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

        response = client.inboxes.drafts.list(input_data.inbox_id, **params)
        drafts = [
            d.__dict__ if hasattr(d, "__dict__") else d
            for d in getattr(response, "drafts", [])
        ]

        yield "drafts", drafts
        yield "count", getattr(response, "count", len(drafts))
        yield "next_page_token", getattr(response, "next_page_token", "")


class AgentMailUpdateDraftBlock(Block):
    """Updates an existing draft in an AgentMail inbox (e.g. reschedule send time)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        draft_id: str = SchemaField(description="The draft ID to update")
        to: list[str] = SchemaField(
            description="Updated recipient list (optional)",
            default_factory=list,
        )
        subject: str = SchemaField(
            description="Updated subject (optional)", default=""
        )
        text: str = SchemaField(
            description="Updated plain text body (optional)", default=""
        )
        html: str = SchemaField(
            description="Updated HTML body (optional)", default="", advanced=True
        )
        send_at: str = SchemaField(
            description="Updated scheduled send time as ISO 8601 (optional)",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        draft_id: str = SchemaField(description="The updated draft ID")
        send_status: str = SchemaField(description="Send status", default="")
        result: dict = SchemaField(description="Full updated draft object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="351f6e51-695a-421a-9032-46a587b10336",
            description="Update a draft in an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        params: dict = {}
        if input_data.to:
            params["to"] = input_data.to
        if input_data.subject:
            params["subject"] = input_data.subject
        if input_data.text:
            params["text"] = input_data.text
        if input_data.html:
            params["html"] = input_data.html
        if input_data.send_at:
            params["send_at"] = input_data.send_at

        draft = client.inboxes.drafts.update(
            inbox_id=input_data.inbox_id,
            draft_id=input_data.draft_id,
            **params,
        )
        result = draft.__dict__ if hasattr(draft, "__dict__") else {}

        yield "draft_id", draft.draft_id
        yield "send_status", getattr(draft, "send_status", "")
        yield "result", result


class AgentMailSendDraftBlock(Block):
    """Sends a draft, converting it into a sent message. The draft is deleted after sending."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        draft_id: str = SchemaField(description="The draft ID to send")

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(
            description="The message ID of the sent email"
        )
        thread_id: str = SchemaField(description="The thread ID")
        result: dict = SchemaField(description="Full sent message object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="37c39e83-475d-4b3d-843a-d923d001b85a",
            description="Send a draft from an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        msg = client.inboxes.drafts.send(
            inbox_id=input_data.inbox_id,
            draft_id=input_data.draft_id,
        )
        result = msg.__dict__ if hasattr(msg, "__dict__") else {}

        yield "message_id", msg.message_id
        yield "thread_id", getattr(msg, "thread_id", "")
        yield "result", result


class AgentMailDeleteDraftBlock(Block):
    """Deletes a draft from an AgentMail inbox. Also cancels scheduled sends."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        draft_id: str = SchemaField(description="The draft ID to delete")

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the deletion was successful")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="9023eb99-3e2f-4def-808b-d9c584b3d9e7",
            description="Delete a draft from an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        client.inboxes.drafts.delete(
            inbox_id=input_data.inbox_id,
            draft_id=input_data.draft_id,
        )
        yield "success", True


class AgentMailListOrgDraftsBlock(Block):
    """Lists all drafts across the entire organization (org-wide)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        limit: int = SchemaField(
            description="Maximum number of drafts to return",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Pagination token from a previous request",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        drafts: list[dict] = SchemaField(description="List of draft objects")
        count: int = SchemaField(description="Number of drafts returned")
        next_page_token: str = SchemaField(
            description="Token for fetching the next page", default=""
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="ed7558ae-3a07-45f5-af55-a25fe88c9971",
            description="List all drafts across the organization (org-wide)",
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

        response = client.drafts.list(**params)
        drafts = [
            d.__dict__ if hasattr(d, "__dict__") else d
            for d in getattr(response, "drafts", [])
        ]

        yield "drafts", drafts
        yield "count", getattr(response, "count", len(drafts))
        yield "next_page_token", getattr(response, "next_page_token", "")
