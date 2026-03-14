"""
AgentMail Message blocks — send, list, get, reply, forward, and update messages.
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


class AgentMailSendMessageBlock(Block):
    """Sends a new email message from an AgentMail inbox, creating a new thread."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to send from"
        )
        to: str = SchemaField(description="Recipient email address")
        subject: str = SchemaField(description="Email subject line")
        text: str = SchemaField(description="Plain text body of the email")
        html: str = SchemaField(
            description="HTML body of the email (optional)",
            default="",
            advanced=True,
        )
        cc: str = SchemaField(
            description="CC recipient email address (optional)",
            default="",
            advanced=True,
        )
        bcc: str = SchemaField(
            description="BCC recipient email address (optional)",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Labels to apply to the message",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(description="The ID of the sent message")
        thread_id: str = SchemaField(description="The thread ID the message belongs to")
        result: dict = SchemaField(description="Full message object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b67469b2-7748-4d81-a223-4ebd332cca89",
            description="Send a new email from an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        params: dict = {
            "to": input_data.to,
            "subject": input_data.subject,
            "text": input_data.text,
        }
        if input_data.html:
            params["html"] = input_data.html
        if input_data.cc:
            params["cc"] = input_data.cc
        if input_data.bcc:
            params["bcc"] = input_data.bcc
        if input_data.labels:
            params["labels"] = input_data.labels

        msg = client.inboxes.messages.send(input_data.inbox_id, **params)
        result = msg.__dict__ if hasattr(msg, "__dict__") else {}

        yield "message_id", msg.message_id
        yield "thread_id", getattr(msg, "thread_id", "")
        yield "result", result


class AgentMailListMessagesBlock(Block):
    """Lists all messages in an AgentMail inbox, with optional label filtering."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to list messages from"
        )
        limit: int = SchemaField(
            description="Maximum number of messages to return",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Pagination token from a previous request",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Filter messages by labels (e.g. ['unread'])",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        messages: list[dict] = SchemaField(description="List of message objects")
        count: int = SchemaField(description="Number of messages returned")
        next_page_token: str = SchemaField(
            description="Token for fetching the next page", default=""
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="721234df-c7a2-4927-b205-744badbd5844",
            description="List messages in an AgentMail inbox",
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

        response = client.inboxes.messages.list(input_data.inbox_id, **params)
        messages = [
            m.__dict__ if hasattr(m, "__dict__") else m
            for m in getattr(response, "messages", [])
        ]

        yield "messages", messages
        yield "count", getattr(response, "count", len(messages))
        yield "next_page_token", getattr(response, "next_page_token", "")


class AgentMailGetMessageBlock(Block):
    """Retrieves a specific message from an AgentMail inbox."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        message_id: str = SchemaField(description="The message ID to retrieve")

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(description="The message ID")
        thread_id: str = SchemaField(description="The thread ID")
        subject: str = SchemaField(description="The email subject")
        text: str = SchemaField(description="Plain text body")
        extracted_text: str = SchemaField(
            description="Reply content without quoted history", default=""
        )
        html: str = SchemaField(description="HTML body", default="")
        result: dict = SchemaField(description="Full message object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="2788bdfa-1527-4603-a5e4-a455c05c032f",
            description="Get a specific message from an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        msg = client.inboxes.messages.get(
            inbox_id=input_data.inbox_id,
            message_id=input_data.message_id,
        )
        result = msg.__dict__ if hasattr(msg, "__dict__") else {}

        yield "message_id", msg.message_id
        yield "thread_id", getattr(msg, "thread_id", "")
        yield "subject", getattr(msg, "subject", "")
        yield "text", getattr(msg, "text", "")
        yield "extracted_text", getattr(msg, "extracted_text", "")
        yield "html", getattr(msg, "html", "")
        yield "result", result


class AgentMailReplyToMessageBlock(Block):
    """Replies to an existing message in an AgentMail inbox, keeping it in the same thread."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to reply from"
        )
        message_id: str = SchemaField(
            description="The message ID to reply to"
        )
        text: str = SchemaField(description="Plain text body of the reply")
        html: str = SchemaField(
            description="HTML body of the reply (optional)",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(description="The ID of the reply message")
        thread_id: str = SchemaField(description="The thread ID")
        result: dict = SchemaField(description="Full reply message object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b9fe53fa-5026-4547-9570-b54ccb487229",
            description="Reply to a message in an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        params: dict = {"text": input_data.text}
        if input_data.html:
            params["html"] = input_data.html

        reply = client.inboxes.messages.reply(
            inbox_id=input_data.inbox_id,
            message_id=input_data.message_id,
            **params,
        )
        result = reply.__dict__ if hasattr(reply, "__dict__") else {}

        yield "message_id", reply.message_id
        yield "thread_id", getattr(reply, "thread_id", "")
        yield "result", result


class AgentMailForwardMessageBlock(Block):
    """Forwards an existing message from an AgentMail inbox to another recipient."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address to forward from"
        )
        message_id: str = SchemaField(
            description="The message ID to forward"
        )
        to: str = SchemaField(description="Recipient email address to forward to")
        subject: str = SchemaField(
            description="Override subject line (optional)",
            default="",
            advanced=True,
        )
        text: str = SchemaField(
            description="Additional plain text to prepend (optional)",
            default="",
            advanced=True,
        )
        html: str = SchemaField(
            description="Additional HTML to prepend (optional)",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(description="The ID of the forwarded message")
        thread_id: str = SchemaField(description="The thread ID")
        result: dict = SchemaField(description="Full forwarded message object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b70c7e33-5d66-4f8e-897f-ac73a7bfce82",
            description="Forward a message from an AgentMail inbox",
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

        fwd = client.inboxes.messages.forward(
            inbox_id=input_data.inbox_id,
            message_id=input_data.message_id,
            **params,
        )
        result = fwd.__dict__ if hasattr(fwd, "__dict__") else {}

        yield "message_id", fwd.message_id
        yield "thread_id", getattr(fwd, "thread_id", "")
        yield "result", result


class AgentMailUpdateMessageBlock(Block):
    """Updates labels on a message in an AgentMail inbox (e.g. mark as read)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API credentials"
        )
        inbox_id: str = SchemaField(
            description="The inbox ID or email address"
        )
        message_id: str = SchemaField(description="The message ID to update")
        add_labels: list[str] = SchemaField(
            description="Labels to add to the message",
            default_factory=list,
        )
        remove_labels: list[str] = SchemaField(
            description="Labels to remove from the message",
            default_factory=list,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(description="The updated message ID")
        result: dict = SchemaField(description="Full updated message object")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="694ff816-4c89-4a5e-a552-8c31be187735",
            description="Update labels on a message in an AgentMail inbox",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        params: dict = {}
        if input_data.add_labels:
            params["add_labels"] = input_data.add_labels
        if input_data.remove_labels:
            params["remove_labels"] = input_data.remove_labels

        msg = client.inboxes.messages.update(
            inbox_id=input_data.inbox_id,
            message_id=input_data.message_id,
            **params,
        )
        result = msg.__dict__ if hasattr(msg, "__dict__") else {}

        yield "message_id", msg.message_id
        yield "result", result
