"""
AgentMail Message blocks — send, list, get, reply, forward, and update messages.

A Message is an individual email within a Thread. Agents can send new messages
(which create threads), reply to existing messages, forward them, and manage
labels for state tracking (e.g. read/unread, campaign tags).
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


class AgentMailSendMessageBlock(Block):
    """
    Send a new email from an AgentMail inbox, automatically creating a new thread.

    Supports plain text and HTML bodies, CC/BCC recipients, and labels for
    organizing messages (e.g. campaign tracking, state management).
    Max 50 combined recipients across to, cc, and bcc.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to send from (e.g. 'agent@agentmail.to')"
        )
        to: list[str] = SchemaField(
            description="Recipient email addresses (e.g. ['user@example.com'])"
        )
        subject: str = SchemaField(description="Email subject line")
        text: str = SchemaField(
            description="Plain text body of the email. Always provide this as a fallback for email clients that don't render HTML."
        )
        html: str = SchemaField(
            description="Rich HTML body of the email. Embed CSS in a <style> tag for best compatibility across email clients.",
            default="",
            advanced=True,
        )
        cc: list[str] = SchemaField(
            description="CC recipient email addresses for human-in-the-loop oversight",
            default_factory=list,
            advanced=True,
        )
        bcc: list[str] = SchemaField(
            description="BCC recipient email addresses (hidden from other recipients)",
            default_factory=list,
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Labels to tag the message for filtering and state management (e.g. ['outreach', 'q4-campaign'])",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(
            description="Unique identifier of the sent message"
        )
        thread_id: str = SchemaField(
            description="Thread ID grouping this message and any future replies"
        )
        result: dict = SchemaField(
            description="Complete sent message object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b67469b2-7748-4d81-a223-4ebd332cca89",
            description="Send a new email from an AgentMail inbox. Creates a new conversation thread. Supports HTML, CC/BCC, and labels.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "to": ["user@example.com"],
                "subject": "Test",
                "text": "Hello",
            },
            test_output=[
                ("message_id", "mock-msg-id"),
                ("thread_id", "mock-thread-id"),
                ("result", dict),
            ],
            test_mock={
                "send_message": lambda *a, **kw: type(
                    "Msg",
                    (),
                    {
                        "message_id": "mock-msg-id",
                        "thread_id": "mock-thread-id",
                        "model_dump": lambda self: {
                            "message_id": "mock-msg-id",
                            "thread_id": "mock-thread-id",
                        },
                    },
                )(),
            },
        )

    @staticmethod
    async def send_message(credentials: APIKeyCredentials, inbox_id: str, **params):
        client = _client(credentials)
        return await client.inboxes.messages.send(inbox_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            total = len(input_data.to) + len(input_data.cc) + len(input_data.bcc)
            if total > 50:
                raise ValueError(
                    f"Max 50 combined recipients across to, cc, and bcc (got {total})"
                )

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

            msg = await self.send_message(credentials, input_data.inbox_id, **params)
            result = msg.model_dump()

            yield "message_id", msg.message_id
            yield "thread_id", msg.thread_id or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailListMessagesBlock(Block):
    """
    List all messages in an AgentMail inbox with optional label filtering.

    Returns a paginated list of messages. Use labels to filter (e.g.
    labels=['unread'] to only get unprocessed messages). Useful for
    polling workflows or building inbox views.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to list messages from"
        )
        limit: int = SchemaField(
            description="Maximum number of messages to return per page (1-100)",
            default=20,
            advanced=True,
        )
        page_token: str = SchemaField(
            description="Token from a previous response to fetch the next page",
            default="",
            advanced=True,
        )
        labels: list[str] = SchemaField(
            description="Only return messages with ALL of these labels (e.g. ['unread'] or ['q4-campaign', 'follow-up'])",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        messages: list[dict] = SchemaField(
            description="List of message objects with subject, sender, text, html, labels, etc."
        )
        count: int = SchemaField(description="Number of messages returned")
        next_page_token: str = SchemaField(
            description="Token for the next page. Empty if no more results.",
            default="",
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="721234df-c7a2-4927-b205-744badbd5844",
            description="List messages in an AgentMail inbox. Filter by labels to find unread, campaign-tagged, or categorized messages.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
            },
            test_output=[
                ("messages", []),
                ("count", 0),
                ("next_page_token", ""),
            ],
            test_mock={
                "list_messages": lambda *a, **kw: type(
                    "Resp",
                    (),
                    {
                        "messages": [],
                        "count": 0,
                        "next_page_token": "",
                    },
                )(),
            },
        )

    @staticmethod
    async def list_messages(credentials: APIKeyCredentials, inbox_id: str, **params):
        client = _client(credentials)
        return await client.inboxes.messages.list(inbox_id, **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"limit": input_data.limit}
            if input_data.page_token:
                params["page_token"] = input_data.page_token
            if input_data.labels:
                params["labels"] = input_data.labels

            response = await self.list_messages(
                credentials, input_data.inbox_id, **params
            )
            messages = [m.model_dump() for m in response.messages]

            yield "messages", messages
            yield "count", (c if (c := response.count) is not None else len(messages))
            yield "next_page_token", response.next_page_token or ""
        except Exception as e:
            yield "error", str(e)


class AgentMailGetMessageBlock(Block):
    """
    Retrieve a specific email message by ID from an AgentMail inbox.

    Returns the full message including subject, body (text and HTML),
    sender, recipients, and attachments. Use extracted_text to get
    only the new reply content without quoted history.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the message belongs to"
        )
        message_id: str = SchemaField(
            description="Message ID to retrieve (e.g. '<abc123@agentmail.to>')"
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(description="Unique identifier of the message")
        thread_id: str = SchemaField(description="Thread this message belongs to")
        subject: str = SchemaField(description="Email subject line")
        text: str = SchemaField(
            description="Full plain text body (may include quoted reply history)"
        )
        extracted_text: str = SchemaField(
            description="Just the new reply content with quoted history stripped. Best for AI processing.",
            default="",
        )
        html: str = SchemaField(description="HTML body of the email", default="")
        result: dict = SchemaField(
            description="Complete message object with all fields including sender, recipients, attachments, labels"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="2788bdfa-1527-4603-a5e4-a455c05c032f",
            description="Retrieve a specific email message by ID. Includes extracted_text for clean reply content without quoted history.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "message_id": "test-msg",
            },
            test_output=[
                ("message_id", "test-msg"),
                ("thread_id", "t1"),
                ("subject", "Hi"),
                ("text", "Hello"),
                ("extracted_text", "Hello"),
                ("html", ""),
                ("result", dict),
            ],
            test_mock={
                "get_message": lambda *a, **kw: type(
                    "Msg",
                    (),
                    {
                        "message_id": "test-msg",
                        "thread_id": "t1",
                        "subject": "Hi",
                        "text": "Hello",
                        "extracted_text": "Hello",
                        "html": "",
                        "model_dump": lambda self: {"message_id": "test-msg"},
                    },
                )(),
            },
        )

    @staticmethod
    async def get_message(
        credentials: APIKeyCredentials,
        inbox_id: str,
        message_id: str,
    ):
        client = _client(credentials)
        return await client.inboxes.messages.get(
            inbox_id=inbox_id, message_id=message_id
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            msg = await self.get_message(
                credentials, input_data.inbox_id, input_data.message_id
            )
            result = msg.model_dump()

            yield "message_id", msg.message_id
            yield "thread_id", msg.thread_id or ""
            yield "subject", msg.subject or ""
            yield "text", msg.text or ""
            yield "extracted_text", msg.extracted_text or ""
            yield "html", msg.html or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailReplyToMessageBlock(Block):
    """
    Reply to an existing email message, keeping the reply in the same thread.

    The reply is automatically added to the same conversation thread as the
    original message. Use this for multi-turn agent conversations.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to send the reply from"
        )
        message_id: str = SchemaField(
            description="Message ID to reply to (e.g. '<abc123@agentmail.to>')"
        )
        text: str = SchemaField(description="Plain text body of the reply")
        html: str = SchemaField(
            description="Rich HTML body of the reply",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(
            description="Unique identifier of the reply message"
        )
        thread_id: str = SchemaField(description="Thread ID the reply was added to")
        result: dict = SchemaField(
            description="Complete reply message object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b9fe53fa-5026-4547-9570-b54ccb487229",
            description="Reply to an existing email in the same conversation thread. Use for multi-turn agent conversations.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "message_id": "test-msg",
                "text": "Reply",
            },
            test_output=[
                ("message_id", "mock-reply-id"),
                ("thread_id", "mock-thread-id"),
                ("result", dict),
            ],
            test_mock={
                "reply_to_message": lambda *a, **kw: type(
                    "Msg",
                    (),
                    {
                        "message_id": "mock-reply-id",
                        "thread_id": "mock-thread-id",
                        "model_dump": lambda self: {"message_id": "mock-reply-id"},
                    },
                )(),
            },
        )

    @staticmethod
    async def reply_to_message(
        credentials: APIKeyCredentials, inbox_id: str, message_id: str, **params
    ):
        client = _client(credentials)
        return await client.inboxes.messages.reply(
            inbox_id=inbox_id, message_id=message_id, **params
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            params: dict = {"text": input_data.text}
            if input_data.html:
                params["html"] = input_data.html

            reply = await self.reply_to_message(
                credentials,
                input_data.inbox_id,
                input_data.message_id,
                **params,
            )
            result = reply.model_dump()

            yield "message_id", reply.message_id
            yield "thread_id", reply.thread_id or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailForwardMessageBlock(Block):
    """
    Forward an existing email message to one or more recipients.

    Sends the original message content to different email addresses.
    Optionally prepend additional text or override the subject line.
    Max 50 combined recipients across to, cc, and bcc.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address to forward from"
        )
        message_id: str = SchemaField(description="Message ID to forward")
        to: list[str] = SchemaField(
            description="Recipient email addresses to forward the message to (e.g. ['user@example.com'])"
        )
        cc: list[str] = SchemaField(
            description="CC recipient email addresses",
            default_factory=list,
            advanced=True,
        )
        bcc: list[str] = SchemaField(
            description="BCC recipient email addresses (hidden from other recipients)",
            default_factory=list,
            advanced=True,
        )
        subject: str = SchemaField(
            description="Override the subject line (defaults to 'Fwd: <original subject>')",
            default="",
            advanced=True,
        )
        text: str = SchemaField(
            description="Additional plain text to prepend before the forwarded content",
            default="",
            advanced=True,
        )
        html: str = SchemaField(
            description="Additional HTML to prepend before the forwarded content",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(
            description="Unique identifier of the forwarded message"
        )
        thread_id: str = SchemaField(description="Thread ID of the forward")
        result: dict = SchemaField(
            description="Complete forwarded message object with all metadata"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b70c7e33-5d66-4f8e-897f-ac73a7bfce82",
            description="Forward an email message to one or more recipients. Supports CC/BCC and optional extra text or subject override.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "message_id": "test-msg",
                "to": ["user@example.com"],
            },
            test_output=[
                ("message_id", "mock-fwd-id"),
                ("thread_id", "mock-thread-id"),
                ("result", dict),
            ],
            test_mock={
                "forward_message": lambda *a, **kw: type(
                    "Msg",
                    (),
                    {
                        "message_id": "mock-fwd-id",
                        "thread_id": "mock-thread-id",
                        "model_dump": lambda self: {"message_id": "mock-fwd-id"},
                    },
                )(),
            },
        )

    @staticmethod
    async def forward_message(
        credentials: APIKeyCredentials, inbox_id: str, message_id: str, **params
    ):
        client = _client(credentials)
        return await client.inboxes.messages.forward(
            inbox_id=inbox_id, message_id=message_id, **params
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            total = len(input_data.to) + len(input_data.cc) + len(input_data.bcc)
            if total > 50:
                raise ValueError(
                    f"Max 50 combined recipients across to, cc, and bcc (got {total})"
                )

            params: dict = {"to": input_data.to}
            if input_data.cc:
                params["cc"] = input_data.cc
            if input_data.bcc:
                params["bcc"] = input_data.bcc
            if input_data.subject:
                params["subject"] = input_data.subject
            if input_data.text:
                params["text"] = input_data.text
            if input_data.html:
                params["html"] = input_data.html

            fwd = await self.forward_message(
                credentials,
                input_data.inbox_id,
                input_data.message_id,
                **params,
            )
            result = fwd.model_dump()

            yield "message_id", fwd.message_id
            yield "thread_id", fwd.thread_id or ""
            yield "result", result
        except Exception as e:
            yield "error", str(e)


class AgentMailUpdateMessageBlock(Block):
    """
    Add or remove labels on an email message for state management.

    Labels are string tags used to track message state (read/unread),
    categorize messages (billing, support), or tag campaigns (q4-outreach).
    Common pattern: add 'read' and remove 'unread' after processing a message.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the message belongs to"
        )
        message_id: str = SchemaField(description="Message ID to update labels on")
        add_labels: list[str] = SchemaField(
            description="Labels to add (e.g. ['read', 'processed', 'high-priority'])",
            default_factory=list,
        )
        remove_labels: list[str] = SchemaField(
            description="Labels to remove (e.g. ['unread', 'pending'])",
            default_factory=list,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(description="The updated message ID")
        result: dict = SchemaField(
            description="Complete updated message object with current labels"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="694ff816-4c89-4a5e-a552-8c31be187735",
            description="Add or remove labels on an email message. Use for read/unread tracking, campaign tagging, or state management.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "message_id": "test-msg",
                "add_labels": ["read"],
            },
            test_output=[
                ("message_id", "test-msg"),
                ("result", dict),
            ],
            test_mock={
                "update_message": lambda *a, **kw: type(
                    "Msg",
                    (),
                    {
                        "message_id": "test-msg",
                        "model_dump": lambda self: {"message_id": "test-msg"},
                    },
                )(),
            },
        )

    @staticmethod
    async def update_message(
        credentials: APIKeyCredentials, inbox_id: str, message_id: str, **params
    ):
        client = _client(credentials)
        return await client.inboxes.messages.update(
            inbox_id=inbox_id, message_id=message_id, **params
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            if not input_data.add_labels and not input_data.remove_labels:
                raise ValueError(
                    "Must specify at least one label operation: add_labels or remove_labels"
                )

            params: dict = {}
            if input_data.add_labels:
                params["add_labels"] = input_data.add_labels
            if input_data.remove_labels:
                params["remove_labels"] = input_data.remove_labels

            msg = await self.update_message(
                credentials,
                input_data.inbox_id,
                input_data.message_id,
                **params,
            )
            result = msg.model_dump()

            yield "message_id", msg.message_id
            yield "result", result
        except Exception as e:
            yield "error", str(e)
