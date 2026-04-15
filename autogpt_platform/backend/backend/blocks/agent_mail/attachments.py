"""
AgentMail Attachment blocks — download file attachments from messages and threads.

Attachments are files associated with messages (PDFs, CSVs, images, etc.).
To send attachments, include them in the attachments parameter when using
AgentMailSendMessageBlock or AgentMailReplyToMessageBlock.

To download, first get the attachment_id from a message's attachments array,
then use these blocks to retrieve the file content as base64.
"""

import base64

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


class AgentMailGetMessageAttachmentBlock(Block):
    """
    Download a file attachment from a specific email message.

    Retrieves the raw file content and returns it as base64-encoded data.
    First get the attachment_id from a message object's attachments array,
    then use this block to download the file.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the message belongs to"
        )
        message_id: str = SchemaField(
            description="Message ID containing the attachment"
        )
        attachment_id: str = SchemaField(
            description="Attachment ID to download (from the message's attachments array)"
        )

    class Output(BlockSchemaOutput):
        content_base64: str = SchemaField(
            description="File content encoded as a base64 string. Decode with base64.b64decode() to get raw bytes."
        )
        attachment_id: str = SchemaField(
            description="The attachment ID that was downloaded"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="a283ffc4-8087-4c3d-9135-8f26b86742ec",
            description="Download a file attachment from an email message. Returns base64-encoded file content.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "message_id": "test-msg",
                "attachment_id": "test-attach",
            },
            test_output=[
                ("content_base64", "dGVzdA=="),
                ("attachment_id", "test-attach"),
            ],
            test_mock={
                "get_attachment": lambda *a, **kw: b"test",
            },
        )

    @staticmethod
    async def get_attachment(
        credentials: APIKeyCredentials,
        inbox_id: str,
        message_id: str,
        attachment_id: str,
    ):
        client = _client(credentials)
        return await client.inboxes.messages.get_attachment(
            inbox_id=inbox_id,
            message_id=message_id,
            attachment_id=attachment_id,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            data = await self.get_attachment(
                credentials=credentials,
                inbox_id=input_data.inbox_id,
                message_id=input_data.message_id,
                attachment_id=input_data.attachment_id,
            )
            if isinstance(data, bytes):
                encoded = base64.b64encode(data).decode()
            elif isinstance(data, str):
                encoded = base64.b64encode(data.encode("utf-8")).decode()
            else:
                raise TypeError(
                    f"Unexpected attachment data type: {type(data).__name__}"
                )

            yield "content_base64", encoded
            yield "attachment_id", input_data.attachment_id
        except Exception as e:
            yield "error", str(e)


class AgentMailGetThreadAttachmentBlock(Block):
    """
    Download a file attachment from a conversation thread.

    Same as GetMessageAttachment but looks up by thread ID instead of
    message ID. Useful when you know the thread but not the specific
    message containing the attachment.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = agent_mail.credentials_field(
            description="AgentMail API key from https://console.agentmail.to"
        )
        inbox_id: str = SchemaField(
            description="Inbox ID or email address the thread belongs to"
        )
        thread_id: str = SchemaField(description="Thread ID containing the attachment")
        attachment_id: str = SchemaField(
            description="Attachment ID to download (from a message's attachments array within the thread)"
        )

    class Output(BlockSchemaOutput):
        content_base64: str = SchemaField(
            description="File content encoded as a base64 string. Decode with base64.b64decode() to get raw bytes."
        )
        attachment_id: str = SchemaField(
            description="The attachment ID that was downloaded"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="06b6a4c4-9d71-4992-9e9c-cf3b352763b5",
            description="Download a file attachment from a conversation thread. Returns base64-encoded file content.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_id": "test-inbox",
                "thread_id": "test-thread",
                "attachment_id": "test-attach",
            },
            test_output=[
                ("content_base64", "dGVzdA=="),
                ("attachment_id", "test-attach"),
            ],
            test_mock={
                "get_attachment": lambda *a, **kw: b"test",
            },
        )

    @staticmethod
    async def get_attachment(
        credentials: APIKeyCredentials,
        inbox_id: str,
        thread_id: str,
        attachment_id: str,
    ):
        client = _client(credentials)
        return await client.inboxes.threads.get_attachment(
            inbox_id=inbox_id,
            thread_id=thread_id,
            attachment_id=attachment_id,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            data = await self.get_attachment(
                credentials=credentials,
                inbox_id=input_data.inbox_id,
                thread_id=input_data.thread_id,
                attachment_id=input_data.attachment_id,
            )
            if isinstance(data, bytes):
                encoded = base64.b64encode(data).decode()
            elif isinstance(data, str):
                encoded = base64.b64encode(data.encode("utf-8")).decode()
            else:
                raise TypeError(
                    f"Unexpected attachment data type: {type(data).__name__}"
                )

            yield "content_base64", encoded
            yield "attachment_id", input_data.attachment_id
        except Exception as e:
            yield "error", str(e)
