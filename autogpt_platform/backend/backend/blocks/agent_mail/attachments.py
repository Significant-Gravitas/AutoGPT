"""
AgentMail Attachment blocks — download file attachments from messages and threads.

Attachments are files associated with messages (PDFs, CSVs, images, etc.).
To send attachments, include them in the attachments parameter when using
AgentMailSendMessageBlock or AgentMailReplyToMessageBlock.

To download, first get the attachment_id from a message's attachments array,
then use these blocks to retrieve the file content as base64.
"""

import base64

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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        data = client.inboxes.messages.get_attachment(
            inbox_id=input_data.inbox_id,
            message_id=input_data.message_id,
            attachment_id=input_data.attachment_id,
        )
        encoded = base64.b64encode(data).decode() if isinstance(data, bytes) else str(data)

        yield "content_base64", encoded
        yield "attachment_id", input_data.attachment_id


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
        thread_id: str = SchemaField(
            description="Thread ID containing the attachment"
        )
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        client = _client(credentials)
        data = client.inboxes.threads.get_attachment(
            inbox_id=input_data.inbox_id,
            thread_id=input_data.thread_id,
            attachment_id=input_data.attachment_id,
        )
        encoded = base64.b64encode(data).decode() if isinstance(data, bytes) else str(data)

        yield "content_base64", encoded
        yield "attachment_id", input_data.attachment_id
