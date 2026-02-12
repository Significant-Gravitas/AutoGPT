import asyncio
import base64
from abc import ABC
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.policy import SMTP
from email.utils import getaddresses, parseaddr
from pathlib import Path
from typing import List, Literal, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel, Field

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file
from backend.util.settings import Settings

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)

settings = Settings()

# No-wrap policy for plain text emails to prevent 78-char hard-wrap
NO_WRAP_POLICY = SMTP.clone(max_line_length=0)


def serialize_email_recipients(recipients: list[str]) -> str:
    """Serialize recipients list to comma-separated string."""
    return ", ".join(recipients)


def _make_mime_text(
    body: str,
    content_type: Optional[Literal["auto", "plain", "html"]] = None,
) -> MIMEText:
    """Create a MIMEText object with proper content type and no hard-wrap for plain text.

    This function addresses the common Gmail issue where plain text emails are
    hard-wrapped at 78 characters, creating awkward narrow columns in modern
    email clients. It also ensures HTML emails are properly identified and sent
    with the correct MIME type.

    Args:
        body: The email body content (plain text or HTML)
        content_type: The content type - "auto" (default), "plain", or "html"
                     - "auto" or None: Auto-detects based on presence of HTML tags
                     - "plain": Forces plain text format without line wrapping
                     - "html": Forces HTML format with standard wrapping

    Returns:
        MIMEText object configured with:
        - Appropriate content subtype (plain or html)
        - UTF-8 charset for proper Unicode support
        - No-wrap policy for plain text (max_line_length=0)
        - Standard wrapping for HTML content

    Examples:
        >>> # Plain text email without wrapping
        >>> mime = _make_mime_text("Long paragraph...", "plain")
        >>> # HTML email with auto-detection
        >>> mime = _make_mime_text("<p>Hello</p>", "auto")
    """
    # Auto-detect content type if not specified or "auto"
    if content_type is None or content_type == "auto":
        # Simple heuristic: check for HTML tags in first 500 chars
        looks_html = "<" in body[:500] and ">" in body[:500]
        actual_type = "html" if looks_html else "plain"
    else:
        actual_type = content_type

    # Create MIMEText with appropriate settings
    if actual_type == "html":
        # HTML content - normal wrapping is OK
        return MIMEText(body, _subtype="html", _charset="utf-8")
    else:
        # Plain text - use no-wrap policy to prevent 78-char hard-wrap
        return MIMEText(body, _subtype="plain", _charset="utf-8", policy=NO_WRAP_POLICY)


async def create_mime_message(
    input_data,
    execution_context: ExecutionContext,
) -> str:
    """Create a MIME message with attachments and return base64-encoded raw message."""

    message = MIMEMultipart()
    message["to"] = serialize_email_recipients(input_data.to)
    message["subject"] = input_data.subject

    if input_data.cc:
        message["cc"] = ", ".join(input_data.cc)
    if input_data.bcc:
        message["bcc"] = ", ".join(input_data.bcc)

    # Use the new helper function with content_type if available
    content_type = getattr(input_data, "content_type", None)
    message.attach(_make_mime_text(input_data.body, content_type))

    # Handle attachments if any
    if input_data.attachments:
        for attach in input_data.attachments:
            local_path = await store_media_file(
                file=attach,
                execution_context=execution_context,
                return_format="for_local_processing",
            )
            assert execution_context.graph_exec_id  # Validated by store_media_file
            abs_path = get_exec_file_path(execution_context.graph_exec_id, local_path)
            part = MIMEBase("application", "octet-stream")
            with open(abs_path, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={Path(abs_path).name}",
            )
            message.attach(part)

    return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")


class Attachment(BaseModel):
    filename: str
    content_type: str
    size: int
    attachment_id: str


class Email(BaseModel):
    threadId: str
    labelIds: list[str]
    id: str
    subject: str
    snippet: str
    from_: str
    to: list[str]  # List of recipient email addresses
    cc: list[str] = Field(default_factory=list)  # CC recipients
    bcc: list[str] = Field(
        default_factory=list
    )  # BCC recipients (rarely available in received emails)
    date: str
    body: str = ""  # Default to an empty string
    sizeEstimate: int
    attachments: List[Attachment]


class Thread(BaseModel):
    id: str
    messages: list[Email]
    historyId: str


class GmailSendResult(BaseModel):
    id: str
    status: str


class GmailDraftResult(BaseModel):
    id: str
    message_id: str
    status: str


class GmailLabelResult(BaseModel):
    label_id: str
    status: str


class Profile(BaseModel):
    emailAddress: str
    messagesTotal: int
    threadsTotal: int
    historyId: str


class GmailBase(Block, ABC):
    """Base class for Gmail blocks with common functionality."""

    def _build_service(self, credentials: GoogleCredentials, **kwargs):
        creds = Credentials(
            token=(
                credentials.access_token.get_secret_value()
                if credentials.access_token
                else None
            ),
            refresh_token=(
                credentials.refresh_token.get_secret_value()
                if credentials.refresh_token
                else None
            ),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.secrets.google_client_id,
            client_secret=settings.secrets.google_client_secret,
            scopes=credentials.scopes,
        )
        return build("gmail", "v1", credentials=creds)

    async def _get_email_body(self, msg, service):
        """Extract email body content with support for multipart messages and HTML conversion."""
        text = await self._walk_for_body(msg["payload"], msg["id"], service)
        return text or "This email does not contain a readable body."

    async def _walk_for_body(self, part, msg_id, service, depth=0):
        """Recursively walk through email parts to find readable body content."""
        # Prevent infinite recursion by limiting depth
        if depth > 10:
            return None

        mime_type = part.get("mimeType", "")
        body = part.get("body", {})

        # Handle text/plain content
        if mime_type == "text/plain" and body.get("data"):
            return self._decode_base64(body["data"])

        # Handle text/html content (convert to plain text)
        if mime_type == "text/html" and body.get("data"):
            html_content = self._decode_base64(body["data"])
            if html_content:
                try:
                    import html2text

                    h = html2text.HTML2Text()
                    h.ignore_links = False
                    h.ignore_images = True
                    return h.handle(html_content)
                except ImportError:
                    # Fallback: return raw HTML if html2text is not available
                    return html_content

        # Handle content stored as attachment
        if body.get("attachmentId"):
            attachment_data = await self._download_attachment_body(
                body["attachmentId"], msg_id, service
            )
            if attachment_data:
                return self._decode_base64(attachment_data)

        # Recursively search in parts
        for sub_part in part.get("parts", []):
            text = await self._walk_for_body(sub_part, msg_id, service, depth + 1)
            if text:
                return text

        return None

    def _decode_base64(self, data):
        """Safely decode base64 URL-safe data with proper padding."""
        if not data:
            return None
        try:
            # Add padding if necessary
            missing_padding = len(data) % 4
            if missing_padding:
                data += "=" * (4 - missing_padding)
            return base64.urlsafe_b64decode(data).decode("utf-8")
        except Exception:
            return None

    async def _download_attachment_body(self, attachment_id, msg_id, service):
        """Download attachment content when email body is stored as attachment."""
        try:
            attachment = await asyncio.to_thread(
                lambda: service.users()
                .messages()
                .attachments()
                .get(userId="me", messageId=msg_id, id=attachment_id)
                .execute()
            )
            return attachment.get("data")
        except Exception:
            return None

    async def _get_attachments(self, service, message):
        attachments = []
        if "parts" in message["payload"]:
            for part in message["payload"]["parts"]:
                if part.get("filename"):
                    attachment = Attachment(
                        filename=part["filename"],
                        content_type=part["mimeType"],
                        size=int(part["body"].get("size", 0)),
                        attachment_id=part["body"]["attachmentId"],
                    )
                    attachments.append(attachment)
        return attachments

    async def download_attachment(self, service, message_id: str, attachment_id: str):
        attachment = await asyncio.to_thread(
            lambda: service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        file_data = base64.urlsafe_b64decode(attachment["data"].encode("UTF-8"))
        return file_data

    async def _get_label_id(self, service, label_name: str) -> str | None:
        """Get label ID by name from Gmail."""
        results = await asyncio.to_thread(
            lambda: service.users().labels().list(userId="me").execute()
        )
        labels = results.get("labels", [])
        for label in labels:
            if label["name"] == label_name:
                return label["id"]
        return None


class GmailReadBlock(GmailBase):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.readonly"]
        )
        query: str = SchemaField(
            description="Search query for reading emails",
            default="is:unread",
        )
        max_results: int = SchemaField(
            description="Maximum number of emails to retrieve",
            default=10,
        )

    class Output(BlockSchemaOutput):
        email: Email = SchemaField(
            description="Email data",
        )
        emails: list[Email] = SchemaField(
            description="List of email data",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="25310c70-b89b-43ba-b25c-4dfa7e2a481c",
            description="A block that retrieves and reads emails from a Gmail account based on search criteria, returning detailed message information including subject, sender, body, and attachments.",
            categories={BlockCategory.COMMUNICATION},
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            input_schema=GmailReadBlock.Input,
            output_schema=GmailReadBlock.Output,
            test_input={
                "query": "is:unread",
                "max_results": 5,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "email",
                    {
                        "threadId": "t1",
                        "labelIds": ["INBOX"],
                        "id": "1",
                        "subject": "Test Email",
                        "snippet": "This is a test email",
                        "from_": "test@example.com",
                        "to": ["recipient@example.com"],
                        "cc": [],
                        "bcc": [],
                        "date": "2024-01-01",
                        "body": "This is a test email",
                        "sizeEstimate": 100,
                        "attachments": [],
                    },
                ),
                (
                    "emails",
                    [
                        {
                            "threadId": "t1",
                            "labelIds": ["INBOX"],
                            "id": "1",
                            "subject": "Test Email",
                            "snippet": "This is a test email",
                            "from_": "test@example.com",
                            "to": ["recipient@example.com"],
                            "cc": [],
                            "bcc": [],
                            "date": "2024-01-01",
                            "body": "This is a test email",
                            "sizeEstimate": 100,
                            "attachments": [],
                        }
                    ],
                ),
            ],
            test_mock={
                "_read_emails": lambda *args, **kwargs: [
                    {
                        "threadId": "t1",
                        "labelIds": ["INBOX"],
                        "id": "1",
                        "subject": "Test Email",
                        "snippet": "This is a test email",
                        "from_": "test@example.com",
                        "to": ["recipient@example.com"],
                        "cc": [],
                        "bcc": [],
                        "date": "2024-01-01",
                        "body": "This is a test email",
                        "sizeEstimate": 100,
                        "attachments": [],
                    }
                ],
                "_send_email": lambda *args, **kwargs: {"id": "1", "status": "sent"},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        messages = await self._read_emails(
            service,
            input_data.query,
            input_data.max_results,
            credentials.scopes,
        )
        for email in messages:
            yield "email", email
        yield "emails", messages

    async def _read_emails(
        self,
        service,
        query: str | None,
        max_results: int | None,
        scopes: list[str] | None,
    ) -> list[Email]:
        scopes = [s.lower() for s in (scopes or [])]
        list_kwargs = {"userId": "me", "maxResults": max_results or 10}
        if query and "https://www.googleapis.com/auth/gmail.metadata" not in scopes:
            list_kwargs["q"] = query

        results = await asyncio.to_thread(
            lambda: service.users().messages().list(**list_kwargs).execute()
        )

        messages = results.get("messages", [])

        email_data = []
        for message in messages:
            format_type = (
                "metadata"
                if "https://www.googleapis.com/auth/gmail.metadata" in scopes
                else "full"
            )
            msg = await asyncio.to_thread(
                lambda: service.users()
                .messages()
                .get(userId="me", id=message["id"], format=format_type)
                .execute()
            )

            headers = {
                header["name"].lower(): header["value"]
                for header in msg["payload"]["headers"]
            }

            attachments = await self._get_attachments(service, msg)

            # Parse all recipients
            to_recipients = [
                addr.strip() for _, addr in getaddresses([headers.get("to", "")])
            ]
            cc_recipients = [
                addr.strip() for _, addr in getaddresses([headers.get("cc", "")])
            ]
            bcc_recipients = [
                addr.strip() for _, addr in getaddresses([headers.get("bcc", "")])
            ]

            email = Email(
                threadId=msg.get("threadId", None),
                labelIds=msg.get("labelIds", []),
                id=msg["id"],
                subject=headers.get("subject", "No Subject"),
                snippet=msg.get("snippet", ""),
                from_=parseaddr(headers.get("from", ""))[1],
                to=to_recipients if to_recipients else [],
                cc=cc_recipients,
                bcc=bcc_recipients,
                date=headers.get("date", ""),
                body=await self._get_email_body(msg, service),
                sizeEstimate=msg.get("sizeEstimate", 0),
                attachments=attachments,
            )
            email_data.append(email)

        return email_data


class GmailSendBlock(GmailBase):
    """
    Sends emails through Gmail with intelligent content type detection.

    Features:
    - Automatic HTML detection: Emails containing HTML tags are sent as text/html
    - No hard-wrap for plain text: Plain text emails preserve natural line flow
    - Manual content type override: Use content_type parameter to force specific format
    - Full Unicode/emoji support with UTF-8 encoding
    - Attachment support for multiple files
    """

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.send"]
        )
        to: list[str] = SchemaField(
            description="Recipient email addresses",
        )
        subject: str = SchemaField(
            description="Email subject",
        )
        body: str = SchemaField(
            description="Email body (plain text or HTML)",
        )
        cc: list[str] = SchemaField(description="CC recipients", default_factory=list)
        bcc: list[str] = SchemaField(description="BCC recipients", default_factory=list)
        content_type: Optional[Literal["auto", "plain", "html"]] = SchemaField(
            description="Content type: 'auto' (default - detects HTML), 'plain', or 'html'",
            default=None,
            advanced=True,
        )
        attachments: list[MediaFileType] = SchemaField(
            description="Files to attach", default_factory=list, advanced=True
        )

    class Output(BlockSchemaOutput):
        result: GmailSendResult = SchemaField(
            description="Send confirmation",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="6c27abc2-e51d-499e-a85f-5a0041ba94f0",
            description="Send emails via Gmail with automatic HTML detection and proper text formatting. Plain text emails are sent without 78-character line wrapping, preserving natural paragraph flow. HTML emails are automatically detected and sent with correct MIME type.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailSendBlock.Input,
            output_schema=GmailSendBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "to": ["recipient@example.com"],
                "subject": "Test Email",
                "body": "This is a test email sent from GmailSendBlock.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"id": "1", "status": "sent"}),
            ],
            test_mock={
                "_send_email": lambda *args, **kwargs: {"id": "1", "status": "sent"},
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        result = await self._send_email(
            service,
            input_data,
            execution_context,
        )
        yield "result", result

    async def _send_email(
        self, service, input_data: Input, execution_context: ExecutionContext
    ) -> dict:
        if not input_data.to or not input_data.subject or not input_data.body:
            raise ValueError(
                "At least one recipient, subject, and body are required for sending an email"
            )
        raw_message = await create_mime_message(input_data, execution_context)
        sent_message = await asyncio.to_thread(
            lambda: service.users()
            .messages()
            .send(userId="me", body={"raw": raw_message})
            .execute()
        )
        return {"id": sent_message["id"], "status": "sent"}


class GmailCreateDraftBlock(GmailBase):
    """
    Creates draft emails in Gmail with intelligent content type detection.

    Features:
    - Automatic HTML detection: Drafts containing HTML tags are formatted as text/html
    - No hard-wrap for plain text: Plain text drafts preserve natural line flow
    - Manual content type override: Use content_type parameter to force specific format
    - Full Unicode/emoji support with UTF-8 encoding
    - Attachment support for multiple files
    """

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.modify"]
        )
        to: list[str] = SchemaField(
            description="Recipient email addresses",
        )
        subject: str = SchemaField(
            description="Email subject",
        )
        body: str = SchemaField(
            description="Email body (plain text or HTML)",
        )
        cc: list[str] = SchemaField(description="CC recipients", default_factory=list)
        bcc: list[str] = SchemaField(description="BCC recipients", default_factory=list)
        content_type: Optional[Literal["auto", "plain", "html"]] = SchemaField(
            description="Content type: 'auto' (default - detects HTML), 'plain', or 'html'",
            default=None,
            advanced=True,
        )
        attachments: list[MediaFileType] = SchemaField(
            description="Files to attach", default_factory=list, advanced=True
        )

    class Output(BlockSchemaOutput):
        result: GmailDraftResult = SchemaField(
            description="Draft creation result",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="e1eeead4-46cb-491e-8281-17b6b9c44a55",
            description="Create draft emails in Gmail with automatic HTML detection and proper text formatting. Plain text drafts preserve natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and formatted correctly.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailCreateDraftBlock.Input,
            output_schema=GmailCreateDraftBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "to": ["recipient@example.com"],
                "subject": "Draft Test Email",
                "body": "This is a test draft email.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    GmailDraftResult(
                        id="draft1", message_id="msg1", status="draft_created"
                    ),
                ),
            ],
            test_mock={
                "_create_draft": lambda *args, **kwargs: {
                    "id": "draft1",
                    "message": {"id": "msg1"},
                },
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        result = await self._create_draft(
            service,
            input_data,
            execution_context,
        )
        yield "result", GmailDraftResult(
            id=result["id"], message_id=result["message"]["id"], status="draft_created"
        )

    async def _create_draft(
        self, service, input_data: Input, execution_context: ExecutionContext
    ) -> dict:
        if not input_data.to or not input_data.subject:
            raise ValueError(
                "At least one recipient and subject are required for creating a draft"
            )

        raw_message = await create_mime_message(input_data, execution_context)
        draft = await asyncio.to_thread(
            lambda: service.users()
            .drafts()
            .create(userId="me", body={"message": {"raw": raw_message}})
            .execute()
        )

        return draft


class GmailListLabelsBlock(GmailBase):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.labels"]
        )

    class Output(BlockSchemaOutput):
        result: list[dict] = SchemaField(
            description="List of labels",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="3e1c2c1c-c689-4520-b956-1f3bf4e02bb7",
            description="A block that retrieves all labels (categories) from a Gmail account for organizing and categorizing emails.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailListLabelsBlock.Input,
            output_schema=GmailListLabelsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    [
                        {"id": "Label_1", "name": "Important"},
                        {"id": "Label_2", "name": "Work"},
                    ],
                ),
            ],
            test_mock={
                "_list_labels": lambda *args, **kwargs: [
                    {"id": "Label_1", "name": "Important"},
                    {"id": "Label_2", "name": "Work"},
                ],
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        result = await self._list_labels(service)
        yield "result", result

    async def _list_labels(self, service) -> list[dict]:
        results = await asyncio.to_thread(
            lambda: service.users().labels().list(userId="me").execute()
        )
        labels = results.get("labels", [])
        return [{"id": label["id"], "name": label["name"]} for label in labels]


class GmailAddLabelBlock(GmailBase):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.modify"]
        )
        message_id: str = SchemaField(
            description="Message ID to add label to",
        )
        label_name: str = SchemaField(
            description="Label name to add",
        )

    class Output(BlockSchemaOutput):
        result: GmailLabelResult = SchemaField(
            description="Label addition result",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="f884b2fb-04f4-4265-9658-14f433926ac9",
            description="A block that adds a label to a specific email message in Gmail, creating the label if it doesn't exist.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailAddLabelBlock.Input,
            output_schema=GmailAddLabelBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "message_id": "12345",
                "label_name": "Important",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {"status": "Label added successfully", "label_id": "Label_1"},
                ),
            ],
            test_mock={
                "_add_label": lambda *args, **kwargs: {
                    "status": "Label added successfully",
                    "label_id": "Label_1",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        result = await self._add_label(
            service, input_data.message_id, input_data.label_name
        )
        yield "result", result

    async def _add_label(self, service, message_id: str, label_name: str) -> dict:
        label_id = await self._get_or_create_label(service, label_name)
        result = await asyncio.to_thread(
            lambda: service.users()
            .messages()
            .modify(userId="me", id=message_id, body={"addLabelIds": [label_id]})
            .execute()
        )
        if not result.get("labelIds"):
            return {
                "status": "Label already applied or not found",
                "label_id": label_id,
            }

        return {"status": "Label added successfully", "label_id": label_id}

    async def _get_or_create_label(self, service, label_name: str) -> str:
        label_id = await self._get_label_id(service, label_name)
        if not label_id:
            label = await asyncio.to_thread(
                lambda: service.users()
                .labels()
                .create(userId="me", body={"name": label_name})
                .execute()
            )
            label_id = label["id"]
        return label_id


class GmailRemoveLabelBlock(GmailBase):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.modify"]
        )
        message_id: str = SchemaField(
            description="Message ID to remove label from",
        )
        label_name: str = SchemaField(
            description="Label name to remove",
        )

    class Output(BlockSchemaOutput):
        result: GmailLabelResult = SchemaField(
            description="Label removal result",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="0afc0526-aba1-4b2b-888e-a22b7c3f359d",
            description="A block that removes a label from a specific email message in a Gmail account.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailRemoveLabelBlock.Input,
            output_schema=GmailRemoveLabelBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "message_id": "12345",
                "label_name": "Important",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {"status": "Label removed successfully", "label_id": "Label_1"},
                ),
            ],
            test_mock={
                "_remove_label": lambda *args, **kwargs: {
                    "status": "Label removed successfully",
                    "label_id": "Label_1",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        result = await self._remove_label(
            service, input_data.message_id, input_data.label_name
        )
        yield "result", result

    async def _remove_label(self, service, message_id: str, label_name: str) -> dict:
        label_id = await self._get_label_id(service, label_name)
        if label_id:
            result = await asyncio.to_thread(
                lambda: service.users()
                .messages()
                .modify(userId="me", id=message_id, body={"removeLabelIds": [label_id]})
                .execute()
            )
            if not result.get("labelIds"):
                return {
                    "status": "Label already removed or not applied",
                    "label_id": label_id,
                }
            return {"status": "Label removed successfully", "label_id": label_id}
        else:
            return {"status": "Label not found", "label_name": label_name}


class GmailGetThreadBlock(GmailBase):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.readonly"]
        )
        threadId: str = SchemaField(description="Gmail thread ID")

    class Output(BlockSchemaOutput):
        thread: Thread = SchemaField(
            description="Gmail thread with decoded message bodies"
        )

    def __init__(self):
        super().__init__(
            id="21a79166-9df7-4b5f-9f36-96f639d86112",
            description="A block that retrieves an entire Gmail thread (email conversation) by ID, returning all messages with decoded bodies for reading complete conversations.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailGetThreadBlock.Input,
            output_schema=GmailGetThreadBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"threadId": "t1", "credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "thread",
                    {
                        "id": "188199feff9dc907",
                        "messages": [
                            {
                                "id": "188199feff9dc907",
                                "to": ["nick@example.co"],
                                "cc": [],
                                "bcc": [],
                                "body": "This email does not contain a text body.",
                                "date": "Thu, 17 Jul 2025 19:22:36 +0100",
                                "from_": "bent@example.co",
                                "snippet": "have a funny looking car -- Bently, Community Administrator For AutoGPT",
                                "subject": "car",
                                "threadId": "188199feff9dc907",
                                "labelIds": ["INBOX"],
                                "attachments": [
                                    {
                                        "size": 5694,
                                        "filename": "frog.jpg",
                                        "content_type": "image/jpeg",
                                        "attachment_id": "ANGjdJ_f777CvJ37TdHYSPIPPqJ0HVNgze1uM8alw5iiqTqAVXjsmBWxOWXrY3Z4W4rEJHfAcHVx54_TbtcZIVJJEqJfAD5LoUOK9_zKCRwwcTJ5TGgjsXcZNSnOJNazM-m4E6buo2-p0WNcA_hqQvuA36nzS31Olx3m2x7BaG1ILOkBcjlKJl4KCcR0AvnfK0S02k8i-bZVqII7XXrNp21f1BDolxH7tiEhkz3d5p-5Lbro24olgOWQwQk0SCJsTWWBMCVgbxU7oLt1QmPcjANxfpvh69Qfap3htvQxFa9P08NDI2YqQkry9yPxVR7ZBJQWrqO35EWmhNySEiX5pfG8SDRmfP9O_BqxTH35nEXmSOvZH9zb214iM-zfSoPSU1F5Fo71",
                                    }
                                ],
                                "sizeEstimate": 14099,
                            }
                        ],
                        "historyId": "645006",
                    },
                )
            ],
            test_mock={
                "_get_thread": lambda *args, **kwargs: {
                    "id": "188199feff9dc907",
                    "messages": [
                        {
                            "id": "188199feff9dc907",
                            "to": ["nick@example.co"],
                            "cc": [],
                            "bcc": [],
                            "body": "This email does not contain a text body.",
                            "date": "Thu, 17 Jul 2025 19:22:36 +0100",
                            "from_": "bent@example.co",
                            "snippet": "have a funny looking car -- Bently, Community Administrator For AutoGPT",
                            "subject": "car",
                            "threadId": "188199feff9dc907",
                            "labelIds": ["INBOX"],
                            "attachments": [
                                {
                                    "size": 5694,
                                    "filename": "frog.jpg",
                                    "content_type": "image/jpeg",
                                    "attachment_id": "ANGjdJ_f777CvJ37TdHYSPIPPqJ0HVNgze1uM8alw5iiqTqAVXjsmBWxOWXrY3Z4W4rEJHfAcHVx54_TbtcZIVJJEqJfAD5LoUOK9_zKCRwwcTJ5TGgjsXcZNSnOJNazM-m4E6buo2-p0WNcA_hqQvuA36nzS31Olx3m2x7BaG1ILOkBcjlKJl4KCcR0AvnfK0S02k8i-bZVqII7XXrNp21f1BDolxH7tiEhkz3d5p-5Lbro24olgOWQwQk0SCJsTWWBMCVgbxU7oLt1QmPcjANxfpvh69Qfap3htvQxFa9P08NDI2YqQkry9yPxVR7ZBJQWrqO35EWmhNySEiX5pfG8SDRmfP9O_BqxTH35nEXmSOvZH9zb214iM-zfSoPSU1F5Fo71",
                                }
                            ],
                            "sizeEstimate": 14099,
                        }
                    ],
                    "historyId": "645006",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        thread = await self._get_thread(
            service, input_data.threadId, credentials.scopes
        )
        yield "thread", thread

    async def _get_thread(
        self, service, thread_id: str, scopes: list[str] | None
    ) -> Thread:
        scopes = [s.lower() for s in (scopes or [])]
        format_type = (
            "metadata"
            if "https://www.googleapis.com/auth/gmail.metadata" in scopes
            else "full"
        )
        thread = await asyncio.to_thread(
            lambda: service.users()
            .threads()
            .get(userId="me", id=thread_id, format=format_type)
            .execute()
        )

        parsed_messages = []
        for msg in thread.get("messages", []):
            headers = {
                h["name"].lower(): h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }
            body = await self._get_email_body(msg, service)
            attachments = await self._get_attachments(service, msg)

            # Parse all recipients
            to_recipients = [
                addr.strip() for _, addr in getaddresses([headers.get("to", "")])
            ]
            cc_recipients = [
                addr.strip() for _, addr in getaddresses([headers.get("cc", "")])
            ]
            bcc_recipients = [
                addr.strip() for _, addr in getaddresses([headers.get("bcc", "")])
            ]

            email = Email(
                threadId=msg.get("threadId", thread_id),
                labelIds=msg.get("labelIds", []),
                id=msg.get("id"),
                subject=headers.get("subject", "No Subject"),
                snippet=msg.get("snippet", ""),
                from_=parseaddr(headers.get("from", ""))[1],
                to=to_recipients if to_recipients else [],
                cc=cc_recipients,
                bcc=bcc_recipients,
                date=headers.get("date", ""),
                body=body,
                sizeEstimate=msg.get("sizeEstimate", 0),
                attachments=attachments,
            )
            parsed_messages.append(email.model_dump())

        thread["messages"] = parsed_messages
        return thread


async def _build_reply_message(
    service, input_data, execution_context: ExecutionContext
) -> tuple[str, str]:
    """
    Builds a reply MIME message for Gmail threads.

    Returns:
        tuple: (base64-encoded raw message, threadId)
    """
    # Get parent message for reply context
    parent = await asyncio.to_thread(
        lambda: service.users()
        .messages()
        .get(
            userId="me",
            id=input_data.parentMessageId,
            format="metadata",
            metadataHeaders=[
                "Subject",
                "References",
                "Message-ID",
                "From",
                "To",
                "Cc",
                "Reply-To",
            ],
        )
        .execute()
    )

    # Build headers dictionary, preserving all values for duplicate headers
    headers = {}
    for h in parent.get("payload", {}).get("headers", []):
        name = h["name"].lower()
        value = h["value"]
        if name in headers:
            # For duplicate headers, keep the first occurrence (most relevant for reply context)
            continue
        headers[name] = value

    # Determine recipients if not specified
    if not (input_data.to or input_data.cc or input_data.bcc):
        if input_data.replyAll:
            recipients = [parseaddr(headers.get("from", ""))[1]]
            recipients += [addr for _, addr in getaddresses([headers.get("to", "")])]
            recipients += [addr for _, addr in getaddresses([headers.get("cc", "")])]
            # Use dict.fromkeys() for O(n) deduplication while preserving order
            input_data.to = list(dict.fromkeys(filter(None, recipients)))
        else:
            # Check Reply-To header first, fall back to From header
            reply_to = headers.get("reply-to", "")
            from_addr = headers.get("from", "")
            sender = parseaddr(reply_to if reply_to else from_addr)[1]
            input_data.to = [sender] if sender else []

    # Set subject with Re: prefix if not already present
    if input_data.subject:
        subject = input_data.subject
    else:
        parent_subject = headers.get("subject", "").strip()
        # Only add "Re:" if not already present (case-insensitive check)
        if parent_subject.lower().startswith("re:"):
            subject = parent_subject
        else:
            subject = f"Re: {parent_subject}" if parent_subject else "Re:"

    # Build references header for proper threading
    references = headers.get("references", "").split()
    if headers.get("message-id"):
        references.append(headers["message-id"])

    # Create MIME message
    msg = MIMEMultipart()
    if input_data.to:
        msg["To"] = ", ".join(input_data.to)
    if input_data.cc:
        msg["Cc"] = ", ".join(input_data.cc)
    if input_data.bcc:
        msg["Bcc"] = ", ".join(input_data.bcc)
    msg["Subject"] = subject
    if headers.get("message-id"):
        msg["In-Reply-To"] = headers["message-id"]
    if references:
        msg["References"] = " ".join(references)

    # Use the helper function for consistent content type handling
    msg.attach(_make_mime_text(input_data.body, input_data.content_type))

    # Handle attachments
    for attach in input_data.attachments:
        local_path = await store_media_file(
            file=attach,
            execution_context=execution_context,
            return_format="for_local_processing",
        )
        assert execution_context.graph_exec_id  # Validated by store_media_file
        abs_path = get_exec_file_path(execution_context.graph_exec_id, local_path)
        part = MIMEBase("application", "octet-stream")
        with open(abs_path, "rb") as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition", f"attachment; filename={Path(abs_path).name}"
        )
        msg.attach(part)

    # Encode message
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return raw, input_data.threadId


class GmailReplyBlock(GmailBase):
    """
    Replies to Gmail threads with intelligent content type detection.

    Features:
    - Automatic HTML detection: Replies containing HTML tags are sent as text/html
    - No hard-wrap for plain text: Plain text replies preserve natural line flow
    - Manual content type override: Use content_type parameter to force specific format
    - Reply-all functionality: Option to reply to all original recipients
    - Thread preservation: Maintains proper email threading with headers
    - Full Unicode/emoji support with UTF-8 encoding
    """

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.readonly",
            ]
        )
        threadId: str = SchemaField(description="Thread ID to reply in")
        parentMessageId: str = SchemaField(
            description="ID of the message being replied to"
        )
        to: list[str] = SchemaField(description="To recipients", default_factory=list)
        cc: list[str] = SchemaField(description="CC recipients", default_factory=list)
        bcc: list[str] = SchemaField(description="BCC recipients", default_factory=list)
        replyAll: bool = SchemaField(
            description="Reply to all original recipients", default=False
        )
        subject: str = SchemaField(description="Email subject", default="")
        body: str = SchemaField(description="Email body (plain text or HTML)")
        content_type: Optional[Literal["auto", "plain", "html"]] = SchemaField(
            description="Content type: 'auto' (default - detects HTML), 'plain', or 'html'",
            default=None,
            advanced=True,
        )
        attachments: list[MediaFileType] = SchemaField(
            description="Files to attach", default_factory=list, advanced=True
        )

    class Output(BlockSchemaOutput):
        messageId: str = SchemaField(description="Sent message ID")
        threadId: str = SchemaField(description="Thread ID")
        message: dict = SchemaField(description="Raw Gmail message object")
        email: Email = SchemaField(
            description="Parsed email object with decoded body and attachments"
        )

    def __init__(self):
        super().__init__(
            id="12bf5a24-9b90-4f40-9090-4e86e6995e60",
            description="Reply to Gmail threads with automatic HTML detection and proper text formatting. Plain text replies maintain natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and sent with correct MIME type.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailReplyBlock.Input,
            output_schema=GmailReplyBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "threadId": "t1",
                "parentMessageId": "m1",
                "body": "Thanks",
                "replyAll": False,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("messageId", "m2"),
                ("threadId", "t1"),
                ("message", {"id": "m2", "threadId": "t1"}),
                (
                    "email",
                    Email(
                        threadId="t1",
                        labelIds=[],
                        id="m2",
                        subject="",
                        snippet="",
                        from_="",
                        to=[],
                        cc=[],
                        bcc=[],
                        date="",
                        body="Thanks",
                        sizeEstimate=0,
                        attachments=[],
                    ),
                ),
            ],
            test_mock={
                "_reply": lambda *args, **kwargs: {
                    "id": "m2",
                    "threadId": "t1",
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        message = await self._reply(
            service,
            input_data,
            execution_context,
        )
        yield "messageId", message["id"]
        yield "threadId", message.get("threadId", input_data.threadId)
        yield "message", message
        email = Email(
            threadId=message.get("threadId", input_data.threadId),
            labelIds=message.get("labelIds", []),
            id=message["id"],
            subject=input_data.subject or "",
            snippet=message.get("snippet", ""),
            from_="",  # From address would need to be retrieved from the message headers
            to=input_data.to if input_data.to else [],
            cc=input_data.cc if input_data.cc else [],
            bcc=input_data.bcc if input_data.bcc else [],
            date="",  # Date would need to be retrieved from the message headers
            body=input_data.body,
            sizeEstimate=message.get("sizeEstimate", 0),
            attachments=[],  # Attachments info not available from send response
        )
        yield "email", email

    async def _reply(
        self, service, input_data: Input, execution_context: ExecutionContext
    ) -> dict:
        # Build the reply message using the shared helper
        raw, thread_id = await _build_reply_message(
            service, input_data, execution_context
        )

        # Send the message
        return await asyncio.to_thread(
            lambda: service.users()
            .messages()
            .send(userId="me", body={"threadId": thread_id, "raw": raw})
            .execute()
        )


class GmailDraftReplyBlock(GmailBase):
    """
    Creates draft replies to Gmail threads with intelligent content type detection.

    Features:
    - Automatic HTML detection: Draft replies containing HTML tags are formatted as text/html
    - No hard-wrap for plain text: Plain text draft replies preserve natural line flow
    - Manual content type override: Use content_type parameter to force specific format
    - Reply-all functionality: Option to reply to all original recipients
    - Thread preservation: Maintains proper email threading with headers
    - Full Unicode/emoji support with UTF-8 encoding
    """

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [
                "https://www.googleapis.com/auth/gmail.modify",
                "https://www.googleapis.com/auth/gmail.readonly",
            ]
        )
        threadId: str = SchemaField(description="Thread ID to reply in")
        parentMessageId: str = SchemaField(
            description="ID of the message being replied to"
        )
        to: list[str] = SchemaField(description="To recipients", default_factory=list)
        cc: list[str] = SchemaField(description="CC recipients", default_factory=list)
        bcc: list[str] = SchemaField(description="BCC recipients", default_factory=list)
        replyAll: bool = SchemaField(
            description="Reply to all original recipients", default=False
        )
        subject: str = SchemaField(description="Email subject", default="")
        body: str = SchemaField(description="Email body (plain text or HTML)")
        content_type: Optional[Literal["auto", "plain", "html"]] = SchemaField(
            description="Content type: 'auto' (default - detects HTML), 'plain', or 'html'",
            default=None,
            advanced=True,
        )
        attachments: list[MediaFileType] = SchemaField(
            description="Files to attach", default_factory=list, advanced=True
        )

    class Output(BlockSchemaOutput):
        draftId: str = SchemaField(description="Created draft ID")
        messageId: str = SchemaField(description="Draft message ID")
        threadId: str = SchemaField(description="Thread ID")
        status: str = SchemaField(description="Draft creation status")

    def __init__(self):
        super().__init__(
            id="d7a9f3e2-8b4c-4d6f-9e1a-3c5b7f8d2a6e",
            description="Create draft replies to Gmail threads with automatic HTML detection and proper text formatting. Plain text draft replies maintain natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and formatted correctly.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailDraftReplyBlock.Input,
            output_schema=GmailDraftReplyBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "threadId": "t1",
                "parentMessageId": "m1",
                "body": "Thanks for your message. I'll review and get back to you.",
                "replyAll": False,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("draftId", "draft1"),
                ("messageId", "m2"),
                ("threadId", "t1"),
                ("status", "draft_created"),
            ],
            test_mock={
                "_create_draft_reply": lambda *args, **kwargs: {
                    "id": "draft1",
                    "message": {"id": "m2", "threadId": "t1"},
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        draft = await self._create_draft_reply(
            service,
            input_data,
            execution_context,
        )
        yield "draftId", draft["id"]
        yield "messageId", draft["message"]["id"]
        yield "threadId", draft["message"].get("threadId", input_data.threadId)
        yield "status", "draft_created"

    async def _create_draft_reply(
        self, service, input_data: Input, execution_context: ExecutionContext
    ) -> dict:
        # Build the reply message using the shared helper
        raw, thread_id = await _build_reply_message(
            service, input_data, execution_context
        )

        # Create draft with proper thread association
        draft = await asyncio.to_thread(
            lambda: service.users()
            .drafts()
            .create(
                userId="me",
                body={
                    "message": {
                        "threadId": thread_id,
                        "raw": raw,
                    }
                },
            )
            .execute()
        )

        return draft


class GmailGetProfileBlock(GmailBase):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.readonly"]
        )

    class Output(BlockSchemaOutput):
        profile: Profile = SchemaField(description="Gmail user profile information")

    def __init__(self):
        super().__init__(
            id="04b0d996-0908-4a4b-89dd-b9697ff253d3",
            description="Get the authenticated user's Gmail profile details including email address and message statistics.",
            categories={BlockCategory.COMMUNICATION},
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            input_schema=GmailGetProfileBlock.Input,
            output_schema=GmailGetProfileBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "profile",
                    {
                        "emailAddress": "test@example.com",
                        "messagesTotal": 1000,
                        "threadsTotal": 500,
                        "historyId": "12345",
                    },
                ),
            ],
            test_mock={
                "_get_profile": lambda *args, **kwargs: {
                    "emailAddress": "test@example.com",
                    "messagesTotal": 1000,
                    "threadsTotal": 500,
                    "historyId": "12345",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        profile = await self._get_profile(service)
        yield "profile", profile

    async def _get_profile(self, service) -> Profile:
        result = await asyncio.to_thread(
            lambda: service.users().getProfile(userId="me").execute()
        )
        return Profile(
            emailAddress=result.get("emailAddress", ""),
            messagesTotal=result.get("messagesTotal", 0),
            threadsTotal=result.get("threadsTotal", 0),
            historyId=result.get("historyId", ""),
        )


class GmailForwardBlock(GmailBase):
    """
    Forwards Gmail messages with intelligent content type detection.

    Features:
    - Preserves original message headers and threading
    - Automatic HTML detection for forwarded content
    - Optional forward message customization
    - Full attachment support from original message
    - Manual content type override option
    """

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.readonly",
            ]
        )
        messageId: str = SchemaField(description="ID of the message to forward")
        to: list[str] = SchemaField(description="Recipients to forward the message to")
        cc: list[str] = SchemaField(description="CC recipients", default_factory=list)
        bcc: list[str] = SchemaField(description="BCC recipients", default_factory=list)
        subject: str = SchemaField(
            description="Optional custom subject (defaults to 'Fwd: [original subject]')",
            default="",
        )
        forwardMessage: str = SchemaField(
            description="Optional message to include before the forwarded content",
            default="",
        )
        includeAttachments: bool = SchemaField(
            description="Include attachments from the original message",
            default=True,
        )
        content_type: Optional[Literal["auto", "plain", "html"]] = SchemaField(
            description="Content type: 'auto' (default - detects HTML), 'plain', or 'html'",
            default=None,
            advanced=True,
        )
        additionalAttachments: list[MediaFileType] = SchemaField(
            description="Additional files to attach",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        messageId: str = SchemaField(description="Forwarded message ID")
        threadId: str = SchemaField(description="Thread ID")
        status: str = SchemaField(description="Forward status")

    def __init__(self):
        super().__init__(
            id="64d2301c-b3f5-4174-8ac0-111ca1e1a7c0",
            description="Forward Gmail messages to other recipients with automatic HTML detection and proper formatting. Preserves original message threading and attachments.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailForwardBlock.Input,
            output_schema=GmailForwardBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "messageId": "m1",
                "to": ["recipient@example.com"],
                "forwardMessage": "FYI - forwarding this to you.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("messageId", "m2"),
                ("threadId", "t1"),
                ("status", "forwarded"),
            ],
            test_mock={
                "_forward_message": lambda *args, **kwargs: {
                    "id": "m2",
                    "threadId": "t1",
                },
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        result = await self._forward_message(
            service,
            input_data,
            execution_context,
        )
        yield "messageId", result["id"]
        yield "threadId", result.get("threadId", "")
        yield "status", "forwarded"

    async def _forward_message(
        self, service, input_data: Input, execution_context: ExecutionContext
    ) -> dict:
        if not input_data.to:
            raise ValueError("At least one recipient is required for forwarding")

        # Get the original message
        original = await asyncio.to_thread(
            lambda: service.users()
            .messages()
            .get(userId="me", id=input_data.messageId, format="full")
            .execute()
        )

        headers = {
            h["name"].lower(): h["value"]
            for h in original.get("payload", {}).get("headers", [])
        }

        # Create subject with Fwd: prefix if not already present
        original_subject = headers.get("subject", "No Subject")
        if input_data.subject:
            subject = input_data.subject
        elif not original_subject.lower().startswith("fwd:"):
            subject = f"Fwd: {original_subject}"
        else:
            subject = original_subject

        # Build forwarded message body
        original_from = headers.get("from", "Unknown")
        original_date = headers.get("date", "Unknown")
        original_to = headers.get("to", "Unknown")

        # Get the original body
        original_body = await self._get_email_body(original, service)

        # Construct the forward header
        forward_header = f"""
---------- Forwarded message ---------
From: {original_from}
Date: {original_date}
Subject: {original_subject}
To: {original_to}
"""

        # Combine optional forward message with original content
        if input_data.forwardMessage:
            body = f"{input_data.forwardMessage}\n\n{forward_header}\n\n{original_body}"
        else:
            body = f"{forward_header}\n\n{original_body}"

        # Create MIME message
        msg = MIMEMultipart()
        msg["To"] = ", ".join(input_data.to)
        if input_data.cc:
            msg["Cc"] = ", ".join(input_data.cc)
        if input_data.bcc:
            msg["Bcc"] = ", ".join(input_data.bcc)
        msg["Subject"] = subject

        # Add body with proper content type
        msg.attach(_make_mime_text(body, input_data.content_type))

        # Include original attachments if requested
        if input_data.includeAttachments:
            attachments = await self._get_attachments(service, original)
            for attachment in attachments:
                # Download and attach each original attachment
                attachment_data = await self.download_attachment(
                    service, input_data.messageId, attachment.attachment_id
                )
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment_data)
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={attachment.filename}",
                )
                msg.attach(part)

        # Add any additional attachments
        for attach in input_data.additionalAttachments:
            local_path = await store_media_file(
                file=attach,
                execution_context=execution_context,
                return_format="for_local_processing",
            )
            assert execution_context.graph_exec_id  # Validated by store_media_file
            abs_path = get_exec_file_path(execution_context.graph_exec_id, local_path)
            part = MIMEBase("application", "octet-stream")
            with open(abs_path, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition", f"attachment; filename={Path(abs_path).name}"
            )
            msg.attach(part)

        # Send the forwarded message
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        return await asyncio.to_thread(
            lambda: service.users()
            .messages()
            .send(userId="me", body={"raw": raw})
            .execute()
        )
