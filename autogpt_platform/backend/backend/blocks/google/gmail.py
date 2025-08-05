import base64
from email.utils import getaddresses, parseaddr
from pathlib import Path
from typing import List

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel, Field

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
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


def serialize_email_recipients(recipients: list[str]) -> str:
    """Serialize recipients list to comma-separated string."""
    return ", ".join(recipients)


async def create_mime_message(
    input_data,
    graph_exec_id: str,
    user_id: str,
) -> str:
    """Create a MIME message with attachments and return base64-encoded raw message."""
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    message = MIMEMultipart()
    message["to"] = serialize_email_recipients(input_data.to)
    message["subject"] = input_data.subject

    if input_data.cc:
        message["cc"] = ", ".join(input_data.cc)
    if input_data.bcc:
        message["bcc"] = ", ".join(input_data.bcc)

    message.attach(MIMEText(input_data.body))

    # Handle attachments if any
    if input_data.attachments:
        for attach in input_data.attachments:
            local_path = await store_media_file(
                user_id=user_id,
                graph_exec_id=graph_exec_id,
                file=attach,
                return_content=False,
            )
            abs_path = get_exec_file_path(graph_exec_id, local_path)
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


class GmailReadBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
            description="This block reads emails from Gmail.",
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
        service = GmailReadBlock._build_service(credentials, **kwargs)
        messages = await self._read_emails(
            service,
            input_data.query,
            input_data.max_results,
            credentials.scopes,
        )
        for email in messages:
            yield "email", email
        yield "emails", messages

    @staticmethod
    def _build_service(credentials: GoogleCredentials, **kwargs):
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
            client_id=Settings().secrets.google_client_id,
            client_secret=Settings().secrets.google_client_secret,
            scopes=credentials.scopes,
        )
        return build("gmail", "v1", credentials=creds)

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

        results = service.users().messages().list(**list_kwargs).execute()

        messages = results.get("messages", [])

        email_data = []
        for message in messages:
            format_type = (
                "metadata"
                if "https://www.googleapis.com/auth/gmail.metadata" in scopes
                else "full"
            )
            msg = (
                service.users()
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
            attachment = (
                service.users()
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
                if part["filename"]:
                    attachment = Attachment(
                        filename=part["filename"],
                        content_type=part["mimeType"],
                        size=int(part["body"].get("size", 0)),
                        attachment_id=part["body"]["attachmentId"],
                    )
                    attachments.append(attachment)
        return attachments

    # Add a new method to download attachment content
    def download_attachment(self, service, message_id: str, attachment_id: str):
        attachment = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        file_data = base64.urlsafe_b64decode(attachment["data"].encode("UTF-8"))
        return file_data


class GmailSendBlock(Block):
    class Input(BlockSchema):
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
            description="Email body",
        )
        cc: list[str] = SchemaField(description="CC recipients", default_factory=list)
        bcc: list[str] = SchemaField(description="BCC recipients", default_factory=list)
        attachments: list[MediaFileType] = SchemaField(
            description="Files to attach", default_factory=list, advanced=True
        )

    class Output(BlockSchema):
        result: GmailSendResult = SchemaField(
            description="Send confirmation",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="6c27abc2-e51d-499e-a85f-5a0041ba94f0",
            description="This block sends an email using Gmail.",
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
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        service = GmailReadBlock._build_service(credentials, **kwargs)
        result = await self._send_email(
            service,
            input_data,
            graph_exec_id,
            user_id,
        )
        yield "result", result

    async def _send_email(
        self, service, input_data: Input, graph_exec_id: str, user_id: str
    ) -> dict:
        if not input_data.to or not input_data.subject or not input_data.body:
            raise ValueError(
                "At least one recipient, subject, and body are required for sending an email"
            )
        raw_message = await create_mime_message(input_data, graph_exec_id, user_id)
        sent_message = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": raw_message})
            .execute()
        )
        return {"id": sent_message["id"], "status": "sent"}


class GmailCreateDraftBlock(Block):
    class Input(BlockSchema):
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
            description="Email body",
        )
        cc: list[str] = SchemaField(description="CC recipients", default_factory=list)
        bcc: list[str] = SchemaField(description="BCC recipients", default_factory=list)
        attachments: list[MediaFileType] = SchemaField(
            description="Files to attach", default_factory=list, advanced=True
        )

    class Output(BlockSchema):
        result: GmailDraftResult = SchemaField(
            description="Draft creation result",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="e1eeead4-46cb-491e-8281-17b6b9c44a55",
            description="This block creates a draft email in Gmail.",
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
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        service = GmailReadBlock._build_service(credentials, **kwargs)
        result = await self._create_draft(
            service,
            input_data,
            graph_exec_id,
            user_id,
        )
        yield "result", GmailDraftResult(
            id=result["id"], message_id=result["message"]["id"], status="draft_created"
        )

    async def _create_draft(
        self, service, input_data: Input, graph_exec_id: str, user_id: str
    ) -> dict:
        if not input_data.to or not input_data.subject:
            raise ValueError(
                "At least one recipient and subject are required for creating a draft"
            )

        raw_message = await create_mime_message(input_data, graph_exec_id, user_id)
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": {"raw": raw_message}})
            .execute()
        )

        return draft


class GmailListLabelsBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.labels"]
        )

    class Output(BlockSchema):
        result: list[dict] = SchemaField(
            description="List of labels",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="3e1c2c1c-c689-4520-b956-1f3bf4e02bb7",
            description="This block lists all labels in Gmail.",
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
        service = GmailReadBlock._build_service(credentials, **kwargs)
        result = await self._list_labels(service)
        yield "result", result

    async def _list_labels(self, service) -> list[dict]:
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        return [{"id": label["id"], "name": label["name"]} for label in labels]


class GmailAddLabelBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.modify"]
        )
        message_id: str = SchemaField(
            description="Message ID to add label to",
        )
        label_name: str = SchemaField(
            description="Label name to add",
        )

    class Output(BlockSchema):
        result: GmailLabelResult = SchemaField(
            description="Label addition result",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="f884b2fb-04f4-4265-9658-14f433926ac9",
            description="This block adds a label to a Gmail message.",
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
        service = GmailReadBlock._build_service(credentials, **kwargs)
        result = await self._add_label(
            service, input_data.message_id, input_data.label_name
        )
        yield "result", result

    async def _add_label(self, service, message_id: str, label_name: str) -> dict:
        label_id = await self._get_or_create_label(service, label_name)
        result = (
            service.users()
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
            label = (
                service.users()
                .labels()
                .create(userId="me", body={"name": label_name})
                .execute()
            )
            label_id = label["id"]
        return label_id

    async def _get_label_id(self, service, label_name: str) -> str | None:
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        for label in labels:
            if label["name"] == label_name:
                return label["id"]
        return None


class GmailRemoveLabelBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.modify"]
        )
        message_id: str = SchemaField(
            description="Message ID to remove label from",
        )
        label_name: str = SchemaField(
            description="Label name to remove",
        )

    class Output(BlockSchema):
        result: GmailLabelResult = SchemaField(
            description="Label removal result",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="0afc0526-aba1-4b2b-888e-a22b7c3f359d",
            description="This block removes a label from a Gmail message.",
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
        service = GmailReadBlock._build_service(credentials, **kwargs)
        result = await self._remove_label(
            service, input_data.message_id, input_data.label_name
        )
        yield "result", result

    async def _remove_label(self, service, message_id: str, label_name: str) -> dict:
        label_id = await self._get_label_id(service, label_name)
        if label_id:
            result = (
                service.users()
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

    async def _get_label_id(self, service, label_name: str) -> str | None:
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        for label in labels:
            if label["name"] == label_name:
                return label["id"]
        return None


class GmailGetThreadBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.readonly"]
        )
        threadId: str = SchemaField(description="Gmail thread ID")

    class Output(BlockSchema):
        thread: Thread = SchemaField(
            description="Gmail thread with decoded message bodies"
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="21a79166-9df7-4b5f-9f36-96f639d86112",
            description="Get a full Gmail thread by ID",
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
        service = GmailReadBlock._build_service(credentials, **kwargs)
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
        thread = (
            service.users()
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
            attachment = (
                service.users()
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


class GmailReplyBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.send"]
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
        body: str = SchemaField(description="Email body")
        attachments: list[MediaFileType] = SchemaField(
            description="Files to attach", default_factory=list, advanced=True
        )

    class Output(BlockSchema):
        messageId: str = SchemaField(description="Sent message ID")
        threadId: str = SchemaField(description="Thread ID")
        message: dict = SchemaField(description="Raw Gmail message object")
        email: Email = SchemaField(
            description="Parsed email object with decoded body and attachments"
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="12bf5a24-9b90-4f40-9090-4e86e6995e60",
            description="Reply to a Gmail thread",
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
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        service = GmailReadBlock._build_service(credentials, **kwargs)
        message = await self._reply(
            service,
            input_data,
            graph_exec_id,
            user_id,
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
        self, service, input_data: Input, graph_exec_id: str, user_id: str
    ) -> dict:
        parent = (
            service.users()
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

        headers = {
            h["name"].lower(): h["value"]
            for h in parent.get("payload", {}).get("headers", [])
        }
        if not (input_data.to or input_data.cc or input_data.bcc):
            if input_data.replyAll:
                recipients = [parseaddr(headers.get("from", ""))[1]]
                recipients += [
                    addr for _, addr in getaddresses([headers.get("to", "")])
                ]
                recipients += [
                    addr for _, addr in getaddresses([headers.get("cc", "")])
                ]
                dedup: list[str] = []
                for r in recipients:
                    if r and r not in dedup:
                        dedup.append(r)
                input_data.to = dedup
            else:
                sender = parseaddr(headers.get("reply-to", headers.get("from", "")))[1]
                input_data.to = [sender] if sender else []
        subject = input_data.subject or (f"Re: {headers.get('subject', '')}".strip())
        references = headers.get("references", "").split()
        if headers.get("message-id"):
            references.append(headers["message-id"])

        from email import encoders
        from email.mime.base import MIMEBase
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

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
        msg.attach(
            MIMEText(input_data.body, "html" if "<" in input_data.body else "plain")
        )

        for attach in input_data.attachments:
            local_path = await store_media_file(
                user_id=user_id,
                graph_exec_id=graph_exec_id,
                file=attach,
                return_content=False,
            )
            abs_path = get_exec_file_path(graph_exec_id, local_path)
            part = MIMEBase("application", "octet-stream")
            with open(abs_path, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition", f"attachment; filename={Path(abs_path).name}"
            )
            msg.attach(part)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        return (
            service.users()
            .messages()
            .send(userId="me", body={"threadId": input_data.threadId, "raw": raw})
            .execute()
        )


class GmailGetProfileBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/gmail.readonly"]
        )

    class Output(BlockSchema):
        profile: Profile = SchemaField(description="Gmail user profile information")
        error: str = SchemaField(description="Error message if any")

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
        service = GmailReadBlock._build_service(credentials, **kwargs)
        profile = await self._get_profile(service)
        yield "profile", profile

    async def _get_profile(self, service) -> Profile:
        result = service.users().getProfile(userId="me").execute()
        return Profile(
            emailAddress=result.get("emailAddress", ""),
            messagesTotal=result.get("messagesTotal", 0),
            threadsTotal=result.get("threadsTotal", 0),
            historyId=result.get("historyId", ""),
        )
