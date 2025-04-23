import base64
from email.utils import parseaddr
from typing import List

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.settings import Settings

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)


class Attachment(BaseModel):
    filename: str
    content_type: str
    size: int
    attachment_id: str


class Email(BaseModel):
    id: str
    subject: str
    snippet: str
    from_: str
    to: str
    date: str
    body: str = ""  # Default to an empty string
    sizeEstimate: int
    attachments: List[Attachment]


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
                        "id": "1",
                        "subject": "Test Email",
                        "snippet": "This is a test email",
                        "from_": "test@example.com",
                        "to": "recipient@example.com",
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
                            "id": "1",
                            "subject": "Test Email",
                            "snippet": "This is a test email",
                            "from_": "test@example.com",
                            "to": "recipient@example.com",
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
                        "id": "1",
                        "subject": "Test Email",
                        "snippet": "This is a test email",
                        "from_": "test@example.com",
                        "to": "recipient@example.com",
                        "date": "2024-01-01",
                        "body": "This is a test email",
                        "sizeEstimate": 100,
                        "attachments": [],
                    }
                ],
                "_send_email": lambda *args, **kwargs: {"id": "1", "status": "sent"},
            },
        )

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        messages = self._read_emails(service, input_data.query, input_data.max_results)
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

    def _read_emails(
        self, service, query: str | None, max_results: int | None
    ) -> list[Email]:
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query or "", maxResults=max_results or 10)
            .execute()
        )
        messages = results.get("messages", [])

        email_data = []
        for message in messages:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=message["id"], format="full")
                .execute()
            )

            headers = {
                header["name"].lower(): header["value"]
                for header in msg["payload"]["headers"]
            }

            attachments = self._get_attachments(service, msg)

            email = Email(
                id=msg["id"],
                subject=headers.get("subject", "No Subject"),
                snippet=msg["snippet"],
                from_=parseaddr(headers.get("from", ""))[1],
                to=parseaddr(headers.get("to", ""))[1],
                date=headers.get("date", ""),
                body=self._get_email_body(msg),
                sizeEstimate=msg["sizeEstimate"],
                attachments=attachments,
            )
            email_data.append(email)

        return email_data

    def _get_email_body(self, msg):
        if "parts" in msg["payload"]:
            for part in msg["payload"]["parts"]:
                if part["mimeType"] == "text/plain":
                    return base64.urlsafe_b64decode(part["body"]["data"]).decode(
                        "utf-8"
                    )
        elif msg["payload"]["mimeType"] == "text/plain":
            return base64.urlsafe_b64decode(msg["payload"]["body"]["data"]).decode(
                "utf-8"
            )

        return "This email does not contain a text body."

    def _get_attachments(self, service, message):
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
        to: str = SchemaField(
            description="Recipient email address",
        )
        subject: str = SchemaField(
            description="Email subject",
        )
        body: str = SchemaField(
            description="Email body",
        )

    class Output(BlockSchema):
        result: dict = SchemaField(
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
                "to": "recipient@example.com",
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

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GmailReadBlock._build_service(credentials, **kwargs)
        send_result = self._send_email(
            service, input_data.to, input_data.subject, input_data.body
        )
        yield "result", send_result

    def _send_email(self, service, to: str, subject: str, body: str) -> dict:
        if not to or not subject or not body:
            raise ValueError("To, subject, and body are required for sending an email")
        message = self._create_message(to, subject, body)
        sent_message = (
            service.users().messages().send(userId="me", body=message).execute()
        )
        return {"id": sent_message["id"], "status": "sent"}

    def _create_message(self, to: str, subject: str, body: str) -> dict:
        import base64
        from email.mime.text import MIMEText

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        return {"raw": raw_message}


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

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GmailReadBlock._build_service(credentials, **kwargs)
        labels = self._list_labels(service)
        yield "result", labels

    def _list_labels(self, service) -> list[dict]:
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
        result: dict = SchemaField(
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

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GmailReadBlock._build_service(credentials, **kwargs)
        result = self._add_label(service, input_data.message_id, input_data.label_name)
        yield "result", result

    def _add_label(self, service, message_id: str, label_name: str) -> dict:
        label_id = self._get_or_create_label(service, label_name)
        service.users().messages().modify(
            userId="me", id=message_id, body={"addLabelIds": [label_id]}
        ).execute()
        return {"status": "Label added successfully", "label_id": label_id}

    def _get_or_create_label(self, service, label_name: str) -> str:
        label_id = self._get_label_id(service, label_name)
        if not label_id:
            label = (
                service.users()
                .labels()
                .create(userId="me", body={"name": label_name})
                .execute()
            )
            label_id = label["id"]
        return label_id

    def _get_label_id(self, service, label_name: str) -> str | None:
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
        result: dict = SchemaField(
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

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = GmailReadBlock._build_service(credentials, **kwargs)
        result = self._remove_label(
            service, input_data.message_id, input_data.label_name
        )
        yield "result", result

    def _remove_label(self, service, message_id: str, label_name: str) -> dict:
        label_id = self._get_label_id(service, label_name)
        if label_id:
            service.users().messages().modify(
                userId="me", id=message_id, body={"removeLabelIds": [label_id]}
            ).execute()
            return {"status": "Label removed successfully", "label_id": label_id}
        else:
            return {"status": "Label not found", "label_name": label_name}

    def _get_label_id(self, service, label_name: str) -> str | None:
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        for label in labels:
            if label["name"] == label_name:
                return label["id"]
        return None
