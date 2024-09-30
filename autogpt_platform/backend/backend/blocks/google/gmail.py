from typing import List, Optional
from pydantic import Field
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)



class GmailBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
            ]
        )
        action: str = SchemaField(
            description="Action to perform: 'read' or 'send'",
        )
        query: Optional[str] = SchemaField(
            description="Search query for reading emails (only for 'read' action)",
        )
        to: Optional[str] = SchemaField(
            description="Recipient email address (only for 'send' action)",
        )
        subject: Optional[str] = SchemaField(
            description="Email subject (only for 'send' action)",
        )
        body: Optional[str] = SchemaField(
            description="Email body (only for 'send' action)",
        )
        max_results: Optional[int] = SchemaField(
            description="Maximum number of emails to retrieve (only for 'read' action)",
        )

    class Output(BlockSchema):
        result: List[dict] = SchemaField(
            description="List of email data or send confirmation",
        )
        error: Optional[str] = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="f90d92ef-9234-432c-a2df-a45a9a5ad43e",
            description="This block performs actions on Gmail, such as reading emails or sending new emails.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailBlock.Input,
            output_schema=GmailBlock.Output,
            test_input={
                "action": "read",
                "query": "is:unread",
                "max_results": 5,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    [
                        {
                            "id": "1",
                            "subject": "Test Email",
                            "snippet": "This is a test email",
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
                    }
                ],
                "_send_email": lambda *args, **kwargs: {"id": "1", "status": "sent"},
            },
        )

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        try:
            if credentials.access_token is None:
                raise ValueError("Access token is required")
            if credentials.refresh_token is None:
                raise ValueError("Refresh token is required")
            creds = Credentials(
                token=credentials.access_token.get_secret_value(),
                refresh_token=credentials.refresh_token.get_secret_value(),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=kwargs.get("client_id"),
                client_secret=kwargs.get("client_secret"),
                scopes=credentials.scopes,
            )

            service = build("gmail", "v1", credentials=creds)

            if input_data.action == "read":
                messages = self._read_emails(
                    service, input_data.query, input_data.max_results
                )
                yield "result", messages
            elif input_data.action == "send":
                send_result = self._send_email(
                    service, input_data.to, input_data.subject, input_data.body
                )
                yield "result", [send_result]
            else:
                yield "error", f"Invalid action: {input_data.action}"

        except Exception as e:
            yield "error", str(e)

    def _read_emails(self, service, query: str, max_results: int) -> List[dict]:
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        messages = results.get("messages", [])

        email_data = []
        for message in messages:
            msg = (
                service.users().messages().get(userId="me", id=message["id"]).execute()
            )
            email_data.append(
                {
                    "id": msg["id"],
                    "subject": next(
                        (
                            header["value"]
                            for header in msg["payload"]["headers"]
                            if header["name"] == "Subject"
                        ),
                        "No Subject",
                    ),
                    "snippet": msg["snippet"],
                }
            )

        return email_data

    def _send_email(self, service, to: str, subject: str, body: str) -> dict:
        message = self._create_message(to, subject, body)
        sent_message = (
            service.users().messages().send(userId="me", body=message).execute()
        )
        return {"id": sent_message["id"], "status": "sent"}

    def _create_message(self, to: str, subject: str, body: str) -> dict:
        from email.mime.text import MIMEText
        import base64

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        return {"raw": raw_message}
