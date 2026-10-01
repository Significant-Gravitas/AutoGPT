import asyncio
import base64

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._gmail_api import (
    GMAIL_READONLY_SCOPE,
    Attachment,
    Email,
    GmailDraft,
    gmail_error,
    message_format,
)
from .gmail import GmailBase


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


_TEST_MESSAGE = {
    "id": "19a2b3c4d5e6f708",
    "threadId": "19a2b3c4d5e6f708",
    "labelIds": ["SENT"],
    "snippet": "Hi Priya, invoice 1042 for September is attached.",
    "sizeEstimate": 58312,
    "payload": {
        "mimeType": "multipart/mixed",
        "headers": [
            {"name": "From", "value": "Sam Rivera <sam@example.com>"},
            {"name": "To", "value": "Priya Shah <priya@client.example.com>"},
            {"name": "Cc", "value": "accounts@client.example.com"},
            {"name": "Bcc", "value": "sam@example.com"},
            {"name": "Subject", "value": "Invoice 1042"},
            {"name": "Date", "value": "Wed, 24 Sep 2026 16:05:12 +0100"},
            {"name": "Message-ID", "value": "<CAB9x2Lk@mail.gmail.com>"},
        ],
        "body": {"size": 0},
        "parts": [
            {
                "mimeType": "text/plain",
                "filename": "",
                "body": {
                    "size": 51,
                    "data": _b64(
                        "Hi Priya, invoice 1042 for September is attached.\r\n"
                    ),
                },
            },
            {
                "mimeType": "application/pdf",
                "filename": "invoice-1042.pdf",
                "body": {"attachmentId": "ANGjdJ9inv1042", "size": 52011},
            },
        ],
    },
}
_TEST_EMAIL = Email(
    threadId="19a2b3c4d5e6f708",
    labelIds=["SENT"],
    id="19a2b3c4d5e6f708",
    subject="Invoice 1042",
    snippet="Hi Priya, invoice 1042 for September is attached.",
    from_="sam@example.com",
    to=["priya@client.example.com"],
    cc=["accounts@client.example.com"],
    bcc=["sam@example.com"],
    date="Wed, 24 Sep 2026 16:05:12 +0100",
    body="Hi Priya, invoice 1042 for September is attached.\r\n",
    sizeEstimate=58312,
    attachments=[
        Attachment(
            filename="invoice-1042.pdf",
            content_type="application/pdf",
            size=52011,
            attachment_id="ANGjdJ9inv1042",
        )
    ],
)
_TEST_DRAFT = {
    "id": "r-5061797546435455397",
    "message": {
        "id": "19a2c0ffee123456",
        "threadId": "19a2c0ffee123456",
        "labelIds": ["DRAFT"],
        "snippet": "Hi team, here is the agenda for Friday.",
        "sizeEstimate": 912,
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "sam@example.com"},
                {"name": "To", "value": "team@example.com"},
                {"name": "Cc", "value": "lead@example.com"},
                {"name": "Bcc", "value": "sam@example.com"},
                {"name": "Subject", "value": "Friday agenda"},
                {"name": "Date", "value": "Thu, 25 Sep 2026 08:30:00 +0100"},
            ],
            "body": {
                "size": 40,
                "data": _b64("Hi team, here is the agenda for Friday."),
            },
        },
    },
}
_TEST_DRAFT_EMAIL = Email(
    threadId="19a2c0ffee123456",
    labelIds=["DRAFT"],
    id="19a2c0ffee123456",
    subject="Friday agenda",
    snippet="Hi team, here is the agenda for Friday.",
    from_="sam@example.com",
    to=["team@example.com"],
    cc=["lead@example.com"],
    bcc=["sam@example.com"],
    date="Thu, 25 Sep 2026 08:30:00 +0100",
    body="Hi team, here is the agenda for Friday.",
    sizeEstimate=912,
    attachments=[],
)


class GmailGetMessageBlock(GmailBase):
    """Get one Gmail email by message ID, Message-ID header or draft ID."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [GMAIL_READONLY_SCOPE]
        )
        message_id: str = SchemaField(
            description=(
                "The email's Gmail message ID (the id from Gmail Read or Get "
                "Thread), or its Message-ID header, with or without the angle brackets"
            ),
            default="",
        )
        draft_id: str = SchemaField(
            description=(
                "Or a draft ID (from Gmail List Drafts or Create Draft), to read "
                "that draft instead"
            ),
            default="",
        )

    class Output(BlockSchemaOutput):
        email: Email = SchemaField(
            description="The email, with its decoded body and attachment details"
        )
        draft_id: str = SchemaField(description="The draft's ID, when a draft was read")

    def __init__(self):
        super().__init__(
            id="2dc5a0ea-e6ad-486f-a0be-67b9dcf3be74",
            description=(
                "Get one Gmail email by its message ID or Message-ID header, or a "
                "draft by its draft ID. Returns the sender, recipients, subject, "
                "date, labels, decoded body and attachment details."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailGetMessageBlock.Input,
            output_schema=GmailGetMessageBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input=[
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "message_id": "19a2b3c4d5e6f708",
                },
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "message_id": "<CAB9x2Lk@mail.gmail.com>",
                },
                {"credentials": TEST_CREDENTIALS_INPUT, "draft_id": _TEST_DRAFT["id"]},
            ],
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("email", _TEST_EMAIL),
                ("email", _TEST_EMAIL),
                ("email", _TEST_DRAFT_EMAIL),
                ("draft_id", _TEST_DRAFT["id"]),
            ],
            test_mock={
                "_fetch_message": lambda *args, **kwargs: _TEST_MESSAGE,
                "_find_by_header": lambda *args, **kwargs: _TEST_MESSAGE["id"],
                "_fetch_draft": lambda *args, **kwargs: _TEST_DRAFT,
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        message_id = input_data.message_id.strip()
        draft_id = input_data.draft_id.strip()
        if bool(message_id) == bool(draft_id):
            raise BlockInputError(
                message="Give either a message ID or a draft ID.",
                block_name=self.name,
                block_id=self.id,
            )
        service = self._build_service(credentials)
        fmt = message_format(credentials.scopes)
        try:
            if draft_id:
                draft = await asyncio.to_thread(
                    self._fetch_draft, service, draft_id, fmt
                )
                message = draft["message"]
            else:
                message = await self._get_message(service, message_id, fmt)
        except HttpError as e:
            item = "draft" if draft_id else "message"
            raise gmail_error(e, self.name, self.id, item) from e
        yield "email", await self._parse_email(message, service)
        if draft_id:
            yield "draft_id", draft_id

    async def _get_message(self, service, message_id: str, fmt: str) -> dict:
        if "@" in message_id:
            found = await asyncio.to_thread(self._find_by_header, service, message_id)
            if found is None:
                raise BlockExecutionError(
                    message=f"No email in this Gmail account has the Message-ID {message_id}.",
                    block_name=self.name,
                    block_id=self.id,
                )
            message_id = found
        return await asyncio.to_thread(self._fetch_message, service, message_id, fmt)

    @staticmethod
    def _fetch_message(service, message_id: str, fmt: str) -> dict:
        return (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format=fmt)
            .execute()
        )

    @staticmethod
    def _find_by_header(service, header_id: str) -> str | None:
        """Find the Gmail message ID for a Message-ID header, Spam and Trash included."""
        result = (
            service.users()
            .messages()
            .list(
                userId="me",
                q=f"rfc822msgid:{header_id.strip().strip('<>')}",
                maxResults=1,
                includeSpamTrash=True,
            )
            .execute()
        )
        messages = result.get("messages", [])
        return messages[0]["id"] if messages else None

    @staticmethod
    def _fetch_draft(service, draft_id: str, fmt: str) -> dict:
        return fetch_draft(service, draft_id, fmt)


class GmailListDraftsBlock(GmailBase):
    """List Gmail drafts, optionally matching a Gmail search."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [GMAIL_READONLY_SCOPE]
        )
        query: str = SchemaField(
            description=(
                "Only drafts matching this Gmail search, e.g. to:priya@example.com "
                "or subject:invoice. Empty lists every draft."
            ),
            default="",
        )
        max_results: int = SchemaField(
            description="Maximum number of drafts to return",
            default=20,
            ge=1,
            le=50,
        )
        page_token: str = SchemaField(
            description="Page token from a previous call, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        drafts: list[GmailDraft] = SchemaField(description="The matching drafts")
        draft: GmailDraft = SchemaField(description="Each matching draft")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more drafts"
        )

    def __init__(self):
        test_draft = GmailDraft(id=_TEST_DRAFT["id"], email=_TEST_DRAFT_EMAIL)
        super().__init__(
            id="112e8098-5b9d-4e63-825a-966ba40a0f20",
            description=(
                "List Gmail drafts, optionally only those matching a Gmail search. "
                "Returns each draft's ID with its recipients, subject and body."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailListDraftsBlock.Input,
            output_schema=GmailListDraftsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "subject:agenda",
                "max_results": 5,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("drafts", [test_draft]),
                ("draft", test_draft),
                ("next_page_token", "next-page"),
            ],
            test_mock={
                "_list_drafts": lambda *args, **kwargs: {
                    "drafts": [
                        {
                            "id": _TEST_DRAFT["id"],
                            "message": {
                                "id": "19a2c0ffee123456",
                                "threadId": "19a2c0ffee123456",
                            },
                        }
                    ],
                    "nextPageToken": "next-page",
                },
                "_fetch_draft": lambda *args, **kwargs: _TEST_DRAFT,
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials)
        fmt = message_format(credentials.scopes)
        try:
            listing = await asyncio.to_thread(
                self._list_drafts,
                service,
                input_data.query,
                input_data.max_results,
                input_data.page_token,
            )
            raw_drafts = [
                await asyncio.to_thread(self._fetch_draft, service, item["id"], fmt)
                for item in listing.get("drafts", [])
            ]
        except HttpError as e:
            raise gmail_error(e, self.name, self.id, "draft") from e

        drafts = [
            GmailDraft(
                id=raw["id"], email=await self._parse_email(raw["message"], service)
            )
            for raw in raw_drafts
        ]
        yield "drafts", drafts
        for draft in drafts:
            yield "draft", draft
        if next_page_token := listing.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    def _list_drafts(service, query: str, max_results: int, page_token: str) -> dict:
        params: dict = {"userId": "me", "maxResults": max_results}
        if query:
            params["q"] = query
        if page_token:
            params["pageToken"] = page_token
        return service.users().drafts().list(**params).execute()

    @staticmethod
    def _fetch_draft(service, draft_id: str, fmt: str) -> dict:
        return fetch_draft(service, draft_id, fmt)


def fetch_draft(service, draft_id: str, fmt: str) -> dict:
    return service.users().drafts().get(userId="me", id=draft_id, format=fmt).execute()
