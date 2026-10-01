import asyncio
from enum import Enum

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._gmail_api import (
    GMAIL_MODIFY_SCOPE,
    GmailChangeResult,
    GmailTarget,
    gmail_error,
    messages_or_threads,
    modify_labels,
    require_id,
    to_change_result,
)
from .gmail import GmailBase


class TrashAction(str, Enum):
    TRASH = "trash"
    RESTORE = "restore"


class SpamAction(str, Enum):
    SPAM = "spam"
    NOT_SPAM = "not_spam"


class ReadState(str, Enum):
    READ = "read"
    UNREAD = "unread"


# Label IDs to (add, remove) for each change.
_SPAM_LABELS: dict[SpamAction, tuple[list[str], list[str]]] = {
    SpamAction.SPAM: (["SPAM"], ["INBOX"]),
    SpamAction.NOT_SPAM: (["INBOX"], ["SPAM"]),
}
_READ_LABELS: dict[ReadState, tuple[list[str], list[str]]] = {
    ReadState.READ: ([], ["UNREAD"]),
    ReadState.UNREAD: (["UNREAD"], []),
}

_TEST_ID = "19a2b3c4d5e6f708"


def _id_field() -> str:
    return SchemaField(
        description=(
            "ID of the message (an email's id), or of the thread (an email's "
            "threadId) when target is thread"
        )
    )


def _target_field() -> GmailTarget:
    return SchemaField(
        description="Change just this message, or every message in the thread",
        default=GmailTarget.MESSAGE,
    )


def _result_field() -> GmailChangeResult:
    return SchemaField(
        description="The message or thread, with its labels after the change"
    )


class GmailTrashBlock(GmailBase):
    """Move a Gmail message or thread to the Trash, or restore it."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [GMAIL_MODIFY_SCOPE]
        )
        message_or_thread_id: str = _id_field()
        target: GmailTarget = _target_field()
        action: TrashAction = SchemaField(
            description="Move it to the Trash, or restore it from the Trash",
            default=TrashAction.TRASH,
        )

    class Output(BlockSchemaOutput):
        result: GmailChangeResult = _result_field()

    def __init__(self):
        trashed_thread = {
            "id": _TEST_ID,
            "historyId": "645120",
            "messages": [
                {"id": _TEST_ID, "threadId": _TEST_ID, "labelIds": ["TRASH", "SENT"]},
                {"id": "19a2b3d1e2f3a4b5", "threadId": _TEST_ID, "labelIds": ["TRASH"]},
            ],
        }
        super().__init__(
            id="88806b69-781d-4468-a543-07606621f98d",
            description=(
                "Move a Gmail message, or a whole thread, to the Trash, or restore "
                "it from the Trash. Gmail deletes mail that stays in the Trash for "
                "30 days."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailTrashBlock.Input,
            output_schema=GmailTrashBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "message_or_thread_id": _TEST_ID,
                "target": GmailTarget.THREAD,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    GmailChangeResult(
                        id=_TEST_ID, thread_id=_TEST_ID, label_ids=["TRASH", "SENT"]
                    ),
                )
            ],
            test_mock={"_trash": lambda *args, **kwargs: trashed_thread},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        item_id = require_id(input_data.message_or_thread_id, self.name, self.id)
        service = self._build_service(credentials)
        try:
            resource = await asyncio.to_thread(
                self._trash, service, input_data.target, item_id, input_data.action
            )
        except HttpError as e:
            raise gmail_error(e, self.name, self.id, input_data.target.value) from e
        yield "result", to_change_result(resource, input_data.target)

    @staticmethod
    def _trash(service, target: GmailTarget, item_id: str, action: TrashAction) -> dict:
        collection = messages_or_threads(service, target)
        move = collection.untrash if action == TrashAction.RESTORE else collection.trash
        return move(userId="me", id=item_id).execute()


class GmailSpamBlock(GmailBase):
    """Report a Gmail message or thread as spam, or mark it as not spam."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [GMAIL_MODIFY_SCOPE]
        )
        message_or_thread_id: str = _id_field()
        target: GmailTarget = _target_field()
        action: SpamAction = SchemaField(
            description=(
                "Move it to Spam, or mark it as not spam, which moves it back "
                "to the Inbox"
            ),
            default=SpamAction.SPAM,
        )

    class Output(BlockSchemaOutput):
        result: GmailChangeResult = _result_field()

    def __init__(self):
        super().__init__(
            id="64354af2-8b3c-4e85-9b00-f84f2e32fd38",
            description=(
                "Report a Gmail message, or a whole thread, as spam, which moves it "
                "to Spam. Or mark it as not spam, which moves it back to the Inbox."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailSpamBlock.Input,
            output_schema=GmailSpamBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "message_or_thread_id": "19a2b3d1e2f3a4b5",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    GmailChangeResult(
                        id="19a2b3d1e2f3a4b5",
                        thread_id=_TEST_ID,
                        label_ids=["SPAM", "UNREAD"],
                    ),
                )
            ],
            test_mock={
                "_modify": lambda *args, **kwargs: {
                    "id": "19a2b3d1e2f3a4b5",
                    "threadId": _TEST_ID,
                    "labelIds": ["SPAM", "UNREAD"],
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        item_id = require_id(input_data.message_or_thread_id, self.name, self.id)
        add, remove = _SPAM_LABELS[input_data.action]
        service = self._build_service(credentials)
        try:
            resource = await asyncio.to_thread(
                self._modify, service, input_data.target, item_id, add, remove
            )
        except HttpError as e:
            raise gmail_error(e, self.name, self.id, input_data.target.value) from e
        yield "result", to_change_result(resource, input_data.target)

    @staticmethod
    def _modify(
        service, target: GmailTarget, item_id: str, add: list[str], remove: list[str]
    ) -> dict:
        return modify_labels(service, target, item_id, add, remove)


class GmailMarkAsReadBlock(GmailBase):
    """Mark a Gmail message or thread as read or unread."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [GMAIL_MODIFY_SCOPE]
        )
        message_or_thread_id: str = _id_field()
        target: GmailTarget = _target_field()
        mark_as: ReadState = SchemaField(
            description="Mark it as read or as unread", default=ReadState.READ
        )

    class Output(BlockSchemaOutput):
        result: GmailChangeResult = _result_field()

    def __init__(self):
        super().__init__(
            id="d9ee46bf-331f-406f-8b36-6cbddb87f4f7",
            description="Mark a Gmail message, or a whole thread, as read or unread.",
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailMarkAsReadBlock.Input,
            output_schema=GmailMarkAsReadBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "message_or_thread_id": "19a2b3d1e2f3a4b5",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    GmailChangeResult(
                        id="19a2b3d1e2f3a4b5",
                        thread_id=_TEST_ID,
                        label_ids=["INBOX", "IMPORTANT"],
                    ),
                )
            ],
            test_mock={
                "_modify": lambda *args, **kwargs: {
                    "id": "19a2b3d1e2f3a4b5",
                    "threadId": _TEST_ID,
                    "labelIds": ["INBOX", "IMPORTANT"],
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        item_id = require_id(input_data.message_or_thread_id, self.name, self.id)
        add, remove = _READ_LABELS[input_data.mark_as]
        service = self._build_service(credentials)
        try:
            resource = await asyncio.to_thread(
                self._modify, service, input_data.target, item_id, add, remove
            )
        except HttpError as e:
            raise gmail_error(e, self.name, self.id, input_data.target.value) from e
        yield "result", to_change_result(resource, input_data.target)

    @staticmethod
    def _modify(
        service, target: GmailTarget, item_id: str, add: list[str], remove: list[str]
    ) -> dict:
        return modify_labels(service, target, item_id, add, remove)
