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
from backend.util.exceptions import BlockInputError

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._gmail_api import (
    GMAIL_LABELS_SCOPE,
    GMAIL_MODIFY_SCOPE,
    GmailChangeResult,
    GmailLabel,
    GmailTarget,
    clean_label_name,
    create_label,
    find_label_by_name,
    gmail_error,
    label_id_for,
    list_labels,
    messages_or_threads,
    modify_labels,
    require_id,
    to_change_result,
    to_gmail_label,
)
from .gmail import GmailBase


class LabelColor(str, Enum):
    NONE = "none"
    BLACK = "black"
    DARK_GRAY = "dark_gray"
    GRAY = "gray"
    LIGHT_GRAY = "light_gray"
    WHITE = "white"
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    MINT = "mint"
    TEAL = "teal"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    DARK_RED = "dark_red"
    DARK_ORANGE = "dark_orange"
    DARK_GREEN = "dark_green"
    DARK_BLUE = "dark_blue"
    DARK_PURPLE = "dark_purple"
    DARK_PINK = "dark_pink"
    BROWN = "brown"


# (background, text) pairs from Gmail's allowed label palette.
LABEL_COLORS: dict[LabelColor, tuple[str, str]] = {
    LabelColor.BLACK: ("#000000", "#ffffff"),
    LabelColor.DARK_GRAY: ("#434343", "#ffffff"),
    LabelColor.GRAY: ("#666666", "#ffffff"),
    LabelColor.LIGHT_GRAY: ("#cccccc", "#000000"),
    LabelColor.WHITE: ("#ffffff", "#000000"),
    LabelColor.RED: ("#fb4c2f", "#ffffff"),
    LabelColor.ORANGE: ("#ffad47", "#000000"),
    LabelColor.YELLOW: ("#fad165", "#000000"),
    LabelColor.GREEN: ("#16a765", "#ffffff"),
    LabelColor.MINT: ("#43d692", "#000000"),
    LabelColor.TEAL: ("#2da2bb", "#ffffff"),
    LabelColor.BLUE: ("#4a86e8", "#ffffff"),
    LabelColor.PURPLE: ("#a479e2", "#ffffff"),
    LabelColor.PINK: ("#f691b2", "#000000"),
    LabelColor.DARK_RED: ("#822111", "#ffffff"),
    LabelColor.DARK_ORANGE: ("#a46a21", "#ffffff"),
    LabelColor.DARK_GREEN: ("#076239", "#ffffff"),
    LabelColor.DARK_BLUE: ("#1c4587", "#ffffff"),
    LabelColor.DARK_PURPLE: ("#41236d", "#ffffff"),
    LabelColor.DARK_PINK: ("#83334c", "#ffffff"),
    LabelColor.BROWN: ("#7a4706", "#ffffff"),
}


class LabelListVisibility(str, Enum):
    SHOW = "show"
    SHOW_IF_UNREAD = "show_if_unread"
    HIDE = "hide"


_LABEL_LIST_VISIBILITY: dict[LabelListVisibility, str] = {
    LabelListVisibility.SHOW: "labelShow",
    LabelListVisibility.SHOW_IF_UNREAD: "labelShowIfUnread",
    LabelListVisibility.HIDE: "labelHide",
}

_TEST_ID = "19a2b3c4d5e6f708"


class GmailUpdateLabelsBlock(GmailBase):
    """Add and remove labels on a Gmail message or thread in one step."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [GMAIL_MODIFY_SCOPE]
        )
        message_or_thread_id: str = SchemaField(
            description=(
                "ID of the message (an email's id), or of the thread (an email's "
                "threadId) when target is thread"
            )
        )
        target: GmailTarget = SchemaField(
            description="Change just this message, or every message in the thread",
            default=GmailTarget.MESSAGE,
        )
        add_labels: list[str] = SchemaField(
            description=(
                "Label names or IDs to add, e.g. Clients/Acme or STARRED. Labels "
                "that don't exist yet are created."
            ),
            default_factory=list,
        )
        remove_labels: list[str] = SchemaField(
            description=(
                "Label names or IDs to remove, e.g. INBOX to archive or UNREAD to "
                "mark as read"
            ),
            default_factory=list,
        )

    class Output(BlockSchemaOutput):
        result: GmailChangeResult = SchemaField(
            description="The message or thread, with its labels after the change"
        )
        created_labels: list[str] = SchemaField(
            description="Names of labels that didn't exist and were created"
        )

    def __init__(self):
        super().__init__(
            id="2ef99a56-82d8-46ca-9981-2e0a05397031",
            description=(
                "Add and remove labels on a Gmail message, or a whole thread, in "
                "one step. Takes label names or IDs, including system labels such "
                "as INBOX, STARRED or UNREAD, and creates any label to add that "
                "doesn't exist yet."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailUpdateLabelsBlock.Input,
            output_schema=GmailUpdateLabelsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "message_or_thread_id": _TEST_ID,
                "add_labels": ["Clients/Acme", "STARRED"],
                "remove_labels": ["INBOX"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    GmailChangeResult(
                        id=_TEST_ID,
                        thread_id=_TEST_ID,
                        label_ids=["Label_43", "STARRED", "IMPORTANT"],
                    ),
                ),
                ("created_labels", ["Clients/Acme"]),
            ],
            test_mock={
                "_update_labels": lambda *args, **kwargs: (
                    {
                        "id": _TEST_ID,
                        "threadId": _TEST_ID,
                        "labelIds": ["Label_43", "STARRED", "IMPORTANT"],
                    },
                    ["Clients/Acme"],
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        item_id = require_id(input_data.message_or_thread_id, self.name, self.id)
        add = _clean_names(input_data.add_labels)
        remove = _clean_names(input_data.remove_labels)
        self._check_changes(add, remove)
        service = self._build_service(credentials)
        try:
            resource, created = await asyncio.to_thread(
                self._update_labels, service, input_data.target, item_id, add, remove
            )
        except HttpError as e:
            raise gmail_error(e, self.name, self.id, input_data.target.value) from e
        yield "result", to_change_result(resource, input_data.target)
        if created:
            yield "created_labels", created

    def _check_changes(self, add: list[str], remove: list[str]) -> None:
        if not add and not remove:
            raise BlockInputError(
                message="Give at least one label to add or remove.",
                block_name=self.name,
                block_id=self.id,
            )
        removing = {name.casefold() for name in remove}
        both = [name for name in add if name.casefold() in removing]
        if both:
            raise BlockInputError(
                message=f"Can't add and remove the same label: {', '.join(both)}",
                block_name=self.name,
                block_id=self.id,
            )

    @staticmethod
    def _update_labels(
        service, target: GmailTarget, item_id: str, add: list[str], remove: list[str]
    ) -> tuple[dict, list[str]]:
        labels = list_labels(service)
        add_ids, created = _ids_to_add(service, labels, add)
        remove_ids = [
            label_id
            for name in remove
            if (label_id := label_id_for(labels, name)) is not None
        ]
        if not add_ids and not remove_ids:
            # Only labels that don't exist were to be removed: nothing changes.
            unchanged = (
                messages_or_threads(service, target)
                .get(userId="me", id=item_id, format="minimal")
                .execute()
            )
            return unchanged, created
        changed = modify_labels(
            service,
            target,
            item_id,
            list(dict.fromkeys(add_ids)),
            list(dict.fromkeys(remove_ids)),
        )
        return changed, created


class GmailCreateLabelBlock(GmailBase):
    """Create a Gmail label, nested under parent labels if its name has '/'."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [GMAIL_LABELS_SCOPE]
        )
        name: str = SchemaField(
            description=(
                "Name of the label. Use '/' to nest it under another label, "
                "e.g. Projects/Alpha."
            )
        )
        color: LabelColor = SchemaField(
            description="Label color", default=LabelColor.NONE
        )
        show_in_label_list: LabelListVisibility = SchemaField(
            description="Show the label in Gmail's label list always, only when it has unread mail, or never",
            default=LabelListVisibility.SHOW,
        )
        show_in_message_list: bool = SchemaField(
            description="Show the label on emails in Gmail's message list",
            default=True,
        )
        create_parent_labels: bool = SchemaField(
            description="Create missing parent labels of a nested name, e.g. Projects for Projects/Alpha",
            default=True,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        label: GmailLabel = SchemaField(
            description="The new label, or the existing label with that name"
        )
        created: bool = SchemaField(
            description="False when a label with that name already existed"
        )

    def __init__(self):
        test_label = {
            "id": "Label_42",
            "name": "Projects/Alpha",
            "type": "user",
            "color": {"backgroundColor": "#4a86e8", "textColor": "#ffffff"},
        }
        super().__init__(
            id="ad658a74-2754-4aff-a8d8-79c2faab5030",
            description=(
                "Create a Gmail label, with an optional color and visibility. A "
                "nested name like Projects/Alpha also creates missing parent "
                "labels. If the name is taken, returns the existing label."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GmailCreateLabelBlock.Input,
            output_schema=GmailCreateLabelBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "name": "Projects/Alpha",
                "color": LabelColor.BLUE,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("label", to_gmail_label(test_label)), ("created", True)],
            test_mock={"_create_label": lambda *args, **kwargs: (test_label, True)},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        name = clean_label_name(input_data.name)
        if not name:
            raise BlockInputError(
                message="Give the label a name.",
                block_name=self.name,
                block_id=self.id,
            )
        service = self._build_service(credentials)
        try:
            label, created = await asyncio.to_thread(
                self._create_label,
                service,
                name,
                label_body(input_data),
                input_data.create_parent_labels,
            )
        except HttpError as e:
            raise gmail_error(e, self.name, self.id, "label") from e
        yield "label", to_gmail_label(label)
        yield "created", created

    @staticmethod
    def _create_label(
        service, name: str, body: dict, with_parents: bool
    ) -> tuple[dict, bool]:
        labels = list_labels(service)
        existing = find_label_by_name(labels, name)
        if existing is not None:
            return existing, False
        return create_label(service, labels, name, body, with_parents)[-1], True


def label_body(input_data: GmailCreateLabelBlock.Input) -> dict:
    """The labels.create settings for the chosen color and visibility."""
    body: dict = {
        "labelListVisibility": _LABEL_LIST_VISIBILITY[input_data.show_in_label_list],
        "messageListVisibility": "show" if input_data.show_in_message_list else "hide",
    }
    if input_data.color != LabelColor.NONE:
        background, text = LABEL_COLORS[input_data.color]
        body["color"] = {"backgroundColor": background, "textColor": text}
    return body


def _clean_names(names: list[str]) -> list[str]:
    return [name for name in map(clean_label_name, names) if name]


def _ids_to_add(
    service, labels: list[dict], names: list[str]
) -> tuple[list[str], list[str]]:
    """Label IDs for names to add, creating user labels that don't exist yet.

    System labels such as STARRED resolve to their IDs and are never created.
    Returns the IDs and the names of any labels created.
    """
    ids: list[str] = []
    created: list[str] = []
    for name in names:
        label_id = label_id_for(labels, name)
        if label_id is None:
            new_labels = create_label(service, labels, name, {}, with_parents=True)
            created += [label["name"] for label in new_labels]
            label_id = new_labels[-1]["id"]
        ids.append(label_id)
    return ids, created
