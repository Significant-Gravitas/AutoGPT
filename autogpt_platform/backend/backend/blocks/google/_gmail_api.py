"""Shared models and helpers for the Gmail blocks."""

from email.utils import getaddresses, parseaddr
from enum import Enum
from typing import List, Optional

from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from backend.util.exceptions import BlockExecutionError, BlockInputError

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"
GMAIL_LABELS_SCOPE = "https://www.googleapis.com/auth/gmail.labels"
GMAIL_METADATA_SCOPE = "https://www.googleapis.com/auth/gmail.metadata"

SYSTEM_LABEL_IDS = frozenset(
    {
        "INBOX",
        "SPAM",
        "TRASH",
        "UNREAD",
        "STARRED",
        "IMPORTANT",
        "SENT",
        "DRAFT",
        "CHAT",
        "CATEGORY_PERSONAL",
        "CATEGORY_SOCIAL",
        "CATEGORY_PROMOTIONS",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
    }
)


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


class GmailDraft(BaseModel):
    """A Gmail draft: its ID and its message."""

    id: str = Field(
        description="Draft ID. It stays the same when the draft is edited; the message ID changes."
    )
    email: Email = Field(
        description="The draft's message: recipients, subject, body and attachments"
    )


class GmailLabel(BaseModel):
    """A Gmail label."""

    id: str = Field(description="Label ID, e.g. Label_12, or a system ID such as INBOX")
    name: str = Field(
        description="Display name. Nested labels use '/', e.g. Projects/Alpha"
    )
    type: str = Field(description="user or system")
    background_color: Optional[str] = Field(
        default=None, description="Background color as #RRGGBB"
    )
    text_color: Optional[str] = Field(default=None, description="Text color as #RRGGBB")


class GmailChangeResult(BaseModel):
    """A message or thread after a change, with the labels it now has."""

    id: str = Field(description="ID of the message or thread that changed")
    thread_id: str = Field(
        description="ID of its thread (the same as id when a thread changed)"
    )
    label_ids: list[str] = Field(
        description="Label IDs it has now. For a thread, the labels on any of its messages."
    )


class GmailTarget(str, Enum):
    MESSAGE = "message"
    THREAD = "thread"


def email_from_message(
    msg: dict,
    body: str,
    attachments: list[Attachment],
    thread_id: str | None = None,
) -> Email:
    """Map a Gmail API message resource to an Email.

    GmailBase supplies the decoded body and the attachments, since reading
    them can take extra API calls. thread_id fills in a missing threadId.
    """
    headers = {
        header["name"].lower(): header["value"]
        for header in msg.get("payload", {}).get("headers", [])
    }
    return Email(
        threadId=msg.get("threadId", thread_id),
        labelIds=msg.get("labelIds", []),
        id=msg["id"],
        subject=headers.get("subject", "No Subject"),
        snippet=msg.get("snippet", ""),
        from_=parseaddr(headers.get("from", ""))[1],
        to=_recipients(headers.get("to", "")),
        cc=_recipients(headers.get("cc", "")),
        bcc=_recipients(headers.get("bcc", "")),
        date=headers.get("date", ""),
        body=body,
        sizeEstimate=msg.get("sizeEstimate", 0),
        attachments=attachments,
    )


def _recipients(header: str) -> list[str]:
    return [addr.strip() for _, addr in getaddresses([header])]


def message_format(scopes: list[str] | None) -> str:
    """Gmail refuses full-format reads from a token that has the metadata scope."""
    granted = [scope.lower() for scope in scopes or []]
    return "metadata" if GMAIL_METADATA_SCOPE in granted else "full"


def gmail_error(
    exc: HttpError, block_name: str, block_id: str, item: str = "item"
) -> BlockExecutionError:
    """Turn a Gmail API error into a message the user can act on."""
    reason = str(exc.reason)
    if exc.status_code == 404:
        message = (
            f"Gmail couldn't find that {item}, or the connected Google account "
            "can't access it."
        )
    elif exc.status_code == 400 and "invalid id" in reason.lower():
        message = f"That isn't a valid Gmail {item} ID."
    elif exc.status_code == 403 and "insufficient" in reason.lower():
        message = (
            "The connected Google account hasn't granted the Gmail access this "
            "block needs. Reconnect Google and approve Gmail access."
        )
    else:
        message = f"Gmail API error {exc.status_code}: {reason}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )


def require_id(value: str, block_name: str, block_id: str) -> str:
    item_id = value.strip()
    if not item_id:
        raise BlockInputError(
            message="Give the ID of the message or thread to change.",
            block_name=block_name,
            block_id=block_id,
        )
    return item_id


def messages_or_threads(service, target: GmailTarget):
    """The Gmail API collection a message or thread change goes to."""
    users = service.users()
    return users.threads() if target == GmailTarget.THREAD else users.messages()


def modify_labels(
    service, target: GmailTarget, item_id: str, add: list[str], remove: list[str]
) -> dict:
    return (
        messages_or_threads(service, target)
        .modify(
            userId="me",
            id=item_id,
            body={"addLabelIds": add, "removeLabelIds": remove},
        )
        .execute()
    )


def to_change_result(resource: dict, target: GmailTarget) -> GmailChangeResult:
    """Read the labels off a changed message or thread resource."""
    if target == GmailTarget.THREAD:
        labels = [
            label_id
            for message in resource.get("messages", [])
            for label_id in message.get("labelIds", [])
        ]
        return GmailChangeResult(
            id=resource["id"],
            thread_id=resource["id"],
            label_ids=list(dict.fromkeys(labels)),
        )
    return GmailChangeResult(
        id=resource["id"],
        thread_id=resource.get("threadId", ""),
        label_ids=resource.get("labelIds", []),
    )


def list_labels(service) -> list[dict]:
    return service.users().labels().list(userId="me").execute().get("labels", [])


def find_label(labels: list[dict], name_or_id: str) -> dict | None:
    """Find a label by ID, or else by name."""
    by_id = next((label for label in labels if label["id"] == name_or_id), None)
    return by_id if by_id is not None else find_label_by_name(labels, name_or_id)


def find_label_by_name(labels: list[dict], name: str) -> dict | None:
    """Find a label by name, ignoring case as Gmail does."""
    wanted = name.casefold()
    return next(
        (label for label in labels if label.get("name", "").casefold() == wanted),
        None,
    )


def label_id_for(labels: list[dict], name_or_id: str) -> str | None:
    """The ID for a label name or ID; system label IDs match in any case."""
    label = find_label(labels, name_or_id)
    if label is not None:
        return label["id"]
    if name_or_id.upper() in SYSTEM_LABEL_IDS:
        return name_or_id.upper()
    return None


def create_label(
    service, labels: list[dict], name: str, body: dict, with_parents: bool
) -> list[dict]:
    """Create a user label, first creating missing parents of a nested name.

    Returns the labels it created, the requested one last, and adds them to
    labels so later lookups find them.
    """
    parts = name.split("/")
    parents = (
        ["/".join(parts[:depth]) for depth in range(1, len(parts))]
        if with_parents
        else []
    )
    missing = [parent for parent in parents if label_id_for(labels, parent) is None]
    created = [_insert_label(service, {"name": parent}) for parent in missing]
    created.append(_insert_label(service, {**body, "name": name}))
    labels.extend(created)
    return created


def _insert_label(service, body: dict) -> dict:
    return service.users().labels().create(userId="me", body=body).execute()


def to_gmail_label(label: dict) -> GmailLabel:
    color = label.get("color", {})
    return GmailLabel(
        id=label["id"],
        name=label.get("name", ""),
        type=label.get("type", "user"),
        background_color=color.get("backgroundColor"),
        text_color=color.get("textColor"),
    )


def clean_label_name(name: str) -> str:
    """Tidy a label name: trim spaces around each '/'-separated part."""
    return "/".join(part.strip() for part in name.split("/") if part.strip())
