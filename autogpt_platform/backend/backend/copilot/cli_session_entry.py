"""A user entry of a Claude CLI session file, rewritten without reformatting.

The CLI keeps a session as JSONL, one entry per line, and records every query
the engine sends as a ``{"type": "user", ...}`` entry whose ``message.content``
is a string or a list of content blocks. Two scrubs rewrite the text of such
entries: the upload scrub in ``sdk/service.py``, which removes the marked
warm-context block every turn now carries, and the restore in
``legacy_session_file.py``, which removes the first-turn block older sessions
stored. Both go through ``rewrite_user_entry``, so an entry either comes back
rewritten or, when nothing in it changed or it is not a user entry, not at
all: the caller then keeps the original line byte for byte, and no untouched
entry is ever reformatted. The restore reads entries with
``user_entry_texts``.
"""

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

TextRewrite = Callable[[str], str]


def rewrite_user_entry(entry: object, rewrite: TextRewrite) -> dict[str, object] | None:
    """*entry* with ``rewrite`` applied to the text of its message, or None.

    None means "no change": *entry* is not a CLI user entry, or ``rewrite``
    left every text in it as it was.
    """
    try:
        parsed = CLIUserEntry.model_validate(entry)
    except ValidationError:
        return None
    message = parsed.message.rewritten(rewrite)
    if message is None:
        return None
    return parsed.model_copy(update={"message": message}).model_dump()


def user_entry_texts(entry: object) -> list[str]:
    """The texts of *entry*'s message: its bare string or its text blocks.

    Empty when *entry* is not a CLI user entry of a shape the models read.
    """
    try:
        return CLIUserEntry.model_validate(entry).message.texts()
    except ValidationError:
        return []


def is_user_entry(entry: object) -> bool:
    """Whether *entry* is a ``{"type": "user", ...}`` line, whatever its
    message looks like."""
    try:
        return _CLIEntryType.model_validate(entry).type == "user"
    except ValidationError:
        return False


class _CLIEntryType(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str


class CLITextBlock(BaseModel):
    """A ``text`` content block of a CLI user message."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text"]
    text: str

    def rewritten(self, rewrite: TextRewrite) -> "CLITextBlock | None":
        """The block with its text rewritten, or None when it must be dropped.

        A block that empties out is dropped rather than emitted empty, since
        Anthropic rejects empty text blocks on ``--resume``.
        """
        text = rewrite(self.text)
        if not text:
            return None
        return self.model_copy(update={"text": text})

    def texts(self) -> list[str]:
        return [self.text]


class CLIOpaqueBlock(RootModel[object]):
    """Any other content block (image, tool_result, …), passed through as-is."""

    def rewritten(self, rewrite: TextRewrite) -> "CLIOpaqueBlock | None":
        return self

    def texts(self) -> list[str]:
        return []


# Left to right, so a well-formed text block never falls through to the
# opaque passthrough, which validates anything.
CLIContentBlock = Annotated[
    CLITextBlock | CLIOpaqueBlock, Field(union_mode="left_to_right")
]


class CLIUserTextMessage(BaseModel):
    """A user message whose content is a bare string."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user"]
    content: str

    def rewritten(self, rewrite: TextRewrite) -> "CLIUserTextMessage | None":
        """The rewritten message, or None to keep the original as it is."""
        content = rewrite(self.content)
        if content == self.content or not content:
            # Unchanged, or nothing would be left: keep the original rather
            # than emit empty content, which --resume rejects.
            return None
        return self.model_copy(update={"content": content})

    def texts(self) -> list[str]:
        return [self.content]


class CLIUserBlocksMessage(BaseModel):
    """A user message whose content is a list of content blocks."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user"]
    content: list[CLIContentBlock]

    def rewritten(self, rewrite: TextRewrite) -> "CLIUserBlocksMessage | None":
        """The rewritten message, or None to keep the original as it is."""
        kept = [
            block
            for block in (item.rewritten(rewrite) for item in self.content)
            if block is not None
        ]
        if not kept:
            # Every block emptied out: keep the original intact.
            return None
        message = self.model_copy(update={"content": kept})
        return None if message == self else message

    def texts(self) -> list[str]:
        return [text for block in self.content for text in block.texts()]


class CLIUserEntry(BaseModel):
    """A ``{"type": "user", ...}`` line of a CLI session file."""

    model_config = ConfigDict(extra="allow")

    type: Literal["user"]
    message: CLIUserTextMessage | CLIUserBlocksMessage
