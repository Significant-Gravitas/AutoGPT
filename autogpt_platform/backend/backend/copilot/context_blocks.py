"""The server-written blocks that open a session's first user message.

The first-turn builders return them as blocks, and the text is rendered where
it is needed: on the first message's row, ahead of the history a turn that
cannot resume sends instead, and in the re-send after a compaction
(``sdk/after_compaction.py``), which shortens a block by its items rather than
by parsing the rendered text.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict


class ContextBlock(BaseModel):
    """One ``<tag>`` block.

    ``items`` are entries a tool lists again (skills, hires, routines,
    installed workflows), so a copy short of room can leave some out.
    ``head`` is the text before them, or the whole body of a block that has
    none; ``tail`` is the text after them.
    """

    model_config = ConfigDict(frozen=True)

    tag: str
    head: str = ""
    items: tuple[str, ...] = ()
    tail: str = ""

    @property
    def body(self) -> str:
        return "\n".join(part for part in (self.head, *self.items, self.tail) if part)

    @property
    def text(self) -> str:
        """The block as a message carries it, followed by a blank line."""
        return f"<{self.tag}>\n{self.body}\n</{self.tag}>\n\n"


def render_context_blocks(blocks: Iterable[ContextBlock]) -> str:
    return "".join(block.text for block in blocks)
