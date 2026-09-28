"""The mark on every warm-context block a chat engine puts into a turn.

Warm context (``context.py`` on a session's first turn, ``context_refresh.py``
on later ones) is shown to the model and never stored. Both engines append the
block to the current turn's model input only: never to the stored user message
nor to the transcript they record. The SDK engine's CLI, though, records the
query exactly as it is sent, so ``sdk/service.py`` scrubs the block from the
CLI session file before it uploads it (``_strip_ephemeral_memory_from_cli_jsonl``).
That scrub has to tell a block the platform put there from a
``<temporal_context>`` tag the user typed, so every block carries this mark:
``append_injected_memory_block`` stamps it and appends the block, and
``strip_injected_memory_text`` removes exactly what it appended.

The mark is an attribute on the open tag carrying a per-process random nonce.
A ``<temporal_context data-agpt-injected="...">`` tag a user types with any
other value is left alone. The nonce is not a secret, and not a proof of
authorship: the model reads it in the prompt, so a user who gets it from the
model and pastes a block carrying it into a later message served by the same
process has that block removed from the uploaded CLI session like an injected
one (their own session only; the stored message keeps it).

Per-process scope is sufficient: a block is injected and scrubbed inside one
turn in one process (the CLI session file is downloaded, appended to, read
back, scrubbed and re-uploaded within one ``stream_chat_completion_sdk`` call),
so no cross-process handoff carries a stamped block. If a turn dies before
upload, nothing is persisted at all. A restart therefore only ever fails
"safe" (a block survives), never by eating user text. The baseline engine
stamps its blocks the same way although it has nothing to scrub: it never
records the model input.

The model still reads a ``<temporal_context ...>`` tag; the attribute is inert.
"""

import logging
import re
import secrets

from .context import CONTEXT_TAG_NAME

logger = logging.getLogger(__name__)

INJECTED_MEMORY_NONCE = secrets.token_hex(16)
INJECTED_MEMORY_MARKER = f'data-agpt-injected="{INJECTED_MEMORY_NONCE}"'

# Matches only a block stamped with this process's nonce: a tag the user
# typed is left alone unless it carries that nonce (the limit the module
# docstring describes). The optional leading ``\n\n`` is the exact separator
# ``append_injected_memory_block`` inserts before the block: removing it with
# the block leaves the user's own text (its leading and trailing whitespace
# and any intentional blank-line runs) byte for byte intact.
INJECTED_MEMORY_BLOCK_RE = re.compile(
    r"(?:\n\n)?<"
    + CONTEXT_TAG_NAME
    + r"\b[^>]*"
    + re.escape(INJECTED_MEMORY_MARKER)
    + r"[^>]*>.*?</"
    + CONTEXT_TAG_NAME
    + r">",
    re.DOTALL,
)

# Open tag matched by name, so the stamp survives attribute or spacing changes
# in the producer. The same ``CONTEXT_TAG_NAME`` the builder emits, so a rename
# there cannot leave this pattern silently matching nothing.
_CONTEXT_OPEN_TAG_RE = re.compile(r"<" + CONTEXT_TAG_NAME + r"\b")


def append_injected_memory_block(text: str, block: str | None) -> str:
    """``text`` with the warm-context ``block`` stamped and appended.

    ``text`` comes back unchanged when there is no block, and when the block
    cannot be stamped: an unstamped block could not be scrubbed from the CLI
    session file, so it would replay on every later turn, a forgotten fact
    with it. The turn goes ahead without memory rather than with a block
    nothing can remove (``mark_injected_memory_block`` logs the miss).
    """
    if not block:
        return text
    marked = mark_injected_memory_block(block)
    if marked is None:
        return text
    return f"{text}\n\n{marked}"


def mark_injected_memory_block(block: str) -> str | None:
    """Stamp the provenance nonce onto a ``<temporal_context>`` block.

    Matches the open tag by NAME rather than as an exact string: the block is
    produced by ``context._format_context``, and an attribute or spacing
    change there must not silently un-stamp it.

    ``None`` when the block has no recognisable open tag. The miss is logged,
    so it surfaces, and the block must not be injected (see
    ``append_injected_memory_block``).
    """
    marked, substitutions = _CONTEXT_OPEN_TAG_RE.subn(
        f"<{CONTEXT_TAG_NAME} {INJECTED_MEMORY_MARKER}", block, count=1
    )
    if not substitutions:
        logger.warning(
            "Warm-context block carries no <temporal_context> open tag; it "
            "cannot be marked for the transcript scrub, so it is not injected"
        )
        return None
    return marked


def strip_injected_memory_text(text: str) -> str:
    """Remove nonce-marked ``<temporal_context>`` blocks (plus the injected
    ``\\n\\n`` separator) from *text*, leaving all other content untouched.

    Removes ONLY what the injector added (no global ``.strip()`` or blank-line
    collapse), so a user's own leading and trailing whitespace and intentional
    blank lines survive. Returns *text* verbatim when no marked block is
    present.
    """
    return INJECTED_MEMORY_BLOCK_RE.sub("", text)
