"""Shared marker for the warm context appended by both chat engines."""

import logging
import re
import secrets

from .context import CONTEXT_TAG_NAME

logger = logging.getLogger(__name__)

# Provenance nonce stamped onto the server-injected follow-up warm-context
# block (see ``_append_follow_up_warm_context``). Only blocks carrying THIS
# process's nonce are scrubbed from the persisted transcript. The nonce is
# random, so a ``<temporal_context data-agpt-injected="...">`` tag a user types
# with any other value is left alone. It is not a secret, and not a proof of
# authorship: the model reads it in the prompt, so a user who gets it from the
# model and pastes a block carrying it into a later message served by the same
# process has that block removed from the uploaded CLI session like an
# injected one (their own session only; the stored message keeps it).
#
# Per-process scope is sufficient: a block is injected and scrubbed inside a
# single ``stream_chat_completion_sdk`` call — the CLI session file is
# downloaded, appended to, read back, scrubbed and re-uploaded within one turn
# in one process — so no cross-process handoff carries a stamped block. If a
# turn dies before upload, nothing is persisted at all. A restart therefore
# only ever fails "safe" (a block survives), never by eating user text.
#
# Internal format: the model still reads a ``<temporal_context ...>`` tag; the
# attribute is inert.
INJECTED_MEMORY_NONCE = secrets.token_hex(16)
INJECTED_MEMORY_MARKER = f'data-agpt-injected="{INJECTED_MEMORY_NONCE}"'
# Matches only a block stamped with this process's nonce: a tag the user
# typed is left alone unless it carries that nonce (the limit
# ``INJECTED_MEMORY_NONCE`` describes).
# The optional leading ``\n\n`` is the exact separator that
# ``_append_follow_up_warm_context`` inserts before the block — removing it
# together with the block leaves the user's own text (its leading/trailing
# whitespace and any intentional blank-line runs) byte-for-byte intact.
INJECTED_MEMORY_BLOCK_RE = re.compile(
    r"(?:\n\n)?<"
    + CONTEXT_TAG_NAME
    + r"\b[^<>]{0,256}?"
    + re.escape(INJECTED_MEMORY_MARKER)
    + r"[^<>]{0,256}>.*?</"
    + CONTEXT_TAG_NAME
    + r">",
    re.DOTALL,
)
# Open tag matched by name, so the stamp survives attribute/spacing changes
# in the producer. Same ``CONTEXT_TAG_NAME`` the builder emits, so a rename
# there cannot leave this pattern silently matching nothing.
_CONTEXT_OPEN_TAG_RE = re.compile(r"<" + CONTEXT_TAG_NAME + r"\b")


def mark_injected_memory_block(block: str) -> str:
    """Stamp the provenance nonce onto a ``<temporal_context>`` block.

    Matches the open tag by NAME rather than as an exact string: the block is
    produced by ``graphiti.context._format_context`` in another module, and an
    attribute or spacing change there must not silently un-stamp it — an
    unstamped block is never scrubbed from the uploaded transcript, so it
    replays as stale context on ``--resume``.

    A block with no recognisable open tag is returned untouched (memory must
    never break a turn) but logged, so the miss surfaces instead of quietly
    growing every transcript.
    """
    marked, substitutions = _CONTEXT_OPEN_TAG_RE.subn(
        f"<{CONTEXT_TAG_NAME} {INJECTED_MEMORY_MARKER}", block, count=1
    )
    if not substitutions:
        logger.warning(
            "Warm-context block carries no <temporal_context> open tag; it "
            "cannot be marked and will not be scrubbed from the transcript"
        )
    return marked
