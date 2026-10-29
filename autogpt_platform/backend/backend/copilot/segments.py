"""Which connection a given turn actually ran on.

A session's metadata says where a chat *starts*. For most of this codebase's
life that was the same thing as where every turn in it ran, because the
connection was chosen at creation and could never change. Once it can --
after a usage limit, on a user's switch -- the two stop being the same, and
a session-level answer starts lying about history: every past turn would
appear to have run on whatever the connection happens to be now.

So each assistant turn records its own route, and the session's metadata
becomes the answer for turns that have none -- segment zero. That covers
every row written before the columns existed, without a backfill that would
have to invent what it could not know.

Nothing here rewrites a stamped turn. A route change applies forward.
"""

from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.model import ChatMessage, ChatSession


class Segment:
    """The connection one turn ran on, and where that answer came from."""

    __slots__ = ("auth_provider", "credential_id", "is_segment_zero")
    auth_provider: CopilotLlmAuthProvider
    credential_id: str | None
    is_segment_zero: bool

    def __init__(
        self,
        auth_provider: CopilotLlmAuthProvider,
        credential_id: str | None,
        *,
        is_segment_zero: bool,
    ) -> None:
        self.auth_provider = auth_provider
        self.credential_id = credential_id
        # True when this came from the session rather than the turn itself:
        # the turn predates per-turn stamping, or is the first of the chat.
        self.is_segment_zero = is_segment_zero

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return (
            self.auth_provider == other.auth_provider
            and self.credential_id == other.credential_id
        )

    def __repr__(self) -> str:
        return (
            f"Segment({self.auth_provider!r}, {self.credential_id!r}, "
            f"is_segment_zero={self.is_segment_zero})"
        )


def segment_of(message: ChatMessage, session: ChatSession) -> Segment:
    """The route this turn ran on, falling back to the session's own.

    A stamp is only trusted when the turn carries an auth provider. A
    credential id without one cannot be interpreted -- it names an account
    without saying whose -- so it is ignored rather than guessed at.
    """
    if message.llm_auth_provider is None:
        return session_segment(session)
    return Segment(
        message.llm_auth_provider,
        message.llm_credential_id,
        is_segment_zero=False,
    )


def session_segment(session: ChatSession) -> Segment:
    """Segment zero: where this chat started."""
    return Segment(
        session.metadata.llm_auth_provider,
        session.metadata.llm_credential_id,
        is_segment_zero=True,
    )


def segment_boundaries(messages: list[ChatMessage], session: ChatSession) -> list[int]:
    """Indexes where the connection changed, in order.

    Only assistant turns carry a route, so user and tool rows are skipped
    rather than treated as a change back to segment zero. The first assistant
    turn is a boundary only when it ran somewhere other than where the chat
    started.
    """
    boundaries: list[int] = []
    current = session_segment(session)
    for index, message in enumerate(messages):
        if message.role != "assistant" or message.llm_auth_provider is None:
            continue
        segment = segment_of(message, session)
        if segment != current:
            boundaries.append(index)
            current = segment
    return boundaries


def stamp_segment(
    messages: list[ChatMessage],
    start_index: int,
    segment: Segment,
) -> None:
    """Record the connection on the assistant turns this run produced.

    Mirrors how ``model`` / ``routing_source`` are stamped: only rows that
    have none are written, so a re-stamp cannot overwrite what an earlier
    run recorded. Rows already flushed to the database mid-turn are flagged
    for back-fill by the save path.
    """
    for message in messages[start_index:]:
        if message.role != "assistant" or message.llm_auth_provider is not None:
            continue
        message.llm_auth_provider = segment.auth_provider
        message.llm_credential_id = segment.credential_id
        if message.sequence is not None:
            message.stamps_pending_save = True
