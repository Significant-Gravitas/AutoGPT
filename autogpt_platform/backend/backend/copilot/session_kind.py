"""What kind of turn is running, decided once as the executor hands it to an
engine.

A turn's kind used to be read off the session row wherever it mattered, each
reader with its own rule: whether anyone is watching the turn, whether it
answers a hire's kickoff message, whether its prompt was written on a chat
platform. ``SessionKind`` holds those answers once. The executor builds it from
the session's stored facts and the queue entry's own, set at dispatch, and every
reader takes it from there. The stored fields are unchanged. (The stored
``ChatSessionMetadata.kind`` is a different thing: a label for dream artifacts.)
"""

from pydantic import BaseModel, ConfigDict

from backend.copilot.expert_kickoff import is_expert_kickoff_turn
from backend.copilot.model import ChatSession, ChatSessionOrigin
from backend.copilot.tools import kickoff_turn_disabled_tools, origin_disabled_tools


class SessionKind(BaseModel):
    """One turn's kind; build it with :meth:`for_turn`."""

    model_config = ConfigDict(frozen=True)

    origin: ChatSessionOrigin | None
    """Who opened the session; ``None`` on a row from before the field."""
    unattended: bool
    """Nobody can answer this turn: the scheduler fired it, or an automation
    opened the chat. A chat from before the origin field counts as watched."""
    expert_kickoff: bool
    """The turn answers a hire's server-written kickoff message."""
    from_chat_platform: bool
    """The prompt was written on a linked chat platform."""
    builder: bool
    """The session is bound to a builder graph."""
    delegated: bool
    """Another session delegated this one's work or handed it over."""
    voice: bool
    """The user spoke this turn's message."""

    @classmethod
    def for_turn(
        cls,
        session: ChatSession,
        *,
        scheduled: bool = False,
        voice: bool = False,
        from_chat_platform: bool = False,
    ) -> "SessionKind":
        """The kind of the turn *session* is about to run. *scheduled*, *voice*
        and *from_chat_platform* are the dispatch's own facts, carried on the
        queue entry: a bot's prompt counts even in a chat started on the web."""
        meta = session.metadata
        return cls(
            origin=meta.origin,
            unattended=scheduled or meta.origin == "automation",
            expert_kickoff=is_expert_kickoff_turn(session),
            from_chat_platform=from_chat_platform or bool(meta.source_platform),
            builder=meta.builder_graph_id is not None,
            delegated=meta.delegated_by_session_id is not None,
            voice=voice,
        )

    def with_turn_rows(self, session: ChatSession) -> "SessionKind":
        """This kind once the turn's own rows are in *session*.

        Whether the turn answers a hire's kickoff message is read off the
        latest user row, and an engine appends some turns' rows itself: the
        replies an auto-continue picks up, or a message nothing saved before
        dispatch. So it is read again then; the dispatch's facts stay.
        """
        return self.model_copy(
            update={"expert_kickoff": is_expert_kickoff_turn(session)}
        )

    def hidden_tools(self) -> frozenset[str]:
        """Tools the turn does not offer, both rules at once: a kickoff is
        narrowed to the onboarding card, and a session no person drives is not
        offered the tools that restaff the team, which its own guard refuses."""
        kickoff = kickoff_turn_disabled_tools() if self.expert_kickoff else frozenset()
        return kickoff | origin_disabled_tools(self.origin)
