"""What Otto is told once a new teammate is on the team.

A new expert has no chat until the user opens one. Told only who joined,
the model filled the gap with "open her thread", which sent the user looking
for a thread that did not exist (SECRT-3401). So each message says where
their chat is: the card's Chat with <name> button, which opens it, or starts
the first one with the expert introducing themselves. A chat-platform
conversation shows no card, so there the user is pointed at AutoGPT instead.
"""

from typing import Literal

NewTeammateKind = Literal["hire", "raise"]


def applied_message(
    kind: NewTeammateKind, name: str, *, has_chat: bool, on_card: bool
) -> str:
    """The confirm's message for a hire or a new expert with no task.

    *has_chat*: a re-hire comes back with the chats it had. *on_card*: the
    user is in AutoGPT, where the confirm renders as a card.
    """
    if kind == "hire":
        joined = (
            f"{name} is hired and on the team; their workflows and skills "
            "finish installing in the background."
        )
    else:
        joined = f"{name} is created and on the team."
    return (
        f"{joined} Tell the user who joined and what they own. "
        f"{_chat_note(name, has_chat=has_chat, on_card=on_card)}"
    )


def _chat_note(name: str, *, has_chat: bool, on_card: bool) -> str:
    if has_chat:
        where = (
            f"the card's Chat with {name} button opens it"
            if on_card
            else "the user can open it in AutoGPT"
        )
        return f"{name} already has a chat with the user: {where}."
    where = (
        f"The card shows a Chat with {name} button that opens their first "
        "chat, where they introduce themselves: point the user to it."
        if on_card
        else f"The user can start it in AutoGPT, where {name} introduces " "themselves."
    )
    return (
        f"{name} has no chat yet. {where} Never say a thread or chat with them "
        "already exists, and never tell the user to look for one."
    )


def joined_verb(kind: NewTeammateKind) -> str:
    return "hired" if kind == "hire" else "created"
