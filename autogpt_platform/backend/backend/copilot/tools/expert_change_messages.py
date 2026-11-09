"""What Otto is told once a new teammate is on the team.

A new expert has no chat until the user opens one. Told only who joined,
the model filled the gap with "open her thread", which sent the user looking
for a thread that did not exist (SECRT-3401). So each message says where the
first chat is: the card's Chat with <name> button, which opens it and has
the expert introduce themselves.
"""

from typing import Literal

NewTeammateKind = Literal["hire", "raise"]


def applied_message(kind: NewTeammateKind, name: str) -> str:
    """The confirm's message for a hire or a new expert with no task."""
    if kind == "hire":
        joined = (
            f"{name} is hired and on the team; their workflows and skills "
            "finish installing in the background."
        )
    else:
        joined = f"{name} is created and on the team."
    return (
        f"{joined} Tell the user who joined and what they own. "
        f"{first_chat_note(name)}"
    )


def first_chat_note(name: str) -> str:
    return (
        f"{name} has no chat yet. The card shows a Chat with {name} button "
        "that opens their first chat, where they introduce themselves: point "
        "the user to it. Never say a thread or chat with them already exists, "
        "and never tell the user to look for one."
    )


def joined_verb(kind: NewTeammateKind) -> str:
    return "hired" if kind == "hire" else "created"
