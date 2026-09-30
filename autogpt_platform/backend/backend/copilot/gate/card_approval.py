"""A proposal card's Approve is the user's answer to the confirm it asks about.

A hire, raise, update or Soul-edit card sends the decision as a chat line naming
the proposal's one-time id (``decisionLine`` in ``ToolChain/ExpertCards.tsx``),
and the confirm takes nothing but that id, so a second card would ask the same
question about the same change.
"""

from typing import Any

from backend.copilot.model import ChatSession

from .held import written_by_gate

_CONFIRMS = frozenset({"confirm_expert_change", "confirm_expert_soul_update"})


def approved_on_card(
    tool_name: str, args: dict[str, Any], session: ChatSession
) -> bool:
    """True when the user's newest message approves exactly this id on its card.

    A typed "yes" or a Decline does not count, so those still reach the mode's
    verdict; the model cannot write a user row, and the gate's own rows are skipped.
    """
    confirmation_id = args.get("confirmation_id")
    if tool_name not in _CONFIRMS or not isinstance(confirmation_id, str):
        return False
    if not confirmation_id:
        return False
    newest = next(
        (
            message
            for message in reversed(session.messages)
            if message.role == "user" and not written_by_gate(message)
        ),
        None,
    )
    if newest is None or not newest.content:
        return False
    suffix = f"(confirmation_id: {confirmation_id})."
    return any(
        line.startswith("Approved: ") and line.endswith(suffix)
        for line in (raw.strip() for raw in newest.content.splitlines())
    )
