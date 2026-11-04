"""Hire and raise cards a task waits on, answered from the board.

A hire or raise proposed in a task's work chat parks a one-time
confirmation_id that ``confirm_expert_change`` only accepts once the user's
own request carried ``Approved: … (confirmation_id: …).`` (``gate/card_approval``).
The chat route records that line when the card is clicked in the chat; a
user working from the board never sees that chat, so the task keeps the
cards (``Contract.approvals``) and ``POST /api/tasks/{id}/approvals`` turns the
user's answer into the same lines.
"""

from backend.api.features.tasks.contract import ContractPatch
from backend.api.features.tasks.models import CardApproval, CardDecision, Contract

_VERBS = {"hire": "hire", "raise": "create"}


def add_approval(
    contract: Contract, card: CardApproval
) -> tuple[Contract, list[ContractPatch]]:
    """Add *card*, replacing an earlier card for the same expert name, since
    a re-proposed hire supersedes the card it replaces."""
    kept = [
        a
        for a in contract.approvals
        if a.confirmation_id != card.confirmation_id and a.name != card.name
    ]
    approvals = [*kept, card]
    if approvals == contract.approvals:
        return contract, []
    return _with(contract, approvals)


def drop_approvals(
    contract: Contract, confirmation_ids: set[str]
) -> tuple[Contract, list[ContractPatch]]:
    approvals = [
        a for a in contract.approvals if a.confirmation_id not in confirmation_ids
    ]
    if len(approvals) == len(contract.approvals):
        return contract, []
    return _with(contract, approvals)


def decision_lines(
    approvals: list[CardApproval], decisions: list[CardDecision]
) -> list[str]:
    """The chat lines the card would have sent, for the decisions that name a
    card this task holds (``decisionLine`` in ``ToolChain/ExpertCards.tsx``)."""
    cards = {a.confirmation_id: a for a in approvals}
    lines = []
    for decision in decisions:
        card = cards.get(decision.confirmation_id)
        if card is None:
            continue
        verb = _VERBS.get(card.kind, "create")
        lines.append(
            f"Approved: {verb} {card.name} (confirmation_id: {card.confirmation_id})."
            if decision.approved
            else f"Not approved: do not {verb} {card.name}, discard that "
            f"proposal (confirmation_id: {card.confirmation_id})."
        )
    return lines


def answer_text(approvals: list[CardApproval], decisions: list[CardDecision]) -> str:
    """One readable line for the activity feed and the closed ask."""
    names = {a.confirmation_id: a.name for a in approvals}
    approved = [
        names[d.confirmation_id]
        for d in decisions
        if d.approved and d.confirmation_id in names
    ]
    declined = [
        names[d.confirmation_id]
        for d in decisions
        if not d.approved and d.confirmation_id in names
    ]
    parts = []
    if approved:
        parts.append(f"Approved {_join(approved)}")
    if declined:
        parts.append(f"Declined {_join(declined)}")
    return "; ".join(parts) or "Answered the cards"


def _join(names: list[str]) -> str:
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _with(
    contract: Contract, approvals: list[CardApproval]
) -> tuple[Contract, list[ContractPatch]]:
    updated = contract.model_copy(update={"approvals": approvals})
    value = [a.model_dump(mode="json") for a in approvals]
    return updated, [ContractPatch(field="approvals", value=value)]
