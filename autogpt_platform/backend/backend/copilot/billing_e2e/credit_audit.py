"""A turn billed to credits, checked against its invoice.

The AutoPilot block's turn past the plan's used-up allowance pays from the
credit balance instead of the plan's spend windows, and its tree is charged as
any turn's is. The SDK engine records the turn's cost once, as its stream
ends: the turn is debited the whole credits then, and the part credit when it
ends, never more in all than the balance leaves it. On OpenRouter that cost
lands from the reconcile task, so a charge that didn't wait for it would find
the turn over and take it all at once.
"""

from backend.copilot.constants import AUTOPILOT_BLOCK_ID
from backend.copilot.model import ChatSession

from .ledger import CreditBilling, PlatformLedger

# A credit is a cent. Written out rather than imported, like the invoice's
# prices, so the rounding under test can't set its own expectation.
_MICRODOLLARS_PER_CREDIT = 10_000
_AUTOPILOT_BLOCK = (AUTOPILOT_BLOCK_ID, "AutoPilotBlock")


def credit_charge_problems(
    credits: CreditBilling,
    turn: int,
    owed: int,
    tolerance: int,
    ledger: PlatformLedger,
) -> list[str]:
    """How the turn's charges strayed from the ``owed`` microdollars its
    invoice comes to, give or take ``tolerance``."""
    spend = [charge.microdollars for charge in ledger.spend_in(turn)]
    tree = [charge.microdollars for charge in ledger.tree_in(turn)]
    debits = [debit.credits for debit in ledger.debits_in(turn)]
    owed_debits = [
        _debits_for(owed + bound, credits.available(turn))
        for bound in (-tolerance, tolerance)
    ]
    problems = []
    if spend:
        problems.append(
            f"turn {turn}: charged {spend} to the plan's spend windows, though "
            f"the turn is billed to credits"
        )
    if len(tree) != 1 or abs(tree[0] - owed) > tolerance:
        problems.append(f"turn {turn}: the tree was charged {tree}; it owes {owed}")
    if debits not in owed_debits:
        problems.append(
            f"turn {turn}: debited {debits} credits for {owed} microdollars; it "
            f"owes {owed_debits[-1]}, the whole credits as the cost landed, "
            f"then the part credit as the turn ended"
        )
    return problems


def debit_problems(
    credits: CreditBilling | None, session: ChatSession, ledger: PlatformLedger
) -> list[str]:
    """Every debit on this chat's user, its chat and the AutoPilot block, and
    none where the plan's allowance covers the turns."""
    if credits is None:
        return [
            f"turn {debit.turn}: debited {debit.credits} credits, though the "
            f"plan's allowance covers the turn"
            for debit in ledger.debits
        ]
    return [
        f"turn {debit.turn}: debited user {debit.user_id!r} for chat "
        f"{debit.metadata.chat_session_id!r} and block "
        f"{(debit.metadata.block_id, debit.metadata.block)}"
        for debit in ledger.debits
        if (debit.user_id, debit.metadata.chat_session_id)
        != (session.user_id, session.session_id)
        or (debit.metadata.block_id, debit.metadata.block) != _AUTOPILOT_BLOCK
    ]


def _debits_for(microdollars: int, available: int | None) -> list[int]:
    """The debits for a cost recorded once: its whole credits, then the part
    credit rounded up, never more in all than ``available``."""
    whole = microdollars // _MICRODOLLARS_PER_CREDIT
    rounded_up = -(-microdollars // _MICRODOLLARS_PER_CREDIT)
    if available is not None:
        whole, rounded_up = min(whole, available), min(rounded_up, available)
    return [debit for debit in (whole, rounded_up - whole) if debit > 0]
