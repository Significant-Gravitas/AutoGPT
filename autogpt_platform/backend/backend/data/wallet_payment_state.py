"""Where a wallet payment for a subscription invoice stands.

The wallet debit is keyed by the invoice id. It is settled once Stripe holds
the invoice paid out of band with it, and refunded once the money went back
under the ``<invoice>:wallet-refund`` key. Both are our own records, so no
decision depends on which fields a Stripe API version puts on the invoice.
Refunded is final: that debit may never pay the invoice afterwards.
"""

from enum import Enum
from typing import cast

from prisma.enums import CreditTransactionType
from prisma.models import CreditTransaction
from pydantic import BaseModel

from backend.util.json import SafeJson

_SETTLED_METADATA_KEY = "wallet_payment"


class WalletPaymentState(str, Enum):
    DEBITED = "debited"
    """The wallet was charged; Stripe does not hold the invoice paid with it yet."""
    SETTLED = "settled"
    """The invoice is paid out of band with the debit; the wallet keeps it."""
    REFUNDED = "refunded"
    """The debit was given back. Terminal."""


class WalletPayment(BaseModel):
    user_id: str
    invoice_id: str
    debited: int
    """Cents taken from the wallet, as a positive number."""
    state: WalletPaymentState
    metadata: dict


def wallet_refund_key(invoice_id: str) -> str:
    return f"{invoice_id}:wallet-refund"


async def find_wallet_payment(user_id: str, invoice_id: str) -> WalletPayment | None:
    rows = await CreditTransaction.prisma().find_many(
        where={
            "userId": user_id,
            "transactionKey": {"in": [invoice_id, wallet_refund_key(invoice_id)]},
        }
    )
    by_key = {row.transactionKey: row for row in rows}
    debit = by_key.get(invoice_id)
    if (
        debit is None
        or debit.type != CreditTransactionType.SUBSCRIPTION
        or debit.amount >= 0
    ):
        return None
    metadata: dict = cast(dict, debit.metadata) or {}
    if wallet_refund_key(invoice_id) in by_key:
        state = WalletPaymentState.REFUNDED
    elif metadata.get(_SETTLED_METADATA_KEY) == WalletPaymentState.SETTLED.value:
        state = WalletPaymentState.SETTLED
    else:
        state = WalletPaymentState.DEBITED
    return WalletPayment(
        user_id=user_id,
        invoice_id=invoice_id,
        debited=-debit.amount,
        state=state,
        metadata=metadata,
    )


async def mark_wallet_payment_settled(payment: WalletPayment) -> None:
    await CreditTransaction.prisma().update(
        where={
            "creditTransactionIdentifier": {
                "transactionKey": payment.invoice_id,
                "userId": payment.user_id,
            }
        },
        data={
            "metadata": SafeJson(
                {
                    **payment.metadata,
                    _SETTLED_METADATA_KEY: WalletPaymentState.SETTLED.value,
                }
            )
        },
    )


async def invoice_paid_from_wallet(user_id: str, invoice_id: str) -> bool:
    """Whether the wallet paid (or is still paying) the invoice, so nothing
    else may treat it as a payment collected from the customer's card."""
    payment = await find_wallet_payment(user_id, invoice_id)
    return payment is not None and payment.state != WalletPaymentState.REFUNDED
