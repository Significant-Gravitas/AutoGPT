"""Pay a failed subscription invoice from the AutoGPT wallet, safely.

Paying is two writes in two systems: debit the wallet, then tell Stripe the
invoice is paid. Either can fail on its own, and Stripe can deliver the same
failure twice or late. The debit is keyed by the invoice id, so it is both
the guard against a second debit and the record that a payment was started;
every later delivery resumes from it until the invoice is paid out of band
or the debit is given back.
"""

import logging

import stripe
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction, User

from backend.data.credit import UserCredit
from backend.data.stripe_client import stripe_call
from backend.util.exceptions import InsufficientBalanceError
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

# Invoice states ``Invoice.pay`` still accepts.
_PAYABLE_INVOICE_STATUSES = ("open", "uncollectible")


async def pay_invoice_from_wallet(
    user_id: str, customer_id: str, sub_id: str, invoice: dict
) -> bool:
    """Debit the wallet for ``invoice`` and mark it paid out of band.

    Returns False, without changing anything, when the balance cannot cover
    the invoice. Raises when Stripe would not mark the invoice paid, leaving
    the debit for the retry to resume from.
    """
    invoice_id: str = invoice["id"]
    try:
        await UserCredit()._add_transaction(
            user_id=user_id,
            amount=-invoice.get("amount_due", 0),
            transaction_type=CreditTransactionType.SUBSCRIPTION,
            fail_insufficient_credits=True,
            transaction_key=invoice_id,
            metadata=SafeJson(
                {
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": sub_id,
                    "reason": "subscription_payment_failure_covered_by_balance",
                }
            ),
        )
    except InsufficientBalanceError:
        return False
    except UniqueViolationError:
        # A concurrent delivery of the same failure took the debit first.
        pass
    debit = await find_wallet_debit(user_id, invoice_id)
    if debit is None:
        raise RuntimeError(f"Wallet debit for invoice {invoice_id} is missing")
    await settle_wallet_payment(user_id, invoice, debit, may_pay=True)
    return True


async def settle_wallet_payment(
    user_id: str, invoice: dict, debit: CreditTransaction, *, may_pay: bool
) -> None:
    """Finish a started wallet payment: mark the invoice paid, or refund the debit.

    Safe to repeat. ``may_pay`` is False once the invoice no longer pays for a
    live plan; the debit is then given back instead of spent on it.
    """
    invoice_id: str = invoice["id"]
    if may_pay and invoice.get("status") in _PAYABLE_INVOICE_STATUSES:
        try:
            # Out of band, so Stripe does not also retry the card that failed.
            # The success handler reads the flag and skips its credit grant.
            await stripe_call(
                stripe.Invoice.pay_async, invoice_id, paid_out_of_band=True
            )
            logger.info(f"Paid invoice {invoice_id} of user {user_id} from wallet")
            return
        except stripe.StripeError:
            invoice = dict(
                await stripe_call(stripe.Invoice.retrieve_async, invoice_id)
            )
            if invoice.get("status") in _PAYABLE_INVOICE_STATUSES:
                logger.warning(
                    f"Wallet debit taken for invoice {invoice_id} (user {user_id})"
                    " but Stripe would not mark it paid; the retry resumes it",
                    exc_info=True,
                )
                raise
    if invoice.get("status") == "paid" and invoice.get("paid_out_of_band"):
        return
    # Paid by card after all, voided, or no longer for a live plan: the
    # wallet must not keep the money.
    await _refund_wallet_debit(
        user_id, invoice_id, debit, f"wallet_payment_not_used:{invoice.get('status')}"
    )


async def refund_wallet_debit_if_paid_by_card(invoice: dict) -> None:
    """Give back an unfinished wallet debit when the card paid the invoice.

    Stripe keeps retrying the card while a wallet payment is unfinished; if a
    retry succeeds first, the customer would otherwise pay twice.
    """
    invoice_id = invoice.get("id") or ""
    customer_id = invoice.get("customer")
    if not invoice_id or not customer_id or invoice.get("paid_out_of_band"):
        return
    user = await User.prisma().find_first(where={"stripeCustomerId": customer_id})
    if not user:
        return
    debit = await find_wallet_debit(user.id, invoice_id)
    if debit is not None:
        await _refund_wallet_debit(user.id, invoice_id, debit, "invoice_paid_by_card")


async def find_wallet_debit(user_id: str, invoice_id: str) -> CreditTransaction | None:
    transaction = await CreditTransaction.prisma().find_unique(
        where={
            "creditTransactionIdentifier": {
                "transactionKey": invoice_id,
                "userId": user_id,
            }
        }
    )
    if (
        transaction is None
        or transaction.type != CreditTransactionType.SUBSCRIPTION
        or transaction.amount >= 0
    ):
        return None
    return transaction


async def _refund_wallet_debit(
    user_id: str, invoice_id: str, debit: CreditTransaction, reason: str
) -> None:
    """Keyed per invoice, so repeated or concurrent calls refund once."""
    try:
        await UserCredit()._add_transaction(
            user_id=user_id,
            amount=-debit.amount,
            transaction_type=CreditTransactionType.SUBSCRIPTION,
            fail_insufficient_credits=False,
            transaction_key=f"{invoice_id}:wallet-refund",
            metadata=SafeJson({"stripe_invoice_id": invoice_id, "reason": reason}),
        )
    except UniqueViolationError:
        return
    logger.warning(
        f"Refunded wallet debit of {-debit.amount} cents to user {user_id}"
        f" for invoice {invoice_id} ({reason})"
    )
