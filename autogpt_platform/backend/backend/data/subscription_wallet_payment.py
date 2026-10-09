"""Pay a failed subscription invoice from the AutoGPT wallet, safely.

Paying is two writes in two systems: debit the wallet, then tell Stripe the
invoice is paid. Either can fail on its own, and Stripe can deliver the same
failure twice or late. The debit is keyed by the invoice id, so it is both
the guard against a second debit and the record that a payment was started;
every later delivery resumes from it until it is settled (the invoice is
paid out of band with it) or refunded. See ``wallet_payment_state``.
"""

import logging

import stripe
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import User

from backend.data.credit import UserCredit
from backend.data.stripe_client import stripe_call
from backend.data.stripe_invoice_payments import paid_by_stripe_collection
from backend.data.wallet_payment_state import (
    WalletPayment,
    WalletPaymentState,
    find_wallet_payment,
    mark_wallet_payment_settled,
    wallet_refund_key,
)
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
        # The balance is checked before the key, so a concurrent delivery
        # that took the debit first shows up here when the wallet cannot
        # cover a second bill; that payment must be finished, not cancelled.
        if await find_wallet_payment(user_id, invoice_id) is None:
            return False
    except UniqueViolationError:
        # A concurrent delivery of the same failure took the debit first.
        pass
    payment = await find_wallet_payment(user_id, invoice_id)
    if payment is None:
        raise RuntimeError(f"Wallet debit for invoice {invoice_id} is missing")
    await settle_wallet_payment(payment, invoice, may_pay=True)
    return True


async def settle_wallet_payment(
    payment: WalletPayment, invoice: dict, *, may_pay: bool
) -> None:
    """Finish a started wallet payment: mark the invoice paid, or refund the debit.

    Safe to repeat. ``may_pay`` is False once the invoice no longer pays for a
    live plan; the debit is then given back instead of spent on it. A settled
    or refunded payment is left as it is: once refunded, the debit must never
    pay the invoice, even if the invoice is reopened and payable again.
    """
    if payment.state != WalletPaymentState.DEBITED:
        return
    invoice_id = payment.invoice_id
    if may_pay and invoice.get("status") in _PAYABLE_INVOICE_STATUSES:
        try:
            # Out of band, so Stripe does not also retry the card that failed.
            await stripe_call(
                stripe.Invoice.pay_async, invoice_id, paid_out_of_band=True
            )
        except stripe.StripeError:
            invoice = dict(await stripe_call(stripe.Invoice.retrieve_async, invoice_id))
            if invoice.get("status") in _PAYABLE_INVOICE_STATUSES:
                logger.warning(
                    f"Wallet debit taken for invoice {invoice_id} (user"
                    f" {payment.user_id}) but Stripe would not mark it paid;"
                    " the retry resumes it",
                    exc_info=True,
                )
                raise
        else:
            await mark_wallet_payment_settled(payment)
            logger.info(
                f"Paid invoice {invoice_id} of user {payment.user_id} from wallet"
            )
            return
    await _settle_unpayable(payment, invoice)


async def reconcile_wallet_payment_on_paid_invoice(invoice: dict) -> None:
    """On an invoice's success event, settle or refund an unfinished wallet debit.

    Stripe keeps retrying the card while a wallet payment is unfinished; if a
    retry succeeds first, the customer would otherwise pay twice. The event
    also closes the window where our out-of-band pay reached Stripe but the
    settlement was not recorded.
    """
    invoice_id = invoice.get("id") or ""
    customer_id = invoice.get("customer")
    if not invoice_id or not customer_id:
        return
    user = await User.prisma().find_first(where={"stripeCustomerId": customer_id})
    if not user:
        return
    payment = await find_wallet_payment(user.id, invoice_id)
    if payment is None or payment.state != WalletPaymentState.DEBITED:
        return
    await _settle_unpayable(payment, invoice)


async def _settle_unpayable(payment: WalletPayment, invoice: dict) -> None:
    """Decide an unfinished debit whose invoice we will not (or cannot) pay."""
    status = invoice.get("status")
    if status != "paid":
        # Voided, or no longer for a live plan: the wallet must not keep it.
        await _refund(payment, f"wallet_payment_not_used:{status}")
        return
    collected = await paid_by_stripe_collection(invoice)
    if collected is None:
        logger.error(
            f"Invoice {payment.invoice_id} is paid but it does not say whether"
            f" by card or by the wallet debit of user {payment.user_id}; keeping"
            " the debit for a manual check"
        )
        return
    if collected:
        await _refund(payment, "invoice_paid_by_card")
        return
    # Paid out of band by our own earlier attempt, before it was recorded.
    await mark_wallet_payment_settled(payment)


async def _refund(payment: WalletPayment, reason: str) -> None:
    """Keyed per invoice, so repeated or concurrent calls refund once."""
    try:
        await UserCredit()._add_transaction(
            user_id=payment.user_id,
            amount=payment.debited,
            transaction_type=CreditTransactionType.SUBSCRIPTION,
            fail_insufficient_credits=False,
            transaction_key=wallet_refund_key(payment.invoice_id),
            metadata=SafeJson(
                {"stripe_invoice_id": payment.invoice_id, "reason": reason}
            ),
        )
    except UniqueViolationError:
        return
    logger.warning(
        f"Refunded wallet debit of {payment.debited} cents to user"
        f" {payment.user_id} for invoice {payment.invoice_id} ({reason})"
    )
