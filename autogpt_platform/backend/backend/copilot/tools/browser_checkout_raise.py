"""Raise an unpaid Link checkout's total when the final price came out higher
than the one the user approved: tax, shipping or a fee shown only at the end.

Link calls this incremental authorization: the request it already has takes
the new total, and the user approves it again. See
https://docs.stripe.com/agentic-commerce/link-agent-wallet/use-link-wallet-pay-online
"""

import time

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.browser_checkout_schemas import RAISE_PARAMETERS
from backend.copilot.tools.browser_checkout_support import (
    available,
    checkout_response,
    failure,
    invalid_plan,
    link_credentials,
    log_failure,
    principal_for,
)
from backend.copilot.tools.models import ToolResponseBase
from backend.util.link_checkout import engine
from backend.util.link_checkout.approval import (
    ApprovalView,
    PendingApproval,
    read_approval,
    reopen_approval,
)
from backend.util.link_checkout.broker_protocol import (
    CheckoutReference,
    CheckoutView,
    Principal,
    RaiseCheckout,
)
from backend.util.link_checkout.models import CHECKOUT_ID, Amount
from backend.util.link_checkout.policy import Purchase, in_app_approval_allowed
from backend.util.link_checkout.preflight import purchase_blocker
from backend.util.link_checkout.refusals import RAISE_NOT_HIGHER, CheckoutRefused


class RaiseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    checkout_id: str = Field(pattern=CHECKOUT_ID)
    amount: Amount
    reason: str = Field(min_length=10, max_length=300)


class BrowserRaiseLinkPaymentTool(BaseTool):
    @property
    def name(self) -> str:
        return "browser_raise_link_payment"

    @property
    def description(self) -> str:
        return (
            "Raise an unpaid checkout's total when the final price came out "
            "higher (tax, shipping, a fee). The user approves the new total "
            "again; then call tool:browser_complete_link_payment."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict:
        return RAISE_PARAMETERS

    @property
    def is_available(self) -> bool:
        return available()

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        try:
            principal = principal_for(user_id, session)
        except ValueError:
            return failure(session, "Private checkout is not available in this chat.")
        try:
            request = RaiseRequest.model_validate(kwargs)
        except ValidationError as invalid:
            return failure(session, invalid_plan(invalid, "raise request"))
        try:
            return await _raise(request, principal, session)
        except CheckoutRefused as refused:
            return failure(session, str(refused))
        except Exception as error:
            log_failure("raise", error)
            return failure(
                session,
                "The total could not be raised, and nothing was charged. Check "
                "the checkout with tool:browser_link_payment_status; to try "
                "again, call this with the same new total.",
            )


async def _raise(
    request: RaiseRequest, principal: Principal, session: ChatSession
) -> ToolResponseBase:
    reference = CheckoutReference(
        **principal.model_dump(), checkout_id=request.checkout_id
    )
    current = await engine.get(reference)
    problem = _cannot_raise(current, request.amount)
    if problem:
        return failure(session, problem)
    async with link_credentials(principal.user_id, current.credentials_id) as c:
        token = c.access_token.get_secret_value()
        blocker = await purchase_blocker(token, request.amount, current.currency)
        if blocker is not None:
            return failure(session, blocker.message)
        # A request already in Link is raised there; only a purchase that
        # exists in the chat alone can stay there.
        in_app = current.spend_request_id is None and await in_app_approval_allowed(
            token,
            Purchase(
                amount=request.amount,
                currency=current.currency,
                payment_method_id=current.payment_method_id,
            ),
        )
        view = await engine.raise_total(
            RaiseCheckout(
                **reference.model_dump(),
                access_token=c.access_token,
                amount=request.amount,
                approval_mode="in_app" if in_app else "link",
            )
        )
    approval = await _record_raise(view, current, request.reason, principal)
    if view.approval_mode == "in_app":
        message = (
            "The new total is shown in the chat for the user to approve. Wait "
            "for them; once they approve, call tool:browser_complete_link_payment "
            "with this checkout_id."
        )
    else:
        message = (
            "Ask the user to approve the new total in Link, then call "
            "tool:browser_complete_link_payment with this checkout_id."
        )
    return checkout_response(view, session, approval, message=message)


def _cannot_raise(current: CheckoutView, amount: int) -> str:
    if current.attempted:
        return (
            "This checkout was already attempted, so its total can't change. "
            "Check it with tool:browser_link_payment_status."
        )
    if current.expires_at <= time.time():
        return (
            "This checkout expired before payment; nothing was charged. Prepare "
            "a new one for the full total."
        )
    if amount <= current.amount:
        return RAISE_NOT_HIGHER
    return ""


async def _record_raise(
    view: CheckoutView, current: CheckoutView, reason: str, principal: Principal
) -> ApprovalView | None:
    """The chat's approval record for the raised total: a new revision for
    the user to approve, or, when the approval moved to Link, a closed one so
    no card in the chat can approve the purchase any more."""
    previous = await read_approval(view.checkout_id)
    if view.approval_mode != "in_app" and previous is None:
        return None
    await reopen_approval(
        PendingApproval(
            checkout_id=view.checkout_id,
            user_id=principal.user_id,
            session_id=principal.session_id,
            merchant_name=view.merchant_name,
            merchant_url=view.merchant_url,
            context=previous.pending.context if previous else "",
            amount=view.amount,
            currency=view.currency,
            test_mode=view.test_mode,
            expires_at=view.expires_at if view.approval_mode == "in_app" else 0.0,
            revision=view.revision,
            previous_amount=current.amount,
            reason=reason,
        )
    )
    return await read_approval(view.checkout_id)
