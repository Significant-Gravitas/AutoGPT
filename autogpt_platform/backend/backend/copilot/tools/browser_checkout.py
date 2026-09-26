"""Pay for a purchase in the agent's browser with the customer's Link wallet,
without the card ever reaching the agent.

The agent prepares the checkout (login, cart, delivery, billing) and names the
payment fields; the checkout engine pins them, the customer approves, and a
separate worker fills the single-use card and submits once. The agent passes
no script and receives no card. See ``backend/util/link_checkout``.
"""

from pydantic import ValidationError

from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.browser_checkout_support import (
    CHECKOUT_ID_PARAMETERS,
    REQUEST_PARAMETERS,
    approval_for,
    available,
    chat_link_credential,
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
    approval_details,
    open_approval,
    read_approval,
)
from backend.util.link_checkout.broker_protocol import (
    AuthorizedCheckout,
    CheckoutReference,
    CheckoutView,
    CreateCheckout,
)
from backend.util.link_checkout.config import live_payments_enabled
from backend.util.link_checkout.models import CheckoutPlan, CompleteCheckout
from backend.util.link_checkout.policy import in_app_approval_allowed
from backend.util.link_checkout.refusals import LIVE_PAYMENTS_DISABLED, CheckoutRefused
from backend.util.request import validate_url_host


class BrowserRequestLinkPaymentTool(BaseTool):
    @property
    def name(self) -> str:
        return "browser_request_link_payment"

    @property
    def description(self) -> str:
        return (
            "Ask the user to approve paying for the open checkout with their "
            "Link wallet. First finish login, cart, delivery and billing, and "
            "pick payment_method_id with the Stripe Link List Payment Methods "
            "block. Pass selectors of the empty card inputs (autocomplete "
            "cc-number, cc-csc, cc-exp) and the pay button; no scripts or card "
            "values. Once the user approves, call "
            "tool:browser_complete_link_payment with the returned checkout_id."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict:
        return REQUEST_PARAMETERS

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
            if not kwargs.get("credentials_id"):
                kwargs["credentials_id"] = await chat_link_credential(
                    principal.user_id, principal.session_id
                )
            plan = CheckoutPlan.model_validate(kwargs)
        except CheckoutRefused as refused:
            return failure(session, str(refused))
        except ValidationError as invalid:
            return failure(session, invalid_plan(invalid))
        except Exception as error:
            log_failure("credential lookup", error)
            return failure(
                session,
                "Could not read the Stripe Link connection. Try again shortly.",
            )
        if not plan.test_mode and not live_payments_enabled():
            return failure(session, LIVE_PAYMENTS_DISABLED)
        try:
            await validate_url_host(plan.checkout_url)
            async with link_credentials(principal.user_id, plan.credentials_id) as c:
                in_app = await in_app_approval_allowed(
                    c.access_token.get_secret_value(), plan
                )
                view = await engine.create(
                    CreateCheckout(
                        **principal.model_dump(),
                        plan=plan,
                        access_token=c.access_token,
                        approval_mode="in_app" if in_app else "link",
                    )
                )
            if view.approval_mode == "in_app":
                await open_approval(_pending(view, plan, principal.user_id, session))
                return checkout_response(
                    view,
                    session,
                    message="The purchase is shown in the chat for the user to "
                    "approve. Wait for them; once they approve, call "
                    "tool:browser_complete_link_payment with this checkout_id.",
                )
            return checkout_response(
                view,
                session,
                message="Ask the user to approve this purchase in Link, then call "
                "tool:browser_complete_link_payment with this checkout_id.",
            )
        except CheckoutRefused as refused:
            return failure(session, str(refused))
        except Exception as error:
            log_failure("preparation", error)
            return failure(
                session,
                "Checkout preparation failed and no card was filled. Check the "
                "browser tab and the Link connection before trying again.",
            )


class BrowserCompleteLinkPaymentTool(BaseTool):
    @property
    def name(self) -> str:
        return "browser_complete_link_payment"

    @property
    def description(self) -> str:
        return (
            "Pay for an approved checkout from tool:browser_request_link_payment. "
            "A private worker fills the single-use card, submits once and discards "
            "the browser; no card is returned. 'submitted' is not proof of "
            "payment: reconcile with tool:browser_link_payment_status and never "
            "repeat a payment."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict:
        return CHECKOUT_ID_PARAMETERS

    @property
    def is_available(self) -> bool:
        return available()

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        try:
            principal = principal_for(user_id, session)
            request = CompleteCheckout.model_validate(kwargs)
            reference = CheckoutReference(
                **principal.model_dump(), checkout_id=request.checkout_id
            )
            current = await engine.get(reference)
            approval = approval_for(
                await read_approval(request.checkout_id), current, principal
            )
            if current.spend_request_id is None and current.approval_mode == "in_app":
                if approval is None or approval.state != "approved":
                    return _not_approved(current, approval, session)
            async with link_credentials(
                principal.user_id, current.credentials_id
            ) as credentials:
                view = await engine.complete(
                    AuthorizedCheckout(
                        **reference.model_dump(),
                        access_token=credentials.access_token,
                        approval=(
                            approval_details(approval)
                            if approval is not None and approval.state == "approved"
                            else None
                        ),
                    )
                )
            return checkout_response(view, session, approval)
        except CheckoutRefused as refused:
            return failure(session, str(refused))
        except Exception as error:
            log_failure("payment", error)
            return failure(
                session,
                "Checkout unavailable, expired or already attempted. Check Link "
                "for the payment status; never repeat a payment with an unknown "
                "outcome.",
            )


def _pending(
    view: CheckoutView, plan: CheckoutPlan, user_id: str, session: ChatSession
) -> PendingApproval:
    return PendingApproval(
        checkout_id=view.checkout_id,
        user_id=user_id,
        session_id=session.session_id,
        merchant_name=plan.merchant_name,
        merchant_url=plan.merchant_url(),
        context=plan.context,
        amount=plan.amount,
        currency=plan.currency,
        test_mode=plan.test_mode,
        expires_at=view.expires_at,
    )


def _not_approved(
    view: CheckoutView, approval: ApprovalView | None, session: ChatSession
) -> ToolResponseBase:
    state = approval.state if approval is not None else "expired"
    status, message = {
        "awaiting": (
            "awaiting_approval",
            "The user has not approved this purchase yet. Nothing was charged. "
            "Wait for their approval in the chat.",
        ),
        "declined": (
            "declined",
            "The user declined this purchase. Nothing was charged; do not retry "
            "it. Ask what they want instead.",
        ),
    }.get(
        state,
        (
            "expired",
            "This purchase request expired before approval. Nothing was charged; "
            "prepare a new checkout if the user still wants it.",
        ),
    )
    return checkout_response(
        view.model_copy(update={"status": status}), session, approval, message=message
    )
