"""Reconcile a Link checkout and hand the chat a clean browser afterwards."""

from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.browser_checkout_support import (
    CHECKOUT_ID_PARAMETERS,
    approval_for,
    available,
    checkout_response,
    failure,
    link_credentials,
    principal_for,
)
from backend.copilot.tools.models import ToolResponseBase
from backend.util.link_checkout import engine
from backend.util.link_checkout.approval import read_approval
from backend.util.link_checkout.broker_protocol import (
    AuthorizedCheckout,
    CheckoutReference,
)
from backend.util.link_checkout.models import CompleteCheckout


class BrowserLinkPaymentStatusTool(BaseTool):
    @property
    def name(self) -> str:
        return "browser_link_payment_status"

    @property
    def description(self) -> str:
        return (
            "Read a Link checkout's approval or payment status. Never pays or "
            "submits, and works on a sealed browser. Only 'succeeded' means paid."
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
            if current.spend_request_id is None:
                # Nothing exists in Link yet; the chat approval is the status.
                approval = approval_for(
                    await read_approval(request.checkout_id), current, principal
                )
                return checkout_response(current, session, approval)
            async with link_credentials(
                principal.user_id, current.credentials_id
            ) as credentials:
                view = await engine.status(
                    AuthorizedCheckout(
                        **reference.model_dump(), access_token=credentials.access_token
                    )
                )
            return checkout_response(view, session)
        except Exception:
            return failure(
                session,
                "Link status is unavailable. Do not submit again while the payment "
                "outcome is unknown.",
            )


class BrowserResetAfterPaymentTool(BaseTool):
    @property
    def name(self) -> str:
        return "browser_reset_after_payment"

    @property
    def description(self) -> str:
        return (
            "Once tool:browser_link_payment_status reports a final status, retire "
            "the sealed payment browser so this chat can browse again. The new "
            "browser starts signed out."
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
            view = await engine.reset(
                CheckoutReference(
                    **principal.model_dump(), checkout_id=request.checkout_id
                )
            )
            return checkout_response(view, session)
        except Exception:
            return failure(
                session,
                "The payment browser cannot be reset yet. Check the Link status "
                "first; it must be final.",
            )
