"""Request and response shapes of the paid usage reset endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.copilot.rate_limit import CoPilotUsagePublic


class UsageResetRequest(BaseModel):
    purchase_key: str = Field(
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="A key the client makes up for this purchase (a UUID). "
        "Sending it again returns the same purchase and charges nothing more.",
    )
    price_credits: int = Field(
        gt=0,
        description="The price the customer was shown and agreed to, from "
        "the usage read-out's reset_offer. Refused if it has changed.",
    )


class UsageResetResponse(BaseModel):
    status: Literal["granted", "payment_required"] = Field(
        description="'granted': the limits are reset. 'payment_required': "
        "nothing is reset yet; send the customer to checkout_url to pay by card."
    )
    paid_with: Literal["credits", "card"]
    credits_charged: int = Field(
        default=0,
        description="Credits taken by this request; 0 for a card payment or "
        "when the key had already bought it.",
    )
    checkout_url: str | None = Field(
        default=None, description="Where to pay by card, when payment is required."
    )
    resets_at: datetime | None = Field(
        default=None,
        description="When a granted reset ends: the weekly reset, unchanged.",
    )
    usage: CoPilotUsagePublic | None = Field(
        default=None,
        description="Usage after a granted reset; null when it couldn't be "
        "read just now (the purchase still went through).",
    )


class UsageResetConfirmRequest(BaseModel):
    session_id: str = Field(
        min_length=8,
        max_length=255,
        pattern=r"^[A-Za-z0-9_]+$",
        description="The Stripe Checkout session the customer returned from.",
    )


class UsageResetConfirmResponse(BaseModel):
    status: Literal["granted", "pending"] = Field(
        description="'pending': Stripe hasn't confirmed the payment (yet); "
        "nothing is reset."
    )
    resets_at: datetime | None = None
    usage: CoPilotUsagePublic | None = None
