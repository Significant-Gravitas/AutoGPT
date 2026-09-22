"""Public, non-identifying reasons an introductory offer could not be started."""

from enum import StrEnum


class TrialRejectionReason(StrEnum):
    INTRO_OFFER_ALREADY_USED = "intro_offer_already_used"
    CARD_VERIFICATION_FAILED = "card_verification_failed"

    @property
    def stripe_comment(self) -> str:
        return f"autogpt_trial:{self.value}"

    @classmethod
    def from_stripe_comment(cls, comment: str | None) -> "TrialRejectionReason | None":
        return next(
            (reason for reason in cls if reason.stripe_comment == comment), None
        )
