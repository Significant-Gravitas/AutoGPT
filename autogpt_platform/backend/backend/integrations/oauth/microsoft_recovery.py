import re
from typing import Literal

import httpx
from pydantic import BaseModel

MicrosoftRecoveryReason = Literal[
    "tenant_setup_required",
    "admin_consent_required",
    "billing_setup_required",
    "app_setup_required",
    "access_denied",
    "sign_in_failed",
]


class MicrosoftRecovery(BaseModel):
    reason: MicrosoftRecoveryReason
    message: str
    error_code: str | None = None


MESSAGES: dict[MicrosoftRecoveryReason, str] = {
    "tenant_setup_required": "Your organization needs to enable Work IQ.",
    "admin_consent_required": "Your Microsoft administrator needs to approve access.",
    "billing_setup_required": "Your organization needs to finish Work IQ billing setup.",
    "app_setup_required": "Microsoft sign-in needs configuration by the application owner.",
    "access_denied": "Microsoft access was declined. Your existing connections are still available.",
    "sign_in_failed": "Microsoft sign-in did not finish. You can try again or open setup help.",
}


class MicrosoftAuthError(ValueError):
    def __init__(self, recovery: MicrosoftRecovery):
        self.recovery = recovery
        super().__init__(recovery.message)


def microsoft_recovery(
    error: str = "", description: str = "", error_codes: list[int] | None = None
) -> MicrosoftRecovery:
    codes = set(error_codes or [])
    codes.update(int(code) for code in re.findall(r"AADSTS(\d+)", description[:4096]))
    reason: MicrosoftRecoveryReason = "sign_in_failed"
    if 650052 in codes:
        reason = "tenant_setup_required"
    elif codes.intersection({65001, 90094, 90093}) or error == "consent_required":
        reason = "admin_consent_required"
    elif codes.intersection({50011, 700016, 7000218, 7000215, 50194}):
        reason = "app_setup_required"
    elif 65004 in codes or error in {"access_denied", "authorization_declined"}:
        reason = "access_denied"
    code = f"AADSTS{min(codes)}" if codes else None
    return MicrosoftRecovery(reason=reason, message=MESSAGES[reason], error_code=code)


def microsoft_recovery_from_response(response: httpx.Response) -> MicrosoftRecovery:
    try:
        data = response.json()
        return microsoft_recovery(
            data.get("error", ""),
            data.get("error_description", ""),
            data.get("error_codes", []),
        )
    except (ValueError, AttributeError, TypeError):
        return microsoft_recovery()
