from dataclasses import dataclass

DEFAULT_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
DEFAULT_EMAIL = "default@example.com"


# Using dataclass here to avoid adding dependency on pydantic
@dataclass(frozen=True)
class User:
    user_id: str
    email: str
    phone_number: str
    role: str

    @classmethod
    def from_payload(cls, payload):
        return cls(
            user_id=payload["sub"],
            email=payload.get("email", ""),
            phone_number=payload.get("phone", ""),
            # Default rather than index: a token without a `role` claim is a
            # token we simply don't grant privileges to, so it should be
            # treated as an ordinary user — not raise KeyError and surface as
            # a 500. verify_user already fails closed on the admin check.
            role=payload.get("role", "user"),
        )


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    org_id: str
    team_id: str | None  # None = org-home context
    is_org_owner: bool
    is_org_admin: bool
    is_org_billing_manager: bool
    is_team_admin: bool
    is_team_billing_manager: bool
    seat_status: str  # ACTIVE, INACTIVE, PENDING, NONE
