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
            role=payload["role"],
        )


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    org_id: str
    workspace_id: str | None  # None = org-home context
    is_org_owner: bool
    is_org_admin: bool
    is_org_billing_manager: bool
    is_workspace_admin: bool
    is_workspace_billing_manager: bool
    seat_status: str  # ACTIVE, INACTIVE, PENDING, NONE
