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
    email_verified: bool = False

    @classmethod
    def from_payload(cls, payload):
        return cls(
            user_id=payload["sub"],
            email=payload.get("email", ""),
            phone_number=payload.get("phone", ""),
            role=payload.get("role", "authenticated"),
            email_verified=payload.get("email_verified", False),
        )
