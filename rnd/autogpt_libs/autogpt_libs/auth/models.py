from dataclasses import dataclass


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
