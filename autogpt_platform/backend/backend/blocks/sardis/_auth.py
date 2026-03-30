import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput
from backend.integrations.providers import ProviderName

SardisCredentials = APIKeyCredentials
SardisCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.SARDIS],
    Literal["api_key"],
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="sardis",
    api_key=SecretStr("mock-sardis-api-key"),
    title="Mock Sardis API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------

_WALLET_ID_RE = re.compile(r"^wal_[a-zA-Z0-9]+$")


def validate_wallet_id(v: str) -> str:
    """Validate that a wallet ID matches the ``wal_<alnum>`` pattern."""
    if not _WALLET_ID_RE.match(v):
        raise ValueError(
            "wallet_id must start with 'wal_' followed by alphanumeric "
            f"characters, got '{v}'"
        )
    return v


_MAX_DESTINATION_LEN = 256
_DESTINATION_RE = re.compile(r"^[a-zA-Z0-9_.\-@:]+$")


def validate_destination(v: str) -> str:
    """Validate destination is a reasonable address/ID (alphanumeric + common separators)."""
    if len(v) > _MAX_DESTINATION_LEN:
        raise ValueError(
            f"destination must be at most {_MAX_DESTINATION_LEN} characters, "
            f"got {len(v)}"
        )
    if not _DESTINATION_RE.match(v):
        raise ValueError(
            "destination must contain only alphanumeric characters "
            f"and . _ - @ : separators, got '{v[:50]}'"
        )
    return v


def validate_amount(v: str) -> str:
    """Validate that an amount is a finite decimal string >= 0.01."""
    try:
        val = Decimal(v)
    except (InvalidOperation, TypeError):
        raise ValueError(f"amount must be a numeric string, got '{v}'")
    if not val.is_finite():
        raise ValueError(f"amount must be a finite numeric string, got '{v}'")
    if val < Decimal("0.01"):
        raise ValueError(f"amount must be >= 0.01, got '{v}'")
    return v


def SardisCredentialsField() -> SardisCredentialsInput:
    """Creates a Sardis credentials input on a block."""
    return CredentialsField(
        description="The Sardis integration requires an API key. "
        "Get one at https://sardis.sh",
    )
