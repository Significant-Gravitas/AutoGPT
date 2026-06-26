from typing import Literal

from pydantic import SecretStr

from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
)
from backend.integrations.providers import ProviderName

StripeCredentials = APIKeyCredentials
StripeCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.STRIPE],
    Literal["api_key"],
]


def StripeCredentialsField() -> StripeCredentialsInput:
    return CredentialsField(
        description="Stripe API secret key (sk_live_... or sk_test_...). "
        "Requires permission to create and delete webhook endpoints.",
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="stripe",
    api_key=SecretStr("mock-stripe-api-key"),
    title="Mock Stripe API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}
