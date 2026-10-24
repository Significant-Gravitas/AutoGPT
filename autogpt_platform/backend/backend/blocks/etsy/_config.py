"""Shared provider and credential configuration for Etsy blocks.

Etsy offers no personal API key: every shop connects through the platform's
Etsy app with OAuth (PKCE), and every request also carries that app's
keystring and shared secret. Without them the blocks are disabled and the
provider offers no sign-in, so nothing promises an Etsy connection that
cannot be made.
"""

from enum import Enum

from backend.sdk import (
    BlockCostType,
    CredentialsMetaInput,
    OAuth2Credentials,
    ProviderBuilder,
    ProviderName,
    SecretStr,
)
from backend.util.settings import Secrets

from ._oauth import EtsyOAuthHandler


class EtsyScope(str, Enum):
    SHOPS_READ = "shops_r"
    LISTINGS_READ = "listings_r"
    LISTINGS_WRITE = "listings_w"


SCOPES = [scope.value for scope in EtsyScope]

secrets = Secrets()
ETSY_OAUTH_IS_CONFIGURED = bool(secrets.etsy_client_id and secrets.etsy_client_secret)

etsy = (
    ProviderBuilder("etsy")
    .with_description("Shop listings, photos and publishing")
    .with_oauth(
        EtsyOAuthHandler,
        scopes=SCOPES,
        client_id_env_var="ETSY_CLIENT_ID",
        client_secret_env_var="ETSY_CLIENT_SECRET",  # pragma: allowlist secret
    )
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)

EtsyCredentials = OAuth2Credentials
EtsyCredentialsInput = CredentialsMetaInput

TEST_CREDENTIALS = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="etsy",
    access_token=SecretStr("12345678.mock-etsy-access-token"),
    refresh_token=SecretStr("12345678.mock-etsy-refresh-token"),
    access_token_expires_at=None,
    title="Mock Etsy shop",
    scopes=SCOPES,
    username="StickerStudio",
)
TEST_CREDENTIALS_INPUT = CredentialsMetaInput(
    provider=ProviderName("etsy"),
    id=TEST_CREDENTIALS.id,
    type=TEST_CREDENTIALS.type,
    title=TEST_CREDENTIALS.title,
).model_dump(mode="json")
