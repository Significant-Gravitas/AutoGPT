from typing import Literal

from pydantic import SecretStr

from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    OAuth2Credentials,
    ProviderName,
)
from backend.integrations.oauth.twitter import TwitterOAuthHandler
from backend.util.settings import Secrets

# --8<-- [start:TwitterOAuthIsConfigured]
secrets = Secrets()
TWITTER_OAUTH_IS_CONFIGURED = bool(
    secrets.twitter_client_id and secrets.twitter_client_secret
)
# --8<-- [end:TwitterOAuthIsConfigured]

TwitterCredentials = OAuth2Credentials
TwitterCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.TWITTER], Literal["oauth2"]
]


# Currently, We are getting all the permission from the Twitter API initally
# In future, If we need to add incremental permission, we can use these requested_scopes
def TwitterCredentialsField(scopes: list[str]) -> TwitterCredentialsInput:
    """
    Creates a Twitter credentials input on a block.

    Params:
        scopes: The authorization scopes needed for the block to work.
    """
    return CredentialsField(
        # required_scopes=set(scopes),
        required_scopes=set(TwitterOAuthHandler.DEFAULT_SCOPES + scopes),
        description="The Twitter integration requires OAuth2 authentication.",
    )


TEST_CREDENTIALS = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="twitter",
    access_token=SecretStr("mock-twitter-access-token"),
    refresh_token=SecretStr("mock-twitter-refresh-token"),
    access_token_expires_at=1234567890,
    scopes=["tweet.read", "tweet.write", "users.read", "offline.access"],
    title="Mock Twitter OAuth2 Credentials",
    username="mock-twitter-username",
    refresh_token_expires_at=1234567890,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
