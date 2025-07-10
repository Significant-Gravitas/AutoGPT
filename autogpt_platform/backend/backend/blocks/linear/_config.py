"""
Shared configuration for all Linear blocks using the new SDK pattern.
"""

import os
from enum import Enum

from backend.sdk import (
    APIKeyCredentials,
    BlockCostType,
    OAuth2Credentials,
    ProviderBuilder,
    SecretStr,
)

from ._oauth import LinearOAuthHandler

# (required) Comma separated list of scopes:

# read - (Default) Read access for the user's account. This scope will always be present.

# write - Write access for the user's account. If your application only needs to create comments, use a more targeted scope

# issues:create - Allows creating new issues and their attachments

# comments:create - Allows creating new issue comments

# timeSchedule:write - Allows creating and modifying time schedules


# admin - Full access to admin level endpoints. You should never ask for this permission unless it's absolutely needed
class LinearScope(str, Enum):
    READ = "read"
    WRITE = "write"
    ISSUES_CREATE = "issues:create"
    COMMENTS_CREATE = "comments:create"
    TIME_SCHEDULE_WRITE = "timeSchedule:write"
    ADMIN = "admin"


# Check if Linear OAuth is configured
client_id = os.getenv("LINEAR_CLIENT_ID")
client_secret = os.getenv("LINEAR_CLIENT_SECRET")
LINEAR_OAUTH_IS_CONFIGURED = bool(client_id and client_secret)

# Build the Linear provider
builder = (
    ProviderBuilder("linear")
    .with_api_key(env_var_name="LINEAR_API_KEY", title="Linear API Key")
    .with_base_cost(1, BlockCostType.RUN)
)

# Linear only supports OAuth authentication
if LINEAR_OAUTH_IS_CONFIGURED:
    builder = builder.with_oauth(
        LinearOAuthHandler,
        scopes=[
            LinearScope.READ,
            LinearScope.WRITE,
            LinearScope.ISSUES_CREATE,
            LinearScope.COMMENTS_CREATE,
        ],
        client_id_env_var="LINEAR_CLIENT_ID",
        client_secret_env_var="LINEAR_CLIENT_SECRET",
    )

# Build the provider
linear = builder.build()


TEST_CREDENTIALS_OAUTH = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="linear",
    title="Mock Linear API key",
    username="mock-linear-username",
    access_token=SecretStr("mock-linear-access-token"),
    access_token_expires_at=None,
    refresh_token=SecretStr("mock-linear-refresh-token"),
    refresh_token_expires_at=None,
    scopes=["mock-linear-scopes"],
)

TEST_CREDENTIALS_API_KEY = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="linear",
    title="Mock Linear API key",
    api_key=SecretStr("mock-linear-api-key"),
    expires_at=None,
)

TEST_CREDENTIALS_INPUT_OAUTH = {
    "provider": TEST_CREDENTIALS_OAUTH.provider,
    "id": TEST_CREDENTIALS_OAUTH.id,
    "type": TEST_CREDENTIALS_OAUTH.type,
    "title": TEST_CREDENTIALS_OAUTH.type,
}

TEST_CREDENTIALS_INPUT_API_KEY = {
    "provider": TEST_CREDENTIALS_API_KEY.provider,
    "id": TEST_CREDENTIALS_API_KEY.id,
    "type": TEST_CREDENTIALS_API_KEY.type,
    "title": TEST_CREDENTIALS_API_KEY.type,
}
