"""
Airtable OAuth handler implementation.
"""

import time
from enum import Enum
from logging import getLogger
from typing import Optional

from backend.sdk import BaseOAuthHandler, OAuth2Credentials, ProviderName, SecretStr

from ._api import (
    OAuthTokenResponse,
    make_oauth_authorize_url,
    oauth_exchange_code_for_tokens,
    oauth_refresh_tokens,
)

logger = getLogger(__name__)


class AirtableScope(str, Enum):
    # Basic scopes
    DATA_RECORDS_READ = "data.records:read"
    DATA_RECORDS_WRITE = "data.records:write"
    DATA_RECORD_COMMENTS_READ = "data.recordComments:read"
    DATA_RECORD_COMMENTS_WRITE = "data.recordComments:write"
    SCHEMA_BASES_READ = "schema.bases:read"
    SCHEMA_BASES_WRITE = "schema.bases:write"
    WEBHOOK_MANAGE = "webhook:manage"
    BLOCK_MANAGE = "block:manage"
    USER_EMAIL_READ = "user.email:read"

    # Enterprise member scopes
    ENTERPRISE_GROUPS_READ = "enterprise.groups:read"
    WORKSPACES_AND_BASES_READ = "workspacesAndBases:read"
    WORKSPACES_AND_BASES_WRITE = "workspacesAndBases:write"
    WORKSPACES_AND_BASES_SHARES_MANAGE = "workspacesAndBases.shares:manage"

    # Enterprise admin scopes
    ENTERPRISE_SCIM_USERS_AND_GROUPS_MANAGE = "enterprise.scim.usersAndGroups:manage"
    ENTERPRISE_AUDIT_LOGS_READ = "enterprise.auditLogs:read"
    ENTERPRISE_CHANGE_EVENTS_READ = "enterprise.changeEvents:read"
    ENTERPRISE_EXPORTS_MANAGE = "enterprise.exports:manage"
    ENTERPRISE_ACCOUNT_READ = "enterprise.account:read"
    ENTERPRISE_ACCOUNT_WRITE = "enterprise.account:write"
    ENTERPRISE_USER_READ = "enterprise.user:read"
    ENTERPRISE_USER_WRITE = "enterprise.user:write"
    ENTERPRISE_GROUPS_MANAGE = "enterprise.groups:manage"
    WORKSPACES_AND_BASES_MANAGE = "workspacesAndBases:manage"
    HYPERDB_RECORDS_READ = "hyperDB.records:read"
    HYPERDB_RECORDS_WRITE = "hyperDB.records:write"


class AirtableOAuthHandler(BaseOAuthHandler):
    """
    OAuth2 handler for Airtable with PKCE support.
    """

    PROVIDER_NAME = ProviderName("airtable")
    DEFAULT_SCOPES = [
        v.value
        for v in [
            AirtableScope.DATA_RECORDS_READ,
            AirtableScope.DATA_RECORDS_WRITE,
            AirtableScope.SCHEMA_BASES_READ,
            AirtableScope.SCHEMA_BASES_WRITE,
            AirtableScope.WEBHOOK_MANAGE,
        ]
    ]

    def __init__(self, client_id: str, client_secret: Optional[str], redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = self.DEFAULT_SCOPES
        self.auth_base_url = "https://airtable.com/oauth2/v1/authorize"
        self.token_url = "https://airtable.com/oauth2/v1/token"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        logger.debug("Generating Airtable OAuth login URL")
        # Generate code_challenge if not provided (PKCE is required)
        if not scopes:
            logger.debug("No scopes provided, using default scopes")
            scopes = self.scopes

        logger.debug(f"Using scopes: {scopes}")
        logger.debug(f"State: {state}")
        logger.debug(f"Code challenge: {code_challenge}")
        if not code_challenge:
            logger.error("Code challenge is required but none was provided")
            raise ValueError("No code challenge provided")

        try:
            url = make_oauth_authorize_url(
                self.client_id, self.redirect_uri, scopes, state, code_challenge
            )
            logger.debug(f"Generated OAuth URL: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate OAuth URL: {str(e)}")
            raise

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        logger.debug("Exchanging authorization code for tokens")
        logger.debug(f"Code: {code[:4]}...")  # Log first 4 chars only for security
        logger.debug(f"Scopes: {scopes}")
        if not code_verifier:
            logger.error("Code verifier is required but none was provided")
            raise ValueError("No code verifier provided")

        try:
            response: OAuthTokenResponse = await oauth_exchange_code_for_tokens(
                client_id=self.client_id,
                code=code,
                code_verifier=code_verifier.encode("utf-8"),
                redirect_uri=self.redirect_uri,
                client_secret=self.client_secret,
            )
            logger.info("Successfully exchanged code for tokens")

            credentials = OAuth2Credentials(
                access_token=SecretStr(response.access_token),
                refresh_token=SecretStr(response.refresh_token),
                access_token_expires_at=int(time.time()) + response.expires_in,
                refresh_token_expires_at=int(time.time()) + response.refresh_expires_in,
                provider=self.PROVIDER_NAME,
                scopes=scopes,
            )
            logger.debug(f"Access token expires in {response.expires_in} seconds")
            logger.debug(
                f"Refresh token expires in {response.refresh_expires_in} seconds"
            )
            return credentials

        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {str(e)}")
            raise

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        logger.debug("Attempting to refresh OAuth tokens")

        if credentials.refresh_token is None:
            logger.error("Cannot refresh tokens - no refresh token available")
            raise ValueError("No refresh token available")

        try:
            response: OAuthTokenResponse = await oauth_refresh_tokens(
                client_id=self.client_id,
                refresh_token=credentials.refresh_token.get_secret_value(),
                client_secret=self.client_secret,
            )
            logger.info("Successfully refreshed tokens")

            new_credentials = OAuth2Credentials(
                id=credentials.id,
                access_token=SecretStr(response.access_token),
                refresh_token=SecretStr(response.refresh_token),
                access_token_expires_at=int(time.time()) + response.expires_in,
                refresh_token_expires_at=int(time.time()) + response.refresh_expires_in,
                provider=self.PROVIDER_NAME,
                scopes=self.scopes,
            )
            logger.debug(f"New access token expires in {response.expires_in} seconds")
            logger.debug(
                f"New refresh token expires in {response.refresh_expires_in} seconds"
            )
            return new_credentials

        except Exception as e:
            logger.error(f"Failed to refresh tokens: {str(e)}")
            raise

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        logger.debug("Token revocation requested")
        logger.info(
            "Airtable doesn't provide a token revocation endpoint - tokens will expire naturally after 60 minutes"
        )
        return False
