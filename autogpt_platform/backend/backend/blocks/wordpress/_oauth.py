import time
from enum import Enum
from logging import getLogger
from typing import Optional
from urllib.parse import quote

from backend.sdk import BaseOAuthHandler, OAuth2Credentials, ProviderName, SecretStr

from ._api import (
    OAuthTokenResponse,
    TokenInfoResponse,
    make_oauth_authorize_url,
    oauth_exchange_code_for_tokens,
    oauth_refresh_tokens,
    validate_token,
)

logger = getLogger(__name__)


class WordPressScope(str, Enum):
    """WordPress OAuth2 scopes.

    Note: If no scope is specified, the token will grant full access to a single blog.
    Special scopes:
    - auth: Access to /me endpoints only, primarily for WordPress.com Connect
    - global: Full access to all blogs the user has on WordPress.com
    """

    # Common endpoint-specific scopes
    POSTS = "posts"
    COMMENTS = "comments"
    LIKES = "likes"
    FOLLOW = "follow"
    STATS = "stats"
    USERS = "users"
    SITES = "sites"
    MEDIA = "media"

    # Special scopes
    AUTH = "auth"  # Access to /me endpoints only
    GLOBAL = "global"  # Full access to all user's blogs


class WordPressOAuthHandler(BaseOAuthHandler):
    """
    OAuth2 handler for WordPress.com and Jetpack sites.

    Supports both single blog and global access tokens.
    Server-side tokens (using 'code' response type) do not expire.
    """

    PROVIDER_NAME = ProviderName("wordpress")
    # Default to no scopes for single blog access
    DEFAULT_SCOPES = []

    def __init__(self, client_id: str, client_secret: Optional[str], redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = self.DEFAULT_SCOPES

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str] = None
    ) -> str:
        logger.debug("Generating WordPress OAuth login URL")
        # WordPress doesn't require PKCE, so code_challenge is not used
        if not scopes:
            logger.debug("No scopes provided, will default to single blog access")
            scopes = self.scopes

        logger.debug(f"Using scopes: {scopes}")
        logger.debug(f"State: {state}")

        try:
            base_url = make_oauth_authorize_url(
                self.client_id, self.redirect_uri, scopes if scopes else None
            )

            separator = "&" if "?" in base_url else "?"
            url = f"{base_url}{separator}state={quote(state)}"
            logger.debug(f"Generated OAuth URL: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate OAuth URL: {str(e)}")
            raise

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str] = None
    ) -> OAuth2Credentials:
        logger.debug("Exchanging authorization code for tokens")
        logger.debug(f"Code: {code[:4]}...")
        logger.debug(f"Scopes: {scopes}")

        # WordPress doesn't use PKCE, so code_verifier is not needed

        try:
            response: OAuthTokenResponse = await oauth_exchange_code_for_tokens(
                client_id=self.client_id,
                client_secret=self.client_secret if self.client_secret else "",
                code=code,
                redirect_uri=self.redirect_uri,
            )
            logger.info("Successfully exchanged code for tokens")

            # Store blog info in metadata
            metadata = {}
            if response.blog_id:
                metadata["blog_id"] = response.blog_id
            if response.blog_url:
                metadata["blog_url"] = response.blog_url

            # WordPress tokens from code flow don't expire
            credentials = OAuth2Credentials(
                access_token=SecretStr(response.access_token),
                refresh_token=(
                    SecretStr(response.refresh_token)
                    if response.refresh_token
                    else None
                ),
                access_token_expires_at=None,
                refresh_token_expires_at=None,
                provider=self.PROVIDER_NAME,
                scopes=scopes if scopes else [],
                metadata=metadata,
            )

            if response.expires_in:
                logger.debug(
                    f"Token expires in {response.expires_in} seconds (client-side token)"
                )
            else:
                logger.debug("Token does not expire (server-side token)")

            return credentials

        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {str(e)}")
            raise

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        """
        Added for completeness, as WordPress tokens don't expire
        """

        logger.debug("Attempting to refresh OAuth tokens")

        # Server-side tokens don't expire
        if credentials.access_token_expires_at is None:
            logger.info("Token does not expire (server-side token), no refresh needed")
            return credentials

        if credentials.refresh_token is None:
            logger.error("Cannot refresh tokens - no refresh token available")
            raise ValueError("No refresh token available")

        try:
            response: OAuthTokenResponse = await oauth_refresh_tokens(
                client_id=self.client_id,
                client_secret=self.client_secret if self.client_secret else "",
                refresh_token=credentials.refresh_token.get_secret_value(),
            )
            logger.info("Successfully refreshed tokens")

            # Preserve blog info from original credentials
            metadata = credentials.metadata or {}
            if response.blog_id:
                metadata["blog_id"] = response.blog_id
            if response.blog_url:
                metadata["blog_url"] = response.blog_url

            new_credentials = OAuth2Credentials(
                access_token=SecretStr(response.access_token),
                refresh_token=(
                    SecretStr(response.refresh_token)
                    if response.refresh_token
                    else credentials.refresh_token
                ),
                access_token_expires_at=(
                    int(time.time()) + response.expires_in
                    if response.expires_in
                    else None
                ),
                refresh_token_expires_at=None,
                provider=self.PROVIDER_NAME,
                scopes=credentials.scopes,
                metadata=metadata,
            )

            if response.expires_in:
                logger.debug(
                    f"New access token expires in {response.expires_in} seconds"
                )
            else:
                logger.debug("New token does not expire")

            return new_credentials

        except Exception as e:
            logger.error(f"Failed to refresh tokens: {str(e)}")
            raise

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        logger.debug("Token revocation requested")
        logger.info(
            "WordPress doesn't provide a token revocation endpoint - server-side tokens don't expire"
        )
        return False

    async def validate_access_token(self, token: str) -> TokenInfoResponse:
        """Validate an access token and get associated metadata."""
        return await validate_token(self.client_id, token)
