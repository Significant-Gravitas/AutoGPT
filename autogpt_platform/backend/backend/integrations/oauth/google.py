import logging
from typing import Optional

from google.auth.external_account_authorized_user import (
    Credentials as ExternalAccountCredentials,
)
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName

from .base import BaseOAuthHandler

logger = logging.getLogger(__name__)


# --8<-- [start:GoogleOAuthHandlerExample]
class GoogleOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at https://developers.google.com/identity/protocols/oauth2/web-server
    """  # noqa

    PROVIDER_NAME = ProviderName.GOOGLE
    EMAIL_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
    DEFAULT_SCOPES = [
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "openid",
    ]
    # --8<-- [end:GoogleOAuthHandlerExample]

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_uri = "https://oauth2.googleapis.com/token"
        self.revoke_uri = "https://oauth2.googleapis.com/revoke"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        all_scopes = list(set(scopes + self.DEFAULT_SCOPES))
        logger.debug(f"Setting up OAuth flow with scopes: {all_scopes}")
        flow = self._setup_oauth_flow(all_scopes)
        flow.redirect_uri = self.redirect_uri
        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            state=state,
            prompt="consent",
        )
        return authorization_url

    def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        logger.debug(f"Exchanging code for tokens with scopes: {scopes}")

        # Use the scopes from the initial request
        flow = self._setup_oauth_flow(scopes)
        flow.redirect_uri = self.redirect_uri

        logger.debug("Fetching token from Google")

        # Disable scope check in fetch_token
        flow.oauth2session.scope = None
        token = flow.fetch_token(code=code)
        logger.debug("Token fetched successfully")

        # Get the actual scopes granted by Google
        granted_scopes: list[str] = token.get("scope", [])

        logger.debug(f"Scopes granted by Google: {granted_scopes}")

        google_creds = flow.credentials
        logger.debug(f"Received credentials: {google_creds}")

        logger.debug("Requesting user email")
        username = self._request_email(google_creds)
        logger.debug(f"User email retrieved: {username}")

        assert google_creds.token
        assert google_creds.refresh_token
        assert google_creds.expiry
        assert granted_scopes

        # Create OAuth2Credentials with the granted scopes
        credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token=SecretStr(google_creds.token),
            refresh_token=(SecretStr(google_creds.refresh_token)),
            access_token_expires_at=(
                int(google_creds.expiry.timestamp()) if google_creds.expiry else None
            ),
            refresh_token_expires_at=None,
            scopes=granted_scopes,
        )
        logger.debug(
            f"OAuth2Credentials object created successfully with scopes: {credentials.scopes}"
        )

        return credentials

    def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        session = AuthorizedSession(credentials)
        session.post(
            self.revoke_uri,
            params={"token": credentials.access_token.get_secret_value()},
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        return True

    def _request_email(
        self, creds: Credentials | ExternalAccountCredentials
    ) -> str | None:
        session = AuthorizedSession(creds)
        response = session.get(self.EMAIL_ENDPOINT)
        if not response.ok:
            logger.error(
                f"Failed to get user email. Status code: {response.status_code}"
            )
            return None
        return response.json()["email"]

    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        # Google credentials should ALWAYS have a refresh token
        assert credentials.refresh_token

        google_creds = Credentials(
            token=credentials.access_token.get_secret_value(),
            refresh_token=credentials.refresh_token.get_secret_value(),
            token_uri=self.token_uri,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=credentials.scopes,
        )
        # Google's OAuth library is poorly typed so we need some of these:
        assert google_creds.refresh_token
        assert google_creds.scopes

        google_creds.refresh(Request())
        assert google_creds.expiry

        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            id=credentials.id,
            title=credentials.title,
            username=credentials.username,
            access_token=SecretStr(google_creds.token),
            refresh_token=SecretStr(google_creds.refresh_token),
            access_token_expires_at=int(google_creds.expiry.timestamp()),
            refresh_token_expires_at=None,
            scopes=google_creds.scopes,
        )

    def _setup_oauth_flow(self, scopes: list[str]) -> Flow:
        return Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": self.token_uri,
                }
            },
            scopes=scopes,
        )
