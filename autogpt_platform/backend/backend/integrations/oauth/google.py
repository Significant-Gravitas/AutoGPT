import logging
from typing import List

from autogpt_libs.supabase_integration_credentials_store import OAuth2Credentials
from google.auth.external_account_authorized_user import (
    Credentials as ExternalAccountCredentials,
)
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from pydantic import SecretStr

from .base import BaseOAuthHandler

logger = logging.getLogger(__name__)


class GoogleOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at https://developers.google.com/identity/protocols/oauth2/web-server
    """  # noqa

    PROVIDER_NAME = "google"
    EMAIL_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_uri = "https://oauth2.googleapis.com/token"

    def get_login_url(self, scopes: List[str], state: str) -> str:
        logger.info(f"Getting login URL with scopes: {scopes}")
        flow = self._setup_oauth_flow(scopes)
        flow.redirect_uri = self.redirect_uri
        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            state=state,
            prompt="consent",
        )
        logger.info(f"Generated authorization URL: {authorization_url}")
        return authorization_url

    def exchange_code_for_tokens(
        self, code: str, scopes: list[str]
    ) -> OAuth2Credentials:
        logger.info("Starting code exchange for tokens")
        try:
            flow = self._setup_oauth_flow(scopes)
            flow.redirect_uri = self.redirect_uri
            logger.debug(f"OAuth flow set up with redirect URI: {self.redirect_uri}")

            logger.info("Fetching token from Google")
            flow.fetch_token(code=code)
            logger.info("Token fetched successfully")

            google_creds = flow.credentials
            logger.debug(f"Received credentials: {google_creds}")
            logger.info(f"Scopes received: {google_creds.scopes}")

            logger.info("Requesting user email")
            username = self._request_email(google_creds)
            logger.info(f"User email retrieved: {username}")

            # Google's OAuth library is poorly typed so we need some of these:
            if not google_creds.token:
                logger.error("No access token received from Google")
                raise ValueError("No access token received from Google")
            if not google_creds.refresh_token:
                logger.warning("No refresh token received from Google")
            if not google_creds.expiry:
                logger.warning("No expiry time received from Google")
            if not google_creds.scopes:
                logger.warning("No scopes received from Google")

            logger.info("Creating OAuth2Credentials object")
            credentials = OAuth2Credentials(
                provider=self.PROVIDER_NAME,
                title="test",
                username=username,
                access_token=SecretStr(google_creds.token),
                refresh_token=(
                    SecretStr(google_creds.refresh_token)
                    if google_creds.refresh_token
                    else None
                ),
                access_token_expires_at=(
                    int(google_creds.expiry.timestamp())
                    if google_creds.expiry
                    else None
                ),
                refresh_token_expires_at=None,
                scopes=google_creds.scopes or [],
            )
            logger.info(
                f"OAuth2Credentials object created successfully with scopes: {credentials.scopes}"
            )

            return credentials

        except Exception as e:
            logger.error(f"Error during code exchange: {str(e)}", exc_info=True)
            raise

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

    def _setup_oauth_flow(self, scopes: List[str]) -> Flow:
        logger.info(f"Setting up OAuth flow with scopes: {scopes}")
        flow = Flow.from_client_config(
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
        logger.info(f"OAuth flow created with scopes: {flow.oauth2session.scope}")
        return flow
