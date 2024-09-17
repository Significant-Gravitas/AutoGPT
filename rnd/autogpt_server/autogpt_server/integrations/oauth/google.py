from autogpt_libs.supabase_integration_credentials_store import OAuth2Credentials
from google.auth.external_account_authorized_user import (
    Credentials as ExternalAccountCredentials,
)
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from pydantic import SecretStr

from .base import BaseOAuthHandler


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

    def get_login_url(self, scopes: list[str], state: str) -> str:
        flow = self._setup_oauth_flow(scopes)
        flow.redirect_uri = self.redirect_uri
        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            state=state,
            prompt="consent",
        )
        return authorization_url

    def exchange_code_for_tokens(self, code: str) -> OAuth2Credentials:
        flow = self._setup_oauth_flow(None)
        flow.redirect_uri = self.redirect_uri
        flow.fetch_token(code=code)

        google_creds = flow.credentials
        username = self._request_email(google_creds)

        # Google's OAuth library is poorly typed so we need some of these:
        assert google_creds.token
        assert google_creds.refresh_token
        assert google_creds.expiry
        assert google_creds.scopes
        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token=SecretStr(google_creds.token),
            refresh_token=SecretStr(google_creds.refresh_token),
            access_token_expires_at=int(google_creds.expiry.timestamp()),
            refresh_token_expires_at=None,
            scopes=google_creds.scopes,
        )

    def _request_email(
        self, creds: Credentials | ExternalAccountCredentials
    ) -> str | None:
        session = AuthorizedSession(creds)
        response = session.get(self.EMAIL_ENDPOINT)
        if not response.ok:
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

    def _setup_oauth_flow(self, scopes: list[str] | None) -> Flow:
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
