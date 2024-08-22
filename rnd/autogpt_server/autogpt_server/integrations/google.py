from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow

from autogpt_libs.supabase_integration_credentials_store import OAuth2Credentials

from .oauth import BaseOAuthHandler


class GoogleOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at https://developers.google.com/identity/protocols/oauth2/web-server
    """  # noqa

    PROVIDER_NAME = "google"

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

        credentials = flow.credentials
        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title="Google",
            access_token=credentials.token,
            refresh_token=credentials.refresh_token,
            access_token_expires_at=int(credentials.expiry.timestamp()),
            refresh_token_expires_at=None,
            scopes=credentials.scopes,
        )

    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        google_creds = Credentials(
            token=credentials.access_token.get_secret_value(),
            refresh_token=credentials.refresh_token.get_secret_value(),
            token_uri=self.token_uri,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=credentials.scopes,
        )

        google_creds.refresh(Request())

        return OAuth2Credentials(
            id=credentials.id,
            provider=self.PROVIDER_NAME,
            title=credentials.title,
            access_token=google_creds.token,
            refresh_token=google_creds.refresh_token,
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
