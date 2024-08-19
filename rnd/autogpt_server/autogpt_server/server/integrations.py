from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel

from .utils import get_user_id

integrations_api = APIRouter()


class GoogleAuthExchangeRequestBody(BaseModel):
    code: str


class GoogleAuthExchangeResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_uri: str
    client_id: str
    client_secret: str
    scopes: str


@integrations_api.post("/auth/google")
def exchange_google_auth_code(
    body: GoogleAuthExchangeRequestBody,
    user_id: Annotated[str, Depends(get_user_id)],
) -> GoogleAuthExchangeResponse:
    # Set up the OAuth 2.0 flow
    flow = Flow.from_client_secrets_file(
        Path(__file__).parent.parent.parent / "google_client_secret.json",
        scopes=[],  # irrelevant since requesting scopes is done by the front end
    )

    # Exchange the authorization code for credentials
    flow.fetch_token(code=body.code)

    # Get the credentials
    credentials = flow.credentials

    # Refresh the token if it's expired
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    # Return the tokens
    return GoogleAuthExchangeResponse(
        access_token=credentials.token,
        refresh_token=credentials.refresh_token,
        token_uri=credentials.token_uri,
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        scopes=credentials.scopes,
    )
