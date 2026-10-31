import importlib

import pytest
from pydantic import SecretStr

from backend.data.model import (
    APIKeyCredentials,
    HostScopedCredentials,
    OAuth2Credentials,
    UserPasswordCredentials,
)
from backend.integrations.credential_input import (
    InvalidCredentialInputError,
    normalize_user_supplied_credentials,
)


def test_api_key_stripped():
    creds = APIKeyCredentials(
        provider="openai", api_key=SecretStr("\tsk-abc  "), title=" T "
    )

    normalize_user_supplied_credentials(creds)

    assert creds.api_key.get_secret_value() == "sk-abc"
    assert creds.title == "T"


@pytest.mark.parametrize("api_key", ["", " ", "\n\t "])
def test_blank_api_key_rejected(api_key: str):
    creds = APIKeyCredentials(provider="openai", api_key=SecretStr(api_key))

    with pytest.raises(InvalidCredentialInputError, match="API key must not be empty"):
        normalize_user_supplied_credentials(creds)


def test_blank_title_becomes_none():
    creds = APIKeyCredentials(provider="openai", api_key=SecretStr("k"), title="  ")

    normalize_user_supplied_credentials(creds)

    assert creds.title is None


def test_password_left_untouched_and_may_be_empty():
    creds = UserPasswordCredentials(
        provider="smtp", username=SecretStr(" bob "), password=SecretStr("")
    )

    normalize_user_supplied_credentials(creds)

    assert creds.username.get_secret_value() == "bob"
    assert creds.password.get_secret_value() == ""


def test_error_never_contains_the_secret():
    creds = HostScopedCredentials(
        provider="http",
        host="api.example.com",
        headers={"X-Key": SecretStr("top-secret"), "Other": SecretStr("  ")},
    )

    with pytest.raises(InvalidCredentialInputError) as exc:
        normalize_user_supplied_credentials(creds)

    assert "top-secret" not in str(exc.value)
    assert str(exc.value) == "Header 'Other' must have a value"


def test_repeated_header_after_strip_rejected():
    creds = HostScopedCredentials(
        provider="http",
        host="api.example.com",
        headers={"X-Key": SecretStr("a"), " X-Key ": SecretStr("b")},
    )

    with pytest.raises(InvalidCredentialInputError, match="repeated"):
        normalize_user_supplied_credentials(creds)


def test_oauth_credentials_unchanged():
    creds = OAuth2Credentials(
        provider="github", access_token=SecretStr(" tok "), scopes=[], title=None
    )

    normalize_user_supplied_credentials(creds)

    assert creds.access_token.get_secret_value() == " tok "


def test_credentials_store_imports_with_empty_settings():
    """System credentials are built at import time with SecretStr("") for unset
    settings. The credential models must keep accepting that, so blank-secret
    checks have to stay at the request boundary, not on the models."""
    credentials_store = importlib.import_module(
        "backend.integrations.credentials_store"
    )

    blank_system_creds = [
        value
        for value in vars(credentials_store).values()
        if isinstance(value, APIKeyCredentials)
        and value.api_key.get_secret_value() == ""
    ]
    assert blank_system_creds, "expected unset system API keys to stay empty"
