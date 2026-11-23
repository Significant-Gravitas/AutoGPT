"""Which connected account a tool call used, as its owner knows it, and never
anything that would let someone else use it."""

import pytest
from pydantic import SecretStr

from backend.copilot import call_account
from backend.copilot.call_account import account_for
from backend.data.model import (
    APIKeyCredentials,
    OAuth2Credentials,
    UserPasswordCredentials,
)
from backend.data.task_activity_meta import TaskActivityAccount
from backend.integrations.credentials_store import ideogram_credentials

_TOKEN = "ya29.a0AfB_byC-access-token-value-1234567890"
_REFRESH = "1//0refresh-token-value-abcdefghijklmnop"
_KEY = "sk-live-0123456789abcdefghijklmnopqrstuv"


def _google(**overrides) -> OAuth2Credentials:
    fields = {
        "id": "cred-google-1",
        "provider": "google",
        "username": "work@acme.com",
        "title": None,
        "access_token": SecretStr(_TOKEN),
        "refresh_token": SecretStr(_REFRESH),
        "scopes": ["https://www.googleapis.com/auth/gmail.send"],
    }
    return OAuth2Credentials(**(fields | overrides))


def _api_key(title: str | None, provider: str = "github") -> APIKeyCredentials:
    return APIKeyCredentials(
        id="cred-key-1", provider=provider, api_key=SecretStr(_KEY), title=title
    )


def test_an_oauth_account_is_named_by_its_email():
    assert account_for(_google()) == TaskActivityAccount(
        service="google", name="Google", label="work@acme.com"
    )


def test_an_api_key_is_named_by_the_title_it_was_saved_under():
    account = account_for(_api_key("Work GitHub"))
    assert account == TaskActivityAccount(
        service="github", name="GitHub", label="Work GitHub"
    )


def test_an_account_without_a_name_of_its_own_shows_the_service_alone():
    assert account_for(_api_key(None)) == TaskActivityAccount(
        service="github", name="GitHub"
    )
    assert account_for(_api_key("GitHub")) == TaskActivityAccount(
        service="github", name="GitHub"
    )


def test_no_secret_scope_or_credential_id_is_ever_captured():
    for credentials in (_google(), _api_key("Work GitHub")):
        account = account_for(credentials)
        assert account is not None
        dumped = account.model_dump_json()
        for secret in (_TOKEN, _REFRESH, _KEY, credentials.id, "gmail.send"):
            assert secret not in dumped


@pytest.mark.parametrize(
    "title",
    [
        _KEY,
        "my key sk-live-0123456789abcdefghijklmnopqrstuv",
        "token for cred-key-1",
        "x" * 81,
    ],
)
def test_a_title_that_could_be_a_secret_or_an_id_is_not_shown(title):
    account = account_for(_api_key(title))
    assert account is not None and account.label is None


def test_a_login_is_never_named_by_its_username():
    login = UserPasswordCredentials(
        id="cred-smtp-1",
        provider="smtp",
        username=SecretStr("admin@mail.acme.com"),
        password=SecretStr("hunter2"),
        title=None,
    )
    account = account_for(login)
    assert account is not None
    assert "admin@mail.acme.com" not in account.model_dump_json()
    assert "hunter2" not in account.model_dump_json()


def test_the_platforms_own_key_names_the_service_but_no_account():
    account = account_for(ideogram_credentials)
    assert account is not None
    assert account.service == "ideogram" and account.label is None


def test_a_model_providers_key_is_not_a_connected_integration():
    assert account_for(_api_key("My OpenAI key", provider="openai")) is None


def test_an_mcp_account_is_named_after_its_catalog_server():
    credentials = _google(
        provider="mcp",
        username="dev@acme.com",
        metadata={"mcp_server_url": "https://mcp.aave.com"},
    )
    account = account_for(credentials)
    assert account is not None
    assert (account.name, account.label) == ("Aave", "dev@acme.com")


def test_a_call_collects_the_accounts_its_code_used_once_each():
    with call_account.collect() as accounts:
        call_account.note(_google())
        call_account.note(_google())
        call_account.note(_api_key("My OpenAI key", provider="openai"))
    assert [a.label for a in accounts] == ["work@acme.com"]


def test_noting_outside_a_call_does_nothing():
    call_account.note(_google())
    with call_account.collect() as accounts:
        pass
    assert accounts == []


def test_an_account_that_cannot_be_described_never_breaks_the_call(mocker):
    mocker.patch(
        "backend.copilot.call_account.service_for_credential",
        side_effect=RuntimeError("catalog down"),
    )
    with call_account.collect() as accounts:
        call_account.note(_google())
    assert accounts == []
