import time

import pytest
from pydantic import SecretStr

from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    HostScopedCredentials,
    OAuth2Credentials,
    UserPasswordCredentials,
)
from backend.integrations.credential_input import (
    InvalidCredentialInputError,
    normalize_user_supplied_credentials,
)


def _api_key(api_key: str, title: str | None = None) -> APIKeyCredentials:
    return APIKeyCredentials(
        id="c", provider="openai", api_key=SecretStr(api_key), title=title
    )


def _login(username: str, password: str) -> UserPasswordCredentials:
    return UserPasswordCredentials(
        id="c",
        provider="smtp",
        username=SecretStr(username),
        password=SecretStr(password),
    )


def _host_scoped(
    headers: dict[str, str], host: str = "api.example.com"
) -> HostScopedCredentials:
    return HostScopedCredentials(
        id="c",
        provider="http",
        host=host,
        headers={name: SecretStr(value) for name, value in headers.items()},
    )


def _oauth(access_token: str, title: str) -> OAuth2Credentials:
    return OAuth2Credentials(
        id="c",
        provider="github",
        access_token=SecretStr(access_token),
        scopes=[],
        title=title,
    )


@pytest.mark.parametrize(
    "submitted, error",
    [
        pytest.param(_api_key(""), "API key must not be empty", id="empty key"),
        pytest.param(_api_key(" \t\n"), "API key must not be empty", id="blank key"),
        pytest.param(
            _login("  ", "hunter2-secret"),
            "Username must not be empty",
            id="blank username",
        ),
        pytest.param(
            _host_scoped({"Authorization": "Bearer x"}, host="  "),
            "Host must not be empty",
            id="blank host",
        ),
        pytest.param(
            _host_scoped({}), "At least one header is required", id="no headers"
        ),
        pytest.param(
            _host_scoped({"  ": "x"}),
            "Header names must not be empty",
            id="blank header name",
        ),
        pytest.param(
            _host_scoped({"X-Key": "top-secret", "Other": "  "}),
            "Header 'Other' must have a value",
            id="blank header value",
        ),
        pytest.param(
            _host_scoped({"X-Key": "a", " X-Key ": "b"}),
            "Header 'X-Key' is repeated",
            id="header repeated once stripped",
        ),
        pytest.param(
            _host_scoped({"X-Key": "a", "x-key": "b"}),
            "Header 'x-key' is repeated",
            id="header repeated in another case",
        ),
    ],
)
def test_blank_or_repeated_values_are_rejected(submitted: Credentials, error: str):
    # The exact message also shows that no submitted secret is quoted back.
    with pytest.raises(InvalidCredentialInputError) as exc:
        normalize_user_supplied_credentials(submitted)

    assert str(exc.value) == error


def test_checking_many_headers_stays_linear():
    """A request can carry any number of headers, and the check runs on the
    event loop; comparing each name with every earlier one took about 6 s
    for 16,000 headers."""
    submitted = _host_scoped({f"X-Header-{i}": "v" for i in range(20_000)})
    started = time.thread_time()

    normalize_user_supplied_credentials(submitted)

    # CPU time on this thread, so a runner that pauses the test doesn't count.
    assert time.thread_time() - started < 1
    assert len(submitted.headers) == 20_000


@pytest.mark.parametrize(
    "submitted, stored",
    [
        pytest.param(
            _api_key("\tsk-abc  ", title=" My key "),
            _api_key("sk-abc", title="My key"),
            id="key and title stripped",
        ),
        pytest.param(
            _api_key("sk-abc", title="  "),
            _api_key("sk-abc", title=None),
            id="blank title dropped",
        ),
        pytest.param(
            _login(" alice ", " pass word "),
            _login("alice", " pass word "),
            id="username stripped, password kept as typed",
        ),
        pytest.param(
            _login("alice", ""), _login("alice", ""), id="empty password kept"
        ),
        pytest.param(
            _host_scoped({" X-Api-Key ": " secret \n"}, host=" api.example.com "),
            _host_scoped({"X-Api-Key": "secret"}),
            id="host and headers stripped",
        ),
        pytest.param(
            _oauth(" tok ", title=" GitHub "),
            _oauth(" tok ", title="GitHub"),
            id="oauth token untouched",
        ),
    ],
)
def test_values_are_stripped_in_place(submitted: Credentials, stored: Credentials):
    normalize_user_supplied_credentials(submitted)

    assert submitted == stored


@pytest.mark.parametrize("offset", [-3600, 0])
def test_api_key_expiry_not_in_future_rejected(offset: int):
    creds = APIKeyCredentials(
        provider="openai",
        api_key=SecretStr("k"),
        expires_at=int(time.time()) + offset,
    )

    with pytest.raises(
        InvalidCredentialInputError, match="Expiry must be in the future"
    ):
        normalize_user_supplied_credentials(creds)


@pytest.mark.parametrize("expires_at", [None, int(time.time()) + 86400])
def test_api_key_future_or_no_expiry_kept(expires_at: int | None):
    creds = APIKeyCredentials(
        provider="openai", api_key=SecretStr("k"), expires_at=expires_at
    )

    normalize_user_supplied_credentials(creds)

    assert creds.expires_at == expires_at
