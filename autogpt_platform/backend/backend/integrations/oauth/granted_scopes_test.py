"""Every handler must persist the scopes the provider granted, not the ones we asked for.

A credential that claims a scope its token lacks passes the up-front coverage
check in `copilot/tools/utils.py` and then 403s at run time, and its inflated
scope list can win `_merge_or_create_credential`'s superset check and overwrite
a credential that really does hold the wider grant.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.airtable._api import OAuthTokenResponse as AirtableTokenResponse
from backend.blocks.airtable._oauth import AirtableOAuthHandler
from backend.blocks.mcp.oauth import MCPOAuthHandler
from backend.blocks.wordpress._api import OAuthTokenResponse as WordPressTokenResponse
from backend.blocks.wordpress._oauth import WordPressOAuthHandler
from backend.data.model import OAuth2Credentials
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.integrations.oauth.base import parse_granted_scopes
from backend.integrations.oauth.github import GitHubOAuthHandler
from backend.integrations.oauth.reddit import RedditOAuthHandler
from backend.integrations.oauth.todoist import TodoistOAuthHandler
from backend.integrations.oauth.twitter import TwitterOAuthHandler

CLIENT = ("client-id", "client-secret", "https://localhost/callback")


@pytest.mark.parametrize(
    "scope,fallback,expected",
    [
        ("identity read", ["a"], ["identity", "read"]),  # RFC 6749 space-delimited
        ("repo,gist", ["a"], ["repo", "gist"]),  # GitHub/Todoist comma-delimited
        ("", ["a", "b"], ["a", "b"]),  # provider reports nothing
        (None, ["a"], ["a"]),  # provider omits the field
    ],
)
def test_parse_granted_scopes(scope, fallback, expected):
    assert parse_granted_scopes(scope, fallback=fallback) == expected


@pytest.mark.asyncio
async def test_reddit_stores_the_narrower_granted_set():
    with mock_requests(
        "backend.integrations.oauth.reddit",
        post=[{"access_token": "at", "refresh_token": "rt", "scope": "identity read"}],
        get=[{"name": "alice"}],
    ):
        credentials = await RedditOAuthHandler(*CLIENT).exchange_code_for_tokens(
            "code", ["identity", "read", "submit"], None
        )

    assert credentials.scopes == ["identity", "read"]


@pytest.mark.asyncio
async def test_twitter_stores_the_narrower_granted_set():
    with mock_requests(
        "backend.integrations.oauth.twitter",
        post=[{"access_token": "at", "expires_in": 7200, "scope": "tweet.read"}],
        get=[{"data": {"username": "alice"}}],
    ):
        credentials = await TwitterOAuthHandler(*CLIENT).exchange_code_for_tokens(
            "code", ["tweet.read", "tweet.write"], "verifier"
        )

    assert credentials.scopes == ["tweet.read"]


@pytest.mark.asyncio
async def test_todoist_stores_granted_scopes_and_falls_back_when_absent():
    async def exchange(token_response: dict) -> OAuth2Credentials:
        with mock_requests(
            "backend.integrations.oauth.todoist",
            post=[token_response, {"user": {"email": "alice@example.com"}}],
        ):
            return await TodoistOAuthHandler(*CLIENT).exchange_code_for_tokens(
                "code", ["data:read", "data:delete"], None
            )

    granted = await exchange({"access_token": "at", "scope": "data:read"})
    assert granted.scopes == ["data:read"]

    # Todoist's documented response carries no `scope` at all.
    silent = await exchange({"access_token": "at"})
    assert silent.scopes == ["data:read", "data:delete"]


@pytest.mark.asyncio
async def test_mcp_stores_the_narrower_granted_set():
    with mock_requests(
        "backend.blocks.mcp.oauth",
        post=[{"access_token": "at", "expires_in": 3600, "scope": "mcp:read"}],
    ):
        credentials = await MCPOAuthHandler(
            client_id="client-id",
            client_secret="client-secret",
            redirect_uri="https://localhost/callback",
            authorize_url="https://mcp.example.com/authorize",
            token_url="https://mcp.example.com/token",
        ).exchange_code_for_tokens("code", ["mcp:read", "mcp:write"], "verifier")

    assert credentials.scopes == ["mcp:read"]


@pytest.mark.asyncio
async def test_airtable_stores_the_narrower_granted_set():
    with patch(
        "backend.blocks.airtable._oauth.oauth_exchange_code_for_tokens",
        AsyncMock(return_value=airtable_token_response("data.records:read")),
    ):
        credentials = await AirtableOAuthHandler(*CLIENT).exchange_code_for_tokens(
            "code", ["data.records:read", "data.records:write"], "verifier"
        )

    assert credentials.scopes == ["data.records:read"]


@pytest.mark.asyncio
async def test_airtable_refresh_keeps_the_stored_scopes():
    """`self.scopes` is the full default set — using it re-widens a narrow grant."""
    stored = OAuth2Credentials(
        id="cred-1",
        provider="airtable",
        access_token=SecretStr("at"),
        refresh_token=SecretStr("rt"),
        scopes=["data.records:read"],
    )
    with patch(
        "backend.blocks.airtable._oauth.oauth_refresh_tokens",
        AsyncMock(return_value=airtable_token_response("data.records:read")),
    ):
        refreshed = await AirtableOAuthHandler(*CLIENT)._refresh_tokens(stored)

    assert refreshed.scopes == ["data.records:read"]
    assert set(refreshed.scopes) != set(AirtableOAuthHandler.DEFAULT_SCOPES)


@pytest.mark.asyncio
async def test_wordpress_stores_granted_scopes_and_falls_back_when_absent():
    async def exchange(scope: str | None) -> OAuth2Credentials:
        response = WordPressTokenResponse(access_token="at", scope=scope)
        with patch(
            "backend.blocks.wordpress._oauth.oauth_exchange_code_for_tokens",
            AsyncMock(return_value=response),
        ):
            return await WordPressOAuthHandler(*CLIENT).exchange_code_for_tokens(
                "code", ["posts", "comments"], None
            )

    granted = await exchange("posts")
    assert granted.scopes == ["posts"]

    # Single-blog tokens report no scope; the request is all we know.
    silent = await exchange(None)
    assert silent.scopes == ["posts", "comments"]


@pytest.mark.asyncio
async def test_github_refresh_keeps_the_existing_scopes_and_id():
    """GitHub App refreshes answer with `"scope": ""` — `"".split(",")` is `[""]`."""
    stored = OAuth2Credentials(
        id="cred-1",
        provider="github",
        access_token=SecretStr("old"),
        refresh_token=SecretStr("rt"),
        scopes=["repo", "gist"],
    )
    with mock_requests(
        "backend.integrations.oauth.github",
        post=[{"access_token": "new", "refresh_token": "rt2", "scope": ""}],
        get=[{"login": "alice"}],
    ):
        refreshed = await GitHubOAuthHandler(*CLIENT)._refresh_tokens(stored)

    assert refreshed.scopes == ["repo", "gist"]
    assert refreshed.id == "cred-1"


@pytest.mark.asyncio
async def test_github_exchange_stores_the_narrower_granted_set():
    with mock_requests(
        "backend.integrations.oauth.github",
        post=[{"access_token": "at", "scope": "repo"}],
        get=[{"login": "alice"}],
    ):
        credentials = await GitHubOAuthHandler(*CLIENT).exchange_code_for_tokens(
            "code", ["repo", "gist"], None
        )

    assert credentials.scopes == ["repo"]


@pytest.mark.asyncio
async def test_first_save_of_a_narrower_grant_is_a_create_not_an_update(mocker):
    """First save never reaches `update_creds`, so its narrowing refusal can't fire."""
    store = IntegrationCredentialsStore()
    persist = patch_store(mocker, store, persisted=[])

    await store.add_creds("user-a", oauth_credentials(["identity"]))

    persist.assert_awaited_once()
    assert persist.await_args.args[1][0].scopes == ["identity"]


@pytest.mark.asyncio
async def test_refresh_that_narrows_the_scopes_is_refused_by_the_store(mocker):
    """Why refresh carries the stored set forward instead of re-reading the grant."""
    store = IntegrationCredentialsStore()
    existing = oauth_credentials(["identity", "read"])
    patch_store(mocker, store, persisted=[existing])

    with pytest.raises(ValueError, match="more restrictive set of scopes"):
        await store.update_creds("user-a", oauth_credentials(["identity"]))

    await store.update_creds("user-a", oauth_credentials(["identity", "read"]))


def mock_requests(
    module: str, *, post: list[dict] | None = None, get: list[dict] | None = None
):
    """Patch `Requests` in *module* to answer each call with the next payload."""

    def response(payload: dict) -> MagicMock:
        return MagicMock(
            ok=True, status=200, json=MagicMock(return_value=payload), text=""
        )

    client = MagicMock()
    client.post = AsyncMock(side_effect=[response(p) for p in post or []])
    client.get = AsyncMock(side_effect=[response(p) for p in get or []])
    return patch(f"{module}.Requests", return_value=client)


def airtable_token_response(scope: str) -> AirtableTokenResponse:
    return AirtableTokenResponse(
        access_token="at",
        refresh_token="rt",
        token_type="Bearer",
        scope=scope,
        expires_in=3600,
        refresh_expires_in=86400,
    )


def oauth_credentials(scopes: list[str]) -> OAuth2Credentials:
    return OAuth2Credentials(
        id="cred-1",
        provider="reddit",
        access_token=SecretStr("at"),
        refresh_token=SecretStr("rt"),
        scopes=scopes,
        username="alice",
    )


def patch_store(mocker, store: IntegrationCredentialsStore, persisted: list):
    mocker.patch.object(
        store, "_get_persisted_user_creds_unlocked", AsyncMock(return_value=persisted)
    )
    mocker.patch.object(store, "locked_user_integrations", AsyncMock())
    return mocker.patch.object(
        store, "_set_user_integration_creds", new_callable=AsyncMock
    )
