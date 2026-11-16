"""A scope reconnect requests the whole set, and a narrow re-grant is not success.

2026-09-16: the GitHub credential lacked read:org, the bot prompted a
reconnect, the new grant still lacked it, and the loop repeated. These tests
walk that loop through the real login route, code exchange and credential
merge, with only the provider's side mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import fastapi
import fastapi.testclient
import pytest
from pydantic import SecretStr

from backend.api.features.integrations import router as router_module
from backend.api.features.integrations.router import (
    _exchange_code_for_credentials,
    _merge_or_create_credential,
    router,
)
from backend.blocks.linear._oauth import LinearOAuthHandler
from backend.copilot.tools.credential_gaps import (
    annotate_credential_gaps,
    find_credential_gap,
)
from backend.data.model import CredentialsFieldInfo, CredentialsType, OAuth2Credentials
from backend.integrations.oauth.github import GitHubOAuthHandler
from backend.integrations.oauth.refresh_failure import (
    RECONNECT_REQUIRED_KEY,
    reconnect_required,
)
from backend.integrations.providers import ProviderName

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

CLIENT = ("client-id", "client-secret", "https://localhost/callback")


@pytest.fixture(autouse=True)
def setup_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


class FakeCredsManager:
    """The router's credentials manager over an in-memory list."""

    def __init__(self, creds: list[OAuth2Credentials]):
        self.creds = {c.id: c for c in creds}
        self.store = MagicMock()
        self.store.get_creds_by_id = AsyncMock(
            side_effect=lambda _user, cred_id: self.creds.get(cred_id)
        )
        self.store.get_creds_by_provider = AsyncMock(
            side_effect=lambda _user, provider: [
                c for c in self.creds.values() if c.provider == provider
            ]
        )
        self.store.store_state_token = AsyncMock(return_value=("state-1", None))

    async def update(self, _user: str, updated: OAuth2Credentials) -> None:
        self.creds[updated.id] = updated

    async def create(self, _user: str, created: OAuth2Credentials) -> None:
        self.creds[created.id] = created


def github(
    scopes: list[str], *, cred_id: str = "gh-1", metadata: dict | None = None
) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=cred_id,
        provider="github",
        title="alice's github",
        username="alice",
        access_token=SecretStr("at"),
        scopes=scopes,
        metadata=metadata or {},
    )


def requirement(provider: str, scopes: set[str]) -> CredentialsFieldInfo:
    return CredentialsFieldInfo[ProviderName, CredentialsType](
        credentials_provider=frozenset([ProviderName(provider)]),
        credentials_types=frozenset(["oauth2"]),
        credentials_scopes=frozenset(scopes),
    )


def card_scopes(existing: OAuth2Credentials, need: CredentialsFieldInfo) -> list[str]:
    """The scopes a setup card built for *need* asks the login to request."""
    annotated, _ = annotate_credential_gaps(
        [existing],
        {"credentials": need},
        {"credentials": {"scopes": sorted(need.required_scopes or [])}},
    )
    return annotated["credentials"]["scopes"]


def login_scopes(provider: str, handler, scopes: list[str]) -> set[str]:
    """Scopes on the provider login URL the real login route builds."""
    with patch.object(
        router_module, "_get_provider_oauth_handler", return_value=handler
    ), patch.object(router_module, "creds_manager", FakeCredsManager([])), patch(
        "backend.api.features.integrations.router.product_analytics"
    ):
        response = client.get(f"/{provider}/login", params={"scopes": ",".join(scopes)})
    assert response.status_code == 200, response.text
    scope = parse_qs(urlparse(response.json()["login_url"]).query)["scope"][0]
    return {s for s in scope.replace(",", " ").split(" ") if s}


def test_github_reconnect_login_requests_read_org_and_keeps_repo():
    scopes = card_scopes(github(["repo"]), requirement("github", {"read:org"}))

    requested = login_scopes("github", GitHubOAuthHandler(*CLIENT), scopes)

    assert requested == {"repo", "read:org"}


def test_linear_reconnect_login_requests_comments_create_and_keeps_the_grant():
    linear = github(["read", "write", "issues:create"]).model_copy(
        update={"provider": "linear", "id": "lin-1"}
    )
    scopes = card_scopes(linear, requirement("linear", {"comments:create"}))

    requested = login_scopes("linear", LinearOAuthHandler(*CLIENT), scopes)

    assert requested == {"read", "write", "issues:create", "comments:create"}


@pytest.mark.asyncio
async def test_a_reconnect_that_returns_the_same_narrow_scopes_is_not_success():
    existing = github(["repo"])
    need = requirement("github", {"repo", "read:org"})
    manager = FakeCredsManager([existing])
    state = MagicMock(scopes=card_scopes(existing, need), code_verifier=None)
    # The user approves, but the provider hands back the same narrow grant.
    handler = MagicMock()
    handler.handle_default_scopes = lambda scopes: scopes
    handler.exchange_code_for_tokens = AsyncMock(
        return_value=github(["repo"], cred_id="gh-new")
    )

    with patch.object(router_module, "creds_manager", manager), patch(
        "backend.api.features.integrations.router.report_credential_failure"
    ) as reported:
        granted = await _exchange_code_for_credentials(
            handler, ProviderName.GITHUB, "code", state
        )
        stored = await _merge_or_create_credential(
            "user-1", ProviderName.GITHUB, granted, None
        )

    reported.assert_called_once()
    assert reported.call_args.args[2] == "granted_scopes_narrower"
    # Nothing downstream may now treat the account as ready: the card built
    # from what is stored still names the missing scope.
    gap = find_credential_gap(list(manager.creds.values()), need)
    assert gap is not None and gap.missing_scopes == ["read:org"]
    assert "read:org" not in stored.scopes


@pytest.mark.asyncio
async def test_a_reconnect_that_grants_the_scope_closes_the_gap_and_clears_the_marker():
    dead = github(
        ["repo"],
        metadata={RECONNECT_REQUIRED_KEY: {"error_code": "invalid_grant"}},
    )
    need = requirement("github", {"repo", "read:org"})
    manager = FakeCredsManager([dead])

    with patch.object(router_module, "creds_manager", manager):
        stored = await _merge_or_create_credential(
            "user-1",
            ProviderName.GITHUB,
            github(["repo", "read:org"], cred_id="gh-new"),
            None,
        )

    # Merged in place into the credential the card pointed at.
    assert stored.id == "gh-1"
    assert reconnect_required(stored) is None
    assert find_credential_gap(list(manager.creds.values()), need) is None
