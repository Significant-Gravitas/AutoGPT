"""
Tests for what a v2 credential is allowed to reach.

Least privilege is the point: a key granted one scope must not be a way to read
data, spend platform money, or destroy platform state that its scopes never
mentioned. Each test here pins one such hole shut.
"""

from datetime import datetime, timezone
from typing import Optional
from unittest import mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission, ContentType
from pydantic import SecretStr

from backend.api.external.v2.errors import add_v2_exception_handlers
from backend.api.external.v2.models import SearchContentType
from backend.api.external.v2.pagination import PageRequest
from backend.api.external.v2.tenancy import TenantContext, require_auth
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.model import APIKeyCredentials, is_sdk_default
from backend.integrations.credentials_store import SYSTEM_CREDENTIAL_IDS

USER_ID = "user-1"
ORG_ID = "org-1"


# ============================================================================
# Search: a private content type costs the scope that guards it elsewhere
# ============================================================================


@pytest.mark.parametrize(
    "content_type,scope",
    [
        (SearchContentType.LIBRARY_AGENT, APIKeyPermission.READ_LIBRARY),
        (SearchContentType.WORKSPACE_FILE, APIKeyPermission.READ_FILES),
    ],
)
async def test_searching_private_content_without_its_scope_is_refused(
    mocker: pytest_mock.MockFixture,
    content_type: SearchContentType,
    scope: APIKeyPermission,
) -> None:
    """A zero-scope key asked for these and got the caller's own rows back."""
    hybrid = _mock_hybrid_search(mocker)

    with pytest.raises(fastapi.HTTPException) as raised:
        await search_with(mocker, [content_type], scopes=[])

    assert raised.value.status_code == 403
    assert scope.value in str(raised.value.detail)
    hybrid.assert_not_awaited()


@pytest.mark.parametrize(
    "content_type,scope",
    [
        (SearchContentType.LIBRARY_AGENT, APIKeyPermission.READ_LIBRARY),
        (SearchContentType.WORKSPACE_FILE, APIKeyPermission.READ_FILES),
    ],
)
async def test_searching_private_content_with_its_scope_is_allowed(
    mocker: pytest_mock.MockFixture,
    content_type: SearchContentType,
    scope: APIKeyPermission,
) -> None:
    hybrid = _mock_hybrid_search(mocker)

    await search_with(mocker, [content_type], scopes=[scope])

    assert hybrid.await_args.kwargs["content_types"] == [
        ContentType(content_type.value)
    ]


async def test_search_without_content_types_asks_only_for_public_ones(
    mocker: pytest_mock.MockFixture,
) -> None:
    """v2 names the default rather than inheriting whatever the engine picks."""
    hybrid = _mock_hybrid_search(mocker)

    await search_with(mocker, None, scopes=[])

    assert hybrid.await_args.kwargs["content_types"] == [
        ContentType.STORE_AGENT,
        ContentType.BLOCK,
        ContentType.INTEGRATION,
        ContentType.DOCUMENTATION,
    ]


def test_chat_sessions_are_not_a_searchable_content_type() -> None:
    """No v2 scope covers chat, so the parameter cannot name it at all."""
    assert "CHAT_SESSION" in {t.value for t in ContentType}
    assert "CHAT_SESSION" not in {t.value for t in SearchContentType}


# ============================================================================
# Sharing: both directions cost the same scopes
# ============================================================================


def test_share_and_unshare_require_the_same_permissions() -> None:
    """Unsharing needed only SHARE_RUN while sharing needed READ_RUN as well."""
    assert _published_scopes("/runs/{run_id}/share", "post") == _published_scopes(
        "/runs/{run_id}/share", "delete"
    )
    assert _published_scopes("/runs/{run_id}/share", "delete") == {
        "READ_RUN",
        "SHARE_RUN",
    }


# ============================================================================
# Credentials: the platform's own credentials are not the caller's to delete
# ============================================================================


async def test_deleting_a_system_credential_is_refused(
    mocker: pytest_mock.MockFixture,
) -> None:
    """`delete_creds_by_id` raises ValueError on these, which v2 reported as a
    400 blaming the caller. The internal route answers 403."""
    from .integrations.credentials import delete_credential

    system_id = sorted(SYSTEM_CREDENTIAL_IDS)[0]
    deleter = _mock_creds_store(mocker, _credential(system_id))

    with pytest.raises(fastapi.HTTPException) as raised:
        await delete_credential(credential_id=system_id, auth=_tenant())

    assert raised.value.status_code == 403
    deleter.assert_not_awaited()


async def test_deleting_a_managed_credential_is_refused(
    mocker: pytest_mock.MockFixture,
) -> None:
    from .integrations.credentials import delete_credential

    deleter = _mock_creds_store(mocker, _credential("managed-1", is_managed=True))

    with pytest.raises(fastapi.HTTPException) as raised:
        await delete_credential(credential_id="managed-1", auth=_tenant())

    assert raised.value.status_code == 403
    deleter.assert_not_awaited()


async def test_deleting_an_sdk_default_credential_is_not_found(
    mocker: pytest_mock.MockFixture,
) -> None:
    from .integrations.credentials import delete_credential

    cred_id = "openai-default"
    assert is_sdk_default(cred_id)
    deleter = _mock_creds_store(mocker, _credential(cred_id))

    with pytest.raises(fastapi.HTTPException) as raised:
        await delete_credential(credential_id=cred_id, auth=_tenant())

    assert raised.value.status_code == 404
    deleter.assert_not_awaited()


async def test_deleting_an_ordinary_credential_still_works(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The guards above must not have closed the endpoint's actual job."""
    from .integrations.credentials import delete_credential

    deleter = _mock_creds_store(mocker, _credential("mine-1"))

    await delete_credential(credential_id="mine-1", auth=_tenant())

    deleter.assert_awaited_once_with(USER_ID, "mine-1")


def test_credential_listing_says_which_credentials_are_the_platform_s() -> None:
    """Without this, managed and user credentials are indistinguishable."""
    from .models import CredentialInfo

    assert CredentialInfo.from_internal(_credential("a", is_managed=True)).is_managed
    assert not CredentialInfo.from_internal(_credential("b")).is_managed


# ============================================================================
# Auth: verified once per request, and the limiter never blocks on its own bugs
# ============================================================================


async def test_the_route_reuses_the_credential_the_middleware_verified() -> None:
    """An API key costs a Scrypt hash to verify; it was paid twice per request."""
    calls = _resolve_calls()

    with mock.patch("backend.api.external.middleware.validate_api_key", new=calls):
        response = _client().get("/probe", headers={"X-API-Key": "k"})

    assert response.status_code == 200, response.text
    assert calls.count == 1


async def test_a_transient_auth_failure_in_the_limiter_does_not_fail_the_request() -> (
    None
):
    """The limiter caught only HTTPException around auth resolution, so any
    other error propagated out of it and the route never ran — a blip in the
    credential store 500'd a request the route itself could have served."""
    calls = _resolve_calls(fail_first=True)

    with mock.patch("backend.api.external.middleware.validate_api_key", new=calls):
        response = _client().get("/probe", headers={"X-API-Key": "k"})

    assert response.status_code == 200, response.text
    assert calls.count == 2


# ---------------------------------------------------------------------------


def _tenant(scopes: Optional[list[APIKeyPermission]] = None) -> TenantContext:
    return TenantContext(
        user_id=USER_ID,
        scopes=list(APIKeyPermission) if scopes is None else scopes,
        type="api_key",
        organization_id=ORG_ID,
    )


def _page() -> PageRequest:
    return PageRequest(limit=20, page=1, cursor=None)


def _mock_hybrid_search(mocker: pytest_mock.MockFixture):
    return mocker.patch(
        "backend.api.external.v2.search.unified_hybrid_search",
        new_callable=mock.AsyncMock,
        return_value=([], 0),
    )


async def search_with(
    mocker: pytest_mock.MockFixture,
    content_types: Optional[list[SearchContentType]],
    scopes: list[APIKeyPermission],
):
    from .search import search

    mocker.patch(
        "backend.api.external.v2.search.search_limiter.check",
        new_callable=mock.AsyncMock,
    )
    return await search(
        query="q",
        content_types=content_types,
        category=None,
        page=_page(),
        auth=_tenant(scopes=scopes),
    )


def _published_scopes(path: str, method: str) -> set[str]:
    """The scopes the OpenAPI document tells a caller the endpoint needs."""
    from .app import v2_app

    operation = v2_app.openapi()["paths"][path][method]
    return {
        scope
        for requirement in operation["security"]
        for scopes in requirement.values()
        for scope in scopes
    }


def _credential(cred_id: str, is_managed: bool = False) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=cred_id,
        provider="openai",
        title="t",
        api_key=SecretStr("secret"),
        is_managed=is_managed,
    )


def _mock_creds_store(mocker: pytest_mock.MockFixture, credential: APIKeyCredentials):
    mocker.patch(
        "backend.api.external.v2.integrations.credentials.creds_manager.store"
        ".get_creds_by_id",
        new_callable=mock.AsyncMock,
        return_value=credential,
    )
    return mocker.patch(
        "backend.api.external.v2.integrations.credentials.creds_manager.delete",
        new_callable=mock.AsyncMock,
    )


def _resolve_calls(fail_first: bool = False):
    """An API-key validator that counts how often the request verified the key."""

    async def validate(key: str) -> APIAuthorizationInfo:
        validate.count += 1
        if fail_first and validate.count == 1:
            raise RuntimeError("credential store is briefly unreachable")
        return APIAuthorizationInfo(
            user_id=USER_ID,
            scopes=list(APIKeyPermission),
            type="api_key",
            created_at=datetime.now(tz=timezone.utc),
            organization_id=ORG_ID,
        )

    validate.count = 0
    return validate


def _client() -> fastapi.testclient.TestClient:
    """A one-route v2-shaped app: the real middleware over the real dependency."""
    from .global_rate_limit import GlobalRateLimitMiddleware

    app = fastapi.FastAPI()

    @app.get("/probe")
    async def probe(auth: TenantContext = fastapi.Security(require_auth)) -> str:
        return auth.user_id

    app.add_middleware(GlobalRateLimitMiddleware)
    add_v2_exception_handlers(app)
    return fastapi.testclient.TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _offline(mocker: pytest_mock.MockFixture) -> None:
    """No test here is about Redis or the org tables."""
    mocker.patch(
        "backend.api.utils.rate_limit.RateLimiter.check",
        new_callable=mock.AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.api.external.v2.tenancy.resolve_credential_tenancy",
        new_callable=mock.AsyncMock,
        return_value=(ORG_ID, None),
    )
