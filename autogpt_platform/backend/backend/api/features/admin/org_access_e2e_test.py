"""An admin turns organizations on for one account, then off again.

Drives the real app end to end: the account's own flags, the organization API's
gate, and the access its team organization passes to someone it invites all
follow the admin's switch.
"""

import uuid

import httpx
import pytest
from autogpt_libs.auth import get_optional_user_id, get_user_id
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.rest_api import app
from backend.data.org_access import has_org_access
from backend.data.user import get_or_create_user
from backend.util.feature_flag import Flag
from backend.util.test import SpinTestServer

# These read the real grants (see `no_org_access_grants` in the conftest).
pytestmark = pytest.mark.real_org_access

SHOW_ORG_SETTINGS = Flag.SHOW_ORG_SETTINGS.value


class _Account:
    def __init__(self, user_id: str, email: str, role: str = "user") -> None:
        self.user_id = user_id
        self.email = email
        self.role = role

    def payload(self) -> dict[str, str]:
        return {"sub": self.user_id, "role": self.role, "email": self.email}


async def _account(label: str, role: str = "user") -> _Account:
    email = f"org-access-e2e-{label}-{uuid.uuid4().hex[:8]}@example.com"
    user = await get_or_create_user(
        {"sub": str(uuid.uuid4()), "email": email, "name": label}
    )
    return _Account(user.id, email, role)


@pytest.fixture
def no_flag_overrides(monkeypatch):
    for name in (
        "FORCE_FLAG_SHOW_ORG_SETTINGS",
        "NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS",
        "FORCE_ALL_FLAGS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
async def client():
    # Restored, not cleared: the session's SpinTestServer keeps its own
    # get_user_id override on this app.
    previous = dict(app.dependency_overrides)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


async def _as(
    client: httpx.AsyncClient, account: _Account, method: str, url: str, **kwargs
):
    app.dependency_overrides[get_jwt_payload] = account.payload
    # Every way a route reads the caller: the token, the user id, and the
    # optional sign-in the flags route reads straight from the token.
    app.dependency_overrides[get_user_id] = lambda: account.user_id
    app.dependency_overrides[get_optional_user_id] = lambda: account.user_id
    has_org_access.cache_delete(account.user_id)
    return await client.request(method, url, **kwargs)


async def _shows_orgs(client: httpx.AsyncClient, account: _Account) -> bool:
    response = await _as(client, account, "GET", "/api/feature-flags")
    assert response.status_code == 200, response.text
    return response.json()["flags"].get(SHOW_ORG_SETTINGS) is True


@pytest.mark.asyncio(loop_scope="session")
async def test_an_admin_switches_organizations_on_and_off_for_one_account(
    server: SpinTestServer, no_flag_overrides, client: httpx.AsyncClient
):
    owner = await _account("owner")
    invitee = await _account("invitee")
    admin = await _account("admin", role="admin")
    slug = f"acme-{uuid.uuid4().hex[:8]}"

    # Off by default: no organization UI, and the API refuses to make one.
    assert not await _shows_orgs(client, owner)
    refused = await _as(
        client, owner, "POST", "/api/orgs", json={"name": "Acme", "slug": slug}
    )
    assert refused.status_code == 403

    # Only an admin can turn it on.
    forbidden = await _as(
        client, owner, "PUT", f"/api/admin/org-access/users/{owner.user_id}", json={}
    )
    assert forbidden.status_code == 403
    enabled = await _as(
        client,
        admin,
        "PUT",
        f"/api/admin/org-access/users/{owner.user_id}",
        json={"note": "Team plan trial"},
    )
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["has_access"] is True

    # On: the owner sees organizations, creates one and invites a teammate,
    # whose invitation brings organizations to their account too.
    assert await _shows_orgs(client, owner)
    created = await _as(
        client, owner, "POST", "/api/orgs", json={"name": "Acme", "slug": slug}
    )
    assert created.status_code == 200, created.text
    org_id = created.json()["id"]
    invited = await _as(
        client,
        owner,
        "POST",
        f"/api/orgs/{org_id}/invitations",
        json={"email": invitee.email},
        headers={"X-Org-Id": org_id},
    )
    assert invited.status_code == 200, invited.text
    assert await _shows_orgs(client, invitee)

    # Off again: both lose organizations and the API refuses once more;
    # the organization itself is kept.
    disabled = await _as(
        client, admin, "DELETE", f"/api/admin/org-access/users/{owner.user_id}"
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["has_access"] is False
    assert not await _shows_orgs(client, owner)
    assert not await _shows_orgs(client, invitee)
    again = await _as(
        client,
        owner,
        "POST",
        "/api/orgs",
        json={"name": "Acme 2", "slug": f"{slug}-2"},
    )
    assert again.status_code == 403
    listed = await _as(
        client, admin, "GET", f"/api/admin/org-access/users/{owner.user_id}"
    )
    assert listed.json()["grant"] is None
