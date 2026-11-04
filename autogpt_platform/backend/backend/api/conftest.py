"""Common test fixtures for server tests.

Database-backed setup fixtures are defined in the parent conftest.py.
"""

from types import SimpleNamespace

import pytest
from prisma.enums import SubscriptionTier
from pytest_snapshot.plugin import Snapshot

from backend.copilot import rate_limit
from backend.data import entitlement
from backend.data.org_plan import OrgPlanSnapshot
from backend.data.pro_activation import UsageActivationState
from backend.data.user import is_user_banned as _REAL_IS_USER_BANNED
from backend.util.feature_flag import Flag

# The one check every paywall asks (``backend.data.entitlement``), and the
# modules that call it by name.
_PAYWALL_CHECK_SITES = (
    "backend.data.entitlement.paywall_reason",
    "backend.api.external.v2.mcp_calls.paywall_reason",
)
_REAL_PAYWALL_REASON = entitlement.paywall_reason
_BAN_CHECK_SITES = (
    "backend.data.user.is_user_banned",
    "backend.api.external.middleware.is_user_banned",
    "backend.api.features.integrations.router.is_user_banned",
    "backend.api.ws_api.is_user_banned",
    "backend.data.auth.oauth.is_user_banned",
    "backend.executor.utils.is_user_banned",
)


@pytest.fixture
def test_user_id() -> str:
    return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


@pytest.fixture
def admin_user_id() -> str:
    return "4e53486c-cf57-477e-ba2a-cb02dc828e1b"


@pytest.fixture
def target_user_id() -> str:
    return "5e53486c-cf57-477e-ba2a-cb02dc828e1c"


@pytest.fixture
def configured_snapshot(snapshot: Snapshot) -> Snapshot:
    """Pre-configured snapshot fixture with standard settings."""
    snapshot.snapshot_dir = "snapshots"
    return snapshot


@pytest.fixture
def mock_jwt_user(test_user_id):
    """Provide mock JWT payload for regular user testing."""
    import fastapi
    from autogpt_libs.auth.models import RequestContext

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {"sub": test_user_id, "role": "user", "email": "test@example.com"}

    def override_get_request_context() -> RequestContext:
        return RequestContext(
            user_id=test_user_id,
            org_id="test-org",
            team_id="test-team",
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    return {
        "get_jwt_payload": override_get_jwt_payload,
        "get_request_context": override_get_request_context,
        "user_id": test_user_id,
    }


@pytest.fixture
def mock_jwt_admin(admin_user_id):
    """Provide mock JWT payload for admin user testing."""
    import fastapi
    from autogpt_libs.auth.models import RequestContext

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {
            "sub": admin_user_id,
            "role": "admin",
            "email": "test-admin@example.com",
        }

    def override_get_request_context() -> RequestContext:
        return RequestContext(
            user_id=admin_user_id,
            org_id="test-org",
            team_id="test-team",
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=True,
            is_team_admin=True,
            is_team_billing_manager=True,
            seat_status="ACTIVE",
        )

    return {
        "get_jwt_payload": override_get_jwt_payload,
        "get_request_context": override_get_request_context,
        "user_id": admin_user_id,
    }


@pytest.fixture(autouse=True)
def _bypass_paywall(mocker):
    """Make every API test treat the user as entitled by default.

    Tests have no real rows backing the JWT or its organization, so without
    this bypass every paywalled route would 503 on the lookup failure inside
    ``enforce_payment_paywall``. Tests of the check itself live beside
    ``backend/data/entitlement.py``; ``paywalled_user`` runs it for real.
    """
    paywall_off = mocker.AsyncMock(return_value=None)
    for site in _PAYWALL_CHECK_SITES:
        mocker.patch(site, new=paywall_off)


@pytest.fixture
def real_entitlement(mocker):
    """Undoes ``_bypass_paywall``: the entitlement check reads the database."""
    for site in _PAYWALL_CHECK_SITES:
        mocker.patch(site, new=_REAL_PAYWALL_REASON)


@pytest.fixture
def paywalled_user(mocker, test_user_id):
    """Make the signed-in test user one without a plan, judged by the real check.

    Undoes ``_bypass_paywall`` and stubs only the stored tier, the payments
    flag and the organization (the user's personal one), so a 402 comes from
    the same entitlement check production runs.
    """

    async def no_plan(user_id: str) -> UsageActivationState:
        return UsageActivationState(
            user_id=user_id, tier=SubscriptionTier.NO_TIER, ready=True
        )

    async def payments_on(flag: Flag, *args, **kwargs) -> bool:
        return flag == Flag.ENABLE_PLATFORM_PAYMENT

    async def personal_org(org_id: str) -> OrgPlanSnapshot:
        return OrgPlanSnapshot(org_id=org_id, exists=True, is_personal=True)

    for site in _PAYWALL_CHECK_SITES:
        mocker.patch(site, new=_REAL_PAYWALL_REASON)
    mocker.patch.object(entitlement, "get_org_plan_snapshot", new=personal_org)
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db",
        return_value=SimpleNamespace(get_usage_activation_state=no_plan),
    )
    mocker.patch.object(rate_limit, "is_feature_enabled", new=payments_on)
    return test_user_id


@pytest.fixture(autouse=True)
def _no_bans(mocker):
    """No account is banned unless a test asks for ``real_ban_lookup``. Once a
    session test has connected Prisma, the real lookup would query it from
    TestClient's own event loop."""
    ban_off = mocker.AsyncMock(return_value=False)
    for site in _BAN_CHECK_SITES:
        mocker.patch(site, new=ban_off)


@pytest.fixture
def real_ban_lookup(mocker):
    """Undoes ``_no_bans``."""
    for site in _BAN_CHECK_SITES:
        mocker.patch(site, new=_REAL_IS_USER_BANNED)


@pytest.fixture
def auth_row(mocker, real_ban_lookup):
    """Give a fresh user id an auth row in one ban state, read by the real
    ``is_user_banned``: ``auth_row("banned" | "until_tomorrow" | "lapsed" |
    "unbanned") -> user_id``."""
    import uuid
    from datetime import datetime, timedelta, timezone

    from prisma.models import AuthUser

    from backend.data import db
    from backend.data import user as user_module

    now = datetime.now(timezone.utc)
    states = {
        "banned": (True, None),
        "until_tomorrow": (True, now + timedelta(days=1)),
        "lapsed": (True, now - timedelta(minutes=1)),
        "unbanned": (None, None),
    }
    rows: dict[str, AuthUser] = {}

    async def find_unique(where):
        return rows.get(where["id"])

    # Connected, so the lookup reads the row here rather than over the RPC.
    mocker.patch.object(db, "is_connected", return_value=True)
    mocker.patch.object(user_module, "AuthUser").prisma.return_value.find_unique = (
        find_unique
    )

    def make(state: str) -> str:
        user_id = str(uuid.uuid4())
        banned, ban_expires = states[state]
        rows[user_id] = AuthUser(
            id=user_id,
            name="Test",
            email=f"{user_id}@example.com",
            emailVerified=True,
            banned=banned,
            banExpires=ban_expires,
            createdAt=now,
            updatedAt=now,
        )
        return user_id

    yield make
    for user_id in rows:
        _REAL_IS_USER_BANNED.cache_delete(user_id)
