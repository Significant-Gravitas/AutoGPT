"""Common test fixtures for server tests.

Note: Common fixtures like test_user_id, admin_user_id, target_user_id,
setup_test_user, and setup_admin_user are defined in the parent conftest.py
(backend/conftest.py) and are available here automatically.
"""

import pytest
from pytest_snapshot.plugin import Snapshot


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
    """Make every API test treat the user as paid by default.

    Tests have no real Supabase row backing the JWT, so without this
    bypass every paywalled route would 503 on the tier-lookup failure
    inside ``enforce_payment_paywall`` and every graph execution would
    raise ``UserPaywalledError`` from ``add_graph_execution`` once the
    paywall is wired up. Tests that specifically exercise the paywall
    (e.g. ``TestEnforcePaymentPaywall``, ``TestIsUserPaywalled``) live
    in ``rate_limit_test.py`` and patch ``_fetch_user_tier`` directly,
    bypassing this helper.
    """
    paywall_off = mocker.AsyncMock(return_value=False)
    mocker.patch("backend.copilot.rate_limit.is_user_paywalled", new=paywall_off)
    mocker.patch("backend.executor.utils.is_user_paywalled", new=paywall_off)
