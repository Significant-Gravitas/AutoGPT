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

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {"sub": test_user_id, "role": "user", "email": "test@example.com"}

    return {"get_jwt_payload": override_get_jwt_payload, "user_id": test_user_id}


@pytest.fixture
def mock_jwt_admin(admin_user_id):
    """Provide mock JWT payload for admin user testing."""
    import fastapi

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {
            "sub": admin_user_id,
            "role": "admin",
            "email": "test-admin@example.com",
        }

    return {"get_jwt_payload": override_get_jwt_payload, "user_id": admin_user_id}


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
    # Patch BOTH bindings: ``rate_limit.is_user_paywalled`` covers the
    # call inside ``enforce_payment_paywall`` (HTTP route dep); the
    # bound name in ``executor.utils.is_user_paywalled`` covers the
    # deep gate inside ``add_graph_execution`` for routes that go all
    # the way through (e.g. graph-execute). Patching only the source
    # module would miss the deep gate because ``utils.py`` does
    # ``from backend.copilot.rate_limit import is_user_paywalled`` at
    # import time.
    paywall_off = mocker.AsyncMock(return_value=False)
    mocker.patch("backend.copilot.rate_limit.is_user_paywalled", new=paywall_off)
    mocker.patch("backend.executor.utils.is_user_paywalled", new=paywall_off)
