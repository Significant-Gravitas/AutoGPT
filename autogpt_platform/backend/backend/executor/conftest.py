"""Common test fixtures for executor tests.

Note: Common fixtures like ``test_user_id`` are defined in the parent
conftest.py (``backend/conftest.py``) and are available here automatically.
"""

import pytest


@pytest.fixture(autouse=True)
def _bypass_paywall(mocker):
    """Make every executor test treat the user as paid by default.

    ``add_graph_execution`` now raises ``UserPaywalledError`` for NO_TIER
    users when ``ENABLE_PLATFORM_PAYMENT`` is on. Tests in this module
    pass synthetic user_ids that don't have a Supabase row, so the tier
    lookup either fails or returns NO_TIER and trips the gate. Bypass it
    here so the executor tests can exercise their own logic without
    concerning themselves with paywall state. Tests that specifically
    exercise the paywall live in ``backend/copilot/rate_limit_test.py``.
    """
    mocker.patch(
        "backend.copilot.rate_limit.is_user_paywalled",
        new=mocker.AsyncMock(return_value=False),
    )
