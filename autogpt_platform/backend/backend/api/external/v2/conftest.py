"""Fixtures shared by the v2 tests."""

from unittest.mock import AsyncMock

import pytest
import pytest_mock


@pytest.fixture(autouse=True)
def roles_cover_resources(mocker: pytest_mock.MockFixture) -> AsyncMock:
    """The caller's role covers the organization's workflows, runs, library and
    schedules unless a test says otherwise, as the paywall is off unless one
    does (`backend/api/conftest.py`): a mocked database holds no membership
    rows. `resource_roles_test.py` tests the check itself, and
    `../resource_roles_db_test.py` all of it against Postgres."""
    return mocker.patch(
        "backend.api.external.v2.tenancy.resource_role_refusal",
        new_callable=AsyncMock,
        return_value=None,
    )
