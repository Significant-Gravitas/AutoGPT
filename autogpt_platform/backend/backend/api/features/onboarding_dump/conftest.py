"""Shared fixtures for the onboarding brain dump tests."""

from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def has_session(mocker: MockerFixture) -> AsyncMock:
    """A brand-new user by default; the greeting is only for those.

    Stubbed for every test in the package, not just the ones that assert
    on it: the intro path asks whether the user has any chat session,
    which is a real query. Left unstubbed, the route tests reach the
    database on the TestClient's own event loop, and the connection they
    leave behind outlives that loop and breaks the next test that
    queries for real.
    """
    mock = AsyncMock(return_value=False)
    mocker.patch(
        "backend.api.features.onboarding_dump.service.user_has_any_session", new=mock
    )
    return mock
