"""Shared fixtures for the onboarding brain dump tests."""

from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import plan


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


@pytest.fixture(autouse=True)
def no_jev_key(mocker: MockerFixture) -> None:
    """No Jev call unless a test asks for one.

    ``plan._api_key`` comes from Settings, which reads ``.env``; a developer
    who set ``TYPESAFE_API_KEY`` for the copilot gate would otherwise send
    every finalize-based test's plan job to the live service, and its order
    and owners would then depend on the model. Tests that want Jev set the
    key themselves.
    """
    mocker.patch.object(plan, "_api_key", "")
