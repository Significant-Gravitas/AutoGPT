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


@pytest.fixture(autouse=True)
def kickoff_tasks(mocker: MockerFixture) -> AsyncMock:
    """Otto starts no tasks unless a test asks.

    A completed take kicks off the user's first suggested tasks, which
    would reach the turn queue, Redis and the database from every test
    that runs the pipeline to the end.
    """
    mock = AsyncMock()
    mocker.patch(
        "backend.api.features.onboarding_dump.service.kickoff_suggested_tasks",
        new=mock,
    )
    return mock
