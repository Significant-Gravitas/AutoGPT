"""Override session-scoped fixtures so unit tests run without the server."""

import pytest


@pytest.fixture(scope="session")
def server():
    yield None
