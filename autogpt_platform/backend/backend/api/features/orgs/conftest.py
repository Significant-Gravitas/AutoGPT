"""Override session-scoped fixtures for org tests.

Org tests mock at the Prisma boundary and don't need the full test server
or its graph cleanup hook.
"""

import pytest


@pytest.fixture(scope="session")
def server():
    """No-op — org tests don't need the full backend server."""
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    """No-op — org tests don't create real graphs."""
    yield
