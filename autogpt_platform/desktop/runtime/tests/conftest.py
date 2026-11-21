"""What every test starts from."""

import pytest

from autogpt_desktop import install, ports


@pytest.fixture(autouse=True)
def not_started_by_a_shell(monkeypatch):
    """The shell tells the runtime which install it belongs to through the
    environment (src/identity.js runtimeEnvironment). A test that is about
    that sets it; no test may depend on what the developer's shell has."""
    monkeypatch.delenv(ports.PUBLIC_PORT_VARIABLE, raising=False)
    monkeypatch.delenv(install.VARIABLE, raising=False)
