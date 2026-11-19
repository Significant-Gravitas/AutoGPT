import pytest

from backend.util.settings import Config


@pytest.mark.parametrize("env, enabled", [(None, False), ("true", True)])
def test_payment_is_off_until_the_deployment_turns_it_on(monkeypatch, env, enabled):
    monkeypatch.delenv("ENABLE_PLATFORM_PAYMENT", raising=False)
    if env is not None:
        monkeypatch.setenv("ENABLE_PLATFORM_PAYMENT", env)
    assert Config(_env_file=None).enable_platform_payment is enabled
