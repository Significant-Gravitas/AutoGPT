"""Stripe SDK global reliability configuration.

Single transient blips (429s / connection resets) currently fail Stripe
requests outright, and slow calls hang for the SDK default (~80s). These
tests lock in the retry + client-timeout configuration that
``backend.data.credit`` applies to the global ``stripe`` module at import
time, and the settings that drive it.

They are pure unit tests: they only inspect module-level state, so the
DB-backed test server fixtures are overridden to no-ops below.
"""

import pytest
import stripe

from backend.util.settings import Settings


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    # Override the autouse SpinTestServer-backed cleanup fixture from
    # conftest.py so these tests do not require a test database.
    yield


def test_stripe_reliability_settings_have_defaults():
    config = Settings().config
    assert config.stripe_max_network_retries == 2
    assert config.stripe_client_timeout_seconds == 20


def test_credit_module_applies_stripe_sdk_config():
    # Importing the credit module applies the SDK config as an import-time
    # side effect; assert the global stripe client picked it up.
    import backend.data.credit  # noqa: F401

    config = Settings().config

    assert stripe.max_network_retries == config.stripe_max_network_retries

    client = stripe.default_http_client
    assert isinstance(client, stripe.RequestsClient)
    assert client._timeout == config.stripe_client_timeout_seconds
