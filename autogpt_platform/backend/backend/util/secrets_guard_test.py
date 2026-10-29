import base64
import hashlib
import os

import pytest

from backend.util import secrets_guard
from backend.util.secrets_guard import check_secrets
from backend.util.settings import Settings

# The values this guard rejects are deliberately NOT reproduced here: putting
# them back in a tracked file is the exact problem SECRT-2611 fixes. To check a
# registered digest against what was actually published, re-derive it from git
# history instead:
#   git show 79168d31f1:autogpt_platform/backend/.env.default \
#     | grep -E '^(ENCRYPTION_KEY|UNSUBSCRIBE_SECRET_KEY)=' \
#     | cut -d= -f2- | tr -d '\n' | shasum -a 256


def _settings(encryption_key: str, unsubscribe_key: str = "fresh-unsubscribe-key"):
    settings = Settings()
    settings.secrets.encryption_key = encryption_key
    settings.secrets.unsubscribe_secret_key = unsubscribe_key
    return settings


def _fresh_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


def test_generated_secrets_pass():
    check_secrets(_settings(_fresh_key()))


def test_missing_encryption_key_is_refused_with_bootstrap_hint():
    with pytest.raises(ValueError, match="make init-env"):
        check_secrets(_settings(""))


def test_blank_unsubscribe_key_only_warns(caplog):
    check_secrets(_settings(_fresh_key(), ""))
    assert "UNSUBSCRIBE_SECRET_KEY" in caplog.text


@pytest.mark.parametrize(
    "name, build_settings",
    [
        ("ENCRYPTION_KEY", lambda value: _settings(value)),
        ("UNSUBSCRIBE_SECRET_KEY", lambda value: _settings(_fresh_key(), value)),
    ],
)
def test_registered_published_value_is_refused(monkeypatch, name, build_settings):
    """A configured value whose digest is registered as published is refused."""
    published = f"pretend-this-was-shipped-in-env-default-{name}"
    monkeypatch.setitem(
        secrets_guard._RETIRED_DIGESTS,
        name,
        {hashlib.sha256(published.encode()).hexdigest()},
    )

    with pytest.raises(ValueError, match=f"{name}.*compromised"):
        check_secrets(build_settings(published))


def test_guard_covers_every_secret_that_env_default_used_to_ship():
    assert set(secrets_guard._RETIRED_DIGESTS) == {
        "ENCRYPTION_KEY",
        "UNSUBSCRIBE_SECRET_KEY",
    }
    for digests in secrets_guard._RETIRED_DIGESTS.values():
        assert digests, "a name with no digests silently disables the guard"
        assert all(len(digest) == 64 for digest in digests)


def _database_url(monkeypatch, settings, password):
    monkeypatch.setenv(
        "DATABASE_URL", f"postgresql://postgres:{password}@db:5432/postgres?schema=x"
    )


def _direct_url(monkeypatch, settings, password):
    monkeypatch.setenv(
        "DIRECT_URL", f"postgresql://postgres:{password}@db:5432/postgres?schema=x"
    )


def _rabbitmq(monkeypatch, settings, password):
    settings.secrets.rabbitmq_default_pass = password


def _redis(monkeypatch, settings, password):
    settings.config.redis_password = password


def _falkordb(monkeypatch, settings, password):
    monkeypatch.setenv("GRAPHITI_FALKORDB_PASSWORD", password)


@pytest.mark.parametrize(
    "name, configure",
    [
        ("DATABASE_URL", _database_url),
        ("DIRECT_URL", _direct_url),
        ("RABBITMQ_DEFAULT_PASS", _rabbitmq),
        ("REDIS_PASSWORD", _redis),
        ("GRAPHITI_FALKORDB_PASSWORD", _falkordb),
    ],
)
def test_published_service_password_warns_and_still_starts(
    monkeypatch, caplog, name, configure
):
    """The service keeps working on it, so the backend starts, but says how to
    replace it."""
    published = f"pretend-this-was-shipped-for-{name}"
    monkeypatch.setattr(
        secrets_guard,
        "_PUBLISHED_CREDENTIAL_DIGESTS",
        {hashlib.sha256(published.encode()).hexdigest()},
    )
    settings = _settings(_fresh_key())
    configure(monkeypatch, settings, published)

    check_secrets(settings)

    warnings = [r.getMessage() for r in caplog.records if "published" in r.getMessage()]
    assert len(warnings) == 1
    assert name in warnings[0] and "make init-env" in warnings[0]


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://postgres:pa[ss@db:5432/postgres",
        "postgresql://postgres:[pass]@db:5432/postgres",
    ],
)
def test_a_connection_string_urllib_cannot_split_does_not_stop_startup(
    monkeypatch, caplog, url
):
    monkeypatch.setenv("DATABASE_URL", url)

    check_secrets(_settings(_fresh_key()))

    assert "database password" not in caplog.text


def test_generated_service_passwords_do_not_warn(monkeypatch, caplog):
    settings = _settings(_fresh_key())
    settings.config.redis_host = "redis-0"
    settings.config.rabbitmq_host = "rabbitmq"
    monkeypatch.setenv("GRAPHITI_FALKORDB_HOST", "falkordb")
    for configure in (_database_url, _direct_url, _rabbitmq, _redis, _falkordb):
        configure(monkeypatch, settings, _fresh_key().rstrip("="))

    check_secrets(settings)

    assert "[SECURITY]" not in caplog.text
    assert "is not set" not in caplog.text


def _blank_falkordb(monkeypatch, settings, host):
    monkeypatch.setenv("GRAPHITI_FALKORDB_HOST", host)
    monkeypatch.setenv("GRAPHITI_FALKORDB_PASSWORD", "")


def _blank_rabbitmq(monkeypatch, settings, host):
    settings.config.rabbitmq_host = host
    settings.secrets.rabbitmq_default_pass = ""


@pytest.mark.parametrize(
    "name, configure, compose_host",
    [
        ("GRAPHITI_FALKORDB_PASSWORD", _blank_falkordb, "falkordb"),
        ("RABBITMQ_DEFAULT_PASS", _blank_rabbitmq, "rabbitmq"),
    ],
)
@pytest.mark.parametrize(
    "host, warns",
    [
        (None, True),  # the Compose service, named as Compose names it
        ("localhost", False),  # CI, or one of one's own
        ("broker.internal", False),
    ],
)
def test_a_compose_service_left_without_a_password_is_pointed_out(
    monkeypatch, caplog, name, configure, compose_host, host, warns
):
    """A backend/.env from before it had a line for the password took it from
    .env.default, which no longer ships one."""
    settings = _settings(_fresh_key())
    configure(monkeypatch, settings, host or compose_host)

    check_secrets(settings)

    warnings = [
        r.getMessage() for r in caplog.records if "is not set" in r.getMessage()
    ]
    assert len(warnings) == int(warns)
    if warns:
        assert name in warnings[0] and "make init-env" in warnings[0]


@pytest.mark.parametrize(
    "host, warns",
    [
        ("redis-0", True),  # the Compose cluster
        ("localhost", False),  # CI, or a cluster of one's own
        ("redis.internal", False),  # a hosted cluster, which may need none
    ],
)
def test_compose_redis_without_a_password_warns(caplog, host, warns):
    settings = _settings(_fresh_key())
    settings.config.redis_host = host
    settings.config.redis_password = ""

    check_secrets(settings)

    assert ("Redis cluster has no password" in caplog.text) is warns


def test_encryption_key_errors_point_at_the_upgrade_path(monkeypatch):
    published = "pretend-this-was-shipped-in-env-default"
    monkeypatch.setitem(
        secrets_guard._RETIRED_DIGESTS,
        "ENCRYPTION_KEY",
        {hashlib.sha256(published.encode()).hexdigest()},
    )

    for encryption_key in ("", published):
        with pytest.raises(ValueError, match="Upgrading: secrets are generated"):
            check_secrets(_settings(encryption_key))
