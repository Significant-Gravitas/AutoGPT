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
