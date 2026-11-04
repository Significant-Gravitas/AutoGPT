"""
Comprehensive tests for auth configuration to ensure 100% line and branch coverage.
These tests verify critical security checks preventing JWT token forgery.

Verification is JWKS-only: JWT_JWKS_URL is the single mandatory setting and
symmetric shared-secret verification has been removed entirely.
"""

import logging
import os

import pytest
from pytest_mock import MockerFixture

from autogpt_libs.auth.config import AuthConfigError, Settings

VALID_JWKS_URL = "https://app.example/api/auth/jwks"


def test_auth_config_error_inheritance():
    """Test that AuthConfigError is properly defined as an Exception."""
    assert issubclass(AuthConfigError, Exception)
    error = AuthConfigError("test message")
    assert str(error) == "test message"


def test_settings_static_after_creation(mocker: MockerFixture):
    """Test that settings maintain their values after creation."""
    mocker.patch.dict(
        os.environ,
        {"JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    original_url = settings.JWT_JWKS_URL

    # Changing environment after creation shouldn't affect settings
    os.environ["JWT_JWKS_URL"] = "https://other.example/api/auth/jwks"

    assert settings.JWT_JWKS_URL == original_url


def test_missing_jwks_url_raises(mocker: MockerFixture):
    """JWT_JWKS_URL is the single mandatory setting: without it Settings()
    must raise about JWT_JWKS_URL."""
    mocker.patch.dict(os.environ, {}, clear=True)

    with pytest.raises(AuthConfigError, match="JWT_JWKS_URL must be set"):
        Settings()


def test_no_crypto_raises_error(mocker: MockerFixture):
    """A missing crypto package raises a clear error, since asymmetric
    verification cannot proceed without it."""
    mocker.patch.dict(
        os.environ,
        {"JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    # Mock has_crypto to return False
    mocker.patch("autogpt_libs.auth.config.has_crypto", False)

    with pytest.raises(AuthConfigError) as exc_info:
        Settings()
    assert "cryptography" in str(exc_info.value)


def test_jwks_url_alone_is_sufficient(mocker: MockerFixture):
    """Test that JWT_JWKS_URL passes validation on its own."""
    mocker.patch.dict(
        os.environ,
        {"JWT_JWKS_URL": "http://localhost:3000/api/auth/jwks"},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_JWKS_URL == "http://localhost:3000/api/auth/jwks"


def test_jwks_algorithms_default(mocker: MockerFixture):
    """Test the default JWKS algorithm allow-list."""
    mocker.patch.dict(
        os.environ,
        {"JWT_JWKS_URL": "http://localhost:3000/api/auth/jwks"},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_JWKS_ALGORITHMS == ["ES256", "RS256", "EdDSA"]


def test_jwks_algorithms_custom(mocker: MockerFixture):
    """Test overriding the JWKS algorithm allow-list."""
    mocker.patch.dict(
        os.environ,
        {
            "JWT_JWKS_URL": "http://localhost:3000/api/auth/jwks",
            "JWT_JWKS_ALGORITHMS": "ES256",
        },
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_JWKS_ALGORITHMS == ["ES256"]


@pytest.mark.parametrize("algorithm", ["HS256", "none", "INVALID"])
def test_jwks_algorithms_rejects_unsafe_entries(mocker: MockerFixture, algorithm: str):
    """Test that symmetric/invalid algorithms are rejected for JWKS use."""
    mocker.patch.dict(
        os.environ,
        {
            "JWT_JWKS_URL": "http://localhost:3000/api/auth/jwks",
            "JWT_JWKS_ALGORITHMS": algorithm,
        },
        clear=True,
    )

    with pytest.raises(AuthConfigError) as exc_info:
        Settings()
    assert "JWT_JWKS_ALGORITHMS" in str(exc_info.value)


@pytest.mark.parametrize(
    "bad_url",
    ["localhost:3000/jwks", "ftp://host/jwks", "/api/auth/jwks", "not a url"],
)
def test_jwks_url_must_be_http(mocker: MockerFixture, bad_url: str):
    """A non-http(s) JWT_JWKS_URL is rejected at config time, not as a
    cryptic PyJWKClientError on the first token."""
    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": bad_url}, clear=True)

    with pytest.raises(AuthConfigError) as exc_info:
        Settings()
    assert "JWT_JWKS_URL" in str(exc_info.value)


@pytest.mark.parametrize(
    "good_url",
    ["http://localhost:3000/api/auth/jwks", "https://app.example/api/auth/jwks"],
)
def test_jwks_url_accepts_http_and_https(mocker: MockerFixture, good_url: str):
    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": good_url}, clear=True)

    settings = Settings()
    assert settings.JWT_JWKS_URL == good_url


def test_jwks_url_cleartext_remote_host_is_rejected(mocker: MockerFixture):
    """A cleartext JWKS URL pointing at a routable host must not boot: an
    attacker in the network path could substitute the keys and forge tokens."""
    insecure_url = "http://auth.example.com/api/auth/jwks"
    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": insecure_url}, clear=True)

    with pytest.raises(AuthConfigError, match="JWKS_ALLOW_INSECURE_TRANSPORT"):
        Settings()


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:3000/api/auth/jwks",
        "http://127.0.0.1:3000/api/auth/jwks",
        "http://[::1]:3000/api/auth/jwks",
        # Docker service name: single-label host on a private network.
        "http://frontend:3000/api/auth/jwks",
        "https://auth.example.com/api/auth/jwks",
    ],
)
def test_jwks_url_trusted_transport_does_not_warn(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture, url: str
):
    """Loopback, container-internal and https URLs stay quiet."""
    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": url}, clear=True)

    with caplog.at_level(logging.WARNING):
        Settings()
        assert "cleartext" not in caplog.text


@pytest.mark.parametrize(
    "malformed_url",
    ["http://[::1/api/auth/jwks", "http://]::1[/api/auth/jwks"],
)
def test_jwks_url_malformed_host_fails_at_boot(
    mocker: MockerFixture, malformed_url: str
):
    """A URL urlparse() can't parse (e.g. unbalanced IPv6 bracket) is unviable
    config and must fail at boot with a clear error, not at first fetch."""
    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": malformed_url}, clear=True)

    with pytest.raises(AuthConfigError, match="Invalid JWT_JWKS_URL"):
        Settings()


@pytest.mark.parametrize("override", ["1", "true", "TRUE", "yes"])
def test_jwks_url_cleartext_allowed_with_override_but_warns(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture, override: str
):
    """JWKS_ALLOW_INSECURE_TRANSPORT lets a trusted-path deployment boot,
    with a warning on record that the transport is cleartext."""
    mocker.patch.dict(
        os.environ,
        {
            "JWT_JWKS_URL": "http://auth.example.com/api/auth/jwks",
            "JWKS_ALLOW_INSECURE_TRANSPORT": override,
        },
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        Settings()
        assert "cleartext" in caplog.text


def test_warns_when_es256_missing_from_jwks_algorithms(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    """The platform frontend signs ES256; excluding it rejects every token."""
    mocker.patch.dict(
        os.environ,
        {
            "JWT_JWKS_URL": "https://app.example/api/auth/jwks",
            "JWT_JWKS_ALGORITHMS": "ES384,ES512",
        },
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        Settings()
        assert "does not include ES256" in caplog.text


def test_no_warning_when_es256_present(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    mocker.patch.dict(
        os.environ,
        {"JWT_JWKS_URL": "https://app.example/api/auth/jwks"},
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        Settings()
        assert "does not include ES256" not in caplog.text
