"""
Comprehensive tests for auth configuration to ensure 100% line and branch coverage.
These tests verify critical security checks preventing JWT token forgery.
"""

import logging
import os

import pytest
from pytest_mock import MockerFixture

from autogpt_libs.auth.config import AuthConfigError, Settings

# Better Auth verification requires a JWKS URL, so these tests set one to
# exercise JWT_VERIFY_KEY / algorithm behavior without tripping the guard.
VALID_JWKS_URL = "https://app.example/api/auth/jwks"


def test_environment_variable_precedence(mocker: MockerFixture):
    """Test that environment variables take precedence over defaults."""
    secret = "environment-secret-key-with-proper-length-123456"
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == secret


def test_environment_variable_backwards_compatible(mocker: MockerFixture):
    """Test that SUPABASE_JWT_SECRET is read if JWT_VERIFY_KEY is not set."""
    secret = "environment-secret-key-with-proper-length-123456"
    mocker.patch.dict(
        os.environ,
        {"SUPABASE_JWT_SECRET": secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == secret


def test_auth_config_error_inheritance():
    """Test that AuthConfigError is properly defined as an Exception."""
    assert issubclass(AuthConfigError, Exception)
    error = AuthConfigError("test message")
    assert str(error) == "test message"


def test_settings_static_after_creation(mocker: MockerFixture):
    """Test that settings maintain their values after creation."""
    secret = "immutable-secret-key-with-proper-length-12345"
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    original_secret = settings.JWT_VERIFY_KEY

    # Changing environment after creation shouldn't affect settings
    os.environ["JWT_VERIFY_KEY"] = "different-secret"

    assert settings.JWT_VERIFY_KEY == original_secret


def test_settings_load_with_valid_secret(mocker: MockerFixture):
    """Test auth enabled with a valid JWT secret."""
    valid_secret = "a" * 32  # 32 character secret
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": valid_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == valid_secret


def test_settings_load_with_strong_secret(mocker: MockerFixture):
    """Test auth enabled with a cryptographically strong secret."""
    strong_secret = "super-secret-jwt-token-with-at-least-32-characters-long"
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": strong_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == strong_secret
    assert len(settings.JWT_VERIFY_KEY) >= 32


@pytest.mark.parametrize(
    "verify_key_env",
    [
        pytest.param({}, id="no-verify-key"),
        pytest.param({"JWT_VERIFY_KEY": ""}, id="empty-verify-key"),
        pytest.param({"JWT_VERIFY_KEY": " \t\n "}, id="whitespace-verify-key"),
        pytest.param({"JWT_VERIFY_KEY": "x" * 40}, id="strong-verify-key"),
    ],
)
def test_missing_jwks_url_raises_regardless_of_verify_key(
    mocker: MockerFixture, verify_key_env: dict[str, str]
):
    """JWT_JWKS_URL is the mandatory setting: without it Settings() must raise
    about JWT_JWKS_URL specifically, and no state of the optional
    JWT_VERIFY_KEY (absent, empty, whitespace, strong) substitutes for it."""
    mocker.patch.dict(os.environ, verify_key_env, clear=True)

    with pytest.raises(AuthConfigError, match="JWT_JWKS_URL must be set"):
        Settings()


def test_secret_weak_logs_warning(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    """Test that weak JWT secret triggers warning log."""
    weak_secret = "short"  # Less than 32 characters
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": weak_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        settings = Settings()
        assert settings.JWT_VERIFY_KEY == weak_secret
        assert "key appears weak" in caplog.text.lower()
        assert "less than 32 characters" in caplog.text


def test_secret_31_char_logs_warning(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    """Test that 31-character secret triggers warning (boundary test)."""
    secret_31 = "a" * 31  # Exactly 31 characters
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": secret_31, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        settings = Settings()
        assert len(settings.JWT_VERIFY_KEY) == 31
        assert "key appears weak" in caplog.text.lower()


def test_secret_32_char_no_warning(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    """Test that 32-character secret does not trigger warning (boundary test)."""
    secret_32 = "a" * 32  # Exactly 32 characters
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": secret_32, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        settings = Settings()
        assert len(settings.JWT_VERIFY_KEY) == 32
        assert "JWT secret appears weak" not in caplog.text


def test_secret_whitespace_stripped(mocker: MockerFixture):
    """Test that JWT secret whitespace is stripped."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": f"  {secret}  ", "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == secret


def test_secret_with_special_characters(mocker: MockerFixture):
    """Test JWT secret with special characters."""
    special_secret = "!@#$%^&*()_+-=[]{}|;:,.<>?`~" + "a" * 10  # 40 chars total
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": special_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == special_secret


def test_secret_with_unicode(mocker: MockerFixture):
    """Test JWT secret with unicode characters."""
    unicode_secret = "秘密🔐キー" + "a" * 25  # Ensure >32 bytes
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": unicode_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == unicode_secret


def test_secret_very_long(mocker: MockerFixture):
    """Test JWT secret with excessive length."""
    long_secret = "a" * 1000  # 1000 character secret
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": long_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == long_secret
    assert len(settings.JWT_VERIFY_KEY) == 1000


def test_secret_with_newline(mocker: MockerFixture):
    """Test JWT secret containing newlines."""
    multiline_secret = "secret\nwith\nnewlines" + "a" * 20
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": multiline_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == multiline_secret


def test_secret_base64_encoded(mocker: MockerFixture):
    """Test JWT secret that looks like base64."""
    base64_secret = "dGhpc19pc19hX3NlY3JldF9rZXlfd2l0aF9wcm9wZXJfbGVuZ3Ro"
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": base64_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == base64_secret


def test_secret_numeric_only(mocker: MockerFixture):
    """Test JWT secret with only numbers."""
    numeric_secret = "1234567890" * 4  # 40 character numeric secret
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": numeric_secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == numeric_secret


def test_algorithm_default_hs256(mocker: MockerFixture):
    """Test that JWT algorithm defaults to HS256."""
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": "a" * 32, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_ALGORITHM == "HS256"


def test_algorithm_whitespace_stripped(mocker: MockerFixture):
    """Test that JWT algorithm whitespace is stripped."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {
            "JWT_VERIFY_KEY": secret,
            "JWT_SIGN_ALGORITHM": "  HS256  ",
            "JWT_JWKS_URL": VALID_JWKS_URL,
        },
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_ALGORITHM == "HS256"


def test_no_crypto_warning(mocker: MockerFixture, caplog: pytest.LogCaptureFixture):
    """Test that a missing crypto package raises a clear error when
    JWT_JWKS_URL is set, since JWT_JWKS_URL is now mandatory and asymmetric
    verification can't silently fall back to a warning anymore."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": secret, "JWT_JWKS_URL": VALID_JWKS_URL},
        clear=True,
    )

    # Mock has_crypto to return False
    mocker.patch("autogpt_libs.auth.config.has_crypto", False)

    with pytest.raises(AuthConfigError) as exc_info:
        Settings()
    assert "cryptography" in str(exc_info.value)


def test_algorithm_invalid_raises_error(mocker: MockerFixture):
    """Test that invalid JWT algorithm raises AuthConfigError."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {
            "JWT_VERIFY_KEY": secret,
            "JWT_SIGN_ALGORITHM": "INVALID_ALG",
            "JWT_JWKS_URL": VALID_JWKS_URL,
        },
        clear=True,
    )

    with pytest.raises(AuthConfigError) as exc_info:
        Settings()
    assert "Invalid JWT_SIGN_ALGORITHM" in str(exc_info.value)
    assert "INVALID_ALG" in str(exc_info.value)


def test_algorithm_none_raises_error(mocker: MockerFixture):
    """Test that 'none' algorithm raises AuthConfigError."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {
            "JWT_VERIFY_KEY": secret,
            "JWT_SIGN_ALGORITHM": "none",
            "JWT_JWKS_URL": VALID_JWKS_URL,
        },
        clear=True,
    )

    with pytest.raises(AuthConfigError) as exc_info:
        Settings()
    assert "Invalid JWT_SIGN_ALGORITHM" in str(exc_info.value)


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
def test_algorithm_symmetric_warning(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture, algorithm: str
):
    """Test warning for symmetric algorithms (HS256, HS384, HS512)."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {
            "JWT_VERIFY_KEY": secret,
            "JWT_SIGN_ALGORITHM": algorithm,
            "JWT_JWKS_URL": VALID_JWKS_URL,
        },
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        settings = Settings()
        assert algorithm in caplog.text
        assert "symmetric shared-key signature algorithm" in caplog.text
        assert settings.JWT_ALGORITHM == algorithm


@pytest.mark.parametrize(
    "algorithm",
    ["ES256", "ES384", "ES512", "RS256", "RS384", "RS512", "PS256", "PS384", "PS512"],
)
def test_algorithm_asymmetric_no_warning(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture, algorithm: str
):
    """Test that asymmetric algorithms do not trigger warning."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {
            "JWT_VERIFY_KEY": secret,
            "JWT_SIGN_ALGORITHM": algorithm,
            "JWT_JWKS_URL": VALID_JWKS_URL,
        },
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        settings = Settings()
        # Should not contain the symmetric algorithm warning
        assert "symmetric shared-key signature algorithm" not in caplog.text
        assert settings.JWT_ALGORITHM == algorithm


def test_jwks_url_alone_is_sufficient(mocker: MockerFixture):
    """Test that JWT_JWKS_URL without a shared secret passes validation."""
    mocker.patch.dict(
        os.environ,
        {"JWT_JWKS_URL": "http://localhost:3000/api/auth/jwks"},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_JWKS_URL == "http://localhost:3000/api/auth/jwks"
    assert settings.JWT_VERIFY_KEY == ""


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


def test_neither_key_nor_jwks_raises_error(mocker: MockerFixture):
    """Test that missing both verification mechanisms raises AuthConfigError."""
    mocker.patch.dict(os.environ, {}, clear=True)

    with pytest.raises(AuthConfigError) as exc_info:
        Settings()
    assert "JWT_JWKS_URL" in str(exc_info.value)
    assert "JWT_VERIFY_KEY" in str(exc_info.value)


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
