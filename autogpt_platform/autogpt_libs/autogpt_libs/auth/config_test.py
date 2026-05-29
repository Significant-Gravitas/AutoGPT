"""
Comprehensive tests for auth configuration to ensure 100% line and branch coverage.
These tests verify critical security checks preventing JWT token forgery.
"""

import logging
import os

import pytest
from pytest_mock import MockerFixture

from autogpt_libs.auth.config import AuthConfigError, Settings


def test_environment_variable_precedence(mocker: MockerFixture):
    """Test that environment variables take precedence over defaults."""
    secret = "environment-secret-key-with-proper-length-123456"
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == secret


def test_environment_variable_backwards_compatible(mocker: MockerFixture):
    """Test that SUPABASE_JWT_SECRET is read if JWT_VERIFY_KEY is not set."""
    secret = "environment-secret-key-with-proper-length-123456"
    mocker.patch.dict(os.environ, {"SUPABASE_JWT_SECRET": secret}, clear=True)

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
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": secret}, clear=True)

    settings = Settings()
    original_secret = settings.JWT_VERIFY_KEY

    # Changing environment after creation shouldn't affect settings
    os.environ["JWT_VERIFY_KEY"] = "different-secret"

    assert settings.JWT_VERIFY_KEY == original_secret


def test_settings_load_with_valid_secret(mocker: MockerFixture):
    """Test auth enabled with a valid JWT secret."""
    valid_secret = "a" * 32  # 32 character secret
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": valid_secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == valid_secret


def test_settings_load_with_strong_secret(mocker: MockerFixture):
    """Test auth enabled with a cryptographically strong secret."""
    strong_secret = "super-secret-jwt-token-with-at-least-32-characters-long"
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": strong_secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == strong_secret
    assert len(settings.JWT_VERIFY_KEY) >= 32


def test_secret_empty_raises_error(mocker: MockerFixture):
    """Test that auth enabled with empty secret raises AuthConfigError."""
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": ""}, clear=True)

    with pytest.raises(Exception) as exc_info:
        Settings()
    assert "JWT_VERIFY_KEY" in str(exc_info.value)


def test_secret_missing_raises_error(mocker: MockerFixture):
    """Test that auth enabled without secret env var raises AuthConfigError."""
    mocker.patch.dict(os.environ, {}, clear=True)

    with pytest.raises(Exception) as exc_info:
        Settings()
    assert "JWT_VERIFY_KEY" in str(exc_info.value)


@pytest.mark.parametrize("secret", [" ", "  ", "\t", "\n", " \t\n "])
def test_secret_only_whitespace_raises_error(mocker: MockerFixture, secret: str):
    """Test that auth enabled with whitespace-only secret raises error."""
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": secret}, clear=True)

    with pytest.raises(ValueError):
        Settings()


def test_secret_weak_logs_warning(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    """Test that weak JWT secret triggers warning log."""
    weak_secret = "short"  # Less than 32 characters
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": weak_secret}, clear=True)

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
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": secret_31}, clear=True)

    with caplog.at_level(logging.WARNING):
        settings = Settings()
        assert len(settings.JWT_VERIFY_KEY) == 31
        assert "key appears weak" in caplog.text.lower()


def test_secret_32_char_no_warning(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    """Test that 32-character secret does not trigger warning (boundary test)."""
    secret_32 = "a" * 32  # Exactly 32 characters
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": secret_32}, clear=True)

    with caplog.at_level(logging.WARNING):
        settings = Settings()
        assert len(settings.JWT_VERIFY_KEY) == 32
        assert "JWT secret appears weak" not in caplog.text


def test_secret_whitespace_stripped(mocker: MockerFixture):
    """Test that JWT secret whitespace is stripped."""
    secret = "a" * 32
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": f"  {secret}  "}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == secret


def test_secret_with_special_characters(mocker: MockerFixture):
    """Test JWT secret with special characters."""
    special_secret = "!@#$%^&*()_+-=[]{}|;:,.<>?`~" + "a" * 10  # 40 chars total
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": special_secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == special_secret


def test_secret_with_unicode(mocker: MockerFixture):
    """Test JWT secret with unicode characters."""
    unicode_secret = "秘密🔐キー" + "a" * 25  # Ensure >32 bytes
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": unicode_secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == unicode_secret


def test_secret_very_long(mocker: MockerFixture):
    """Test JWT secret with excessive length."""
    long_secret = "a" * 1000  # 1000 character secret
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": long_secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == long_secret
    assert len(settings.JWT_VERIFY_KEY) == 1000


def test_secret_with_newline(mocker: MockerFixture):
    """Test JWT secret containing newlines."""
    multiline_secret = "secret\nwith\nnewlines" + "a" * 20
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": multiline_secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == multiline_secret


def test_secret_base64_encoded(mocker: MockerFixture):
    """Test JWT secret that looks like base64."""
    base64_secret = "dGhpc19pc19hX3NlY3JldF9rZXlfd2l0aF9wcm9wZXJfbGVuZ3Ro"
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": base64_secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == base64_secret


def test_secret_numeric_only(mocker: MockerFixture):
    """Test JWT secret with only numbers."""
    numeric_secret = "1234567890" * 4  # 40 character numeric secret
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": numeric_secret}, clear=True)

    settings = Settings()
    assert settings.JWT_VERIFY_KEY == numeric_secret


def test_algorithm_default_hs256(mocker: MockerFixture):
    """Test that JWT algorithm defaults to HS256."""
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": "a" * 32}, clear=True)

    settings = Settings()
    assert settings.JWT_ALGORITHM == "HS256"


def test_algorithm_whitespace_stripped(mocker: MockerFixture):
    """Test that JWT algorithm whitespace is stripped."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": secret, "JWT_SIGN_ALGORITHM": "  HS256  "},
        clear=True,
    )

    settings = Settings()
    assert settings.JWT_ALGORITHM == "HS256"


def test_no_crypto_warning(mocker: MockerFixture, caplog: pytest.LogCaptureFixture):
    """Test warning when crypto package is not available."""
    secret = "a" * 32
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": secret}, clear=True)

    # Mock has_crypto to return False
    mocker.patch("autogpt_libs.auth.config.has_crypto", False)

    with caplog.at_level(logging.WARNING):
        Settings()
        assert "Asymmetric JWT verification is not available" in caplog.text
        assert "cryptography" in caplog.text


def test_algorithm_invalid_raises_error(mocker: MockerFixture):
    """Test that invalid JWT algorithm raises AuthConfigError."""
    secret = "a" * 32
    mocker.patch.dict(
        os.environ,
        {"JWT_VERIFY_KEY": secret, "JWT_SIGN_ALGORITHM": "INVALID_ALG"},
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
        {"JWT_VERIFY_KEY": secret, "JWT_SIGN_ALGORITHM": "none"},
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
        {"JWT_VERIFY_KEY": secret, "JWT_SIGN_ALGORITHM": algorithm},
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
        {"JWT_VERIFY_KEY": secret, "JWT_SIGN_ALGORITHM": algorithm},
        clear=True,
    )

    with caplog.at_level(logging.WARNING):
        settings = Settings()
        # Should not contain the symmetric algorithm warning
        assert "symmetric shared-key signature algorithm" not in caplog.text
        assert settings.JWT_ALGORITHM == algorithm
