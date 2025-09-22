"""
Unit tests for APIKeyAuthenticator class.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from backend.server.utils.api_key_auth import APIKeyAuthenticator
from backend.util.exceptions import MissingConfigError


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = Mock(spec=Request)
    request.state = Mock()
    request.headers = {}
    return request


@pytest.fixture
def api_key_auth():
    """Create a basic APIKeyAuthenticator instance."""
    return APIKeyAuthenticator(
        header_name="X-API-Key", expected_token="test-secret-token"
    )


@pytest.fixture
def api_key_auth_custom_validator():
    """Create APIKeyAuthenticator with custom validator."""

    def custom_validator(api_key: str) -> bool:
        return api_key == "custom-valid-key"

    return APIKeyAuthenticator(header_name="X-API-Key", validator=custom_validator)


@pytest.fixture
def api_key_auth_async_validator():
    """Create APIKeyAuthenticator with async custom validator."""

    async def async_validator(api_key: str) -> bool:
        return api_key == "async-valid-key"

    return APIKeyAuthenticator(header_name="X-API-Key", validator=async_validator)


@pytest.fixture
def api_key_auth_object_validator():
    """Create APIKeyAuthenticator that returns objects from validator."""

    async def object_validator(api_key: str):
        if api_key == "user-key":
            return {"user_id": "123", "permissions": ["read", "write"]}
        return None

    return APIKeyAuthenticator(header_name="X-API-Key", validator=object_validator)


# ========== Basic Initialization Tests ========== #


def test_init_with_expected_token():
    """Test initialization with expected token."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="test-token")

    assert auth.model.name == "X-API-Key"
    assert auth.expected_token == "test-token"
    assert auth.custom_validator is None
    assert auth.status_if_missing == HTTP_401_UNAUTHORIZED
    assert auth.message_if_invalid == "Invalid API key"


def test_init_with_custom_validator():
    """Test initialization with custom validator."""

    def validator(key: str) -> bool:
        return True

    auth = APIKeyAuthenticator(header_name="Authorization", validator=validator)

    assert auth.model.name == "Authorization"
    assert auth.expected_token is None
    assert auth.custom_validator == validator
    assert auth.status_if_missing == HTTP_401_UNAUTHORIZED
    assert auth.message_if_invalid == "Invalid API key"


def test_init_with_custom_parameters():
    """Test initialization with custom status and message."""
    auth = APIKeyAuthenticator(
        header_name="X-Custom-Key",
        expected_token="token",
        status_if_missing=HTTP_403_FORBIDDEN,
        message_if_invalid="Access denied",
    )

    assert auth.model.name == "X-Custom-Key"
    assert auth.status_if_missing == HTTP_403_FORBIDDEN
    assert auth.message_if_invalid == "Access denied"


def test_scheme_name_generation():
    """Test that scheme_name is generated correctly."""
    auth = APIKeyAuthenticator(header_name="X-Custom-Header", expected_token="token")

    assert auth.scheme_name == "APIKeyAuthenticator-X-Custom-Header"


# ========== Authentication Flow Tests ========== #


@pytest.mark.asyncio
async def test_api_key_missing(api_key_auth, mock_request):
    """Test behavior when API key is missing from request."""
    # Mock the parent class method to return None (no API key)
    with pytest.raises(HTTPException) as exc_info:
        await api_key_auth(mock_request)

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "No API key in request"


@pytest.mark.asyncio
async def test_api_key_valid(api_key_auth, mock_request):
    """Test behavior with valid API key."""
    # Mock the parent class to return the API key
    with patch.object(
        api_key_auth.__class__.__bases__[0],
        "__call__",
        return_value="test-secret-token",
    ):
        result = await api_key_auth(mock_request)

    assert result is True


@pytest.mark.asyncio
async def test_api_key_invalid(api_key_auth, mock_request):
    """Test behavior with invalid API key."""
    # Mock the parent class to return an invalid API key
    with patch.object(
        api_key_auth.__class__.__bases__[0], "__call__", return_value="invalid-token"
    ):
        with pytest.raises(HTTPException) as exc_info:
            await api_key_auth(mock_request)

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"


# ========== Custom Validator Tests ========== #


@pytest.mark.asyncio
async def test_custom_status_and_message(mock_request):
    """Test custom status code and message."""
    auth = APIKeyAuthenticator(
        header_name="X-API-Key",
        expected_token="valid-token",
        status_if_missing=HTTP_403_FORBIDDEN,
        message_if_invalid="Access forbidden",
    )

    # Test missing key
    with pytest.raises(HTTPException) as exc_info:
        await auth(mock_request)

    assert exc_info.value.status_code == HTTP_403_FORBIDDEN
    assert exc_info.value.detail == "No API key in request"

    # Test invalid key
    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value="invalid-token"
    ):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

    assert exc_info.value.status_code == HTTP_403_FORBIDDEN
    assert exc_info.value.detail == "Access forbidden"


@pytest.mark.asyncio
async def test_custom_sync_validator(api_key_auth_custom_validator, mock_request):
    """Test with custom synchronous validator."""
    # Mock the parent class to return the API key
    with patch.object(
        api_key_auth_custom_validator.__class__.__bases__[0],
        "__call__",
        return_value="custom-valid-key",
    ):
        result = await api_key_auth_custom_validator(mock_request)

    assert result is True


@pytest.mark.asyncio
async def test_custom_sync_validator_invalid(
    api_key_auth_custom_validator, mock_request
):
    """Test custom synchronous validator with invalid key."""
    with patch.object(
        api_key_auth_custom_validator.__class__.__bases__[0],
        "__call__",
        return_value="invalid-key",
    ):
        with pytest.raises(HTTPException) as exc_info:
            await api_key_auth_custom_validator(mock_request)

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_custom_async_validator(api_key_auth_async_validator, mock_request):
    """Test with custom async validator."""
    with patch.object(
        api_key_auth_async_validator.__class__.__bases__[0],
        "__call__",
        return_value="async-valid-key",
    ):
        result = await api_key_auth_async_validator(mock_request)

    assert result is True


@pytest.mark.asyncio
async def test_custom_async_validator_invalid(
    api_key_auth_async_validator, mock_request
):
    """Test custom async validator with invalid key."""
    with patch.object(
        api_key_auth_async_validator.__class__.__bases__[0],
        "__call__",
        return_value="invalid-key",
    ):
        with pytest.raises(HTTPException) as exc_info:
            await api_key_auth_async_validator(mock_request)

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_validator_returns_object(api_key_auth_object_validator, mock_request):
    """Test validator that returns an object instead of boolean."""
    with patch.object(
        api_key_auth_object_validator.__class__.__bases__[0],
        "__call__",
        return_value="user-key",
    ):
        result = await api_key_auth_object_validator(mock_request)

    expected_result = {"user_id": "123", "permissions": ["read", "write"]}
    assert result == expected_result
    # Verify the object is stored in request state
    assert mock_request.state.api_key == expected_result


@pytest.mark.asyncio
async def test_validator_returns_none(api_key_auth_object_validator, mock_request):
    """Test validator that returns None (falsy)."""
    with patch.object(
        api_key_auth_object_validator.__class__.__bases__[0],
        "__call__",
        return_value="invalid-key",
    ):
        with pytest.raises(HTTPException) as exc_info:
            await api_key_auth_object_validator(mock_request)

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_validator_database_lookup_simulation(mock_request):
    """Test simulation of database lookup validator."""
    # Simulate database records
    valid_api_keys = {
        "key123": {"user_id": "user1", "active": True},
        "key456": {"user_id": "user2", "active": False},
    }

    async def db_validator(api_key: str):
        record = valid_api_keys.get(api_key)
        return record if record and record["active"] else None

    auth = APIKeyAuthenticator(header_name="X-API-Key", validator=db_validator)

    # Test valid active key
    with patch.object(auth.__class__.__bases__[0], "__call__", return_value="key123"):
        result = await auth(mock_request)
        assert result == {"user_id": "user1", "active": True}
        assert mock_request.state.api_key == {"user_id": "user1", "active": True}

    # Test inactive key
    mock_request.state = Mock()  # Reset state
    with patch.object(auth.__class__.__bases__[0], "__call__", return_value="key456"):
        with pytest.raises(HTTPException):
            await auth(mock_request)

    # Test non-existent key
    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value="nonexistent"
    ):
        with pytest.raises(HTTPException):
            await auth(mock_request)


# ========== Default Validator Tests ========== #


@pytest.mark.asyncio
async def test_default_validator_key_valid(api_key_auth):
    """Test default validator with valid token."""
    result = await api_key_auth.default_validator("test-secret-token")
    assert result is True


@pytest.mark.asyncio
async def test_default_validator_key_invalid(api_key_auth):
    """Test default validator with invalid token."""
    result = await api_key_auth.default_validator("wrong-token")
    assert result is False


@pytest.mark.asyncio
async def test_default_validator_missing_expected_token():
    """Test default validator when expected_token is not set."""
    auth = APIKeyAuthenticator(header_name="X-API-Key")

    with pytest.raises(MissingConfigError) as exc_info:
        await auth.default_validator("any-token")

    assert "expected_token is not set" in str(exc_info.value)
    assert "either specify it or provide a custom validator" in str(exc_info.value)


@pytest.mark.asyncio
async def test_default_validator_uses_constant_time_comparison(api_key_auth):
    """
    Test that default validator uses secrets.compare_digest for timing attack protection
    """
    with patch("secrets.compare_digest") as mock_compare:
        mock_compare.return_value = True

        await api_key_auth.default_validator("test-token")

        mock_compare.assert_called_once_with("test-token", "test-secret-token")


@pytest.mark.asyncio
async def test_api_key_empty(mock_request):
    """Test behavior with empty string API key."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    with patch.object(auth.__class__.__bases__[0], "__call__", return_value=""):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_key_whitespace_only(mock_request):
    """Test behavior with whitespace-only API key."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value="   \t\n  "
    ):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_key_very_long(mock_request):
    """Test behavior with extremely long API key (potential DoS protection)."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    # Create a very long API key (10MB)
    long_api_key = "a" * (10 * 1024 * 1024)

    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value=long_api_key
    ):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_key_with_null_bytes(mock_request):
    """Test behavior with API key containing null bytes."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    api_key_with_null = "valid\x00token"

    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value=api_key_with_null
    ):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_key_with_control_characters(mock_request):
    """Test behavior with API key containing control characters."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    # API key with various control characters
    api_key_with_control = "valid\r\n\t\x1b[31mtoken"

    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value=api_key_with_control
    ):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_key_with_unicode_characters(mock_request):
    """Test behavior with Unicode characters in API key."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    # API key with Unicode characters
    unicode_api_key = "validтокен🔑"

    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value=unicode_api_key
    ):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_key_with_unicode_characters_normalization_attack(mock_request):
    """Test that Unicode normalization doesn't bypass validation."""
    # Create auth with composed Unicode character
    auth = APIKeyAuthenticator(
        header_name="X-API-Key", expected_token="café"  # é is composed
    )

    # Try with decomposed version (c + a + f + e + ´)
    decomposed_key = "cafe\u0301"  # é as combining character

    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value=decomposed_key
    ):
        # Should fail because secrets.compare_digest doesn't normalize
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_key_with_binary_data(mock_request):
    """Test behavior with binary data in API key."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    # Binary data that might cause encoding issues
    binary_api_key = bytes([0xFF, 0xFE, 0xFD, 0xFC, 0x80, 0x81]).decode(
        "latin1", errors="ignore"
    )

    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value=binary_api_key
    ):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_key_with_regex_dos_attack_pattern(mock_request):
    """Test behavior with API key of repeated characters (pattern attack)."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    # Pattern that might cause regex DoS in poorly implemented validators
    repeated_key = "a" * 1000 + "b" * 1000 + "c" * 1000

    with patch.object(
        auth.__class__.__bases__[0], "__call__", return_value=repeated_key
    ):
        with pytest.raises(HTTPException) as exc_info:
            await auth(mock_request)

        assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_api_keys_with_newline_variations(mock_request):
    """Test different newline characters in API key."""
    auth = APIKeyAuthenticator(header_name="X-API-Key", expected_token="valid-token")

    newline_variations = [
        "valid\ntoken",  # Unix newline
        "valid\r\ntoken",  # Windows newline
        "valid\rtoken",  # Mac newline
        "valid\x85token",  # NEL (Next Line)
        "valid\x0Btoken",  # Vertical Tab
        "valid\x0Ctoken",  # Form Feed
    ]

    for api_key in newline_variations:
        with patch.object(
            auth.__class__.__bases__[0], "__call__", return_value=api_key
        ):
            with pytest.raises(HTTPException) as exc_info:
                await auth(mock_request)

            assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
            assert exc_info.value.detail == "Invalid API key"
