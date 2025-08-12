import pytest
from ldclient import LDClient

from autogpt_libs.feature_flag.client import (
    feature_flag,
    is_feature_enabled,
    mock_flag_variation,
)


@pytest.fixture
def ld_client(mocker):
    client = mocker.Mock(spec=LDClient)
    mocker.patch("ldclient.get", return_value=client)
    client.is_initialized.return_value = True
    return client


@pytest.mark.asyncio
async def test_feature_flag_enabled(ld_client):
    ld_client.variation.return_value = True

    @feature_flag("test-flag")
    async def test_function(user_id: str):
        return "success"

    result = test_function(user_id="test-user")
    assert result == "success"
    ld_client.variation.assert_called_once()


@pytest.mark.asyncio
async def test_feature_flag_unauthorized_response(ld_client):
    ld_client.variation.return_value = False

    @feature_flag("test-flag")
    async def test_function(user_id: str):
        return "success"

    result = test_function(user_id="test-user")
    assert result == {"error": "disabled"}


def test_mock_flag_variation(ld_client):
    with mock_flag_variation("test-flag", True):
        assert ld_client.variation("test-flag", None, False)

    with mock_flag_variation("test-flag", False):
        assert ld_client.variation("test-flag", None, False)


def test_is_feature_enabled(ld_client):
    """Test the is_feature_enabled helper function."""
    ld_client.is_initialized.return_value = True
    ld_client.variation.return_value = True

    result = is_feature_enabled("test-flag", "user123", default=False)
    assert result is True

    ld_client.variation.assert_called_once()
    call_args = ld_client.variation.call_args
    assert call_args[0][0] == "test-flag"  # flag_key
    assert call_args[0][2] is False  # default value


def test_is_feature_enabled_not_initialized(ld_client):
    """Test is_feature_enabled when LaunchDarkly is not initialized."""
    ld_client.is_initialized.return_value = False

    result = is_feature_enabled("test-flag", "user123", default=True)
    assert result is True  # Should return default

    ld_client.variation.assert_not_called()


def test_is_feature_enabled_exception(mocker):
    """Test is_feature_enabled when get_client() raises an exception."""
    mocker.patch(
        "autogpt_libs.feature_flag.client.get_client",
        side_effect=Exception("Client error"),
    )

    result = is_feature_enabled("test-flag", "user123", default=True)
    assert result is True  # Should return default
