import pytest
from fastapi import HTTPException
from ldclient import LDClient

from backend.util.feature_flag import (
    Flag,
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

    result = await test_function(user_id="test-user")
    assert result == "success"
    ld_client.variation.assert_called_once()


@pytest.mark.asyncio
async def test_feature_flag_unauthorized_response(ld_client):
    ld_client.variation.return_value = False

    @feature_flag("test-flag")
    async def test_function(user_id: str):
        return "success"

    with pytest.raises(HTTPException) as exc_info:
        await test_function(user_id="test-user")
    assert exc_info.value.status_code == 404


def test_mock_flag_variation(ld_client):
    with mock_flag_variation("test-flag", True):
        assert ld_client.variation("test-flag", None, False) is True

    with mock_flag_variation("test-flag", False):
        assert ld_client.variation("test-flag", None, True) is False


@pytest.mark.asyncio
async def test_is_feature_enabled(ld_client):
    """Test the is_feature_enabled helper function."""
    ld_client.is_initialized.return_value = True
    ld_client.variation.return_value = True

    result = await is_feature_enabled(Flag.AUTOMOD, "user123", default=False)
    assert result is True

    ld_client.variation.assert_called_once()
    call_args = ld_client.variation.call_args
    assert call_args[0][0] == "AutoMod"  # flag_key
    assert call_args[0][2] is False  # default value


@pytest.mark.asyncio
async def test_is_feature_enabled_not_initialized(ld_client):
    """Test is_feature_enabled when LaunchDarkly is not initialized."""
    ld_client.is_initialized.return_value = False

    result = await is_feature_enabled(Flag.AGENT_ACTIVITY, "user123", default=True)
    assert result is True  # Should return default

    ld_client.variation.assert_not_called()


@pytest.mark.asyncio
async def test_is_feature_enabled_exception(mocker):
    """Test is_feature_enabled when get_client() raises an exception."""
    mocker.patch(
        "backend.util.feature_flag.get_client",
        side_effect=Exception("Client error"),
    )

    result = await is_feature_enabled(Flag.AGENT_ACTIVITY, "user123", default=True)
    assert result is True  # Should return default


def test_flag_enum_values():
    """Test that Flag enum has expected values."""
    assert Flag.AUTOMOD == "AutoMod"
    assert Flag.AI_ACTIVITY_STATUS == "ai-agent-execution-summary"
    assert Flag.BETA_BLOCKS == "beta-blocks"
    assert Flag.AGENT_ACTIVITY == "agent-activity"


@pytest.mark.asyncio
async def test_is_feature_enabled_with_flag_enum(mocker):
    """Test is_feature_enabled function with Flag enum."""
    mock_get_feature_flag_value = mocker.patch(
        "backend.util.feature_flag.get_feature_flag_value"
    )
    mock_get_feature_flag_value.return_value = True

    result = await is_feature_enabled(Flag.AUTOMOD, "user123")

    assert result is True
    # Should call with the flag's string value
    mock_get_feature_flag_value.assert_called_once()
