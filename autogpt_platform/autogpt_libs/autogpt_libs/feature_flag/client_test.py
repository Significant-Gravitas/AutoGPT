import pytest
from ldclient import LDClient

from autogpt_libs.feature_flag.client import feature_flag, mock_flag_variation


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
