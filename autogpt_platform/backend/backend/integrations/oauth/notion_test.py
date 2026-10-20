from unittest.mock import AsyncMock, MagicMock

from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.notion import NotionOAuthHandler
from backend.integrations.providers import ProviderName


async def test_exchange_keeps_the_refresh_token_notion_returns(mocker: MockerFixture):
    _mock_token_endpoint(mocker, _token_response())

    credentials = await _handler().exchange_code_for_tokens("code", [], None)

    assert credentials.refresh_token is not None
    assert credentials.refresh_token.get_secret_value() == "refresh-1"


async def test_refresh_sends_the_refresh_token_and_stores_the_rotated_pair(
    mocker: MockerFixture,
):
    post = _mock_token_endpoint(
        mocker, _token_response(access_token="access-2", refresh_token="refresh-2")
    )
    current = OAuth2Credentials(
        id="credential-id",
        provider=ProviderName.NOTION,
        title="Renamed by the user",
        access_token=SecretStr("access-1"),
        refresh_token=SecretStr("refresh-1"),
        scopes=[],
    )

    fresh = await _handler().refresh_tokens(current)

    assert post.await_args.kwargs["json"] == {
        "grant_type": "refresh_token",
        "refresh_token": "refresh-1",
    }
    assert (fresh.id, fresh.title) == ("credential-id", "Renamed by the user")
    assert fresh.access_token.get_secret_value() == "access-2"
    assert fresh.refresh_token is not None
    assert fresh.refresh_token.get_secret_value() == "refresh-2"


def _handler() -> NotionOAuthHandler:
    return NotionOAuthHandler(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.com/callback",
    )


def _token_response(**overrides: str) -> dict:
    return {
        "access_token": "access-1",
        "token_type": "bearer",
        "refresh_token": "refresh-1",
        "bot_id": "bot-id",
        "workspace_id": "workspace-id",
        "workspace_name": "Acme",
        "workspace_icon": None,
        "owner": {"type": "workspace", "workspace": True},
        "duplicated_template_id": None,
        **overrides,
    }


def _mock_token_endpoint(mocker: MockerFixture, payload: dict) -> AsyncMock:
    post = AsyncMock(return_value=MagicMock(json=MagicMock(return_value=payload)))
    mocker.patch(
        "backend.integrations.oauth.notion.Requests",
        return_value=MagicMock(post=post),
    )
    return post
