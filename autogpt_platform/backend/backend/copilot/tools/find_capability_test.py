"""``find_capability`` in an expert session reports what the session can run,
not what the account owns, so search and ``run_capability`` agree."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.capabilities.resolve import load_connection_state
from backend.copilot.context import set_execution_context
from backend.data.model import APIKeyCredentials

from ._test_data import make_session
from .find_capability import NEEDS_EXPERT_GRANT, FindCapabilityTool
from .models import CapabilityListResponse

USER = "user-find-cap"
GITHUB_STATE = ConnectionState(providers=frozenset({"github"}))
GITHUB_UNGRANTED = ConnectionState(ungranted=GITHUB_STATE)


@pytest.fixture(autouse=True)
def _clean_context():
    set_execution_context(USER, make_session(USER))
    yield
    set_execution_context(None, None)


@pytest.fixture(autouse=True)
def _no_skills():
    """Skills are read through Redis and the workspace; keep them out."""
    with (
        patch(
            "backend.copilot.tools.session_registry.is_skills_feature_enabled",
            AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.session_registry.list_all_skills",
            AsyncMock(return_value=[]),
        ),
    ):
        yield


def _api_key(id: str, provider: str) -> APIKeyCredentials:
    return APIKeyCredentials(id=id, provider=provider, title=id, api_key=SecretStr("k"))


def _github_hits(result: CapabilityListResponse) -> list[dict]:
    return [c for c in result.capabilities if c["name"].startswith("Github")]


async def test_expert_search_reports_needs_grant_for_an_owned_but_ungranted_credential():
    """The account holds a GitHub credential the expert was never granted:
    the listing must say so, never ``connected: true`` — that is what sent
    the model to a sign-in card the run then refused."""
    session = make_session(USER, expert_id="expert-1")
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=GITHUB_UNGRANTED),
    ) as loader:
        result = await FindCapabilityTool()._execute(
            USER, session, query="github pull request"
        )
    assert isinstance(result, CapabilityListResponse)
    loader.assert_awaited_once_with(USER, "expert-1")
    hits = _github_hits(result)
    assert hits
    assert all(c["connected"] == NEEDS_EXPERT_GRANT for c in hits)
    assert not any(c.get("connected") is True for c in result.capabilities)
    assert NEEDS_EXPERT_GRANT in result.message
    assert "grant" in result.message.lower()
    assert "do not ask the user to sign in" in result.message.lower()


async def test_expert_search_reports_connected_for_a_granted_credential():
    session = make_session(USER, expert_id="expert-1")
    granted = ConnectionState(
        providers=frozenset({"github"}), ungranted=ConnectionState()
    )
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=granted),
    ):
        result = await FindCapabilityTool()._execute(
            USER, session, query="github pull request"
        )
    assert isinstance(result, CapabilityListResponse)
    assert all(c["connected"] is True for c in _github_hits(result))
    assert NEEDS_EXPERT_GRANT not in result.message


async def test_expert_search_reports_false_for_a_provider_the_account_lacks():
    session = make_session(USER, expert_id="expert-1")
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=ConnectionState(ungranted=ConnectionState())),
    ):
        result = await FindCapabilityTool()._execute(
            USER, session, query="github pull request"
        )
    assert isinstance(result, CapabilityListResponse)
    assert all(c["connected"] is False for c in _github_hits(result))
    assert NEEDS_EXPERT_GRANT not in result.message


async def test_personal_session_is_not_narrowed():
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=GITHUB_STATE),
    ) as loader:
        result = await FindCapabilityTool()._execute(
            USER, make_session(USER), query="github pull request"
        )
    assert isinstance(result, CapabilityListResponse)
    loader.assert_awaited_once_with(USER, None)
    assert all(c["connected"] is True for c in _github_hits(result))


async def test_loader_narrows_an_expert_to_its_grants():
    """Account: a GitHub credential the expert has, a second GitHub one it
    lacks, and a Slack one it lacks. Only the granted provider is connected;
    the rest is reported as ungranted."""
    store = MagicMock()
    store.get_all_creds = AsyncMock(
        return_value=[
            _api_key("gh-granted", "github"),
            _api_key("gh-spare", "github"),
            _api_key("slack-1", "slack"),
        ]
    )
    experts = MagicMock()
    experts.expert_allowed_credential_ids = AsyncMock(return_value=["gh-granted"])
    with (
        patch(
            "backend.copilot.capabilities.resolve.IntegrationCredentialsManager",
            return_value=MagicMock(store=store),
        ),
        patch("backend.data.db_accessors.experts_db", return_value=experts),
    ):
        state = await load_connection_state(USER, "expert-1")
        personal = await load_connection_state(USER)
    experts.expert_allowed_credential_ids.assert_awaited_once_with(USER, "expert-1")
    assert state.providers == frozenset({"github"})
    assert state.ungranted is not None
    assert state.ungranted.providers == frozenset({"github", "slack"})
    assert personal.providers == frozenset({"github", "slack"})
    assert personal.ungranted is None


async def test_loader_fails_closed_when_the_grant_lookup_fails():
    """A broken allow-list read must not report the account's credentials as
    usable: the run-time gate would refuse them anyway."""
    store = MagicMock()
    store.get_all_creds = AsyncMock(return_value=[_api_key("gh-1", "github")])
    experts = MagicMock()
    experts.expert_allowed_credential_ids = AsyncMock(side_effect=RuntimeError("db"))
    with (
        patch(
            "backend.copilot.capabilities.resolve.IntegrationCredentialsManager",
            return_value=MagicMock(store=store),
        ),
        patch("backend.data.db_accessors.experts_db", return_value=experts),
    ):
        state = await load_connection_state(USER, "expert-1")
    assert state.providers == frozenset()
    assert state.ungranted is not None
    assert state.ungranted.providers == frozenset({"github"})
