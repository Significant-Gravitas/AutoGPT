"""Tests for SearchFeatureRequestsTool and CreateFeatureRequestTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .feature_requests import CreateFeatureRequestTool, SearchFeatureRequestsTool
from .models import (
    ErrorResponse,
    FeatureRequestCreatedResponse,
    FeatureRequestSearchResponse,
    NoResultsResponse,
)

_TEST_USER_ID = "test-user-feature-requests"
_TEST_USER_EMAIL = "testuser@example.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FAKE_PROJECT_ID = "test-project-id"
_FAKE_TEAM_ID = "test-team-id"


def _mock_linear_config(*, query_return=None, mutate_return=None):
    """Return a patched _get_linear_config that yields a mock LinearClient."""
    client = AsyncMock()
    if query_return is not None:
        client.query.return_value = query_return
    if mutate_return is not None:
        client.mutate.return_value = mutate_return
    return (
        patch(
            "backend.copilot.tools.feature_requests._get_linear_config",
            return_value=(client, _FAKE_PROJECT_ID, _FAKE_TEAM_ID),
        ),
        client,
    )


def _search_response(nodes: list[dict]) -> dict:
    return {"searchIssues": {"nodes": nodes}}


def _customer_upsert_response(
    customer_id: str = "cust-1", name: str = _TEST_USER_EMAIL, success: bool = True
) -> dict:
    return {
        "customerUpsert": {
            "success": success,
            "customer": {"id": customer_id, "name": name, "externalIds": [name]},
        }
    }


def _issue_create_response(
    issue_id: str = "issue-1",
    identifier: str = "FR-1",
    title: str = "New Feature",
    success: bool = True,
) -> dict:
    return {
        "issueCreate": {
            "success": success,
            "issue": {
                "id": issue_id,
                "identifier": identifier,
                "title": title,
                "url": f"https://linear.app/issue/{identifier}",
            },
        }
    }


def _need_create_response(
    need_id: str = "need-1",
    issue_id: str = "issue-1",
    identifier: str = "FR-1",
    title: str = "New Feature",
    success: bool = True,
) -> dict:
    return {
        "customerNeedCreate": {
            "success": success,
            "need": {
                "id": need_id,
                "body": "description",
                "customer": {"id": "cust-1", "name": _TEST_USER_EMAIL},
                "issue": {
                    "id": issue_id,
                    "identifier": identifier,
                    "title": title,
                    "url": f"https://linear.app/issue/{identifier}",
                },
            },
        }
    }


# ===========================================================================
# SearchFeatureRequestsTool
# ===========================================================================


class TestSearchFeatureRequestsTool:
    """Tests for SearchFeatureRequestsTool._execute."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_successful_search(self):
        session = make_session(user_id=_TEST_USER_ID)
        nodes = [
            {
                "id": "id-1",
                "identifier": "FR-1",
                "title": "Dark mode",
            },
            {
                "id": "id-2",
                "identifier": "FR-2",
                "title": "Dark theme",
            },
        ]
        patcher, _ = _mock_linear_config(query_return=_search_response(nodes))
        with patcher:
            tool = SearchFeatureRequestsTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID, session=session, query="dark mode"
            )

        assert isinstance(resp, FeatureRequestSearchResponse)
        assert resp.count == 2
        assert resp.results[0].id == "id-1"
        assert resp.results[1].identifier == "FR-2"
        assert resp.query == "dark mode"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_results(self):
        session = make_session(user_id=_TEST_USER_ID)
        patcher, _ = _mock_linear_config(query_return=_search_response([]))
        with patcher:
            tool = SearchFeatureRequestsTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID, session=session, query="nonexistent"
            )

        assert isinstance(resp, NoResultsResponse)
        assert "nonexistent" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_empty_query_returns_error(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = SearchFeatureRequestsTool()
        resp = await tool._execute(user_id=_TEST_USER_ID, session=session, query="   ")

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "query" in resp.error.lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_query_returns_error(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = SearchFeatureRequestsTool()
        resp = await tool._execute(user_id=_TEST_USER_ID, session=session)

        assert isinstance(resp, ErrorResponse)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_api_failure(self):
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.query.side_effect = RuntimeError("Linear API down")
        with patcher:
            tool = SearchFeatureRequestsTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID, session=session, query="test"
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "Linear API down" in resp.error

    @pytest.mark.asyncio(loop_scope="session")
    async def test_malformed_node_returns_error(self):
        """A node missing required keys should be caught by the try/except."""
        session = make_session(user_id=_TEST_USER_ID)
        # Node missing 'identifier' key
        bad_nodes = [{"id": "id-1", "title": "Missing identifier"}]
        patcher, _ = _mock_linear_config(query_return=_search_response(bad_nodes))
        with patcher:
            tool = SearchFeatureRequestsTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID, session=session, query="test"
            )

        assert isinstance(resp, ErrorResponse)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_linear_client_init_failure(self):
        session = make_session(user_id=_TEST_USER_ID)
        with patch(
            "backend.copilot.tools.feature_requests._get_linear_config",
            side_effect=RuntimeError("No API key"),
        ):
            tool = SearchFeatureRequestsTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID, session=session, query="test"
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "No API key" in resp.error


# ===========================================================================
# CreateFeatureRequestTool
# ===========================================================================


class TestCreateFeatureRequestTool:
    """Tests for CreateFeatureRequestTool._execute."""

    @pytest.fixture(autouse=True)
    def _patch_email_lookup(self):
        mock_user_db = MagicMock()
        mock_user_db.get_user_email_by_id = AsyncMock(return_value=_TEST_USER_EMAIL)
        with patch(
            "backend.copilot.tools.feature_requests.user_db",
            return_value=mock_user_db,
        ):
            yield

    # ---- Happy paths -------------------------------------------------------

    @pytest.mark.asyncio(loop_scope="session")
    async def test_create_new_issue(self):
        """Full happy path: upsert customer -> create issue -> attach need."""
        session = make_session(user_id=_TEST_USER_ID)

        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            _issue_create_response(),
            _need_create_response(),
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="New Feature",
                description="Please add this",
            )

        assert isinstance(resp, FeatureRequestCreatedResponse)
        assert resp.is_new_issue is True
        assert resp.issue_identifier == "FR-1"
        assert resp.customer_name == _TEST_USER_EMAIL
        assert client.mutate.call_count == 3

    @pytest.mark.asyncio(loop_scope="session")
    async def test_add_need_to_existing_issue(self):
        """When existing_issue_id is provided, skip issue creation."""
        session = make_session(user_id=_TEST_USER_ID)

        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            _need_create_response(issue_id="existing-1", identifier="FR-99"),
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Existing Feature",
                description="Me too",
                existing_issue_id="existing-1",
            )

        assert isinstance(resp, FeatureRequestCreatedResponse)
        assert resp.is_new_issue is False
        assert resp.issue_id == "existing-1"
        # Only 2 mutations: customer upsert + need create (no issue create)
        assert client.mutate.call_count == 2

    # ---- Validation errors -------------------------------------------------

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_title(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = CreateFeatureRequestTool()
        resp = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            title="",
            description="some desc",
        )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "required" in resp.error.lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_description(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = CreateFeatureRequestTool()
        resp = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            title="Some title",
            description="",
        )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "required" in resp.error.lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_user_id(self):
        session = make_session(user_id=_TEST_USER_ID)
        tool = CreateFeatureRequestTool()
        resp = await tool._execute(
            user_id=None,
            session=session,
            title="Some title",
            description="Some desc",
        )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "user_id" in resp.error.lower()

    # ---- Linear client init failure ----------------------------------------

    @pytest.mark.asyncio(loop_scope="session")
    async def test_linear_client_init_failure(self):
        session = make_session(user_id=_TEST_USER_ID)
        with patch(
            "backend.copilot.tools.feature_requests._get_linear_config",
            side_effect=RuntimeError("No API key"),
        ):
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "No API key" in resp.error

    # ---- Customer upsert failures ------------------------------------------

    @pytest.mark.asyncio(loop_scope="session")
    async def test_customer_upsert_api_error(self):
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = RuntimeError("Customer API error")

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "Customer API error" in resp.error

    @pytest.mark.asyncio(loop_scope="session")
    async def test_customer_upsert_not_success(self):
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.return_value = _customer_upsert_response(success=False)

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_customer_malformed_response(self):
        """Customer dict missing 'id' key should be caught."""
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        # success=True but customer has no 'id'
        client.mutate.return_value = {
            "customerUpsert": {
                "success": True,
                "customer": {"name": _TEST_USER_ID},
            }
        }

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)

    # ---- Issue creation failures -------------------------------------------

    @pytest.mark.asyncio(loop_scope="session")
    async def test_issue_create_api_error(self):
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            RuntimeError("Issue create failed"),
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "Issue create failed" in resp.error

    @pytest.mark.asyncio(loop_scope="session")
    async def test_issue_create_not_success(self):
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            _issue_create_response(success=False),
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)
        assert "Failed to create feature request issue" in resp.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_issue_create_malformed_response(self):
        """issueCreate success=True but missing 'issue' key."""
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            {"issueCreate": {"success": True}},  # no 'issue' key
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)

    # ---- Customer need attachment failures ---------------------------------

    @pytest.mark.asyncio(loop_scope="session")
    async def test_need_create_api_error_new_issue(self):
        """Need creation fails after new issue was created -> orphaned issue info."""
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            _issue_create_response(issue_id="orphan-1", identifier="FR-10"),
            RuntimeError("Need attach failed"),
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.error is not None
        assert "Need attach failed" in resp.error
        assert resp.details is not None
        assert resp.details["issue_id"] == "orphan-1"
        assert resp.details["issue_identifier"] == "FR-10"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_need_create_api_error_existing_issue(self):
        """Need creation fails on existing issue -> no orphaned info."""
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            RuntimeError("Need attach failed"),
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
                existing_issue_id="existing-1",
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.details is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_need_create_not_success_includes_orphaned_info(self):
        """customerNeedCreate returns success=False -> includes orphaned issue."""
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            _issue_create_response(issue_id="orphan-2", identifier="FR-20"),
            _need_create_response(success=False),
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.details is not None
        assert resp.details["issue_id"] == "orphan-2"
        assert resp.details["issue_identifier"] == "FR-20"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_need_create_not_success_existing_issue_no_details(self):
        """customerNeedCreate fails on existing issue -> no orphaned info."""
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            _need_create_response(success=False),
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
                existing_issue_id="existing-1",
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.details is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_need_create_malformed_response(self):
        """need_result missing 'need' key after success=True."""
        session = make_session(user_id=_TEST_USER_ID)
        patcher, client = _mock_linear_config()
        client.mutate.side_effect = [
            _customer_upsert_response(),
            _issue_create_response(),
            {"customerNeedCreate": {"success": True}},  # no 'need' key
        ]

        with patcher:
            tool = CreateFeatureRequestTool()
            resp = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                title="Title",
                description="Desc",
            )

        assert isinstance(resp, ErrorResponse)
        assert resp.details is not None
        assert resp.details["issue_id"] == "issue-1"
