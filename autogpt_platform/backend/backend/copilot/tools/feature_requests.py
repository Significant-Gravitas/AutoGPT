"""Feature request tools - search and create feature requests via Linear."""

import logging
from typing import Any

from pydantic import SecretStr

from backend.blocks.linear._api import LinearClient
from backend.copilot.model import ChatSession
from backend.data.db_accessors import user_db
from backend.data.model import APIKeyCredentials
from backend.util.settings import Settings

from .base import BaseTool
from .models import (
    ErrorResponse,
    FeatureRequestCreatedResponse,
    FeatureRequestInfo,
    FeatureRequestSearchResponse,
    NoResultsResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 10

# GraphQL queries/mutations
SEARCH_ISSUES_QUERY = """
query SearchFeatureRequests($term: String!, $filter: IssueFilter, $first: Int) {
  searchIssues(term: $term, filter: $filter, first: $first) {
    nodes {
      id
      identifier
      title
    }
  }
}
"""

CUSTOMER_UPSERT_MUTATION = """
mutation CustomerUpsert($input: CustomerUpsertInput!) {
  customerUpsert(input: $input) {
    success
    customer {
      id
      name
      externalIds
    }
  }
}
"""

ISSUE_CREATE_MUTATION = """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      id
      identifier
      title
      url
    }
  }
}
"""

CUSTOMER_NEED_CREATE_MUTATION = """
mutation CustomerNeedCreate($input: CustomerNeedCreateInput!) {
  customerNeedCreate(input: $input) {
    success
    need {
      id
      body
      customer {
        id
        name
      }
      issue {
        id
        identifier
        title
        url
      }
    }
  }
}
"""


_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _get_linear_config() -> tuple[LinearClient, str, str]:
    """Return a configured Linear client, project ID, and team ID.

    Raises RuntimeError if any required setting is missing.
    """
    secrets = _get_settings().secrets
    if not secrets.copilot_linear_api_key:
        raise RuntimeError("COPILOT_LINEAR_API_KEY is not configured")
    if not secrets.linear_feature_request_project_id:
        raise RuntimeError("LINEAR_FEATURE_REQUEST_PROJECT_ID is not configured")
    if not secrets.linear_feature_request_team_id:
        raise RuntimeError("LINEAR_FEATURE_REQUEST_TEAM_ID is not configured")

    credentials = APIKeyCredentials(
        id="system-linear",
        provider="linear",
        api_key=SecretStr(secrets.copilot_linear_api_key),
        title="System Linear API Key",
    )
    client = LinearClient(credentials=credentials)
    return (
        client,
        secrets.linear_feature_request_project_id,
        secrets.linear_feature_request_team_id,
    )


class SearchFeatureRequestsTool(BaseTool):
    """Tool for searching existing feature requests in Linear."""

    @property
    def name(self) -> str:
        return "search_feature_requests"

    @property
    def description(self) -> str:
        return (
            "Search existing feature requests to check if a similar request "
            "already exists before creating a new one. Returns matching feature "
            "requests with their ID, title, and description."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term to find matching feature requests.",
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        query = kwargs.get("query", "").strip()
        session_id = session.session_id if session else None

        if not query:
            return ErrorResponse(
                message="Please provide a search query.",
                error="Missing query parameter",
                session_id=session_id,
            )

        try:
            client, project_id, _team_id = _get_linear_config()
            data = await client.query(
                SEARCH_ISSUES_QUERY,
                {
                    "term": query,
                    "filter": {
                        "project": {"id": {"eq": project_id}},
                    },
                    "first": MAX_SEARCH_RESULTS,
                },
            )

            nodes = data.get("searchIssues", {}).get("nodes", [])

            if not nodes:
                return NoResultsResponse(
                    message=f"No feature requests found matching '{query}'.",
                    suggestions=[
                        "Try different keywords",
                        "Use broader search terms",
                        "You can create a new feature request if none exists",
                    ],
                    session_id=session_id,
                )

            results = [
                FeatureRequestInfo(
                    id=node["id"],
                    identifier=node["identifier"],
                    title=node["title"],
                )
                for node in nodes
            ]

            return FeatureRequestSearchResponse(
                message=f"Found {len(results)} feature request(s) matching '{query}'.",
                results=results,
                count=len(results),
                query=query,
                session_id=session_id,
            )
        except Exception as e:
            logger.exception("Failed to search feature requests")
            return ErrorResponse(
                message="Failed to search feature requests.",
                error=str(e),
                session_id=session_id,
            )


class CreateFeatureRequestTool(BaseTool):
    """Tool for creating feature requests (or adding needs to existing ones)."""

    @property
    def name(self) -> str:
        return "create_feature_request"

    @property
    def description(self) -> str:
        return (
            "Create a new feature request or add a customer need to an existing one. "
            "Always search first with search_feature_requests to avoid duplicates. "
            "If a matching request exists, pass its ID as existing_issue_id to add "
            "the user's need to it instead of creating a duplicate. "
            "IMPORTANT: Never include personally identifiable information (PII) in "
            "the title or description — no names, emails, phone numbers, company "
            "names, or other identifying details. Write titles and descriptions in "
            "generic, feature-focused language."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Title for the feature request. Must be generic and "
                        "feature-focused — do not include any user names, emails, "
                        "company names, or other PII."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Detailed description of what the user wants and why. "
                        "Must not contain any personally identifiable information "
                        "(PII) — describe the feature need generically without "
                        "referencing specific users, companies, or contact details."
                    ),
                },
                "existing_issue_id": {
                    "type": "string",
                    "description": (
                        "If adding a need to an existing feature request, "
                        "provide its Linear issue ID (from search results). "
                        "Omit to create a new feature request."
                    ),
                },
            },
            "required": ["title", "description"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _find_or_create_customer(
        self, client: LinearClient, user_id: str, name: str
    ) -> dict:
        """Find existing customer by user_id or create a new one via upsert.

        Args:
            client: Linear API client.
            user_id: Stable external ID used to deduplicate customers.
            name: Human-readable display name (e.g. the user's email).
        """
        data = await client.mutate(
            CUSTOMER_UPSERT_MUTATION,
            {
                "input": {
                    "name": name,
                    "externalId": user_id,
                },
            },
        )
        result = data.get("customerUpsert", {})
        if not result.get("success"):
            raise RuntimeError(f"Failed to upsert customer: {data}")
        return result["customer"]

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        title = kwargs.get("title", "").strip()
        description = kwargs.get("description", "").strip()
        existing_issue_id = kwargs.get("existing_issue_id")
        session_id = session.session_id if session else None

        if not title or not description:
            return ErrorResponse(
                message="Both title and description are required.",
                error="Missing required parameters",
                session_id=session_id,
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required to create feature requests.",
                error="Missing user_id",
                session_id=session_id,
            )

        try:
            client, project_id, team_id = _get_linear_config()
        except Exception as e:
            logger.exception("Failed to initialize Linear client")
            return ErrorResponse(
                message="Failed to create feature request.",
                error=str(e),
                session_id=session_id,
            )

        # Resolve a human-readable name (email) for the Linear customer record.
        # Fall back to user_id if the lookup fails or returns None.
        try:
            customer_display_name = (
                await user_db().get_user_email_by_id(user_id) or user_id
            )
        except Exception:
            customer_display_name = user_id

        # Step 1: Find or create customer for this user
        try:
            customer = await self._find_or_create_customer(
                client, user_id, customer_display_name
            )
            customer_id = customer["id"]
            customer_name = customer["name"]
        except Exception as e:
            logger.exception("Failed to upsert customer in Linear")
            return ErrorResponse(
                message="Failed to create feature request.",
                error=str(e),
                session_id=session_id,
            )

        # Step 2: Create or reuse issue
        issue_id: str | None = None
        issue_identifier: str | None = None
        if existing_issue_id:
            # Add need to existing issue - we still need the issue details for response
            is_new_issue = False
            issue_id = existing_issue_id
        else:
            # Create new issue in the feature requests project
            try:
                data = await client.mutate(
                    ISSUE_CREATE_MUTATION,
                    {
                        "input": {
                            "title": title,
                            "description": description,
                            "teamId": team_id,
                            "projectId": project_id,
                        },
                    },
                )
                result = data.get("issueCreate", {})
                if not result.get("success"):
                    return ErrorResponse(
                        message="Failed to create feature request issue.",
                        error=str(data),
                        session_id=session_id,
                    )
                issue = result["issue"]
                issue_id = issue["id"]
                issue_identifier = issue.get("identifier")
            except Exception as e:
                logger.exception("Failed to create feature request issue")
                return ErrorResponse(
                    message="Failed to create feature request.",
                    error=str(e),
                    session_id=session_id,
                )
            is_new_issue = True

        # Step 3: Create customer need on the issue
        try:
            data = await client.mutate(
                CUSTOMER_NEED_CREATE_MUTATION,
                {
                    "input": {
                        "customerId": customer_id,
                        "issueId": issue_id,
                        "body": description,
                        "priority": 0,
                    },
                },
            )
            need_result = data.get("customerNeedCreate", {})
            if not need_result.get("success"):
                orphaned = (
                    {"issue_id": issue_id, "issue_identifier": issue_identifier}
                    if is_new_issue
                    else None
                )
                return ErrorResponse(
                    message="Failed to attach customer need to the feature request.",
                    error=str(data),
                    details=orphaned,
                    session_id=session_id,
                )
            need = need_result["need"]
            issue_info = need["issue"]
        except Exception as e:
            logger.exception("Failed to create customer need")
            orphaned = (
                {"issue_id": issue_id, "issue_identifier": issue_identifier}
                if is_new_issue
                else None
            )
            return ErrorResponse(
                message="Failed to attach customer need to the feature request.",
                error=str(e),
                details=orphaned,
                session_id=session_id,
            )

        return FeatureRequestCreatedResponse(
            message=(
                f"{'Created new feature request' if is_new_issue else 'Added your request to existing feature request'}: "
                f"{issue_info['title']}."
            ),
            issue_id=issue_info["id"],
            issue_identifier=issue_info["identifier"],
            issue_title=issue_info["title"],
            issue_url=issue_info.get("url", ""),
            is_new_issue=is_new_issue,
            customer_name=customer_name,
            session_id=session_id,
        )
