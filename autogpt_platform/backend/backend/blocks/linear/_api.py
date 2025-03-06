from __future__ import annotations

import json
from typing import Any, Dict, Optional

from backend.blocks.linear._auth import LinearCredentials
from backend.blocks.linear.models import (
    CreateCommentResponse,
    CreateIssueResponse,
    Issue,
    Project,
)
from backend.util.request import Requests


class LinearAPIException(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class LinearClient:
    """Client for the Linear API

    If you're looking for the schema: https://studio.apollographql.com/public/Linear-API/variant/current/schema
    """

    API_URL = "https://api.linear.app/graphql"

    def __init__(
        self,
        credentials: LinearCredentials | None = None,
        custom_requests: Optional[Requests] = None,
    ):
        if custom_requests:
            self._requests = custom_requests
        else:

            headers: Dict[str, str] = {
                "Content-Type": "application/json",
            }
            if credentials:
                headers["Authorization"] = credentials.auth_header()

            self._requests = Requests(
                extra_headers=headers,
                trusted_origins=["https://api.linear.app"],
                raise_for_status=False,
            )

    def _execute_graphql_request(
        self, query: str, variables: dict | None = None
    ) -> Any:
        """
        Executes a GraphQL request against the Linear API and returns the response data.

        Args:
            query: The GraphQL query string.
            variables (optional): Any GraphQL query variables

        Returns:
            The parsed JSON response data, or raises a LinearAPIException on error.
        """
        payload: Dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        response = self._requests.post(self.API_URL, json=payload)

        if not response.ok:

            try:
                error_data = response.json()
                error_message = error_data.get("errors", [{}])[0].get("message", "")
            except json.JSONDecodeError:
                error_message = response.text

            raise LinearAPIException(
                f"Linear API request failed ({response.status_code}): {error_message}",
                response.status_code,
            )

        response_data = response.json()
        if "errors" in response_data:

            error_messages = [
                error.get("message", "") for error in response_data["errors"]
            ]
            raise LinearAPIException(
                f"Linear API returned errors: {', '.join(error_messages)}",
                response.status_code,
            )

        return response_data["data"]

    def query(self, query: str, variables: Optional[dict] = None) -> dict:
        """Executes a GraphQL query.

        Args:
            query: The GraphQL query string.
            variables: Query variables, if any.

        Returns:
             The response data.
        """
        return self._execute_graphql_request(query, variables)

    def mutate(self, mutation: str, variables: Optional[dict] = None) -> dict:
        """Executes a GraphQL mutation.

        Args:
            mutation: The GraphQL mutation string.
            variables: Query variables, if any.

        Returns:
            The response data.
        """
        return self._execute_graphql_request(mutation, variables)

    def try_create_comment(self, issue_id: str, comment: str) -> CreateCommentResponse:
        try:
            mutation = """
                mutation CommentCreate($input: CommentCreateInput!) {
                  commentCreate(input: $input) {
                    success
                    comment {
                      id
                      body
                    }
              }
            }
        """

            variables = {
                "input": {
                    "body": comment,
                    "issueId": issue_id,
                }
            }

            added_comment = self.mutate(mutation, variables)
            # Select the commentCreate field from the mutation response
            return CreateCommentResponse(**added_comment["commentCreate"])
        except LinearAPIException as e:
            raise e

    def try_get_team_by_name(self, team_name: str) -> str:
        try:
            query = """
            query GetTeamId($searchTerm: String!) {
              teams(filter: { 
                or: [
                  { name: { eqIgnoreCase: $searchTerm } },
                  { key: { eqIgnoreCase: $searchTerm } }
                ]
              }) {
                nodes {
                  id
                  name
                  key
                }
              }
            }
            """

            variables: dict[str, Any] = {
                "searchTerm": team_name,
            }

            team_id = self.query(query, variables)
            return team_id["teams"]["nodes"][0]["id"]
        except LinearAPIException as e:
            raise e

    def try_create_issue(
        self,
        team_id: str,
        title: str,
        description: str | None = None,
        priority: int | None = None,
        project_id: str | None = None,
    ) -> CreateIssueResponse:
        try:
            mutation = """
               mutation IssueCreate($input: IssueCreateInput!) {
                issueCreate(input: $input) {
                  issue {
                    title
                    description
                    id
                    identifier
                    priority
                  }
                }
            }
            """

            variables: dict[str, Any] = {
                "input": {
                    "teamId": team_id,
                    "title": title,
                }
            }

            if project_id:
                variables["input"]["projectId"] = project_id

            if description:
                variables["input"]["description"] = description

            if priority:
                variables["input"]["priority"] = priority

            added_issue = self.mutate(mutation, variables)
            return CreateIssueResponse(**added_issue["issueCreate"])
        except LinearAPIException as e:
            raise e

    def try_search_projects(self, term: str) -> list[Project]:
        try:
            query = """
                query SearchProjects($term: String!, $includeComments: Boolean!) {
                    searchProjects(term: $term, includeComments: $includeComments) {
                        nodes {
                            id
                            name
                            description
                            priority
                            progress
                            content
                        }
                    }
                }
            """

            variables: dict[str, Any] = {
                "term": term,
                "includeComments": True,
            }

            projects = self.query(query, variables)
            return [
                Project(**project) for project in projects["searchProjects"]["nodes"]
            ]
        except LinearAPIException as e:
            raise e

    def try_search_issues(self, term: str) -> list[Issue]:
        try:
            query = """
                query SearchIssues($term: String!, $includeComments: Boolean!) {
                    searchIssues(term: $term, includeComments: $includeComments) {
                        nodes {
                            id
                            identifier
                            title
                            description
                            priority
                        }
                    }
                }
            """

            variables: dict[str, Any] = {
                "term": term,
                "includeComments": True,
            }

            issues = self.query(query, variables)
            return [Issue(**issue) for issue in issues["searchIssues"]["nodes"]]
        except LinearAPIException as e:
            raise e
