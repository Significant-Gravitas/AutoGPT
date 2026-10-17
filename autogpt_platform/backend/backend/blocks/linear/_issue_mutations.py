from pydantic import BaseModel

from ._api import LinearClient
from ._issue_updates import IssueChanges
from .models import Issue


class LinearIssueClient(LinearClient):
    async def update_issue(self, issue_id: str, changes: IssueChanges) -> Issue | None:
        mutation = """
            mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
                issueUpdate(id: $id, input: $input) {
                    success
                    issue {
                        id identifier title description priority
                        state { id name type }
                        assignee { id name }
                    }
                }
            }
        """
        result = await self.mutate(
            mutation, {"id": issue_id, "input": changes.to_api_input()}
        )
        payload = IssueUpdatePayload.model_validate(result["issueUpdate"])
        return payload.issue if payload.success else None

    async def archive_issue(self, issue_id: str) -> bool:
        mutation = """
            mutation ArchiveIssue($id: String!) {
                issueArchive(id: $id) { success }
            }
        """
        result = await self.mutate(mutation, {"id": issue_id})
        return IssueActionPayload.model_validate(result["issueArchive"]).success

    async def delete_issue(self, issue_id: str) -> bool:
        mutation = """
            mutation DeleteIssue($id: String!) {
                issueDelete(id: $id) { success }
            }
        """
        result = await self.mutate(mutation, {"id": issue_id})
        return IssueActionPayload.model_validate(result["issueDelete"]).success


class IssueActionPayload(BaseModel):
    success: bool


class IssueUpdatePayload(IssueActionPayload):
    issue: Issue | None = None
