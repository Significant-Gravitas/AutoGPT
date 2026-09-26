from datetime import UTC, datetime
from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._api import ConductorClient, WorkspaceState
from ._config import conductor

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorListWorkspacesBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        project_id: str = SchemaField(
            description="Only workspaces of this project (repository). Leave empty "
            "for all projects.",
            default="",
            advanced=False,
        )
        state: list[WorkspaceState] = SchemaField(
            description="Only workspaces in these states, e.g. ready, sleeping",
            default_factory=list,
            advanced=False,
        )
        name: str = SchemaField(
            description="Filter by workspace name", default="", advanced=False
        )
        repo: str = SchemaField(description="Filter by repository URL", default="")
        creator: str = SchemaField(description="Filter by creator user ID", default="")
        since: str = SchemaField(
            description="Only workspaces whose last activity is on or after this "
            "ISO-8601 date or timestamp",
            default="",
        )
        include_archived: bool = SchemaField(
            description="Include archived workspaces", default=False
        )
        limit: int = SchemaField(
            description="Maximum number of workspaces", default=50, ge=1, le=500
        )
        offset: int = SchemaField(description="Pagination offset", default=0, ge=0)

    class Output(BlockSchemaOutput):
        workspaces: list[dict] = SchemaField(
            description="Workspaces: id, projectId, name, state, repoUrl, deepLink, "
            "creatorName, lastActivityAt"
        )
        workspace: dict = SchemaField(description="Each workspace, one at a time")
        has_more: bool = SchemaField(description="Whether more pages exist")
        next_offset: int = SchemaField(
            description="Offset to request the next page; with project_id a page "
            "can be filtered down to nothing while has_more is still true"
        )

    def __init__(self):
        super().__init__(
            id="2410943a-0e63-47d9-a205-9f0dc8fc9b9b",
            description="List Conductor workspaces, optionally filtered by project, "
            "state, name, repository, creator or activity date.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.READ,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": conductor.get_test_credentials().model_dump(),
                "state": ["ready"],
            },
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("workspaces", lambda ws: len(ws) == 1 and ws[0]["id"] == "ws_1"),
                ("workspace", lambda w: w["id"] == "ws_1"),
                ("has_more", False),
                ("next_offset", 1),
            ],
            test_mock={
                "_fetch": lambda *args, **kwargs: {
                    "data": [
                        {
                            "id": "ws_1",
                            "name": "fix-login",
                            "state": "ready",
                            "repoUrl": "https://github.com/x/y",
                            "createdAt": "2026-09-26T00:00:00Z",
                            "deepLink": "conductor://workspace/ws_1",
                        }
                    ],
                    "offset": 0,
                    "hasMore": False,
                }
            },
        )

    async def _fetch(
        self, credentials: APIKeyCredentials, input_data: Input
    ) -> dict[str, Any]:
        client = ConductorClient(credentials)
        if input_data.project_id:
            # The per-project route accepts no filters; see run().
            params: dict[str, Any] = {
                "limit": input_data.limit,
                "offset": input_data.offset,
            }
        else:
            params = {
                "limit": input_data.limit,
                "offset": input_data.offset,
                "includeArchived": input_data.include_archived,
                "state": [s.value for s in input_data.state],
                "name": input_data.name,
                "repo": input_data.repo,
                "creator": input_data.creator,
                "since": input_data.since,
            }
        return await client.list_workspaces(params, input_data.project_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        since = _parse_timestamp(input_data.since)
        if input_data.since and since is None:
            raise BlockInputError(
                message="since must be an ISO-8601 date or timestamp",
                block_name=self.name,
                block_id=self.id,
            )
        try:
            listing = await self._fetch(credentials, input_data)
        except Exception as e:
            raise BlockExecutionError(
                message=f"List workspaces failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        workspaces = list(listing.get("data") or [])
        if input_data.project_id:
            # The per-project route only paginates, so every filter the
            # cross-project route would apply is applied to the page here.
            workspaces = [w for w in workspaces if _matches(w, input_data, since)]

        yield "workspaces", workspaces
        for workspace in workspaces:
            yield "workspace", workspace
        yield "has_more", bool(listing.get("hasMore", False))
        yield "next_offset", input_data.offset + len(listing.get("data") or [])


def _matches(
    workspace: dict[str, Any],
    input_data: ConductorListWorkspacesBlock.Input,
    since: datetime | None,
) -> bool:
    """Mirror the cross-project route's filters: exact state and creator,
    case-insensitive name and repository substrings, activity since a date,
    and archived workspaces hidden unless asked for explicitly."""
    state = str(workspace.get("state") or "")
    wanted = {s.value for s in input_data.state}
    if wanted and state not in wanted:
        return False
    if not wanted and not input_data.include_archived and state == "archived":
        return False
    if input_data.creator and workspace.get("creatorId") != input_data.creator:
        return False
    name = str(workspace.get("name") or "").lower()
    if input_data.name and input_data.name.lower() not in name:
        return False
    repo = str(workspace.get("repoUrl") or "").lower()
    if input_data.repo and input_data.repo.lower() not in repo:
        return False
    if since is None:
        return True
    last_activity = _parse_timestamp(str(workspace.get("lastActivityAt") or ""))
    return last_activity is not None and last_activity >= since


def _parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO-8601 date or timestamp; naive values are taken as UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
