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
from backend.util.exceptions import BlockExecutionError

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
            description="Only workspaces active since this ISO-8601 timestamp",
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
        next_offset: int = SchemaField(description="Offset to request the next page")

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
            params: dict[str, Any] = {
                "limit": input_data.limit,
                "offset": input_data.offset,
                "includeArchived": input_data.include_archived,
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
            # The per-project route only paginates; apply the other filters here.
            wanted = {s.value for s in input_data.state}
            needle = input_data.name.lower()
            workspaces = [
                w
                for w in workspaces
                if (not wanted or w.get("state") in wanted)
                and (not needle or needle in str(w.get("name", "")).lower())
            ]

        yield "workspaces", workspaces
        for workspace in workspaces:
            yield "workspace", workspace
        yield "has_more", bool(listing.get("hasMore", False))
        yield "next_offset", input_data.offset + len(listing.get("data") or [])
