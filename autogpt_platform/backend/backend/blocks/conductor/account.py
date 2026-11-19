import asyncio
from collections.abc import Awaitable, Callable
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

from ._api import PAGE_SIZE, ConductorClient
from ._config import conductor


async def paginate(
    fetch: Callable[[int, int], Awaitable[dict[str, Any]]], limit: int
) -> list[dict[str, Any]]:
    """Walk a `{data, offset, hasMore}` listing until `limit` items or the end."""
    items: list[dict[str, Any]] = []
    offset = 0
    while len(items) < limit:
        page = await fetch(min(PAGE_SIZE, limit - len(items)), offset)
        data = page.get("data") or []
        items.extend(data)
        if not page.get("hasMore") or not data:
            break
        offset += len(data)
    return items[:limit]


class ConductorGetAccountBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description="Conductor API key from app.conductor.build/users/api-keys"
        )
        limit: int = SchemaField(
            description="Maximum number of projects, sections and routines to list",
            default=100,
            ge=1,
            le=1000,
        )

    class Output(BlockSchemaOutput):
        user: dict = SchemaField(
            description="The authenticated identity: userId, name, email, organizationId"
        )
        projects: list[dict] = SchemaField(
            description="Repositories you can create workspaces in: id, name, gitRemote"
        )
        sections: list[dict] = SchemaField(
            description="Your cloud sections: id, name, emoji, workspaceIds"
        )
        routines: list[dict] = SchemaField(
            description="Your routines: id, name, prompt, repoUrl, agent, model, "
            "enabled, runCount, triggers (webhook URLs are not included)"
        )

    def __init__(self):
        super().__init__(
            id="f0f522e1-99eb-4bf3-b394-202494a35189",
            description="Get an overview of your Conductor account in one call: who "
            "you are, the projects (repositories) you can open workspaces in, your "
            "sections and your routines. Use this first to find project IDs.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.READ,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": conductor.get_test_credentials().model_dump(),
            },
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("user", {"userId": "user_1", "authMethod": "api-key"}),
                ("projects", [{"id": "proj_1", "name": "autogpt", "gitRemote": "x"}]),
                ("sections", []),
                ("routines", []),
            ],
            test_mock={
                "_fetch": lambda *args, **kwargs: {
                    "user": {"userId": "user_1", "authMethod": "api-key"},
                    "projects": [{"id": "proj_1", "name": "autogpt", "gitRemote": "x"}],
                    "sections": [],
                    "routines": [],
                }
            },
        )

    async def _fetch(
        self, credentials: APIKeyCredentials, limit: int
    ) -> dict[str, Any]:
        client = ConductorClient(credentials)
        user, projects, sections, routines = await asyncio.gather(
            client.get_me(),
            paginate(client.list_projects, limit),
            paginate(client.list_sections, limit),
            paginate(client.list_routines, limit),
        )
        return {
            "user": user,
            "projects": projects,
            "sections": sections,
            "routines": routines,
        }

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            data = await self._fetch(credentials, input_data.limit)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Get account failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        yield "user", data["user"]
        yield "projects", data["projects"]
        yield "sections", data["sections"]
        yield "routines", data["routines"]
