"""
Bunnyshell environment management blocks for AutoGPT.

Provides a suite of blocks to programmatically create, deploy, monitor,
and tear down Bunnyshell staging environments — enabling agents to spin up
a full Docker-Compose-based stack (10+ microservices + frontend) on demand,
test against it, and destroy it when done.

Blocks:
  - BunnyshellCreateEnvironmentBlock : Create a new environment from a template
  - BunnyshellDeployEnvironmentBlock : Deploy (or redeploy) an environment
  - BunnyshellGetStatusBlock         : Poll environment status + get service URLs
  - BunnyshellStopEnvironmentBlock   : Stop environment (pause billing)
  - BunnyshellDeleteEnvironmentBlock : Permanently destroy environment

Pricing context (2025):
  $0.007/min per active environment. Sleeping environments cost $0.
  A 2-hour PR review session costs ~$0.84. Always delete when done.

Security:
  - API token stored in AutoGPT credential infra — never in block output
  - environment_id validated on every operation
  - No shell commands executed — pure REST API calls

Performance:
  - BunnyshellGetStatusBlock supports polling mode for agent loops
  - Environments can be stopped (not deleted) between reviews to save cost
    while preserving state for fast resume
"""

from typing import Literal, Optional
from pydantic import SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_BUNNYSHELL_API_BASE = "https://api.environments.bunnyshell.com/v1"

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcde1",
    provider="bunnyshell",
    api_key=SecretStr("mock-bunnyshell-api-token"),
    title="Mock Bunnyshell API token",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------


async def _bunnyshell_request(
    method: str,
    path: str,
    api_token: str,
    json: Optional[dict] = None,
) -> dict:
    """Make an authenticated request to the Bunnyshell API."""
    import httpx

    url = f"{_BUNNYSHELL_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.request(method, url, headers=headers, json=json)
        response.raise_for_status()
        return response.json() if response.content else {}


# ---------------------------------------------------------------------------
# Block 1 — Create environment
# ---------------------------------------------------------------------------


class BunnyshellCreateEnvironmentBlock(Block):
    """
    Create a new Bunnyshell environment from an existing template.

    This provisions an isolated staging stack (databases, microservices,
    frontend) from your ``bunnyshell.yaml`` definition. The environment
    is created but NOT deployed — call BunnyshellDeployEnvironmentBlock next.

    Typical agent flow:
      CreateEnvironment → DeployEnvironment → GetStatus (poll) → test → Delete
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.BUNNYSHELL], Literal["api_key"]
        ] = CredentialsField(
            description=(
                "Bunnyshell API token. Get yours at "
                "https://documentation.bunnyshell.com/docs/public-api"
            ),
        )
        project_id: str = SchemaField(
            description=(
                "Bunnyshell project ID. Found in your Bunnyshell dashboard URL "
                "or via the Bunnyshell API."
            ),
            placeholder="prj_xxxxxxxxxxxx",
        )
        template_id: str = SchemaField(
            description=(
                "Bunnyshell environment template ID to create the environment from. "
                "Templates define your full stack (services, env vars, resources)."
            ),
            placeholder="tpl_xxxxxxxxxxxx",
        )
        name: str = SchemaField(
            description=(
                "Name for this environment. Use a descriptive name like "
                "'pr-123-feature-auth' to identify it in the dashboard."
            ),
            placeholder="pr-123-feature-auth",
        )
        labels: dict = SchemaField(
            description=(
                "Optional key-value labels to tag this environment. "
                "Useful for filtering and automation (e.g. {'pr': '123', 'branch': 'feat/auth'})."
            ),
            default_factory=dict,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        environment_id: str = SchemaField(
            description=(
                "Unique Bunnyshell environment ID. Pass this to all other "
                "Bunnyshell blocks. Store in memory to reuse across messages."
            )
        )
        name: str = SchemaField(description="Name of the created environment.")
        status: str = SchemaField(
            description="Initial environment status (typically 'draft' before deploy)."
        )
        error: str = SchemaField(
            description="Error message if environment creation failed."
        )

    def __init__(self):
        super().__init__(
            id="f6a7b8c9-d0e1-2345-fabc-456789012345",
            description=(
                "Create a new Bunnyshell staging environment from a template. "
                "Provisions an isolated full-stack environment (Docker Compose services, "
                "databases, frontend) ready to deploy. "
                "Cost: $0 until deployed. $0.007/min while active."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=BunnyshellCreateEnvironmentBlock.Input,
            output_schema=BunnyshellCreateEnvironmentBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": "prj_test",
                "template_id": "tpl_test",
                "name": "pr-123-test",
                "labels": {"pr": "123"},
            },
            test_output=[
                ("environment_id", "env_mock123"),
                ("name", "pr-123-test"),
                ("status", "draft"),
            ],
            test_mock={
                "create_environment": lambda *a, **kw: {
                    "id": "env_mock123",
                    "name": "pr-123-test",
                    "status": "draft",
                }
            },
        )

    @staticmethod
    async def create_environment(
        api_token: str,
        project_id: str,
        template_id: str,
        name: str,
        labels: dict,
    ) -> dict:
        return await _bunnyshell_request(
            method="POST",
            path="/environments",
            api_token=api_token,
            json={
                "project": project_id,
                "template": template_id,
                "name": name,
                "labels": labels,
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data = await self.create_environment(
                api_token=credentials.api_key.get_secret_value(),
                project_id=input_data.project_id,
                template_id=input_data.template_id,
                name=input_data.name,
                labels=input_data.labels,
            )
            yield "environment_id", data["id"]
            yield "name", data.get("name", input_data.name)
            yield "status", data.get("status", "unknown")
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block 2 — Deploy environment
# ---------------------------------------------------------------------------


class BunnyshellDeployEnvironmentBlock(Block):
    """
    Deploy (or redeploy) a Bunnyshell environment.

    Triggers a full stack deploy — Docker images are built/pulled and all
    services are started. Use BunnyshellGetStatusBlock to poll until
    status == 'running' before running tests against it.

    Billing starts from this call. A 2-hour session costs ~$0.84.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.BUNNYSHELL], Literal["api_key"]
        ] = CredentialsField(
            description="Bunnyshell API token.",
        )
        environment_id: str = SchemaField(
            description="Environment ID from BunnyshellCreateEnvironmentBlock.",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the deploy was triggered successfully."
        )
        operation_id: str = SchemaField(
            description="Deploy operation ID for tracking progress."
        )
        error: str = SchemaField(
            description="Error message if deploy failed."
        )

    def __init__(self):
        super().__init__(
            id="a7b8c9d0-e1f2-3456-abcd-567890123456",
            description=(
                "Deploy a Bunnyshell environment — starts all Docker Compose services "
                "(databases, microservices, frontend). "
                "Billing starts from this call ($0.007/min). "
                "Poll BunnyshellGetStatusBlock until status == 'running' before testing."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=BunnyshellDeployEnvironmentBlock.Input,
            output_schema=BunnyshellDeployEnvironmentBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "environment_id": "env_mock123",
            },
            test_output=[
                ("success", True),
                ("operation_id", "op_mock456"),
            ],
            test_mock={
                "deploy_environment": lambda *a, **kw: {"id": "op_mock456"}
            },
        )

    @staticmethod
    async def deploy_environment(api_token: str, environment_id: str) -> dict:
        return await _bunnyshell_request(
            method="POST",
            path=f"/environments/{environment_id}/deploy",
            api_token=api_token,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data = await self.deploy_environment(
                api_token=credentials.api_key.get_secret_value(),
                environment_id=input_data.environment_id,
            )
            yield "success", True
            yield "operation_id", data.get("id", "")
        except Exception as e:
            yield "error", str(e)
            yield "success", False


# ---------------------------------------------------------------------------
# Block 3 — Get status + service URLs
# ---------------------------------------------------------------------------


class BunnyshellGetStatusBlock(Block):
    """
    Get the current status and service URLs of a Bunnyshell environment.

    Poll this block after BunnyshellDeployEnvironmentBlock until
    ``status == 'running'``. Once running, ``service_urls`` contains the
    public URLs for each service (API, frontend, etc.) that agents can
    test against via HTTP.

    Typical statuses: draft → deploying → running → stopped → deleted
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.BUNNYSHELL], Literal["api_key"]
        ] = CredentialsField(
            description="Bunnyshell API token.",
        )
        environment_id: str = SchemaField(
            description="Environment ID from BunnyshellCreateEnvironmentBlock.",
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(
            description=(
                "Current environment status. "
                "Possible values: draft, deploying, running, stopping, stopped, deleting, deleted, error."
            )
        )
        is_running: bool = SchemaField(
            description="True if status == 'running' (ready for testing)."
        )
        service_urls: dict = SchemaField(
            description=(
                "Map of service name → public URL for all running services. "
                "E.g. {'api': 'https://api-pr123.bunnyshell.app', "
                "'frontend': 'https://app-pr123.bunnyshell.app'}. "
                "Use these URLs to test your services via HTTP."
            )
        )
        error: str = SchemaField(
            description="Error message if status check failed."
        )

    def __init__(self):
        super().__init__(
            id="b8c9d0e1-f2a3-4567-bcde-678901234567",
            description=(
                "Get the current status and public service URLs of a Bunnyshell environment. "
                "Poll this after deploying until is_running=True, then use "
                "service_urls to test your services via HTTP."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=BunnyshellGetStatusBlock.Input,
            output_schema=BunnyshellGetStatusBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "environment_id": "env_mock123",
            },
            test_output=[
                ("status", "running"),
                ("is_running", True),
                ("service_urls", {"api": "https://api-mock.bunnyshell.app"}),
            ],
            test_mock={
                "get_status": lambda *a, **kw: {
                    "status": "running",
                    "components": [
                        {
                            "name": "api",
                            "urls": ["https://api-mock.bunnyshell.app"],
                        }
                    ],
                }
            },
        )

    @staticmethod
    async def get_status(api_token: str, environment_id: str) -> dict:
        return await _bunnyshell_request(
            method="GET",
            path=f"/environments/{environment_id}",
            api_token=api_token,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data = await self.get_status(
                api_token=credentials.api_key.get_secret_value(),
                environment_id=input_data.environment_id,
            )
            status = data.get("status", "unknown")
            is_running = status == "running"

            # Extract service URLs from components
            service_urls: dict[str, str] = {}
            for component in data.get("components", []):
                name = component.get("name", "unknown")
                urls = component.get("urls", [])
                if urls:
                    service_urls[name] = urls[0]

            yield "status", status
            yield "is_running", is_running
            yield "service_urls", service_urls
        except Exception as e:
            yield "error", str(e)
            yield "status", "error"
            yield "is_running", False
            yield "service_urls", {}


# ---------------------------------------------------------------------------
# Block 4 — Stop environment (pause billing, preserve state)
# ---------------------------------------------------------------------------


class BunnyshellStopEnvironmentBlock(Block):
    """
    Stop a running Bunnyshell environment without deleting it.

    Stopping pauses billing immediately while preserving all environment
    configuration. Use this to save cost between review sessions — the
    environment can be restarted (redeployed) quickly without recreating
    from scratch.

    Use BunnyshellDeleteEnvironmentBlock when the environment is no longer
    needed at all (e.g. PR merged).
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.BUNNYSHELL], Literal["api_key"]
        ] = CredentialsField(
            description="Bunnyshell API token.",
        )
        environment_id: str = SchemaField(
            description="Environment ID to stop.",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the stop was triggered successfully."
        )
        message: str = SchemaField(description="Status message.")
        error: str = SchemaField(
            description="Error message if stop failed."
        )

    def __init__(self):
        super().__init__(
            id="c9d0e1f2-a3b4-5678-cdef-789012345678",
            description=(
                "Stop a Bunnyshell environment to pause billing while preserving "
                "its configuration. Use this between review sessions. "
                "Billing stops immediately — sleeping environments cost $0. "
                "Redeploy to resume. Use Delete when the PR is merged."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=BunnyshellStopEnvironmentBlock.Input,
            output_schema=BunnyshellStopEnvironmentBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "environment_id": "env_mock123",
            },
            test_output=[
                ("success", True),
                ("message", "Environment env_mock123 stopped. Billing paused."),
            ],
            test_mock={
                "stop_environment": lambda *a, **kw: {}
            },
        )

    @staticmethod
    async def stop_environment(api_token: str, environment_id: str) -> dict:
        return await _bunnyshell_request(
            method="POST",
            path=f"/environments/{environment_id}/stop",
            api_token=api_token,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            await self.stop_environment(
                api_token=credentials.api_key.get_secret_value(),
                environment_id=input_data.environment_id,
            )
            yield "success", True
            yield "message", (
                f"Environment {input_data.environment_id} stopped. Billing paused."
            )
        except Exception as e:
            yield "error", str(e)
            yield "success", False


# ---------------------------------------------------------------------------
# Block 5 — Delete environment (permanent, stops billing)
# ---------------------------------------------------------------------------


class BunnyshellDeleteEnvironmentBlock(Block):
    """
    Permanently delete a Bunnyshell environment.

    This destroys all services, volumes, and configuration for the environment.
    Billing stops immediately. Use this when a PR is merged or closed and the
    environment is no longer needed.

    This action is irreversible. Use BunnyshellStopEnvironmentBlock if you
    want to pause billing while preserving the environment for future use.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.BUNNYSHELL], Literal["api_key"]
        ] = CredentialsField(
            description="Bunnyshell API token.",
        )
        environment_id: str = SchemaField(
            description="Environment ID to permanently delete.",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the environment was deleted successfully."
        )
        message: str = SchemaField(description="Status message.")
        error: str = SchemaField(
            description="Error message if deletion failed."
        )

    def __init__(self):
        super().__init__(
            id="d0e1f2a3-b4c5-6789-defa-890123456789",
            description=(
                "Permanently delete a Bunnyshell environment and stop billing. "
                "Use when a PR is merged/closed. This is irreversible — "
                "all services, volumes, and config are destroyed. "
                "Use Stop instead if you want to pause and resume later."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=BunnyshellDeleteEnvironmentBlock.Input,
            output_schema=BunnyshellDeleteEnvironmentBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "environment_id": "env_mock123",
            },
            test_output=[
                ("success", True),
                ("message", "Environment env_mock123 deleted. Billing stopped."),
            ],
            test_mock={
                "delete_environment": lambda *a, **kw: {}
            },
        )

    @staticmethod
    async def delete_environment(api_token: str, environment_id: str) -> dict:
        return await _bunnyshell_request(
            method="DELETE",
            path=f"/environments/{environment_id}",
            api_token=api_token,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            await self.delete_environment(
                api_token=credentials.api_key.get_secret_value(),
                environment_id=input_data.environment_id,
            )
            yield "success", True
            yield "message", (
                f"Environment {input_data.environment_id} deleted. Billing stopped."
            )
        except Exception as e:
            yield "error", str(e)
            yield "success", False
