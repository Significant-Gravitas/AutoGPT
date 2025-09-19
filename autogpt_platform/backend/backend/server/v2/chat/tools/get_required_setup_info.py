"""Tool for getting required setup information for an agent."""

import logging
from typing import Any

from backend.data import graph as graph_db
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.sdk.registry import AutoRegistry
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    ErrorResponse,
    ExecutionModeInfo,
    InputField,
    SetupInfo,
    SetupRequirementInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
)
from backend.server.v2.store import db as store_db

logger = logging.getLogger(__name__)


class GetRequiredSetupInfoTool(BaseTool):
    """Tool for getting required setup information including credentials and inputs."""

    @property
    def name(self) -> str:
        return "get_required_setup_info"

    @property
    def description(self) -> str:
        return """Check if an agent can be set up with the provided input data and credentials.
        Call this AFTER get_agent_details to validate that you have all required inputs.
        Pass the input dictionary you plan to use with run_agent or setup_agent to verify it's complete."""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The marketplace agent slug (e.g., 'username/agent-name' or just 'agent-name' to search)",
                },
                "agent_version": {
                    "type": "integer",
                    "description": "Optional specific version of the agent (defaults to latest)",
                },
                "inputs": {
                    "type": "object",
                    "description": "The input dictionary you plan to provide. Should contain ALL required inputs from get_agent_details",
                    "additionalProperties": True,
                },
            },
            "required": ["agent_id"],
        }

    @property
    def requires_auth(self) -> bool:
        """This tool requires authentication."""
        return True

    async def _execute(
        self,
        user_id: str | None,
        session_id: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Get required setup information for an agent.

        Args:
            user_id: Authenticated user ID
            session_id: Chat session ID
            agent_id: Agent/Graph ID
            agent_version: Optional version

        Returns:
            JSON formatted setup requirements

        """
        agent_id = kwargs.get("agent_id", "").strip()
        agent_version = kwargs.get("agent_version")

        if not agent_id:
            return ErrorResponse(
                message="Please provide an agent ID",
                session_id=session_id,
            )

        try:
            graph = None

            # Check if it's a marketplace slug format (username/agent_name)
            if "/" in agent_id:
                try:
                    # Parse username/agent_name from slug
                    username, agent_name = agent_id.split("/", 1)
                    store_agent = await store_db.get_store_agent_details(
                        username, agent_name
                    )
                    if store_agent:
                        # Get graph from store listing
                        graph_meta = await store_db.get_available_graph(
                            store_agent.store_listing_version_id
                        )
                        # Now get the full graph with that ID
                        graph = await graph_db.get_graph(
                            graph_id=graph_meta.id,
                            version=graph_meta.version,
                            user_id=None,  # Public access
                            include_subgraphs=True,
                        )
                        logger.info(f"Found agent {agent_id} in marketplace")
                except Exception as e:
                    logger.debug(f"Failed to get from marketplace: {e}")
            else:
                # Try direct graph ID lookup
                graph = await graph_db.get_graph(
                    graph_id=agent_id,
                    version=agent_version,
                    user_id=user_id,
                    include_subgraphs=True,
                )

                if not graph:
                    # Try to get from marketplace/public
                    graph = await graph_db.get_graph(
                        graph_id=agent_id,
                        version=agent_version,
                        user_id=None,
                        include_subgraphs=True,
                    )

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )

            setup_info = SetupInfo(
                agent_id=graph.id,
                agent_name=graph.name,
                version=graph.version,
            )

            # Get credential manager
            creds_manager = IntegrationCredentialsManager()

            # Analyze credential requirements
            if (
                hasattr(graph, "credentials_input_schema")
                and graph.credentials_input_schema
            ):
                user_credentials = {}
                system_credentials = {}

                try:
                    # Get user's existing credentials
                    if user_id:
                        user_creds_list = await creds_manager.store.get_all_creds(
                            user_id
                        )
                    else:
                        user_creds_list = []
                    user_credentials = {c.provider: c for c in user_creds_list}
                    logger.info(
                        f"User has credentials for providers: {list(user_credentials.keys())}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get user credentials: {e}")

                # Get system-provided default credentials
                try:
                    system_creds_list = AutoRegistry.get_all_credentials()
                    system_credentials = {c.provider: c for c in system_creds_list}

                    # WORKAROUND: Check for common LLM providers that don't use SDK pattern
                    import os

                    from pydantic import SecretStr

                    from backend.data.model import APIKeyCredentials

                    # Check for OpenAI
                    if "openai" not in system_credentials:
                        openai_key = os.getenv("OPENAI_API_KEY")
                        if openai_key:
                            system_credentials["openai"] = APIKeyCredentials(
                                id="openai-system",
                                provider="openai",
                                api_key=SecretStr(openai_key),
                                title="System OpenAI API Key",
                            )

                    # Check for Anthropic
                    if "anthropic" not in system_credentials:
                        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                        if anthropic_key:
                            system_credentials["anthropic"] = APIKeyCredentials(
                                id="anthropic-system",
                                provider="anthropic",
                                api_key=SecretStr(anthropic_key),
                                title="System Anthropic API Key",
                            )

                    # Check for other common providers
                    provider_env_map = {
                        "groq": "GROQ_API_KEY",
                        "ollama": "OLLAMA_API_KEY",
                        "open_router": "OPEN_ROUTER_API_KEY",
                    }

                    for provider, env_var in provider_env_map.items():
                        if provider not in system_credentials:
                            api_key = os.getenv(env_var)
                            if api_key:
                                system_credentials[provider] = APIKeyCredentials(
                                    id=f"{provider}-system",
                                    provider=provider,
                                    api_key=SecretStr(api_key),
                                    title=f"System {provider} API Key",
                                )

                    logger.info(
                        f"System provides credentials for: {list(system_credentials.keys())}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get system credentials: {e}")

                # Handle the nested schema structure
                credentials_to_check = {}
                if isinstance(graph.credentials_input_schema, dict):
                    # Check if it's a JSON schema with properties
                    if "properties" in graph.credentials_input_schema:
                        credentials_to_check = graph.credentials_input_schema[
                            "properties"
                        ]
                    else:
                        # Fallback to treating the whole schema as credentials
                        credentials_to_check = graph.credentials_input_schema

                cred_in_schema = {}
                for cred_key, cred_schema in credentials_to_check.items():
                    cred_req = SetupRequirementInfo(
                        key=cred_key,
                        provider=cred_key,
                        required=True,
                        user_has=False,
                    )

                    # Parse credential schema to extract the actual provider
                    actual_provider = None
                    if isinstance(cred_schema, dict):
                        # Try to extract provider from credentials_provider field
                        if "credentials_provider" in cred_schema:
                            providers = cred_schema["credentials_provider"]
                            if isinstance(providers, list) and len(providers) > 0:
                                # Extract the actual provider name from the enum
                                actual_provider = str(providers[0])
                                # Handle ProviderName enum format
                                if "ProviderName." in actual_provider:
                                    actual_provider = (
                                        actual_provider.split("'")[1]
                                        if "'" in actual_provider
                                        else actual_provider.split(".")[-1].lower()
                                    )
                                cred_req.provider = actual_provider
                        elif "provider" in cred_schema:
                            cred_req.provider = cred_schema["provider"]
                            actual_provider = cred_schema["provider"]

                        if "type" in cred_schema:
                            cred_req.type = cred_schema["type"]  # oauth, api_key
                        if "scopes" in cred_schema:
                            cred_req.scopes = cred_schema["scopes"]
                        if "description" in cred_schema:
                            cred_req.description = cred_schema["description"]

                    # Check if user has this credential using the actual provider name
                    provider_name = actual_provider or cred_req.provider
                    logger.debug(
                        f"Checking credential {cred_key}: provider={provider_name}, available={list(user_credentials.keys())}"
                    )

                    # Check user credentials first, then system credentials
                    if provider_name in user_credentials:
                        cred_req.user_has = True
                        cred_req.credential_id = user_credentials[provider_name].id
                        logger.info(f"User has credential for {provider_name}")
                    elif provider_name in system_credentials:
                        cred_req.user_has = True
                        cred_req.credential_id = f"system-{provider_name}"
                        logger.info(f"System provides credential for {provider_name}")
                    else:
                        cred_in_schema[cred_key] = cred_schema
                        logger.info(
                            f"User missing credential for {provider_name} (not provided by system either)"
                        )

                    setup_info.requirements["credentials"].append(cred_req)
                setup_info.user_readiness.missing_credentials = cred_in_schema

            # Analyze input requirements
            if hasattr(graph, "input_schema") and graph.input_schema:
                if isinstance(graph.input_schema, dict):
                    properties = graph.input_schema.get("properties", {})
                    required = graph.input_schema.get("required", [])

                    for key, schema in properties.items():
                        input_req = InputField(
                            name=key,
                            type=schema.get("type", "string"),
                            required=key in required,
                            description=schema.get("description", ""),
                        )

                        # Add default value if present
                        if "default" in schema:
                            input_req.default = schema["default"]

                        # Add enum values if present
                        if "enum" in schema:
                            input_req.options = schema["enum"]

                        # Add format hints
                        if "format" in schema:
                            input_req.format = schema["format"]

                        setup_info.requirements["inputs"].append(input_req)

            # Determine supported execution modes
            execution_modes = []

            # Manual execution is always supported
            execution_modes.append(
                ExecutionModeInfo(
                    type="manual",
                    description="Run the agent immediately with provided inputs",
                    supported=True,
                )
            )

            # Check for scheduled execution support
            execution_modes.append(
                ExecutionModeInfo(
                    type="scheduled",
                    description="Run the agent on a recurring schedule (cron)",
                    supported=True,
                    config_required={
                        "cron": "Cron expression (e.g., '0 9 * * 1' for Mondays at 9 AM)",
                        "timezone": "User timezone (converted to UTC)",
                    },
                )
            )

            # Check for webhook support
            webhook_supported = False
            if hasattr(graph, "has_external_trigger"):
                webhook_supported = graph.has_external_trigger
            elif hasattr(graph, "webhook_input_node") and graph.webhook_input_node:
                webhook_supported = True

            if webhook_supported:
                webhook_mode = ExecutionModeInfo(
                    type="webhook",
                    description="Trigger the agent via external webhook",
                    supported=True,
                    config_required={},
                )

                # Add trigger setup info if available
                if hasattr(graph, "trigger_setup_info") and graph.trigger_setup_info:
                    webhook_mode.trigger_info = (
                        graph.trigger_setup_info.dict()  # type: ignore
                        if hasattr(graph.trigger_setup_info, "dict")
                        else graph.trigger_setup_info  # type: ignore
                    )

                execution_modes.append(webhook_mode)
            else:
                execution_modes.append(
                    ExecutionModeInfo(
                        type="webhook",
                        description="Webhook triggers not supported for this agent",
                        supported=False,
                    )
                )

            setup_info.requirements["execution_modes"] = execution_modes

            # Check overall readiness
            has_all_creds = len(setup_info.user_readiness.missing_credentials) == 0
            setup_info.user_readiness.has_all_credentials = has_all_creds

            # Agent is ready if all required credentials are present
            setup_info.user_readiness.ready_to_run = has_all_creds

            # Add setup instructions
            if not setup_info.user_readiness.ready_to_run:
                instructions = []
                if setup_info.user_readiness.missing_credentials:
                    instructions.append(
                        f"Add credentials for: {', '.join(setup_info.user_readiness.missing_credentials)}",
                    )
                setup_info.setup_instructions = instructions
            else:
                setup_info.setup_instructions = ["Agent is ready to set up and run!"]

            return SetupRequirementsResponse(
                message=f"Setup requirements for '{graph.name}' retrieved successfully",
                setup_info=setup_info,
                session_id=session_id,
                graph_id=graph.id,
                graph_version=graph.version,
            )

        except Exception as e:
            logger.error(f"Error getting setup requirements: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to get setup requirements: {e!s}",
                session_id=session_id,
            )


if __name__ == "__main__":
    import asyncio

    from backend.data.db import prisma

    setup_tool = GetRequiredSetupInfoTool()
    print(setup_tool.parameters)

    async def main():
        await prisma.connect()

        # Test with a logged-in user who HAS credentials
        print("\n=== Testing with logged-in user WITH credentials ===")
        result1 = await setup_tool._execute(
            agent_id="autogpt-store/slug-a",
            user_id="c640e784-7355-4afb-bed6-299cea1e5945",
            session_id="session1",
        )
        print(f"Result type: {result1.type}")
        if hasattr(result1, "setup_info"):
            print(f"User ready: {result1.setup_info.user_readiness.ready_to_run}")
            creds = result1.setup_info.requirements.get("credentials", [])
            print("Required credentials:")
            for cred in creds:
                print(
                    f"  - {cred.provider}: {'✓ Has' if cred.user_has else '✗ Missing'}"
                )

        # Test with a logged-in user WITHOUT credentials
        print("\n=== Testing with logged-in user WITHOUT credentials ===")
        result2 = await setup_tool._execute(
            agent_id="autogpt-store/slug-a",
            user_id="3e53486c-cf57-477e-ba2a-cb02dc828e1a",
            session_id="session2",
        )
        print(f"Result type: {result2.type}")
        if hasattr(result2, "setup_info"):
            print(f"User ready: {result2.setup_info.user_readiness.ready_to_run}")
            creds = result2.setup_info.requirements.get("credentials", [])
            print("Required credentials:")
            for cred in creds:
                print(
                    f"  - {cred.provider}: {'✓ Has' if cred.user_has else '✗ Missing'}"
                )

        # Test with an anonymous user
        print("\n=== Testing with anonymous user ===")
        result3 = await setup_tool._execute(
            agent_id="autogpt-store/slug-a",
            user_id="anon_user123",
            session_id="session3",
        )
        print(f"Result type: {result3.type}")
        if hasattr(result3, "setup_info"):
            print(f"User ready: {result3.setup_info.user_readiness.ready_to_run}")
            creds = result3.setup_info.requirements.get("credentials", [])
            print("Required credentials:")
            for cred in creds:
                print(
                    f"  - {cred.provider}: {'✓ Has' if cred.user_has else '✗ Missing'}"
                )

        await prisma.disconnect()

    asyncio.run(main())
