"""Tool for getting detailed information about a specific agent."""

import logging
from typing import Any

from backend.data import graph as graph_db
from backend.server.v2.store import db as store_db
from backend.sdk.registry import AutoRegistry

from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    AgentDetails,
    AgentDetailsNeedLoginResponse,
    AgentDetailsNeedCredentialsResponse,
    AgentDetailsResponse,
    ErrorResponse,
    ExecutionOptions,
    InputField,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class GetAgentDetailsTool(BaseTool):
    """Tool for getting detailed information about an agent."""

    @property
    def name(self) -> str:
        return "get_agent_details"

    @property
    def description(self) -> str:
        return "Get detailed information about a specific agent including inputs, credentials required, and execution options."

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
            },
            "required": ["agent_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session_id: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Get detailed information about an agent.

        Args:
            user_id: User ID (may be anonymous)
            session_id: Chat session ID
            agent_id: Agent ID or slug
            agent_version: Optional version number

        Returns:
            Pydantic response model

        """
        agent_id = kwargs.get("agent_id", "").strip()
        agent_version = kwargs.get("agent_version")

        if not agent_id:
            return ErrorResponse(
                message="Please provide an agent ID",
                session_id=session_id,
            )

        try:
            # Always try to get from marketplace first
            graph = None
            store_agent = None
            in_library = False
            is_marketplace = False

            # Check if it's a slug format (username/agent_name)
            if "/" in agent_id:
                try:
                    # Parse username/agent_name from slug
                    username, agent_name = agent_id.split("/", 1)
                    store_agent = await store_db.get_store_agent_details(
                        username, agent_name
                    )
                    logger.info(f"Found agent {agent_id} in marketplace")
                except Exception as e:
                    logger.debug(f"Failed to get from marketplace: {e}")
            else:
                # Try to find by agent slug alone (search all agents)
                try:
                    # Search for the agent in the store
                    search_results = await store_db.search_store_agents(
                        search_query=agent_id,
                        limit=1
                    )
                    if search_results.agents:
                        first_agent = search_results.agents[0]
                        # Now get the full details using the slug
                        if "/" in first_agent.slug:
                            username, agent_name = first_agent.slug.split("/", 1)
                            store_agent = await store_db.get_store_agent_details(
                                username, agent_name
                            )
                            logger.info(f"Found agent by search in marketplace")
                except Exception as e:
                    logger.debug(f"Failed to search marketplace: {e}")
            
            # If we found a store agent, get its graph
            if store_agent:
                try:
                    # Use get_available_graph to get the graph from store listing version
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
                    is_marketplace = True
                except Exception as e:
                    logger.error(f"Failed to get graph for store agent: {e}")

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )

            # Parse input schema
            input_fields = {}
            if hasattr(graph, "input_schema") and graph.input_schema:
                if isinstance(graph.input_schema, dict):
                    properties = graph.input_schema.get("properties", {})
                    required = graph.input_schema.get("required", [])

                    input_required = []
                    input_optional = []

                    for key, schema in properties.items():
                        field = InputField(
                            name=key,
                            type=schema.get("type", "string"),
                            description=schema.get("description", ""),
                            required=key in required,
                            default=schema.get("default"),
                            options=schema.get("enum"),
                            format=schema.get("format"),
                        )

                        if key in required:
                            input_required.append(field)
                        else:
                            input_optional.append(field)

                    input_fields = {
                        "schema": graph.input_schema,
                        "required": input_required,
                        "optional": input_optional,
                    }

            # Parse credential requirements and check availability
            credentials = []
            needs_auth = False
            missing_credentials = []
            if (
                hasattr(graph, "credentials_input_schema")
                and graph.credentials_input_schema
            ):
                # Get system-provided credentials
                system_credentials = {}
                try:
                    system_creds_list = AutoRegistry.get_all_credentials()
                    system_credentials = {c.provider: c for c in system_creds_list}
                    
                    # WORKAROUND: Check for common LLM providers that don't use SDK pattern
                    import os
                    from backend.data.model import APIKeyCredentials
                    from pydantic import SecretStr
                    
                    # Check for OpenAI
                    if "openai" not in system_credentials:
                        openai_key = os.getenv("OPENAI_API_KEY")
                        if openai_key:
                            system_credentials["openai"] = APIKeyCredentials(
                                id="openai-system",
                                provider="openai",
                                api_key=SecretStr(openai_key),
                                title="System OpenAI API Key"
                            )
                    
                    # Check for Anthropic  
                    if "anthropic" not in system_credentials:
                        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                        if anthropic_key:
                            system_credentials["anthropic"] = APIKeyCredentials(
                                id="anthropic-system",
                                provider="anthropic",
                                api_key=SecretStr(anthropic_key),
                                title="System Anthropic API Key"
                            )
                    
                    # Check for other common providers
                    for provider, env_var in [("groq", "GROQ_API_KEY"), 
                                               ("ollama", "OLLAMA_API_KEY"),
                                               ("open_router", "OPEN_ROUTER_API_KEY")]:
                        if provider not in system_credentials:
                            api_key = os.getenv(env_var)
                            if api_key:
                                system_credentials[provider] = APIKeyCredentials(
                                    id=f"{provider}-system",
                                    provider=provider,
                                    api_key=SecretStr(api_key),
                                    title=f"System {provider} API Key"
                                )
                    
                    logger.debug(f"System provides credentials for: {list(system_credentials.keys())}")
                except Exception as e:
                    logger.warning(f"Failed to get system credentials: {e}")
                
                # Get user's credentials if authenticated
                user_credentials = {}
                if user_id and not user_id.startswith("anon_"):
                    try:
                        from backend.integrations.creds_manager import IntegrationCredentialsManager
                        creds_manager = IntegrationCredentialsManager()
                        user_creds_list = await creds_manager.store.get_all_creds(user_id)
                        user_credentials = {c.provider: c for c in user_creds_list}
                        logger.debug(f"User has credentials for: {list(user_credentials.keys())}")
                    except Exception as e:
                        logger.warning(f"Failed to get user credentials: {e}")
                
                # Handle nested schema structure
                credentials_to_check = {}
                if isinstance(graph.credentials_input_schema, dict):
                    if "properties" in graph.credentials_input_schema:
                        credentials_to_check = graph.credentials_input_schema["properties"]
                    else:
                        credentials_to_check = graph.credentials_input_schema
                
                # Process credentials from the schema dict into a list
                credentials = []
                for cred_key, cred_schema in credentials_to_check.items():
                    # Extract the actual provider name
                    actual_provider = None
                    if isinstance(cred_schema, dict):
                        # Try to extract provider from credentials_provider field
                        if "credentials_provider" in cred_schema:
                            providers = cred_schema["credentials_provider"]
                            if isinstance(providers, list) and len(providers) > 0:
                                actual_provider = str(providers[0])
                                if "ProviderName." in actual_provider:
                                    actual_provider = actual_provider.split("'")[1] if "'" in actual_provider else actual_provider.split(".")[-1].lower()
                        
                        cred_meta = {
                            "id": cred_key,
                            "provider": actual_provider or cred_schema.get("credentials_provider", cred_key),
                            "type": cred_schema.get("credentials_type", "api_key"),
                            "title": cred_schema.get("title") or cred_schema.get("description"),
                        }
                        credentials.append(cred_meta)
                        
                        # Check if this credential is available
                        provider_name = actual_provider or cred_key
                        if provider_name not in user_credentials and provider_name not in system_credentials:
                            missing_credentials.append(provider_name)
                            logger.debug(f"Missing credential for provider: {provider_name}")
                
                # Only needs auth if there are missing credentials
                needs_auth = bool(missing_credentials)

            # Determine execution options
            execution_options = ExecutionOptions(
                manual=True,  # Always support manual execution
                scheduled=True,  # Most agents support scheduling
                webhook=False,  # Check for webhook support
            )

            # Check for webhook/trigger support
            if hasattr(graph, "has_external_trigger"):
                execution_options.webhook = graph.has_external_trigger
            elif hasattr(graph, "webhook_input_node") and graph.webhook_input_node:
                execution_options.webhook = True

            # Build trigger info if available
            trigger_info = None
            if hasattr(graph, "trigger_setup_info") and graph.trigger_setup_info:
                trigger_info = {
                    "supported": True,
                    "config": (
                        graph.trigger_setup_info.dict()
                        if hasattr(graph.trigger_setup_info, "dict")
                        else graph.trigger_setup_info
                    ),
                }

            # Build stats if available
            stats = None
            if hasattr(graph, "executions_count"):
                stats = {
                    "total_runs": graph.executions_count,  # type: ignore
                    "last_run": (
                        graph.last_execution.isoformat()  # type: ignore
                        if hasattr(graph, "last_execution") and graph.last_execution  # type: ignore
                        else None
                    ),
                }

            # Create agent details
            details = AgentDetails(
                id=graph.id,
                name=graph.name,
                description=graph.description,
                version=graph.version,
                is_latest=graph.is_active if hasattr(graph, "is_active") else True,
                in_library=in_library,
                is_marketplace=is_marketplace,
                inputs=input_fields,
                credentials=credentials, # type: ignore
                execution_options=execution_options,
                trigger_info=trigger_info,
                stats=stats,
            )

            # Check if user needs to log in or set up credentials
            if needs_auth:
                if not user_id or user_id.startswith("anon_"):
                    # Anonymous user needs to log in first
                    # Build a descriptive message about what credentials are needed
                    cred_list = []
                    for cred in credentials:
                        cred_desc = f"{cred.get('provider', 'Unknown')}"
                        if cred.get('type'):
                            cred_desc += f" ({cred.get('type')})"
                        cred_list.append(cred_desc)
                    
                    cred_message = f"This agent requires the following credentials: {', '.join(cred_list)}. Please sign in to set up and run this agent."
                    
                    return AgentDetailsNeedLoginResponse(
                        message=cred_message,
                        session_id=session_id,
                        agent=details,
                        agent_info={
                            "agent_id": agent_id,
                            "agent_version": agent_version,
                            "name": details.name,
                            "graph_id": graph.id,
                        },
                        graph_id=graph.id,
                        graph_version=graph.version,
                    )
                else:
                    # Authenticated user needs to set up credentials
                    # Return the credentials schema so the frontend can show the setup UI
                    cred_message = f"The agent '{details.name}' requires credentials to be configured. Please provide the required credentials to continue."
                    
                    return AgentDetailsNeedCredentialsResponse(
                        message=cred_message,
                        session_id=session_id,
                        agent=details,
                        credentials_schema=graph.credentials_input_schema,
                        agent_info={
                            "agent_id": agent_id,
                            "agent_version": agent_version,
                            "name": details.name,
                            "graph_id": graph.id,
                        },
                        graph_id=graph.id,
                        graph_version=graph.version,
                    )

            # Build a descriptive message about the agent
            message_parts = [f"Agent '{graph.name}' details loaded successfully."]
            
            if credentials:
                cred_list = []
                for cred in credentials:
                    cred_desc = f"{cred.get('provider', 'Unknown')}"
                    if cred.get('type'):
                        cred_desc += f" ({cred.get('type')})"
                    cred_list.append(cred_desc)
                message_parts.append(f"Required credentials: {', '.join(cred_list)}")
            
            # Be very explicit about required inputs
            if input_fields:
                if input_fields.get("required"):
                    message_parts.append("\n**REQUIRED INPUTS:**")
                    for field in input_fields["required"]:
                        desc = f"  - {field.name} ({field.type})"
                        if field.description:
                            desc += f": {field.description}"
                        message_parts.append(desc)
                    
                    # Build example dict format
                    example_dict = {}
                    for field in input_fields["required"]:
                        if field.type == "string":
                            example_dict[field.name] = f"<{field.name}_value>"
                        elif field.type == "number" or field.type == "integer":
                            example_dict[field.name] = 123
                        elif field.type == "boolean":
                            example_dict[field.name] = True
                        else:
                            example_dict[field.name] = f"<{field.type}_value>"
                    
                    message_parts.append(f"\n**IMPORTANT:** To run this agent, you MUST pass these inputs as a dictionary to run_agent, setup_agent, and get_required_setup_info tools.")
                    message_parts.append(f"Example format: inputs={example_dict}")
                
                if input_fields.get("optional"):
                    message_parts.append("\n**OPTIONAL INPUTS:**")
                    for field in input_fields["optional"]:
                        desc = f"  - {field.name} ({field.type})"
                        if field.description:
                            desc += f": {field.description}"
                        if field.default is not None:
                            desc += f" [default: {field.default}]"
                        message_parts.append(desc)
            
            return AgentDetailsResponse(
                message=" ".join(message_parts),
                session_id=session_id,
                agent=details,
                user_authenticated=not (not user_id or user_id.startswith("anon_")),
                graph_id=graph.id,
                graph_version=graph.version,
            )

        except Exception as e:
            logger.error(f"Error getting agent details: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to get agent details: {e!s}",
                error=str(e),
                session_id=session_id,
            )


if __name__ == "__main__":
    import asyncio
    from backend.data.db import prisma

    find_agent_tool = GetAgentDetailsTool()
    print(find_agent_tool.parameters)

    async def main():
        await prisma.connect()
        
        # Test with a logged-in user
        print("\n=== Testing agent with logged-in user ===")
        result1 = await find_agent_tool._execute(
            agent_id="autogpt-store/slug-a", 
            user_id="3e53486c-cf57-477e-ba2a-cb02dc828e1a", 
            session_id="session1"
        )
        print(f"Result type: {result1.type}")
        print(f"Has credentials schema: {'credentials_schema' in result1.__dict__}")
        if hasattr(result1, 'message'):
            print(f"Result type: {result1.type}")
            print(f"Message: {result1.message}...")
        
        # Test with an anonymous user  
        print("\n=== Testing with anonymous user ===")
        result2 = await find_agent_tool._execute(
            agent_id="autogpt-store/slug-a", 
            user_id="anon_user123", 
            session_id="session2"
        )
        print(f"Result type: {result2.type}")
        print(f"Message: {result2.message}...")
        
        await prisma.disconnect()
    
    asyncio.run(main())