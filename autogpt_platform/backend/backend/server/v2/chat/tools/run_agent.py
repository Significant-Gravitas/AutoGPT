"""Tool for running an agent manually (one-off execution)."""

import asyncio
import logging
from typing import Any

import prisma.enums

from backend.data import graph as graph_db
from backend.data.credit import get_user_credit_model
from backend.data.execution import get_graph_execution, get_graph_execution_meta
from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.executor import utils as execution_utils
from backend.integrations.providers import ProviderName
from backend.server.v2.library import db as library_db
from backend.sdk.registry import AutoRegistry

from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    ErrorResponse,
    ExecutionStartedResponse,
    InsufficientCreditsResponse,
    ToolResponseBase,
    ValidationErrorResponse,
)

logger = logging.getLogger(__name__)


class RunAgentTool(BaseTool):
    """Tool for executing an agent manually with immediate results."""

    @property
    def name(self) -> str:
        return "run_agent"

    @property
    def description(self) -> str:
        return """Run an agent immediately (one-off manual execution). 
        IMPORTANT: Before calling this tool, you MUST first call get_agent_details to determine what inputs are required.
        The 'inputs' parameter must be a dictionary containing ALL required input values identified by get_agent_details.
        Example: If get_agent_details shows required inputs 'search_query' and 'max_results', you must pass:
        inputs={"search_query": "user's query", "max_results": 10}"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the agent to run (graph ID or marketplace slug)",
                },
                "agent_version": {
                    "type": "integer",
                    "description": "Optional version number of the agent",
                },
                "inputs": {
                    "type": "object",
                    "description": "REQUIRED: Dictionary of input values. Must include ALL required inputs from get_agent_details. Format: {\"input_name\": value}",
                    "additionalProperties": True,
                },
                "credentials": {
                    "type": "object",
                    "description": "Credentials for the agent (if needed)",
                    "additionalProperties": True,
                },
                "wait_for_result": {
                    "type": "boolean",
                    "description": "Whether to wait for execution to complete (max 30s)",
                    "default": False,
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
        """Execute an agent manually.

        Args:
            user_id: Authenticated user ID
            session_id: Chat session ID
            **kwargs: Execution parameters

        Returns:
            JSON formatted execution result

        """
        agent_id = kwargs.get("agent_id", "").strip()
        agent_version = kwargs.get("agent_version")
        inputs = kwargs.get("inputs", {})
        credentials = kwargs.get("credentials", {})
        wait_for_result = kwargs.get("wait_for_result", False)

        if not agent_id:
            return ErrorResponse(
                message="Please provide an agent ID",
                session_id=session_id,
            )

        try:
            # Check if user is authenticated (required for running agents)
            if not user_id:
                return ErrorResponse(
                    message="Authentication required to run agents",
                    session_id=session_id,
                )

            # Check credit balance
            credit_model = get_user_credit_model()
            balance = await credit_model.get_credits(user_id)

            if balance <= 0:
                return InsufficientCreditsResponse(
                    message="Insufficient credits. Please top up your account.",
                    balance=balance,
                    session_id=session_id,
                )

            # Check if agent_id looks like a marketplace slug
            graph = None
            marketplace_graph = None
            
            if "/" in agent_id:
                # Looks like a marketplace slug, try to get from store first
                from backend.server.v2.store import db as store_db
                try:
                    username, agent_name = agent_id.split("/", 1)
                    agent_details = await store_db.get_store_agent_details(username, agent_name)
                    if agent_details:
                        # Get the graph from the store listing version
                        graph_meta = await store_db.get_available_graph(
                            agent_details.store_listing_version_id
                        )
                        marketplace_graph = await graph_db.get_graph(
                            graph_id=graph_meta.id,
                            version=graph_meta.version,
                            user_id=None,  # Public access
                            include_subgraphs=True,
                        )
                        logger.info(f"Found marketplace agent by slug: {agent_id}")
                except Exception as e:
                    logger.debug(f"Failed to get agent by slug: {e}")
            
            # If we have a marketplace graph from the slug lookup, handle it
            if marketplace_graph:
                # Check if already in user's library
                library_agent = await library_db.get_library_agent_by_graph_id(
                    user_id=user_id,
                    graph_id=marketplace_graph.id,
                    graph_version=marketplace_graph.version,
                )
                
                if library_agent:
                    logger.info(f"Agent {agent_id} already in user library, using existing entry")
                    # Get the graph from the library agent
                    graph = await graph_db.get_graph(
                        graph_id=library_agent.graph_id,
                        version=library_agent.graph_version,
                        user_id=user_id,
                        include_subgraphs=True,
                    )
                else:
                    logger.info(f"Adding marketplace agent {agent_id} to user library")
                    await library_db.create_library_agent(
                        graph=marketplace_graph,
                        user_id=user_id,
                        create_library_agents_for_sub_graphs=True,
                    )
                    graph = marketplace_graph
            else:
                # Not found via slug, try as direct graph ID
                graph = await graph_db.get_graph(
                    graph_id=agent_id,
                    version=agent_version,
                    user_id=user_id,
                    include_subgraphs=True,
                )
                
                if not graph:
                    # Try as marketplace agent by ID
                    marketplace_graph = await graph_db.get_graph(
                        graph_id=agent_id,
                        version=agent_version,
                        user_id=None,  # Public access
                        include_subgraphs=True,
                    )
                    
                    if marketplace_graph:
                        # Check if already in user's library
                        library_agent = await library_db.get_library_agent_by_graph_id(
                            user_id=user_id,
                            graph_id=marketplace_graph.id,
                            graph_version=marketplace_graph.version,
                        )
                        
                        if library_agent:
                            logger.info(f"Agent {agent_id} already in user library, using existing entry")
                            # Get the graph from the library agent
                            graph = await graph_db.get_graph(
                                graph_id=library_agent.graph_id,
                                version=library_agent.graph_version,
                                user_id=user_id,
                                include_subgraphs=True,
                            )
                        else:
                            logger.info(f"Adding marketplace agent {agent_id} to user library")
                            await library_db.create_library_agent(
                                graph=marketplace_graph,
                                user_id=user_id,
                                create_library_agents_for_sub_graphs=True,
                            )
                            graph = marketplace_graph

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )

            # Get system-provided credentials
            system_credentials = {}
            try:
                system_creds_list = AutoRegistry.get_all_credentials()
                system_credentials = {c.provider: c for c in system_creds_list}
                
                # WORKAROUND: Check for common LLM providers that don't use SDK pattern
                import os
                from datetime import datetime, timedelta
                from pydantic import SecretStr
                
                # System credentials never expire - set to far future (Unix timestamp)
                expires_at = int((datetime.utcnow() + timedelta(days=36500)).timestamp())  # 100 years
                
                # Check for OpenAI
                if "openai" not in system_credentials:
                    openai_key = os.getenv("OPENAI_API_KEY")
                    if openai_key:
                        system_credentials["openai"] = APIKeyCredentials(
                            id="system-openai",
                            provider="openai",
                            api_key=SecretStr(openai_key),
                            title="System OpenAI API Key",
                            expires_at=expires_at
                        )
                
                # Check for Anthropic  
                if "anthropic" not in system_credentials:
                    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                    if anthropic_key:
                        system_credentials["anthropic"] = APIKeyCredentials(
                            id="system-anthropic",
                            provider="anthropic",
                            api_key=SecretStr(anthropic_key),
                            title="System Anthropic API Key",
                            expires_at=expires_at
                        )
                
                # Check for other common providers
                for provider, env_var in [("groq", "GROQ_API_KEY"), 
                                           ("ollama", "OLLAMA_API_KEY"),
                                           ("open_router", "OPEN_ROUTER_API_KEY")]:
                    if provider not in system_credentials:
                        api_key = os.getenv(env_var)
                        if api_key:
                            system_credentials[provider] = APIKeyCredentials(
                                id=f"system-{provider}",
                                provider=provider,
                                api_key=SecretStr(api_key),
                                title=f"System {provider} API Key",
                                expires_at=expires_at
                            )
                
                logger.info(f"System provides credentials for: {list(system_credentials.keys())}")
            except Exception as e:
                logger.warning(f"Failed to get system credentials: {e}")
            
            # Convert credentials to CredentialsMetaInput format
            # Fill in missing credentials with system-provided ones
            input_credentials = {}
            
            # First, process user-provided credentials
            for key, value in credentials.items():
                if isinstance(value, dict):
                    input_credentials[key] = CredentialsMetaInput(**value)
                else:
                    # Assume it's a credential ID
                    input_credentials[key] = CredentialsMetaInput(
                        id=value,
                        provider=key,  # Use the key as provider name
                        type="api_key",
                    )
            
            # Get user credentials if authenticated
            user_credentials = {}
            if user_id and not user_id.startswith("anon_"):
                try:
                    from backend.integrations.creds_manager import IntegrationCredentialsManager
                    creds_manager = IntegrationCredentialsManager()
                    user_creds_list = await creds_manager.store.get_all_creds(user_id)
                    for cred in user_creds_list:
                        user_credentials[cred.provider] = cred
                    logger.info(f"User has credentials for: {list(user_credentials.keys())}")
                except Exception as e:
                    logger.warning(f"Failed to get user credentials: {e}")
            
            # Use the graph's aggregated credentials to properly map credentials
            # This ensures we use the same keys that the graph expects
            graph_cred_inputs = graph.aggregate_credentials_inputs()
            logger.info(f"Graph aggregate credentials: {list(graph_cred_inputs.keys())}")
            logger.info(f"User provided credentials: {list(input_credentials.keys())}")
            logger.info(f"Available system credentials: {list(system_credentials.keys())}")
            logger.info(f"Available user credentials: {list(user_credentials.keys())}")
            
            # Process each aggregated credential field
            for agg_key, (field_info, node_fields) in graph_cred_inputs.items():
                if agg_key not in input_credentials:
                    # Extract provider from field_info (it's a frozenset, get the first element)
                    provider_set = field_info.provider
                    if isinstance(provider_set, (set, frozenset)) and len(provider_set) > 0:
                        # Get the first provider from the set
                        provider_enum = next(iter(provider_set))
                        # Get the string value from the enum
                        provider_name = provider_enum.value if hasattr(provider_enum, 'value') else str(provider_enum)
                    else:
                        provider_name = str(provider_set) if provider_set else None
                    
                    logger.info(f"Checking credential {agg_key} for provider {provider_name}")
                    
                    # Try to find credential from user or system
                    credential_found = False
                    
                    # First check user credentials
                    if provider_name and provider_name in user_credentials:
                        logger.info(f"Using user credential for {provider_name} (key: {agg_key})")
                        user_cred = user_credentials[provider_name]
                        # Use the provider_enum we already extracted from the frozenset
                        if isinstance(provider_set, (set, frozenset)) and len(provider_set) > 0:
                            provider_enum = next(iter(provider_set))
                            input_credentials[agg_key] = CredentialsMetaInput(
                                id=user_cred.id,
                                provider=provider_enum,
                                type=user_cred.type if hasattr(user_cred, 'type') else "api_key",
                            )
                            credential_found = True
                            logger.info(f"Added user credential to input_credentials[{agg_key}]")
                    
                    # If not found in user creds, check system credentials
                    if not credential_found and provider_name and provider_name in system_credentials:
                        logger.info(f"Using system credential for {provider_name} (key: {agg_key})")
                        # Use the provider_enum we already extracted from the frozenset
                        if isinstance(provider_set, (set, frozenset)) and len(provider_set) > 0:
                            provider_enum = next(iter(provider_set))
                            input_credentials[agg_key] = CredentialsMetaInput(
                                id=f"system-{provider_name}",
                                provider=provider_enum,
                                type="api_key",
                            )
                            credential_found = True
                            logger.info(f"Added system credential to input_credentials[{agg_key}]")
                    
                    if not credential_found:
                        logger.warning(f"Could not find credential for {agg_key} (provider: {provider_name}) in user or system stores")

            # Check if the graph needs inputs that weren't provided
            if hasattr(graph, "input_schema") and graph.input_schema:
                required_inputs = []
                optional_inputs = []
                
                # Parse the input schema
                input_schema = graph.input_schema
                if isinstance(input_schema, dict):
                    properties = input_schema.get("properties", {})
                    required = input_schema.get("required", [])
                    
                    for key, schema in properties.items():
                        if key not in inputs:
                            input_info = {
                                "name": key,
                                "type": schema.get("type", "string"),
                                "description": schema.get("description", ""),
                            }
                            
                            if key in required:
                                required_inputs.append(input_info)
                            else:
                                optional_inputs.append(input_info)
                
                # If there are required inputs missing, return an error
                if required_inputs:
                    return ValidationErrorResponse(
                        message="Missing required inputs for agent execution",
                        session_id=session_id,
                        error="Missing required inputs",
                        details={
                            "missing_inputs": required_inputs,
                            "optional_inputs": optional_inputs,
                        }
                    )
            
            # Execute the graph
            logger.info(
                f"Executing agent {graph.name} (ID: {graph.id}) for user {user_id}"
            )
            logger.info(f"Final credentials being passed: {list(input_credentials.keys())}")
            for key, cred in input_credentials.items():
                logger.debug(f"  {key}: id={cred.id}, provider={cred.provider}, type={cred.type}")

            graph_exec = await execution_utils.add_graph_execution(
                graph_id=graph.id,
                user_id=user_id,
                inputs=inputs,
                graph_version=graph.version,
                graph_credentials_inputs=input_credentials,
            )

            result = ExecutionStartedResponse(
                message=f"Agent '{graph.name}' execution started",
                execution_id=graph_exec.id,
                graph_id=graph.id,
                graph_name=graph.name,
                status="QUEUED",
                session_id=session_id,
            )

            # Optionally wait for completion (with timeout)
            if wait_for_result:
                logger.info(f"Waiting for execution {graph_exec.id} to complete...")
                start_time = asyncio.get_event_loop().time()
                timeout = 30  # 30 seconds max wait

                while asyncio.get_event_loop().time() - start_time < timeout:
                    # Get execution status
                    exec_status = await get_graph_execution_meta(user_id, graph_exec.id)

                    if exec_status and exec_status.status in [
                        "COMPLETED",
                        "FAILED",
                    ]:
                        result.status = exec_status.status
                        result.ended_at = (
                            exec_status.ended_at.isoformat()
                            if exec_status.ended_at
                            else None
                        )

                        if exec_status.status == "COMPLETED":
                            result.message = "Agent completed successfully"

                            # Try to get outputs
                            try:
                                full_exec = await get_graph_execution(
                                    user_id=user_id,
                                    execution_id=graph_exec.id,
                                    include_node_executions=True,
                                )
                                if (
                                    full_exec
                                    and hasattr(full_exec, "output_data")
                                    and full_exec.output_data  # type: ignore
                                ):
                                    result.outputs = full_exec.output_data  # type: ignore
                            except Exception as e:
                                logger.warning(f"Failed to get execution outputs: {e}")
                        else:
                            result.message = "Agent execution failed"
                            if (
                                hasattr(exec_status, "stats")
                                and exec_status.stats
                                and hasattr(exec_status.stats, "error")
                            ):
                                result.error = exec_status.stats.error
                        break

                    # Wait before checking again
                    await asyncio.sleep(2)
                else:
                    # Timeout reached
                    result.status = "RUNNING"
                    result.message = "Execution still running. Check status later."
                    result.timeout_reached = True

            return result

        except Exception as e:
            logger.error(f"Error executing agent: {e}", exc_info=True)

            # Check for specific error types
            if "validation" in str(e).lower():
                return ValidationErrorResponse(
                    message="Input validation failed",
                    error=str(e),
                    session_id=session_id,
                )

            return ErrorResponse(
                message=f"Failed to execute agent: {e!s}",
                session_id=session_id,
            )

if __name__ == "__main__":
    import asyncio
    import json
    from backend.data.db import prisma

    async def main():
        await prisma.connect()
        
        run_agent_tool = RunAgentTool()
        print("RunAgentTool parameters:")
        print(json.dumps(run_agent_tool.parameters, indent=2))
        
        # Test user IDs
        test_user_with_creds = "c640e784-7355-4afb-bed6-299cea1e5945"
        test_user_without_creds = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
        anon_user = "anon_test123"
        
        # For testing, we'll use the run_agent tool to pass in marketplace slugs
        # The tool will handle converting them to graph IDs internally
        print("\n" + "="*60)
        print("Setting up test agent IDs...")
        print("="*60)
        
        # Use marketplace slugs for testing - the tool will handle conversion
        test_agent_slug = "autogpt-store/slug-a"  # LinkedIn Post Generator agent
        test_agent_name = "LinkedIn Post Generator"
        
        print(f"Using test agent: {test_agent_name} (slug: {test_agent_slug})")
        print("Note: The run_agent tool will convert marketplace slugs to graph IDs internally")
        
        print("\n" + "="*60)
        print("Testing run_agent tool with different scenarios")
        print("="*60)
        
        # Test 1: Run with authenticated user with credentials
        print("\n1. Testing with authenticated user (has credentials):")
        print(f"   User ID: {test_user_with_creds}")
        print(f"   Agent: {test_agent_name} (slug: {test_agent_slug})")
        result1 = await run_agent_tool._execute(
            user_id=test_user_with_creds,
            session_id="test-session-1",
            agent_id=test_agent_slug,  # Use slug, tool will convert to graph ID
            inputs={},  # Use empty inputs for generic testing
            wait_for_result=False
        )
        print(f"   Result type: {result1.type}")
        if hasattr(result1, 'message'):
            print(f"   Message: {result1.message}")
        if result1.type == "execution_started" and hasattr(result1, 'execution_id'):
            print(f"   Execution ID: {result1.execution_id}")
        
        # Test 2: Run with authenticated user without credentials
        print("\n2. Testing with authenticated user (missing credentials):")
        print(f"   User ID: {test_user_without_creds}")
        print(f"   Agent: {test_agent_name} (slug: {test_agent_slug})")
        result2 = await run_agent_tool._execute(
            user_id=test_user_without_creds,
            session_id="test-session-2",
            agent_id=test_agent_slug,
            inputs={},
            wait_for_result=False
        )
        print(f"   Result type: {result2.type}")
        if hasattr(result2, 'message'):
            print(f"   Message: {result2.message}")
        
        # Test 3: Run with anonymous user
        print("\n3. Testing with anonymous user:")
        print(f"   User ID: {anon_user}")
        print(f"   Agent: {test_agent_name} (slug: {test_agent_slug})")
        result3 = await run_agent_tool._execute(
            user_id=None,  # Anonymous
            session_id="test-session-3",
            agent_id=test_agent_slug,
            inputs={},
            wait_for_result=False
        )
        print(f"   Result type: {result3.type}")
        if hasattr(result3, 'message'):
            print(f"   Message: {result3.message}")
        
        # Test 4: Run agent that only needs system credentials
        print("\n4. Testing agent with system-provided credentials:")
        print(f"   Using same agent: {test_agent_name} (slug: {test_agent_slug})")
        print("   This tests if system credentials are being used...")
        result4 = await run_agent_tool._execute(
            user_id=test_user_without_creds,
            session_id="test-session-4",
            agent_id=test_agent_slug,
            inputs={},
            wait_for_result=False
        )
        print(f"   Result type: {result4.type}")
        if hasattr(result4, 'message'):
            print(f"   Message: {result4.message}")
        
        # Test 5: Check library deduplication
        print("\n5. Testing library deduplication (running same agent twice):")
        print(f"   Using agent: {test_agent_name} (slug: {test_agent_slug})")
        print("   First run (should add to library if not present):")
        result5a = await run_agent_tool._execute(
            user_id=test_user_with_creds,
            session_id="test-session-5a",
            agent_id=test_agent_slug,
            inputs={},
            wait_for_result=False
        )
        print(f"   Result type: {result5a.type}")
        if result5a.type == "execution_started" and hasattr(result5a, 'execution_id'):
            print(f"   Execution ID: {result5a.execution_id}")
        
        print("   Second run (should use existing library entry):")
        result5b = await run_agent_tool._execute(
            user_id=test_user_with_creds,
            session_id="test-session-5b",
            agent_id=test_agent_slug,  # Same agent
            inputs={},
            wait_for_result=False
        )
        print(f"   Result type: {result5b.type}")
        if result5b.type == "execution_started" and hasattr(result5b, 'execution_id'):
            print(f"   Execution ID: {result5b.execution_id}")
        
        # Test 6: Invalid agent
        print("\n6. Testing with invalid agent ID:")
        result6 = await run_agent_tool._execute(
            user_id=test_user_with_creds,
            session_id="test-session-6",
            agent_id="invalid/agent-that-does-not-exist",
            inputs={},
            wait_for_result=False
        )
        print(f"   Result type: {result6.type}")
        if hasattr(result6, 'message'):
            print(f"   Message: {result6.message}")
        
        print("\n" + "="*60)
        print("Testing complete!")
        print("="*60)
        
        await prisma.disconnect()
    
    asyncio.run(main())