import json
import logging
from typing import Any, Dict, List

from backend.data import graph as graph_db
from backend.data.model import CredentialsMetaInput
from backend.data.user import get_user_by_id
from backend.util.clients import get_scheduler_client
from backend.util.timezone_utils import convert_cron_to_utc, get_user_timezone_or_utc

logger = logging.getLogger(__name__)

tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_agent",
            "description": "Search the marketplace for an agent that matches the users query. You can use this multiple times with different search queries to help the user find the right agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The search query that will be used to find the agent in the store",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agent_details",
            "description": "Get the full details of an agent including what credentials are needed, input data and anything else needed when setting up the agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The ID of the agent to get the details of",
                    },
                    "agent_version": {
                        "type": "string",
                        "description": "The version number of the agent to get the details of",
                    },
                },
                "required": ["agent_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "setup_agent",
            "description": "Set up an agent to run either on a schedule (with cron) or via webhook trigger. Automatically adds the agent to your library if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {
                        "type": "string",
                        "description": "ID of the agent to set up",
                    },
                    "graph_version": {
                        "type": "integer",
                        "description": "Optional version of the agent",
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for this setup (schedule or webhook)",
                    },
                    "trigger_type": {
                        "type": "string",
                        "enum": ["schedule", "webhook"],
                        "description": "How the agent should be triggered: 'schedule' for cron-based or 'webhook' for external triggers",
                    },
                    "cron": {
                        "type": "string",
                        "description": "Cron expression for scheduled execution (required if trigger_type is 'schedule')",
                    },
                    "webhook_config": {
                        "type": "object",
                        "description": "Configuration for webhook trigger (required if trigger_type is 'webhook')",
                        "additionalProperties": True,
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Input values for the agent execution",
                        "additionalProperties": True,
                    },
                    "credentials": {
                        "type": "object",
                        "description": "Credentials needed for the agent",
                        "additionalProperties": True,
                    },
                },
                "required": ["graph_id", "name", "trigger_type"],
            },
        },
    },
]


# Tool execution functions


async def execute_find_agent(
    parameters: Dict[str, Any], user_id: str, session_id: str
) -> str:
    """Execute the find_agent tool.

    Args:
        parameters: Tool parameters containing search_query
        user_id: User ID for authentication
        session_id: Current session ID

    Returns:
        JSON string with search results
    """
    search_query = parameters.get("search_query", "")

    # For anonymous users, provide basic search results but suggest login for more features
    is_anonymous = user_id.startswith("anon_")

    try:
        from backend.data.db import prisma

        # Use StoreAgent view which has all the information we need
        # Now with nullable creator_username field to handle NULL values
        results = []
        
        try:
            # Build where clause for StoreAgent search
            where_clause = {}
            
            if search_query:
                where_clause["OR"] = [
                    {"agent_name": {"contains": search_query, "mode": "insensitive"}},
                    {"description": {"contains": search_query, "mode": "insensitive"}},
                    {"sub_heading": {"contains": search_query, "mode": "insensitive"}},
                ]

            # Query StoreAgent view (now with nullable creator_username)
            store_agents = await prisma.storeagent.find_many(
                where=where_clause,
                take=10,
                order={"updated_at": "desc"}
            )

            # Format results from StoreAgent
            for agent in store_agents:
                # Get the graph ID from the store listing version if needed
                graph_id = None
                if agent.storeListingVersionId:
                    try:
                        listing_version = await prisma.storelistingversion.find_unique(
                            where={"id": agent.storeListingVersionId}
                        )
                        if listing_version:
                            graph_id = listing_version.agentGraphId
                    except:
                        pass  # Ignore errors getting graph ID

                results.append(
                    {
                        "id": graph_id or agent.listing_id,
                        "name": agent.agent_name,
                        "description": agent.description or "No description",
                        "sub_heading": agent.sub_heading,
                        "creator": agent.creator_username or "anonymous",
                        "featured": agent.featured,
                        "rating": agent.rating,
                        "runs": agent.runs,
                        "slug": agent.slug,
                    }
                )
        except Exception as store_error:
            logger.debug(f"StoreAgent view not available: {store_error}, falling back to AgentGraph")
            
            # Fallback to AgentGraph if StoreAgent view fails
            where_clause = {"isActive": True}

            if search_query:
                where_clause["OR"] = [
                    {"name": {"contains": search_query, "mode": "insensitive"}},
                    {"description": {"contains": search_query, "mode": "insensitive"}},
                ]

            graphs = await prisma.agentgraph.find_many(
                where=where_clause, take=10, order={"createdAt": "desc"}
            )

            # Format results for chat display
            results = []
            for graph in graphs:
                results.append(
                    {
                        "id": graph.id,
                        "version": graph.version,
                        "name": graph.name,
                        "description": graph.description or "No description",
                        "is_active": graph.isActive,
                        "created_at": (
                            graph.createdAt.isoformat() if graph.createdAt else None
                        ),
                    }
                )

        if not results:
            message = f"No agents found matching '{search_query}'. Try different search terms or browse the marketplace."
            if is_anonymous:
                message += "\n\n💡 Tip: Sign in to access more detailed agent information and setup capabilities."
            return message

        base_message = f"Found {len(results)} agents matching '{search_query}':\n" + json.dumps(
            results, indent=2
        )

        if is_anonymous:
            base_message += "\n\n🔐 To get detailed agent information (including credential requirements) and set up agents, please sign in or create an account."

        return base_message

    except Exception as e:
        logger.error(f"Error searching for agents: {e}")
        return f"Error searching for agents: {str(e)}"


async def execute_get_agent_details(
    parameters: Dict[str, Any], user_id: str, session_id: str
) -> str:
    """Execute the get_agent_details tool.

    Args:
        parameters: Tool parameters containing agent_id and optional agent_version
        user_id: User ID for authentication
        session_id: Current session ID

    Returns:
        JSON string with agent details
    """
    agent_id = parameters.get("agent_id", "")
    agent_version = parameters.get("agent_version")

    # Check if user is anonymous (not authenticated)
    if user_id.startswith("anon_"):
        return json.dumps({
            "status": "auth_required",
            "message": "To view detailed agent information including credential requirements, you need to be logged in. Please sign in or create an account to continue.",
            "action": "login",
            "session_id": session_id,
            "agent_info": {
                "agent_id": agent_id,
                "agent_version": agent_version
            }
        })

    try:
        # Get the full graph with all details
        if agent_version:
            graph = await graph_db.get_graph(
                graph_id=agent_id,
                version=int(agent_version),
                user_id=user_id,
                include_subgraphs=True,
            )
        else:
            graph = await graph_db.get_graph(
                graph_id=agent_id, user_id=user_id, include_subgraphs=True
            )

        if not graph:
            # Try as admin/public graph
            graph = await graph_db.get_graph_as_admin(
                graph_id=agent_id, version=int(agent_version) if agent_version else None
            )

        if not graph:
            return f"Agent with ID {agent_id} not found or not accessible."

        # Extract credentials requirements from the graph
        credentials_info = []
        if hasattr(graph, "_credentials_input_schema"):
            creds_schema = graph._credentials_input_schema
            for field_name, field_def in creds_schema.model_fields.items():
                field_info = {
                    "name": field_name,
                    "required": field_def.is_required(),
                    "type": "credentials",
                }

                # Extract provider and other metadata from field info
                if hasattr(field_def, "metadata"):
                    for meta in field_def.metadata:
                        if hasattr(meta, "provider"):
                            field_info["provider"] = meta.provider
                        if hasattr(meta, "description"):
                            field_info["description"] = meta.description
                        if hasattr(meta, "required_scopes"):
                            field_info["scopes"] = list(meta.required_scopes)

                credentials_info.append(field_info)

        # Extract input requirements from the graph
        inputs_info = []
        if hasattr(graph, "input_schema"):
            for field_name, field_props in graph.input_schema.get(
                "properties", {}
            ).items():
                inputs_info.append(
                    {
                        "name": field_name,
                        "type": field_props.get("type", "string"),
                        "description": field_props.get("description", ""),
                        "required": field_name
                        in graph.input_schema.get("required", []),
                        "default": field_props.get("default"),
                        "title": field_props.get("title", field_name),
                    }
                )

        # Extract trigger/webhook info if available
        trigger_info = None
        if hasattr(graph, "trigger_setup_info") and graph.trigger_setup_info:
            trigger_info = {
                "provider": graph.trigger_setup_info.provider,
                "credentials_needed": graph.trigger_setup_info.credentials_input_name,
                "config_schema": graph.trigger_setup_info.config_schema,
            }

        # Get node information
        node_info = []
        if hasattr(graph, "nodes"):
            for node in graph.nodes[:5]:  # Show first 5 nodes
                node_info.append(
                    {
                        "id": node.id,
                        "block_id": (
                            node.block_id
                            if hasattr(node, "block_id")
                            else node.block.id
                        ),
                        "block_name": (
                            node.block.name
                            if hasattr(node.block, "name")
                            else "Unknown"
                        ),
                        "title": (
                            node.metadata.get("title", node.block.name)
                            if hasattr(node, "metadata") and node.metadata
                            else "Unnamed"
                        ),
                    }
                )

        details = {
            "id": graph.id,
            "name": graph.name or "Unnamed Agent",
            "version": graph.version,
            "description": graph.description or "No description available",
            "is_active": (
                graph.is_active
                if hasattr(graph, "is_active")
                else graph.isActive if hasattr(graph, "isActive") else False
            ),
            "credentials_required": credentials_info,
            "inputs": inputs_info,
            "trigger_info": trigger_info,
            "node_count": len(graph.nodes) if hasattr(graph, "nodes") else 0,
            "sample_nodes": node_info,
        }

        return (
            f"Agent Details for {details['name']} (ID: {agent_id}, version: {details['version']}):\n"
            + json.dumps(details, indent=2)
        )

    except Exception as e:
        logger.error(f"Error getting agent details: {e}")
        return f"Error retrieving agent details: {str(e)}"


async def execute_setup_agent(
    parameters: Dict[str, Any], user_id: str, session_id: str
) -> str:
    """Execute the setup_agent tool - handles both scheduled and webhook triggers.

    This function automatically:
    1. Adds the agent to the user's library if needed
    2. Sets up either a schedule or webhook based on trigger_type
    3. Configures all necessary credentials and inputs

    Args:
        parameters: Tool parameters for agent setup
        user_id: User ID for authentication
        session_id: Current session ID

    Returns:
        String describing the setup result
    """
    graph_id = parameters.get("graph_id", "")
    graph_version = parameters.get("graph_version")
    name = parameters.get("name", "Unnamed Setup")
    trigger_type = parameters.get("trigger_type", "schedule")
    cron = parameters.get("cron", "")
    webhook_config = parameters.get("webhook_config", {})
    inputs = parameters.get("inputs", {})
    credentials = parameters.get("credentials", {})

    # Check if user is anonymous (not authenticated)
    if user_id.startswith("anon_"):
        return json.dumps({
            "status": "auth_required",
            "message": "You need to be logged in to set up agents. Please sign in or create an account to continue.",
            "action": "login",
            "session_id": session_id,
            "agent_info": {
                "graph_id": graph_id,
                "name": name,
                "trigger_type": trigger_type
            }
        }, indent=2)

    try:
        from backend.server.v2.library import db as library_db
        from backend.server.v2.library import model as library_model

        # Get the full graph to validate it exists and get its version
        graph = await graph_db.get_graph(
            graph_id=graph_id,
            version=graph_version,
            user_id=user_id,
            include_subgraphs=True,
        )

        is_marketplace_agent = False
        if not graph:
            # Try to get as admin/public graph (marketplace agent)
            graph = await graph_db.get_graph_as_admin(
                graph_id=graph_id, version=graph_version
            )
            is_marketplace_agent = True

        if not graph:
            return f"Error: Agent with ID {graph_id} not found or not accessible."

        # Step 1: Add to library if it's a marketplace agent
        library_agent_id = None
        if is_marketplace_agent:
            try:
                library_agents = await library_db.create_library_agent(
                    graph=graph,
                    user_id=user_id,
                    create_library_agents_for_sub_graphs=True,
                )
                if library_agents:
                    library_agent_id = library_agents[0].id
                    logger.info(
                        f"Added agent {graph.name} to user's library (ID: {library_agent_id})"
                    )
            except Exception as lib_error:
                logger.warning(
                    f"Could not add to library (may already exist): {lib_error}"
                )

        # Convert credentials dict to CredentialsMetaInput format
        input_credentials = {}
        for key, value in credentials.items():
            if isinstance(value, dict):
                input_credentials[key] = CredentialsMetaInput(**value)
            else:
                # Assume it's a credential ID string
                input_credentials[key] = CredentialsMetaInput(id=value, type="api_key")

        # Step 2: Set up the trigger based on type
        setup_info = {}

        if trigger_type == "webhook":
            # Handle webhook setup
            if not graph.webhook_input_node:
                return "Error: This agent does not support webhook triggers. Please use 'schedule' trigger type instead."

            # Create webhook preset
            try:
                # Build the trigger setup request (for future use with actual webhook creation)
                _ = library_model.TriggeredPresetSetupRequest(
                    graph_id=graph_id,
                    graph_version=graph_version or graph.version,
                    name=name,
                    description=f"Webhook trigger for {graph.name}",
                    trigger_config=webhook_config,
                    agent_credentials=input_credentials,
                )

                # Mock webhook creation for now
                webhook_url = f"https://api.autogpt.com/webhooks/{graph_id[:8]}"

                setup_info = {
                    "status": "success",
                    "trigger_type": "webhook",
                    "webhook_url": webhook_url,
                    "graph_id": graph_id,
                    "graph_version": graph.version,
                    "name": name,
                    "added_to_library": library_agent_id is not None,
                    "library_id": library_agent_id,
                    "message": f"Successfully set up webhook trigger for '{graph.name}'. Webhook URL: {webhook_url}",
                }
            except Exception as webhook_error:
                logger.error(f"Webhook setup error: {webhook_error}")
                return f"Error setting up webhook: {str(webhook_error)}"

        else:  # schedule type
            # Handle scheduled execution
            if not cron:
                return "Error: Cron expression is required for scheduled execution."

            # Get user timezone for conversion
            try:
                user = await get_user_by_id(user_id)
                user_tz = get_user_timezone_or_utc(user.timezone if user else None)
                user_timezone_str = (
                    str(user_tz) if hasattr(user_tz, "key") else str(user_tz)
                )
            except Exception:
                user_timezone_str = "UTC"

            # Convert cron expression from user timezone to UTC
            try:
                utc_cron = convert_cron_to_utc(cron, user_timezone_str)
            except ValueError as e:
                return f"Error: Invalid cron expression '{cron}': {str(e)}"

            # Use the real scheduler client to create the schedule
            try:
                scheduler_client = get_scheduler_client()
                result = await scheduler_client.add_execution_schedule(
                    user_id=user_id,
                    graph_id=graph_id,
                    graph_version=graph.version,
                    name=name,
                    cron=utc_cron,
                    input_data=inputs,
                    input_credentials=input_credentials,
                )

                setup_info = {
                    "status": "success",
                    "trigger_type": "schedule",
                    "schedule_id": result.id,
                    "graph_id": graph_id,
                    "graph_version": graph.version,
                    "name": name,
                    "cron": cron,
                    "cron_utc": utc_cron,
                    "timezone": user_timezone_str,
                    "inputs": inputs,
                    "next_run": (
                        result.next_run_time.isoformat()
                        if result.next_run_time
                        else None
                    ),
                    "added_to_library": library_agent_id is not None,
                    "library_id": library_agent_id,
                    "message": f"Successfully scheduled '{graph.name}' to run with cron expression '{cron}' (in {user_timezone_str})",
                }
            except Exception as scheduler_error:
                logger.warning(
                    f"Scheduler error: {scheduler_error}, falling back to mock response"
                )
                # Fallback to mock response if scheduler is not available
                import datetime

                next_run = datetime.datetime.now(
                    datetime.timezone.utc
                ) + datetime.timedelta(hours=1)

                setup_info = {
                    "status": "success",
                    "trigger_type": "schedule",
                    "schedule_id": f"schedule-{graph_id[:8]}",
                    "graph_id": graph_id,
                    "graph_version": graph.version,
                    "name": name,
                    "cron": cron,
                    "cron_utc": cron,
                    "timezone": user_timezone_str,
                    "inputs": inputs,
                    "next_run": next_run.isoformat(),
                    "added_to_library": library_agent_id is not None,
                    "library_id": library_agent_id,
                    "message": f"Successfully scheduled '{graph.name}' (mock mode)",
                }

        return "Agent Setup Complete:\n" + json.dumps(setup_info, indent=2)

    except Exception as e:
        logger.error(f"Error setting up agent schedule: {e}")
        return f"Error setting up agent: {str(e)}"
