"""Tool for setting up an agent with credentials and configuration."""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import pytz
from apscheduler.triggers.cron import CronTrigger
from pydantic import SecretStr

from backend.data import graph as graph_db
from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.executor.scheduler import SchedulerClient
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks.utils import setup_webhook_for_block
from backend.sdk.registry import AutoRegistry
from backend.server.v2.library import db as library_db

from .base import BaseTool
from .models import (
    ErrorResponse,
    PresetCreatedResponse,
    ScheduleCreatedResponse,
    ToolResponseBase,
    WebhookCreatedResponse,
)

logger = logging.getLogger(__name__)


class SetupAgentTool(BaseTool):
    """Tool for setting up an agent with scheduled execution or webhook triggers."""

    @property
    def name(self) -> str:
        return "setup_agent"

    @property
    def description(self) -> str:
        return """Set up an agent with credentials and configure it for scheduled execution or webhook triggers.
        IMPORTANT: Before calling this tool, you MUST first call get_agent_details to determine what inputs are required.
        
        For SCHEDULED execution:
        - Cron format: "minute hour day month weekday" (e.g., "0 9 * * 1-5" = 9am weekdays)
        - Common patterns: "0 * * * *" (hourly), "0 0 * * *" (daily at midnight), "0 9 * * 1" (Mondays at 9am)
        - Timezone: Use IANA timezone names like "America/New_York", "Europe/London", "Asia/Tokyo"
        - The 'inputs' parameter must contain ALL required inputs from get_agent_details as a dictionary
        
        For WEBHOOK triggers:
        - The agent will be triggered by external events
        - Still requires all input values from get_agent_details"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent ID (graph ID) to set up",
                },
                "setup_type": {
                    "type": "string",
                    "enum": ["schedule", "webhook", "preset"],
                    "description": "Type of setup: 'schedule' for cron, 'webhook' for triggers, 'preset' for saved configuration",
                },
                "name": {
                    "type": "string",
                    "description": "Name for this setup/schedule (e.g., 'Daily Report', 'Weekly Summary')",
                },
                "description": {
                    "type": "string",
                    "description": "Description of this setup",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression (5 fields: minute hour day month weekday). Examples: '0 9 * * 1-5' (9am weekdays), '*/30 * * * *' (every 30 min)",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone (e.g., 'America/New_York', 'Europe/London', 'UTC'). Defaults to UTC if not specified.",
                },
                "inputs": {
                    "type": "object",
                    "description": "REQUIRED: Dictionary with ALL required inputs from get_agent_details. Format: {\"input_name\": value}",
                    "additionalProperties": True,
                },
                "credentials": {
                    "type": "object",
                    "description": "Credentials configuration",
                    "additionalProperties": True,
                },
                "webhook_config": {
                    "type": "object",
                    "description": "Webhook configuration (required if setup_type is 'webhook')",
                    "additionalProperties": True,
                },
            },
            "required": ["agent_id", "setup_type"],
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
        """Set up an agent with configuration.

        Args:
            user_id: Authenticated user ID
            session_id: Chat session ID
            **kwargs: Setup parameters

        Returns:
            JSON formatted setup result

        """
        # Check if user is authenticated (required for setting up agents)
        if not user_id:
            return ErrorResponse(
                message="Authentication required to set up agents",
                session_id=session_id,
            )

        agent_id = kwargs.get("agent_id", "").strip()
        setup_type = kwargs.get("setup_type", "").strip()
        name = kwargs.get("name", f"Setup for {agent_id}")
        description = kwargs.get("description", "")
        inputs = kwargs.get("inputs", {})
        credentials = kwargs.get("credentials", {})

        if not agent_id:
            return ErrorResponse(
                message="Please provide an agent ID",
                session_id=session_id,
            )

        if not setup_type:
            return ErrorResponse(
                message="Please specify setup type: 'schedule', 'webhook', or 'preset'",
                session_id=session_id,
            )

        try:
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
                    version=None,  # Use latest
                    user_id=user_id,
                    include_subgraphs=True,
                )
                
                if not graph:
                    # Try as marketplace agent by ID
                    marketplace_graph = await graph_db.get_graph(
                        graph_id=agent_id,
                        version=None,
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
                # Get SDK-registered credentials
                system_creds_list = AutoRegistry.get_all_credentials()
                for cred in system_creds_list:
                    system_credentials[cred.provider] = cred
                
                # System credentials never expire - set to far future (Unix timestamp)
                expires_at = int((datetime.utcnow() + timedelta(days=36500)).timestamp())
                
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
            
            # Get user credentials if authenticated
            user_credentials = {}
            try:
                from backend.integrations.creds_manager import IntegrationCredentialsManager
                creds_manager = IntegrationCredentialsManager()
                user_creds_list = await creds_manager.store.get_all_creds(user_id)
                for cred in user_creds_list:
                    user_credentials[cred.provider] = cred
                logger.info(f"User has credentials for: {list(user_credentials.keys())}")
            except Exception as e:
                logger.warning(f"Failed to get user credentials: {e}")

            # Convert provided credentials to CredentialsMetaInput format
            input_credentials = {}
            for key, value in credentials.items():
                if isinstance(value, dict):
                    input_credentials[key] = CredentialsMetaInput(**value)
                elif isinstance(value, str):
                    # Assume it's a credential ID
                    input_credentials[key] = CredentialsMetaInput(
                        id=value,
                        provider=key,  # Use the key as provider name
                        type="api_key",  # Default type
                    )
            
            # Use the graph's aggregated credentials to properly map credentials
            # This ensures we use the same keys that the graph expects
            graph_cred_inputs = graph.aggregate_credentials_inputs()
            logger.info(f"Graph aggregate credentials: {list(graph_cred_inputs.keys())}")
            logger.info(f"User provided credentials: {list(input_credentials.keys())}")
            
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

            result = {}

            if setup_type == "schedule":
                # Set up scheduled execution
                cron = kwargs.get("cron")
                if not cron:
                    return ErrorResponse(
                        message="Cron expression is required for scheduled execution",
                        session_id=session_id,
                    )

                # Validate cron expression
                try:
                    CronTrigger.from_crontab(cron)
                except Exception as e:
                    return ErrorResponse(
                        message=f"Invalid cron expression '{cron}': {e!s}",
                        session_id=session_id,
                    )

                # Convert timezone if provided
                timezone = kwargs.get("timezone", "UTC")
                try:
                    pytz.timezone(timezone)
                except Exception:
                    return ErrorResponse(
                        message=f"Invalid timezone '{timezone}'",
                        session_id=session_id,
                    )

                # Create schedule via scheduler client
                scheduler_client = SchedulerClient()
                schedule_info = await scheduler_client.add_execution_schedule(
                    user_id=user_id,
                    graph_id=graph.id,
                    graph_version=graph.version,
                    cron=cron,
                    input_data=inputs,
                    input_credentials=input_credentials,
                    name=name,
                )

                result = ScheduleCreatedResponse(
                    message=f"Schedule '{name}' created successfully",
                    schedule_id=schedule_info.id,
                    name=name,
                    cron=cron,
                    timezone=timezone,
                    next_run=schedule_info.next_run_time,
                    graph_id=graph.id,
                    graph_name=graph.name,
                    session_id=session_id,
                )

            elif setup_type == "webhook":
                # Set up webhook trigger
                if not graph.webhook_input_node:
                    return ErrorResponse(
                        message=f"Agent '{graph.name}' does not support webhook triggers",
                        session_id=session_id,
                    )

                webhook_config = kwargs.get("webhook_config", {})

                # Combine webhook config with credentials
                trigger_config = {**webhook_config, **input_credentials}

                # Set up webhook
                new_webhook, feedback = await setup_webhook_for_block(
                    user_id=user_id,
                    trigger_block=graph.webhook_input_node.block,
                    trigger_config=trigger_config,
                )

                if not new_webhook:
                    return ErrorResponse(
                        message=f"Failed to create webhook: {feedback}",
                        session_id=session_id,
                    )

                # Create preset with webhook
                preset = await library_db.create_preset(
                    user_id=user_id,
                    preset={  # type: ignore
                        "graph_id": graph.id,
                        "graph_version": graph.version,
                        "name": name,
                        "description": description,
                        "inputs": inputs,
                        "credentials": input_credentials,
                        "webhook_id": new_webhook.id,
                        "is_active": True,
                    },
                )

                result = WebhookCreatedResponse(
                    message=f"Webhook trigger '{name}' created successfully",
                    webhook_id=new_webhook.id,
                    webhook_url=new_webhook.webhook_url,  # type: ignore
                    preset_id=preset.id,
                    name=name,
                    graph_id=graph.id,
                    graph_name=graph.name,
                    session_id=session_id,
                )

            elif setup_type == "preset":
                # Create a preset configuration for manual execution
                preset = await library_db.create_preset(
                    user_id=user_id,
                    preset={  # type: ignore
                        "graph_id": graph.id,
                        "graph_version": graph.version,
                        "name": name,
                        "description": description,
                        "inputs": inputs,
                        "credentials": input_credentials,
                        "is_active": True,
                    },
                )

                result = PresetCreatedResponse(
                    message=f"Preset configuration '{name}' created successfully",
                    preset_id=preset.id,
                    name=name,
                    graph_id=graph.id,
                    graph_name=graph.name,
                    session_id=session_id,
                )

            else:
                return ErrorResponse(
                    message=f"Unknown setup type: {setup_type}",
                    session_id=session_id,
                )

            return result

        except Exception as e:
            logger.error(f"Error setting up agent: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to set up agent: {e!s}",
                session_id=session_id,
            )


if __name__ == "__main__":
    """Added a test script here to check all the functionality of the setup_agent tool"""
    import asyncio
    import json
    from backend.data.db import prisma

    async def main():
        await prisma.connect()
        
        setup_tool = SetupAgentTool()
        print("SetupAgentTool parameters:")
        print(json.dumps(setup_tool.parameters, indent=2))
        
        # Test user IDs
        test_user = "c640e784-7355-4afb-bed6-299cea1e5945"
        
        print("\n" + "="*60)
        print("Testing setup_agent tool")
        print("="*60)
        
        # Test 1: Schedule setup with LinkedIn agent
        print("\n1. Testing schedule setup with LinkedIn Post Generator:")
        result1 = await setup_tool._execute(
            user_id=test_user,
            session_id="test-session-1",
            agent_id="autogpt-store/slug-a",  # LinkedIn Post Generator
            setup_type="schedule",
            name="Daily LinkedIn Posts",
            description="Generate LinkedIn posts every morning",
            cron="0 9 * * *",  # Daily at 9am
            timezone="America/New_York",
            inputs={"topic": "AI and automation"},
            credentials={},  # Will use user's credentials
        )
        print(f"   Result type: {result1.type}")
        if hasattr(result1, 'message'):
            print(f"   Message: {result1.message}")
        if hasattr(result1, 'schedule_id'):
            print(f"   Schedule ID: {result1.schedule_id}")
        
        # Test 2: Webhook setup with Github PR Reviewer (auto-setup webhook)
        print("\n2. Testing webhook setup with Github PR Reviewer (auto-setup):")
        result2 = await setup_tool._execute(
            user_id=test_user,
            session_id="test-session-2", 
            agent_id="github-pr-reviewer",  # Auto-setup webhook agent
            setup_type="webhook",
            name="PR Review Webhook",
            description="Automatically review PRs when created/updated",
            webhook_config={
                "repository": "owner/repo",
                "events": ["pull_request"],
            },
            credentials={},  # Github credentials would be needed
        )
        print(f"   Result type: {result2.type}")
        if hasattr(result2, 'message'):
            print(f"   Message: {result2.message}")
        if hasattr(result2, 'webhook_url'):
            print(f"   Webhook URL: {result2.webhook_url}")
        
        # Test 3: Webhook setup with Stripe Payment Tracker (manual setup)
        print("\n3. Testing webhook setup with Stripe Payment Tracker (manual):")
        result3 = await setup_tool._execute(
            user_id=test_user,
            session_id="test-session-3",
            agent_id="stripe-payment-tracker",  # Manual webhook setup agent
            setup_type="webhook",
            name="Stripe Payment Webhook", 
            description="Track Stripe payment events",
            webhook_config={
                "endpoint": "payment.succeeded",
            },
            credentials={},  # Stripe credentials would be needed
        )
        print(f"   Result type: {result3.type}")
        if hasattr(result3, 'message'):
            print(f"   Message: {result3.message}")
        if hasattr(result3, 'webhook_url'):
            print(f"   Webhook URL: {result3.webhook_url}")
        
        # Test 4: Create preset configuration
        print("\n4. Testing preset creation for manual execution:")
        result4 = await setup_tool._execute(
            user_id=test_user,
            session_id="test-session-4",
            agent_id="autogpt-store/slug-a",
            setup_type="preset",
            name="LinkedIn Post Preset",
            description="Saved configuration for LinkedIn posts",
            inputs={
                "topic": "Technology trends",
                "tone": "professional",
            },
            credentials={},
        )
        print(f"   Result type: {result4.type}")
        if hasattr(result4, 'message'):
            print(f"   Message: {result4.message}")
        if hasattr(result4, 'preset_id'):
            print(f"   Preset ID: {result4.preset_id}")
        
        # Test 5: Invalid cron expression
        print("\n5. Testing with invalid cron expression:")
        result5 = await setup_tool._execute(
            user_id=test_user,
            session_id="test-session-5",
            agent_id="autogpt-store/slug-a",
            setup_type="schedule",
            name="Invalid Schedule",
            cron="not-a-valid-cron",
        )
        print(f"   Result type: {result5.type}")
        if hasattr(result5, 'message'):
            print(f"   Message: {result5.message}")
        
        # Test 6: No authentication
        print("\n6. Testing without authentication:")
        result6 = await setup_tool._execute(
            user_id=None,
            session_id="test-session-6",
            agent_id="test-agent",
            setup_type="schedule",
            name="Test Schedule",
            cron="0 * * * *",
        )
        print(f"   Result type: {result6.type}")
        if hasattr(result6, 'message'):
            print(f"   Message: {result6.message}")
        
        print("\n" + "="*60)
        print("Testing complete!")
        print("="*60)
        
        await prisma.disconnect()
    
    asyncio.run(main())