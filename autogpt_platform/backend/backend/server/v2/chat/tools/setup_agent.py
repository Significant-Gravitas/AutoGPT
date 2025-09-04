"""Tool for setting up an agent with credentials and configuration."""

import logging
from typing import Any

import pytz
from apscheduler.triggers.cron import CronTrigger

from backend.data import graph as graph_db
from backend.data.model import CredentialsMetaInput
from backend.executor.scheduler import SchedulerClient
from backend.integrations.webhooks.utils import setup_webhook_for_block
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
        return "Set up an agent with credentials and configure it for scheduled execution or webhook triggers. Requires authentication."

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
                    "description": "Name for this setup/schedule",
                },
                "description": {
                    "type": "string",
                    "description": "Description of this setup",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression for scheduled execution (required if setup_type is 'schedule')",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone for the schedule (e.g., 'America/New_York'). Defaults to UTC.",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input values for the agent",
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
            # Get the graph
            graph = await graph_db.get_graph(
                graph_id=agent_id,
                version=None,  # Use latest
                user_id=user_id,
                include_subgraphs=True,
            )

            if not graph:
                # Try marketplace/public
                graph = await graph_db.get_graph(
                    graph_id=agent_id,
                    version=None,
                    user_id=None,
                    include_subgraphs=True,
                )

                if graph:
                    # Add to user's library if from marketplace
                    logger.info(f"Adding marketplace agent {agent_id} to user library")
                    await library_db.create_library_agent(
                        graph=graph,
                        user_id=user_id,
                        create_library_agents_for_sub_graphs=True,
                    )

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )

            # Convert credentials to CredentialsMetaInput format
            input_credentials = {}
            for key, value in credentials.items():
                if isinstance(value, dict):
                    input_credentials[key] = CredentialsMetaInput(**value)
                elif isinstance(value, str):
                    # Assume it's a credential ID
                    input_credentials[key] = CredentialsMetaInput(
                        id=value,
                        type="api_key",  # Default type
                    )

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
                    preset={
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
                    webhook_url=new_webhook.webhook_url,
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
                    preset={
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
