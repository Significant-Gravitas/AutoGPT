import asyncio
import json
import logging
from typing import Any, Dict, List

from pydantic import ValidationError

from backend.util.request import Requests

from backend.data.execution import ExecutionStatus
from backend.server.v2.AutoMod.models import (
    AutoModRequest,
    AutoModResponse,
    ModerationConfig,
)
from backend.util.exceptions import ModerationError
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


class AutoModManager:

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> ModerationConfig:
        """Load AutoMod configuration from settings"""
        settings = Settings()
        return ModerationConfig(
            enabled=settings.config.automod_enabled,
            api_url=settings.config.automod_api_url,
            api_key=settings.secrets.automod_api_key,
            timeout=settings.config.automod_timeout,
            retry_attempts=settings.config.automod_retry_attempts,
            retry_delay=settings.config.automod_retry_delay,
            fail_open=settings.config.automod_fail_open,
        )

    async def moderate_graph_execution_inputs(
        self, db_client, graph_exec, send_update_func, timeout: int = 10
    ) -> Exception | None:
        """
        Complete input moderation flow for graph execution
        Returns: Exception if moderation failed, None if passed
        """
        if not self.config.enabled:
            return None

        # Get graph model and collect all inputs
        graph_model = await db_client.get_graph(
            graph_exec.graph_id,
            user_id=graph_exec.user_id,
            version=graph_exec.graph_version,
        )

        if not graph_model or not graph_model.nodes:
            return None

        all_inputs = []
        for node in graph_model.nodes:
            if node.input_default:
                all_inputs.extend(str(v) for v in node.input_default.values() if v)
            if (masks := graph_exec.nodes_input_masks) and (mask := masks.get(node.id)):
                all_inputs.extend(str(v) for v in mask.values() if v)

        if not all_inputs:
            return None

        # Combine all content and moderate directly
        content = " ".join(all_inputs)

        # Run moderation
        try:
            moderation_passed = await self._moderate_content(
                content,
                {
                    "user_id": graph_exec.user_id,
                    "graph_id": graph_exec.graph_id,
                    "graph_exec_id": graph_exec.graph_exec_id,
                    "moderation_type": "execution_input",
                },
            )

            if not moderation_passed:
                # Get existing executions to mark as failed
                existing_executions = await db_client.get_node_executions(
                    graph_exec.graph_exec_id,
                    statuses=[
                        ExecutionStatus.QUEUED,
                        ExecutionStatus.RUNNING,
                        ExecutionStatus.INCOMPLETE,
                    ],
                )
                
                # Update execution stats before raising error
                cleared_inputs = {}
                cleared_outputs = {}
                
                for exec_entry in existing_executions:
                    # Clear all inputs
                    if exec_entry.input_data:
                        for name in exec_entry.input_data.keys():
                            cleared_inputs[name] = "Failed due to content moderation"
                    # Clear all outputs
                    if exec_entry.output_data:
                        for name in exec_entry.output_data.keys():
                            cleared_outputs[name] = "Failed due to content moderation"

                # Mark all as failed with clear moderation message and clear inputs and outputs
                await db_client.update_node_execution_status_batch(
                    [exec_entry.node_exec_id for exec_entry in existing_executions],
                    status=ExecutionStatus.FAILED,
                    stats={
                        "error": "Failed due to content moderation",
                        "cleared_outputs": cleared_outputs,
                        "cleared_inputs": cleared_inputs,
                    },
                )

                # Send websocket updates for each failed node to update the UI
                for exec_entry in existing_executions:
                    if updated_exec := await db_client.get_node_execution(exec_entry.node_exec_id):
                        await send_update_func(updated_exec)
                
                return ModerationError(
                    "Execution failed due to content moderation",
                    graph_exec.user_id,
                    graph_exec.graph_exec_id,
                )

            return None

        except Exception as e:
            logger.error(f"Input moderation execution failed: {e}")
            return ModerationError(
                "Execution failed due to content moderation",
                graph_exec.user_id,
                graph_exec.graph_exec_id,
            )

    async def moderate_graph_execution_outputs(
        self,
        db_client,
        graph_exec_id: str,
        user_id: str,
        graph_id: str,
        send_update_func,
        timeout: int = 10,
    ) -> Exception | None:
        """
        Complete output moderation flow for graph execution
        Returns: Exception if moderation failed, None if passed
        """
        if not self.config.enabled:
            return None

        # Get completed executions and collect outputs
        completed_executions = await db_client.get_node_executions(
            graph_exec_id, statuses=[ExecutionStatus.COMPLETED], include_exec_data=True
        )

        if not completed_executions:
            return None

        all_outputs = []
        for exec_entry in completed_executions:
            if exec_entry.output_data:
                all_outputs.extend(str(v) for v in exec_entry.output_data.values() if v)

        if not all_outputs:
            return None

        # Combine all content and moderate directly
        content = " ".join(all_outputs)

        # Run moderation
        try:
            moderation_passed = await self._moderate_content(
                content,
                {
                    "user_id": user_id,
                    "graph_id": graph_id,
                    "graph_exec_id": graph_exec_id,
                    "moderation_type": "execution_output",
                },
            )

            if not moderation_passed:
                # Update execution stats before raising error
                cleared_inputs = {}
                cleared_outputs = {}
                
                for exec_entry in completed_executions:
                    # Clear all inputs
                    if exec_entry.input_data:
                        for name in exec_entry.input_data.keys():
                            cleared_inputs[name] = "Failed due to content moderation"
                    # Clear all outputs
                    if exec_entry.output_data:
                        for name in exec_entry.output_data.keys():
                            cleared_outputs[name] = "Failed due to content moderation"

                # Mark all as failed with clear moderation message and clear inputs and outputs
                await db_client.update_node_execution_status_batch(
                    [exec_entry.node_exec_id for exec_entry in completed_executions],
                    status=ExecutionStatus.FAILED,
                    stats={
                        "error": "Failed due to content moderation",
                        "cleared_outputs": cleared_outputs,
                        "cleared_inputs": cleared_inputs,
                    },
                )

                # Send websocket updates for each failed node to update the UI
                for exec_entry in completed_executions:
                    if updated_exec := await db_client.get_node_execution(exec_entry.node_exec_id):
                        await send_update_func(updated_exec)
                
                return ModerationError(
                    "Execution failed due to content moderation",
                    user_id,
                    graph_exec_id,
                )

            return None

        except Exception as e:
            logger.error(f"Output moderation execution failed: {e}")
            return ModerationError(
                "Execution failed due to content moderation",
                user_id,
                graph_exec_id,
            )


    async def _moderate_content(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Moderate content using AutoMod API"""
        try:
            request_data = AutoModRequest(
                type="text",
                content=content,
                metadata=metadata,
            )

            # Make API request with retries
            last_exception = None
            for attempt in range(self.config.retry_attempts):
                try:
                    response = await self._make_request(request_data)

                    if response.success and response.status == "approved":
                        logger.debug(
                            f"Content approved for {metadata.get('graph_exec_id', 'unknown')}"
                        )
                        return True
                    else:
                        reasons = [
                            r.reason for r in response.moderation_results if r.reason
                        ]
                        error_msg = f"Content rejected by AutoMod: {'; '.join(reasons)}"
                        logger.warning(f"Content rejected: {error_msg}")
                        return False

                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"AutoMod API request failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                    )
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (2**attempt))

            # All retries failed
            logger.error(f"AutoMod moderation failed: {last_exception}")
            return self.config.fail_open

        except Exception as e:
            logger.error(f"AutoMod moderation error: {e}")
            return self.config.fail_open

    async def _make_request(self, request_data: AutoModRequest) -> AutoModResponse:
        """Make HTTP request to AutoMod API"""
        url = f"{self.config.api_url}/moderate"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.config.api_key,
        }

        # Use the centralized request utility with SSRF protection and retry logic
        requests = Requests(
            trusted_origins=[self.config.api_url],
            raise_for_status=True,
            retry_max_wait=self.config.timeout,
        )
        
        try:
            response = await requests.post(
                url, 
                headers=headers, 
                json=request_data.model_dump(),
                timeout=self.config.timeout
            )
            
            try:
                response_data = response.json()
                return AutoModResponse.model_validate(response_data)
            except (json.JSONDecodeError, ValidationError) as e:
                raise Exception(f"Invalid response from AutoMod API: {e}")
                
        except Exception as e:
            # The Requests class already handles status errors, so this catches all other issues
            raise Exception(f"AutoMod API request failed: {e}")


# Global instance
automod_manager = AutoModManager()
