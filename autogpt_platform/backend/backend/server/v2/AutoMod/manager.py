import asyncio
import json
import logging
from typing import Any, Dict, List, Tuple

import aiohttp
from pydantic import ValidationError

from backend.data.execution import ExecutionStatus
from backend.server.v2.AutoMod.models import (
    AutoModRequest,
    AutoModResponse,
    ModerationConfig,
)
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
            moderate_inputs=settings.config.automod_moderate_inputs,
            moderate_outputs=settings.config.automod_moderate_outputs,
        )

    def moderate_graph_execution_inputs(
        self, db_client, graph_exec, event_loop, send_update_func, timeout: int = 10
    ) -> Tuple[bool, Exception | None]:
        """
        Complete input moderation flow for graph execution
        Returns: (success, error_if_failed)
        """
        if not self.config.moderate_inputs or not self.config.enabled:
            return True, None

        # Get graph model and collect all inputs
        graph_model = db_client.get_graph(
            graph_exec.graph_id,
            user_id=graph_exec.user_id,
            version=graph_exec.graph_version,
        )

        if not graph_model or not graph_model.nodes:
            return True, None

        all_inputs = []
        for node in graph_model.nodes:
            if node.input_default:
                all_inputs.extend(str(v) for v in node.input_default.values() if v)
            if (masks := graph_exec.nodes_input_masks) and (mask := masks.get(node.id)):
                all_inputs.extend(str(v) for v in mask.values() if v)

        if not all_inputs:
            return True, None

        # Combine all content and moderate directly
        content = " ".join(all_inputs)

        # Run moderation
        try:
            moderation_passed = asyncio.run_coroutine_threadsafe(
                self._moderate_content(
                    content,
                    {
                        "user_id": graph_exec.user_id,
                        "graph_id": graph_exec.graph_id,
                        "graph_exec_id": graph_exec.graph_exec_id,
                        "moderation_type": "execution_input",
                    },
                ),
                event_loop,
            ).result(timeout=timeout)

            if not moderation_passed:
                # Get existing executions to mark as failed
                existing_executions = db_client.get_node_executions(
                    graph_exec.graph_exec_id,
                    statuses=[
                        ExecutionStatus.QUEUED,
                        ExecutionStatus.RUNNING,
                        ExecutionStatus.INCOMPLETE,
                    ],
                )
                self.handle_moderation_failure(
                    db_client=db_client,
                    executions=existing_executions,
                    send_update_func=send_update_func,
                )
                return False, Exception("Execution failed due to content moderation")

            return True, None

        except Exception as e:
            logger.error(f"Input moderation execution failed: {e}")
            return False, Exception("Execution failed due to content moderation")

    def moderate_graph_execution_outputs(
        self,
        db_client,
        graph_exec_id: str,
        user_id: str,
        graph_id: str,
        event_loop,
        send_update_func,
        timeout: int = 10,
    ) -> Tuple[bool, Exception | None]:
        """
        Complete output moderation flow for graph execution
        Returns: (success, error_if_failed)
        """
        if not self.config.moderate_outputs or not self.config.enabled:
            return True, None

        # Get completed executions and collect outputs
        completed_executions = db_client.get_node_executions(
            graph_exec_id, statuses=[ExecutionStatus.COMPLETED], include_exec_data=True
        )

        if not completed_executions:
            return True, None

        all_outputs = []
        for exec_entry in completed_executions:
            if exec_entry.output_data:
                all_outputs.extend(str(v) for v in exec_entry.output_data.values() if v)

        if not all_outputs:
            return True, None

        # Combine all content and moderate directly
        content = " ".join(all_outputs)

        # Run moderation
        try:
            moderation_passed = asyncio.run_coroutine_threadsafe(
                self._moderate_content(
                    content,
                    {
                        "user_id": user_id,
                        "graph_id": graph_id,
                        "graph_exec_id": graph_exec_id,
                        "moderation_type": "execution_output",
                    },
                ),
                event_loop,
            ).result(timeout=timeout)

            if not moderation_passed:
                self.handle_moderation_failure(
                    db_client=db_client,
                    executions=completed_executions,
                    send_update_func=send_update_func,
                )
                return False, Exception("Execution failed due to content moderation")

            return True, None

        except Exception as e:
            logger.error(f"Output moderation execution failed: {e}")
            return False, Exception("Execution failed due to content moderation")

    def handle_moderation_failure(self, db_client, executions: List, send_update_func):
        """Handle moderation failure by updating executions and sending websocket updates"""
        if not executions:
            return

        # Collect all input and output names from the executions to clear
        cleared_inputs = {}
        cleared_outputs = {}

        for exec_entry in executions:
            # Clear all inputs
            if exec_entry.input_data:
                for name in exec_entry.input_data.keys():
                    cleared_inputs[name] = "Failed due to content moderation"

            # Clear all outputs
            if exec_entry.output_data:
                for name in exec_entry.output_data.keys():
                    cleared_outputs[name] = "Failed due to content moderation"

        # Mark all as failed with clear moderation message and clear inputs and outputs
        db_client.update_node_execution_status_batch(
            [exec_entry.node_exec_id for exec_entry in executions],
            status=ExecutionStatus.FAILED,
            stats={
                "error": "Failed due to content moderation",
                "moderation_cleared": True,
                "cleared_outputs": cleared_outputs,
                "cleared_inputs": cleared_inputs,
            },
        )

        # Send websocket updates for each failed node to update the UI
        for exec_entry in executions:
            if updated_exec := db_client.get_node_execution(exec_entry.node_exec_id):
                send_update_func(updated_exec)

    def _extract_text_simple(self, data: Any) -> str:
        """Extract text content simply - just convert to string"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return " ".join(str(v) for v in data.values() if v)
        elif isinstance(data, list):
            return " ".join(str(item) for item in data if item)
        else:
            return str(data) if data else ""

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

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url, headers=headers, json=request_data.model_dump()
            ) as response:
                response_text = await response.text()

                if response.status != 200:
                    raise Exception(
                        f"AutoMod API returned status {response.status}: {response_text}"
                    )

                try:
                    response_data = json.loads(response_text)
                    return AutoModResponse.model_validate(response_data)
                except (json.JSONDecodeError, ValidationError) as e:
                    raise Exception(f"Invalid response from AutoMod API: {e}")


# Global instance
automod_manager = AutoModManager()
