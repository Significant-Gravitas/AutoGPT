import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from backend.executor import DatabaseManagerAsyncClient

from pydantic import ValidationError

from backend.data.execution import ExecutionStatus
from backend.util.exceptions import ModerationError
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.request import Requests
from backend.util.settings import Settings

from .models import AutoModRequest, AutoModResponse, ModerationConfig

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
        self, db_client: "DatabaseManagerAsyncClient", graph_exec, timeout: int = 10
    ) -> Exception | None:
        """
        Complete input moderation flow for graph execution
        Returns: error_if_failed (None means success)
        """
        if not self.config.enabled:
            return None

        # Check if AutoMod feature is enabled for this user
        if not await is_feature_enabled(Flag.AUTOMOD, graph_exec.user_id):
            logger.debug(f"AutoMod feature not enabled for user {graph_exec.user_id}")
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
        logger.warning(
            f"Moderating inputs for graph execution {graph_exec.graph_exec_id}"
        )
        try:
            moderation_passed, content_id = await self._moderate_content(
                content,
                {
                    "user_id": graph_exec.user_id,
                    "graph_id": graph_exec.graph_id,
                    "graph_exec_id": graph_exec.graph_exec_id,
                    "moderation_type": "execution_input",
                },
            )

            if not moderation_passed:
                logger.warning(
                    f"Moderation failed for graph execution {graph_exec.graph_exec_id}"
                )
                # Update node statuses for frontend display before raising error
                await self._update_failed_nodes_for_moderation(
                    db_client, graph_exec.graph_exec_id, "input", content_id
                )

                return ModerationError(
                    message="Execution failed due to input content moderation",
                    user_id=graph_exec.user_id,
                    graph_exec_id=graph_exec.graph_exec_id,
                    moderation_type="input",
                    content_id=content_id,
                )

            return None

        except asyncio.TimeoutError:
            logger.warning(
                f"Input moderation timed out for graph execution {graph_exec.graph_exec_id}, bypassing moderation"
            )
            return None  # Bypass moderation on timeout
        except Exception as e:
            logger.warning(f"Input moderation execution failed: {e}")
            return ModerationError(
                message="Execution failed due to input content moderation error",
                user_id=graph_exec.user_id,
                graph_exec_id=graph_exec.graph_exec_id,
                moderation_type="input",
            )

    async def moderate_graph_execution_outputs(
        self,
        db_client: "DatabaseManagerAsyncClient",
        graph_exec_id: str,
        user_id: str,
        graph_id: str,
        timeout: int = 10,
    ) -> Exception | None:
        """
        Complete output moderation flow for graph execution
        Returns: error_if_failed (None means success)
        """
        if not self.config.enabled:
            return None

        # Check if AutoMod feature is enabled for this user
        if not await is_feature_enabled(Flag.AUTOMOD, user_id):
            logger.debug(f"AutoMod feature not enabled for user {user_id}")
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
        logger.warning(f"Moderating outputs for graph execution {graph_exec_id}")
        try:
            moderation_passed, content_id = await self._moderate_content(
                content,
                {
                    "user_id": user_id,
                    "graph_id": graph_id,
                    "graph_exec_id": graph_exec_id,
                    "moderation_type": "execution_output",
                },
            )

            if not moderation_passed:
                logger.warning(f"Moderation failed for graph execution {graph_exec_id}")
                # Update node statuses for frontend display before raising error
                await self._update_failed_nodes_for_moderation(
                    db_client, graph_exec_id, "output", content_id
                )

                return ModerationError(
                    message="Execution failed due to output content moderation",
                    user_id=user_id,
                    graph_exec_id=graph_exec_id,
                    moderation_type="output",
                    content_id=content_id,
                )

            return None

        except asyncio.TimeoutError:
            logger.warning(
                f"Output moderation timed out for graph execution {graph_exec_id}, bypassing moderation"
            )
            return None  # Bypass moderation on timeout
        except Exception as e:
            logger.warning(f"Output moderation execution failed: {e}")
            return ModerationError(
                message="Execution failed due to output content moderation error",
                user_id=user_id,
                graph_exec_id=graph_exec_id,
                moderation_type="output",
            )

    async def _update_failed_nodes_for_moderation(
        self,
        db_client: "DatabaseManagerAsyncClient",
        graph_exec_id: str,
        moderation_type: Literal["input", "output"],
        content_id: str | None = None,
    ):
        """Update node execution statuses for frontend display when moderation fails"""
        # Import here to avoid circular imports
        from backend.executor.manager import send_async_execution_update

        if moderation_type == "input":
            # For input moderation, mark queued/running/incomplete nodes as failed
            target_statuses = [
                ExecutionStatus.QUEUED,
                ExecutionStatus.RUNNING,
                ExecutionStatus.INCOMPLETE,
            ]
        else:
            # For output moderation, mark completed nodes as failed
            target_statuses = [ExecutionStatus.COMPLETED]

        # Get the executions that need to be updated
        executions_to_update = await db_client.get_node_executions(
            graph_exec_id, statuses=target_statuses, include_exec_data=True
        )

        if not executions_to_update:
            return

        # Create error message with content_id if available
        error_message = "Failed due to content moderation"
        if content_id:
            error_message += f" (Moderation ID: {content_id})"

        # Prepare database update tasks
        exec_updates = []
        for exec_entry in executions_to_update:
            # Collect all input and output names to clear
            cleared_inputs = {}
            cleared_outputs = {}

            if exec_entry.input_data:
                for name in exec_entry.input_data.keys():
                    cleared_inputs[name] = [error_message]

            if exec_entry.output_data:
                for name in exec_entry.output_data.keys():
                    cleared_outputs[name] = [error_message]

            # Add update task to list
            exec_updates.append(
                db_client.update_node_execution_status(
                    exec_entry.node_exec_id,
                    status=ExecutionStatus.FAILED,
                    stats={
                        "error": error_message,
                        "cleared_inputs": cleared_inputs,
                        "cleared_outputs": cleared_outputs,
                    },
                )
            )

        # Execute all database updates in parallel
        updated_execs = await asyncio.gather(*exec_updates)

        # Send all websocket updates in parallel
        await asyncio.gather(
            *[
                send_async_execution_update(updated_exec)
                for updated_exec in updated_execs
            ]
        )

    async def _moderate_content(
        self, content: str, metadata: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Moderate content using AutoMod API

        Returns:
            Tuple of (approval_status, content_id)
            - approval_status: True if approved or timeout occurred, False if rejected
            - content_id: Reference ID from moderation API, or None if not available

        Raises:
            asyncio.TimeoutError: When moderation times out (should be bypassed)
        """
        try:
            request_data = AutoModRequest(
                type="text",
                content=content,
                metadata=metadata,
            )

            response = await self._make_request(request_data)

            if response.success and response.status == "approved":
                logger.debug(
                    f"Content approved for {metadata.get('graph_exec_id', 'unknown')}"
                )
                return True, response.content_id
            else:
                reasons = [r.reason for r in response.moderation_results if r.reason]
                error_msg = f"Content rejected by AutoMod: {'; '.join(reasons)}"
                logger.warning(f"Content rejected: {error_msg}")
                return False, response.content_id

        except asyncio.TimeoutError:
            # Re-raise timeout to be handled by calling methods
            logger.warning(
                f"AutoMod API timeout for {metadata.get('graph_exec_id', 'unknown')}"
            )
            raise
        except Exception as e:
            logger.error(f"AutoMod moderation error: {e}")
            return self.config.fail_open, None

    async def _make_request(self, request_data: AutoModRequest) -> AutoModResponse:
        """Make HTTP request to AutoMod API using the standard request utility"""
        url = f"{self.config.api_url}/moderate"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.config.api_key.strip(),
        }

        # Create requests instance with timeout and retry configuration
        requests = Requests(
            extra_headers=headers,
            retry_max_wait=float(self.config.timeout),
        )

        try:
            response = await requests.post(
                url, json=request_data.model_dump(), timeout=self.config.timeout
            )

            response_data = response.json()
            return AutoModResponse.model_validate(response_data)

        except asyncio.TimeoutError:
            # Re-raise timeout error to be caught by _moderate_content
            raise
        except (json.JSONDecodeError, ValidationError) as e:
            raise Exception(f"Invalid response from AutoMod API: {e}")
        except Exception as e:
            # Check if this is an aiohttp timeout that we should convert
            if "timeout" in str(e).lower():
                raise asyncio.TimeoutError(f"AutoMod API request timed out: {e}")
            raise Exception(f"AutoMod API request failed: {e}")


# Global instance
automod_manager = AutoModManager()
