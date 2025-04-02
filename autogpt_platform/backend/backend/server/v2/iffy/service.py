import logging

import requests
from autogpt_libs.utils.cache import thread_cached

from backend.util import json
from backend.util.openrouter import open_router_moderate_content
from backend.util.service import get_service_client
from backend.util.settings import BehaveAs, Settings

from .models import BlockContentForModeration, IffyPayload, ModerationResult, UserData

logger = logging.getLogger(__name__)
settings = Settings()


@thread_cached
def get_db():
    from backend.executor.database import DatabaseManager

    return get_service_client(DatabaseManager)


class IffyService:
    """Service class for handling content moderation through Iffy API"""

    @staticmethod
    def get_user_data(user_id: str) -> UserData:
        """Get user data for Iffy API from user_id"""
        # Initialize with default values
        user_data: UserData = {
            "clientId": user_id,
            "email": None,
            "name": None,
        }

        try:
            user = get_db().get_user_info_by_id(user_id)
            if user:
                user_data.update(
                    {
                        "id": user["id"],
                        "name": user["name"],
                        "email": user["email"],
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to get user details for {user_id}: {str(e)}")

        return user_data

    @staticmethod
    def moderate_content(
        user_id: str, block_content: BlockContentForModeration
    ) -> ModerationResult:
        """
        Send block content to Iffy for content moderation.
        Only used in cloud mode - local mode skips moderation entirely.

        Args:
            user_id: The ID of the user executing the block
            block_content: The content of the block to be moderated (BlockContentForModeration model)

        Returns:
            ModerationResult: Result of the moderation check
        """

        IFFY_API_KEY = settings.secrets.iffy_api_key
        IFFY_API_URL = settings.secrets.iffy_api_url

        if settings.config.behave_as == BehaveAs.LOCAL:
            logger.info("Content moderation skipped - running in local mode")
            return ModerationResult(
                is_safe=True, reason="Moderation skipped - running in local mode"
            )

        # Validate Iffy API URL and key at the start
        if not IFFY_API_URL or not IFFY_API_KEY:
            logger.warning(
                "Iffy API URL or key not configured, falling back to OpenRouter moderation"
            )
            input_data = json.dumps(block_content.input_data)
            is_safe, reason = open_router_moderate_content(input_data)
            return ModerationResult(
                is_safe=is_safe,
                reason=f"Iffy not configured. OpenRouter result: {reason}",
            )

        try:
            # Validate URL format
            if not IFFY_API_URL.startswith(("http://", "https://")):
                logger.error(f"Invalid Iffy API URL format: {IFFY_API_URL}")
                input_data = json.dumps(block_content.input_data)
                is_safe, reason = open_router_moderate_content(input_data)
                return ModerationResult(
                    is_safe=is_safe, reason="Invalid Iffy API URL format"
                )

            headers = {
                "Authorization": f"Bearer {IFFY_API_KEY}",
                "Content-Type": "application/json",
            }

            input_data = json.dumps(block_content.input_data)
            user_data = IffyService.get_user_data(user_id)

            # Prepare the metadata
            metadata = {
                "graphId": str(block_content.graph_id),
                "graphExecutionId": str(block_content.graph_exec_id),
                "nodeId": str(block_content.node_id),
                "blockId": str(block_content.block_id),
                "blockName": str(block_content.block_name),
            }

            name = f"{block_content.block_name}-{block_content.block_id}"
            graph_execution_id = (
                f"{block_content.graph_exec_id}-{block_content.node_id}"
            )

            # Create the payload
            payload = IffyPayload(
                clientId=graph_execution_id,
                name=name,
                metadata=metadata,
                content={"text": input_data, "imageUrls": []},
                # Only include user data values that are not None
                user={k: v for k, v in user_data.items() if v is not None},
            )

            logger.info(
                f"Sending content to Iffy for moderation - User: {user_data['name'] or user_id}, Block: {name}"
            )

            base_url = IFFY_API_URL.rstrip("/")
            api_path = "/api/v1/ingest"
            response = requests.post(
                f"{base_url}{api_path}", json=payload.model_dump(), headers=headers
            )

            if response.status_code != 200:
                logger.info(
                    f"Iffy moderation failed, falling back to OpenRouter. Status: {response.status_code}, Response: {response.text}"
                )
                is_safe, reason = open_router_moderate_content(input_data)
                if is_safe:
                    logger.info(f"OpenRouter moderation passed. Block: {name}")
                else:
                    logger.info(
                        f"OpenRouter moderation flagged content. Block: {name}, Reason: {reason}"
                    )
                return ModerationResult(is_safe=is_safe, reason=reason)

            logger.info(f"Successfully sent content to Iffy. Block: {name}")
            return ModerationResult(is_safe=True, reason="")

        except Exception as e:
            logger.error(
                f"Error in primary moderation service: {str(e)}", exc_info=True
            )
            try:
                block_name = f"{block_content.block_name}-{block_content.block_id}"
                input_data = json.dumps(block_content.input_data)
                is_safe, reason = open_router_moderate_content(input_data)
                if is_safe:
                    logger.info(
                        f"OpenRouter moderation passed after Iffy failure. Block: {block_name}"
                    )
                else:
                    logger.warning(
                        f"OpenRouter moderation flagged content after Iffy failure. Block: {block_name}, Reason: {reason}"
                    )
                return ModerationResult(is_safe=is_safe, reason=reason)
            except Exception as e2:
                block_name = (
                    getattr(block_content, "block_name", "unknown")
                    + "-"
                    + str(getattr(block_content, "block_id", "unknown"))
                )
                reason = f"Both moderation services failed. Error: {str(e2)}"
                logger.error(f"{reason}. Block: {block_name}", exc_info=True)
                return ModerationResult(is_safe=False, reason=reason)
