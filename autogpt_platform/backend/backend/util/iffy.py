import aiohttp
import logging
from typing import Any, Dict, Tuple, TypedDict, Optional
import json
from backend.data.user import get_user_by_id
from backend.util.settings import Settings
from backend.data.db import prisma
from backend.util.openrouter import moderate_content

logger = logging.getLogger(__name__)
settings = Settings()

IFFY_API_KEY = settings.secrets.iffy_api_key
IFFY_API_URL = settings.secrets.iffy_api_url

class UserData(TypedDict):
    clientId: str
    email: Optional[str]
    name: Optional[str]
    username: Optional[str]

async def send_to_iffy(user_id: str, block_content: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Send block content to Iffy for content moderation.
    Only used in cloud mode - local mode skips moderation entirely.
    
    Args:
        user_id: The ID of the user executing the block
        block_content: The content of the block to be moderated
        
    Returns:
        Tuple[bool, str]: (is_safe, reason)
        - is_safe: True if content was sent to Iffy successfully or passed OpenRouter moderation,
                  False if both services failed or content was flagged
        - reason: Description of the result or error
    """
    if settings.config.behave_as == "local":
        logger.info("Content moderation skipped - running in local mode")
        return True, "Moderation skipped - running in local mode"

    # Validate Iffy API URL and key at the start
    if not IFFY_API_URL or not IFFY_API_KEY:
        logger.warning("Iffy API URL or key not configured, falling back to OpenRouter moderation")
        is_safe, reason = await moderate_content(json.dumps(block_content.get('input_data', {}), indent=2))
        if not is_safe:
            logger.error(f"OpenRouter moderation failed after Iffy configuration issue: {reason}")
        return is_safe, f"Iffy not configured. OpenRouter result: {reason}"

    try:
        # Validate URL format
        if not IFFY_API_URL.startswith(('http://', 'https://')):
            logger.error(f"Invalid Iffy API URL format: {IFFY_API_URL}")
            return await moderate_content(json.dumps(block_content.get('input_data', {}), indent=2)), "Invalid Iffy API URL format"

        headers = {
            "Authorization": f"Bearer {IFFY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        input_data = json.dumps(block_content.get('input_data', {}), indent=2)

        # Get user details with proper connection handling
        user_data: UserData = {
            "clientId": user_id,
            "email": None,
            "name": None,
            "username": None,
        }

        try:
            if not prisma.is_connected():
                await prisma.connect()
            user = await get_user_by_id(user_id)
            if user:
                user_data.update({
                    "email": user.email or user_data["email"],
                    "name": user.name,
                    "username": user.username if hasattr(user, 'username') else user.name,
                })
        except Exception as e:
            logger.warning(f"Failed to get user details for {user_id}: {str(e)}")


        metadata = {
            "graphId": str(block_content.get('graph_id', '')),
            "graphExecutionId": str(block_content['graph_exec_id']),
            "nodeId": str(block_content['node_id']),
            "blockId": str(block_content['block_id']),
            "blockName": str(block_content['block_name']),
        }

        name = f"{block_content['block_name']}-{block_content['block_id']}"
        graphExecutionId = f"{block_content['graph_exec_id']}-{block_content['node_id']}"

        payload = {
            "clientId": graphExecutionId,
            "name": name,
            "entity": "block_execution",
            "metadata": metadata,
            "content": {
                "text": input_data,
                "imageUrls": []
            },
            # Only include user data if it's not None
            "user": {k: v for k, v in user_data.items() if v is not None}
        }
        
        logger.info(f"Sending content to Iffy for moderation - User: {user_data['name'] or user_id}, Block: {name}")
        
        async with aiohttp.ClientSession() as session:
            base_url = IFFY_API_URL.rstrip('/')
            api_path = '/api/v1/ingest'
            async with session.post(f"{base_url}{api_path}", json=payload, headers=headers) as response:
                response_text = await response.text()
                if response.status != 200:
                    logger.info(f"Iffy moderation failed, falling back to OpenRouter. Status: {response.status}, Response: {response_text}")
                    # Fall back to OpenRouter moderation
                    is_safe, reason = await moderate_content(input_data)
                    if is_safe:
                        logger.info(f"OpenRouter moderation passed. Block: {name}")
                    else:
                        logger.info(f"OpenRouter moderation flagged content. Block: {name}, Reason: {reason}")
                    return is_safe, reason
                else:
                    logger.info(f"Successfully sent content to Iffy. Block: {name}")
                    return True, ""
                    
    except Exception as e:
        logger.error(f"Error in primary moderation service: {str(e)}", exc_info=True)
        try:
            # Last attempt with OpenRouter
            result = await moderate_content(json.dumps(block_content.get('input_data', {}), indent=2))
            is_safe, reason = result
            if is_safe:
                logger.info(f"OpenRouter moderation passed after Iffy failure. Block: {name}")
            else:
                logger.warning(f"OpenRouter moderation flagged content after Iffy failure. Block: {name}, Reason: {reason}")
            return is_safe, reason
        except Exception as e2:
            reason = f"Both moderation services failed. Error: {str(e2)}"
            logger.error(f"{reason}. Block: {name}", exc_info=True)
            return False, reason
        
    finally:
        # Ensure we disconnect from Prisma if we connected
        try:
            if prisma.is_connected():
                await prisma.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting from Prisma: {str(e)}") 