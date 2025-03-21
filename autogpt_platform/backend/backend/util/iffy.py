import logging
from typing import Any, Dict, Tuple
import json
from backend.server.v2.iffy.service import IffyService
from backend.util.settings import Settings
from backend.util.openrouter import open_router_moderate_content

logger = logging.getLogger(__name__)
settings = Settings()

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
    # Use the IffyService to handle moderation
    try:
        result = await IffyService._moderate_content(user_id, block_content)
        return result.is_safe, result.reason
    except Exception as e:
        logger.error(f"Error in IffyService: {str(e)}", exc_info=True)
        # Fall back to OpenRouter as a last resort
        try:
            is_safe, reason = await open_router_moderate_content(json.dumps(block_content.get('input_data', {}), indent=2))
            return is_safe, f"IffyService error, OpenRouter fallback: {reason}"
        except Exception as e2:
            logger.error(f"Both moderation services failed: {str(e2)}", exc_info=True)
            return False, f"All moderation services failed: {str(e)}, {str(e2)}" 