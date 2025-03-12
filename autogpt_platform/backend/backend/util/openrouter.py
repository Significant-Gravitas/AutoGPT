import logging
from typing import Tuple
import openai
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

API_KEY = settings.secrets.open_router_api_key

MODERATION_PROMPT = """You are a content moderation AI. Your task is to analyze the following content and determine if it contains any inappropriate, harmful, or malicious content. Please respond with one of these exact words:
- FLAGGED: If the content contains harmful, inappropriate, or malicious content
- SAFE: If the content appears to be safe

Content to moderate:MODERATION_PROMPT 
{content}

Respond with only one word from the above choices."""

async def moderate_content(content: str) -> Tuple[bool, str]:
    """
    Use OpenRouter's API to moderate content using an LLM.
    Uses OpenRouter's auto-routing to select the best available model.
    
    Args:
        content: The content to be moderated
        
    Returns:
        Tuple[bool, str]: (is_safe, reason)
        - is_safe: True if content is safe, False if flagged
        - reason: The raw response from the LLM
    """
    # If API key is not configured, fail immediately
    if not API_KEY:
        logger.error("OpenRouter API key not configured")
        return False, "OpenRouter API key not configured"
        
    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
        )

        # Use up to 3 retries with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://agpt.co",
                        "X-Title": "AutoGPT",
                    },
                    model="openrouter/auto",  # Use auto-routing for best available model
                    messages=[
                        {"role": "system", "content": "You are a content moderation AI. Respond only with FLAGGED or SAFE."},
                        {"role": "user", "content": MODERATION_PROMPT.format(content=content)}
                    ],
                    max_tokens=10,
                    temperature=0.1,
                    timeout=10,
                )

                if not response.choices:
                    logger.error("No response from OpenRouter moderation")
                    return False, "No response from moderation service"

                result = response.choices[0].message.content.strip().upper()
                
                # Only consider content safe if explicitly marked as SAFE
                is_safe = result == "SAFE"
                
                if not is_safe:
                    logger.warning(f"Content moderation result: {result}")
                
                return is_safe, result

            except openai.APITimeoutError:
                if attempt == max_retries - 1:
                    logger.error("OpenRouter moderation timed out after all retries")
                    return False, "Moderation service timeout"
                logger.warning(f"OpenRouter timeout, attempt {attempt + 1} of {max_retries}")
                continue

            except openai.APIConnectionError:
                if attempt == max_retries - 1:
                    logger.error("OpenRouter connection error after all retries")
                    return False, "Moderation service connection error"
                logger.warning(f"OpenRouter connection error, attempt {attempt + 1} of {max_retries}")
                continue

            except Exception as e:
                logger.error(f"Unexpected error in OpenRouter moderation attempt {attempt + 1}: {str(e)}", exc_info=True)
                if attempt == max_retries - 1:
                    raise

    except Exception as e:
        logger.error(f"Error in OpenRouter moderation: {str(e)}", exc_info=True)
        return False, f"Moderation error: {str(e)}" 