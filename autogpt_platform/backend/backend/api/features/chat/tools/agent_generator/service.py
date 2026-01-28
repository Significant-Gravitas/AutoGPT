"""External Agent Generator service client.

This module provides a client for communicating with the external Agent Generator
microservice. When AGENTGENERATOR_HOST is configured, the agent generation functions
will delegate to the external service instead of using the built-in LLM-based implementation.
"""

import logging
from typing import Any

import httpx

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None
_settings: Settings | None = None


def _get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def is_external_service_configured() -> bool:
    """Check if external Agent Generator service is configured."""
    settings = _get_settings()
    return bool(settings.config.agentgenerator_host)


def _get_base_url() -> str:
    """Get the base URL for the external service."""
    settings = _get_settings()
    host = settings.config.agentgenerator_host
    port = settings.config.agentgenerator_port
    return f"http://{host}:{port}"


def _get_client() -> httpx.AsyncClient:
    """Get or create the HTTP client for the external service."""
    global _client
    if _client is None:
        settings = _get_settings()
        _client = httpx.AsyncClient(
            base_url=_get_base_url(),
            timeout=httpx.Timeout(settings.config.agentgenerator_timeout),
        )
    return _client


async def decompose_goal_external(
    description: str, context: str = ""
) -> dict[str, Any] | None:
    """Call the external service to decompose a goal.

    Args:
        description: Natural language goal description
        context: Additional context (e.g., answers to previous questions)

    Returns:
        Dict with either:
        - {"type": "clarifying_questions", "questions": [...]}
        - {"type": "instructions", "steps": [...]}
        - {"type": "unachievable_goal", ...}
        - {"type": "vague_goal", ...}
        Or None on error
    """
    client = _get_client()

    # Build the request payload
    payload: dict[str, Any] = {"description": description}
    if context:
        # The external service uses user_instruction for additional context
        payload["user_instruction"] = context

    try:
        response = await client.post("/api/decompose-description", json=payload)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.error(f"External service returned error: {data.get('error')}")
            return None

        # Map the response to the expected format
        response_type = data.get("type")
        if response_type == "instructions":
            return {"type": "instructions", "steps": data.get("steps", [])}
        elif response_type == "clarifying_questions":
            return {
                "type": "clarifying_questions",
                "questions": data.get("questions", []),
            }
        elif response_type == "unachievable_goal":
            return {
                "type": "unachievable_goal",
                "reason": data.get("reason"),
                "suggested_goal": data.get("suggested_goal"),
            }
        elif response_type == "vague_goal":
            return {
                "type": "vague_goal",
                "suggested_goal": data.get("suggested_goal"),
            }
        else:
            logger.error(
                f"Unknown response type from external service: {response_type}"
            )
            return None

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling external agent generator: {e}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request error calling external agent generator: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling external agent generator: {e}")
        return None


async def generate_agent_external(
    instructions: dict[str, Any]
) -> dict[str, Any] | None:
    """Call the external service to generate an agent from instructions.

    Args:
        instructions: Structured instructions from decompose_goal

    Returns:
        Agent JSON dict or None on error
    """
    client = _get_client()

    try:
        response = await client.post(
            "/api/generate-agent", json={"instructions": instructions}
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.error(f"External service returned error: {data.get('error')}")
            return None

        return data.get("agent_json")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling external agent generator: {e}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request error calling external agent generator: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling external agent generator: {e}")
        return None


async def generate_agent_patch_external(
    update_request: str, current_agent: dict[str, Any]
) -> dict[str, Any] | None:
    """Call the external service to generate a patch for an existing agent.

    Args:
        update_request: Natural language description of changes
        current_agent: Current agent JSON

    Returns:
        Updated agent JSON, clarifying questions dict, or None on error
    """
    client = _get_client()

    try:
        response = await client.post(
            "/api/update-agent",
            json={
                "update_request": update_request,
                "current_agent_json": current_agent,
            },
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.error(f"External service returned error: {data.get('error')}")
            return None

        # Check if it's clarifying questions
        if data.get("type") == "clarifying_questions":
            return {
                "type": "clarifying_questions",
                "questions": data.get("questions", []),
            }

        # Otherwise return the updated agent JSON
        return data.get("agent_json")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling external agent generator: {e}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request error calling external agent generator: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling external agent generator: {e}")
        return None


async def get_blocks_external() -> list[dict[str, Any]] | None:
    """Get available blocks from the external service.

    Returns:
        List of block info dicts or None on error
    """
    client = _get_client()

    try:
        response = await client.get("/api/blocks")
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.error("External service returned error getting blocks")
            return None

        return data.get("blocks", [])

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting blocks from external service: {e}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request error getting blocks from external service: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting blocks from external service: {e}")
        return None


async def health_check() -> bool:
    """Check if the external service is healthy.

    Returns:
        True if healthy, False otherwise
    """
    if not is_external_service_configured():
        return False

    client = _get_client()

    try:
        response = await client.get("/health")
        response.raise_for_status()
        data = response.json()
        return data.get("status") == "healthy" and data.get("blocks_loaded", False)
    except Exception as e:
        logger.warning(f"External agent generator health check failed: {e}")
        return False


async def close_client() -> None:
    """Close the HTTP client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
