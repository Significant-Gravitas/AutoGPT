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


def _create_error_response(
    error_message: str,
    error_type: str = "unknown",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized error response dict.

    Args:
        error_message: Human-readable error message
        error_type: Machine-readable error type
        details: Optional additional error details

    Returns:
        Error dict with type="error" and error details
    """
    response: dict[str, Any] = {
        "type": "error",
        "error": error_message,
        "error_type": error_type,
    }
    if details:
        response["details"] = details
    return response


def _classify_http_error(e: httpx.HTTPStatusError) -> tuple[str, str]:
    """Classify an HTTP error into error_type and message.

    Args:
        e: The HTTP status error

    Returns:
        Tuple of (error_type, error_message)
    """
    status = e.response.status_code
    if status == 429:
        return "rate_limit", f"Agent Generator rate limited: {e}"
    elif status == 503:
        return "service_unavailable", f"Agent Generator unavailable: {e}"
    elif status == 504 or status == 408:
        return "timeout", f"Agent Generator timed out: {e}"
    else:
        return "http_error", f"HTTP error calling Agent Generator: {e}"


def _classify_request_error(e: httpx.RequestError) -> tuple[str, str]:
    """Classify a request error into error_type and message.

    Args:
        e: The request error

    Returns:
        Tuple of (error_type, error_message)
    """
    error_str = str(e).lower()
    if "timeout" in error_str or "timed out" in error_str:
        return "timeout", f"Agent Generator request timed out: {e}"
    elif "connect" in error_str:
        return "connection_error", f"Could not connect to Agent Generator: {e}"
    else:
        return "request_error", f"Request error calling Agent Generator: {e}"


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
    description: str,
    context: str = "",
    library_agents: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Call the external service to decompose a goal.

    Args:
        description: Natural language goal description
        context: Additional context (e.g., answers to previous questions)
        library_agents: User's library agents available for sub-agent composition

    Returns:
        Dict with either:
        - {"type": "clarifying_questions", "questions": [...]}
        - {"type": "instructions", "steps": [...]}
        - {"type": "unachievable_goal", ...}
        - {"type": "vague_goal", ...}
        - {"type": "error", "error": "...", "error_type": "..."} on error
        Or None on unexpected error
    """
    client = _get_client()

    if context:
        description = f"{description}\n\nAdditional context from user:\n{context}"

    payload: dict[str, Any] = {"description": description}
    if library_agents:
        payload["library_agents"] = library_agents

    try:
        response = await client.post("/api/decompose-description", json=payload)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            error_msg = data.get("error", "Unknown error from Agent Generator")
            error_type = data.get("error_type", "unknown")
            logger.error(
                f"Agent Generator decomposition failed: {error_msg} "
                f"(type: {error_type})"
            )
            return _create_error_response(error_msg, error_type)

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
        elif response_type == "error":
            # Pass through error from the service
            return _create_error_response(
                data.get("error", "Unknown error"),
                data.get("error_type", "unknown"),
            )
        else:
            logger.error(
                f"Unknown response type from external service: {response_type}"
            )
            return _create_error_response(
                f"Unknown response type from Agent Generator: {response_type}",
                "invalid_response",
            )

    except httpx.HTTPStatusError as e:
        error_type, error_msg = _classify_http_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except httpx.RequestError as e:
        error_type, error_msg = _classify_request_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except Exception as e:
        error_msg = f"Unexpected error calling Agent Generator: {e}"
        logger.error(error_msg)
        return _create_error_response(error_msg, "unexpected_error")


async def generate_agent_external(
    instructions: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
    operation_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    """Call the external service to generate an agent from instructions.

    Args:
        instructions: Structured instructions from decompose_goal
        library_agents: User's library agents available for sub-agent composition
        operation_id: Operation ID for async processing (enables Redis Streams callback)
        task_id: Task ID for async processing (enables Redis Streams callback)

    Returns:
        Agent JSON dict, {"status": "accepted"} for async, or error dict {"type": "error", ...} on error
    """
    client = _get_client()

    # Build request payload
    payload: dict[str, Any] = {"instructions": instructions}
    if library_agents:
        payload["library_agents"] = library_agents
    if operation_id and task_id:
        payload["operation_id"] = operation_id
        payload["task_id"] = task_id

    try:
        response = await client.post("/api/generate-agent", json=payload)

        # Handle 202 Accepted for async processing
        if response.status_code == 202:
            logger.info(
                f"Agent Generator accepted async request "
                f"(operation_id={operation_id}, task_id={task_id})"
            )
            return {
                "status": "accepted",
                "operation_id": operation_id,
                "task_id": task_id,
            }

        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            error_msg = data.get("error", "Unknown error from Agent Generator")
            error_type = data.get("error_type", "unknown")
            logger.error(
                f"Agent Generator generation failed: {error_msg} (type: {error_type})"
            )
            return _create_error_response(error_msg, error_type)

        return data.get("agent_json")

    except httpx.HTTPStatusError as e:
        error_type, error_msg = _classify_http_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except httpx.RequestError as e:
        error_type, error_msg = _classify_request_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except Exception as e:
        error_msg = f"Unexpected error calling Agent Generator: {e}"
        logger.error(error_msg)
        return _create_error_response(error_msg, "unexpected_error")


async def generate_agent_patch_external(
    update_request: str,
    current_agent: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
    operation_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    """Call the external service to generate a patch for an existing agent.

    Args:
        update_request: Natural language description of changes
        current_agent: Current agent JSON
        library_agents: User's library agents available for sub-agent composition
        operation_id: Operation ID for async processing (enables Redis Streams callback)
        task_id: Task ID for async processing (enables Redis Streams callback)

    Returns:
        Updated agent JSON, clarifying questions dict, {"status": "accepted"} for async, or error dict on error
    """
    client = _get_client()

    # Build request payload
    payload: dict[str, Any] = {
        "update_request": update_request,
        "current_agent_json": current_agent,
    }
    if library_agents:
        payload["library_agents"] = library_agents
    if operation_id and task_id:
        payload["operation_id"] = operation_id
        payload["task_id"] = task_id

    try:
        response = await client.post("/api/update-agent", json=payload)

        # Handle 202 Accepted for async processing
        if response.status_code == 202:
            logger.info(
                f"Agent Generator accepted async update request "
                f"(operation_id={operation_id}, task_id={task_id})"
            )
            return {
                "status": "accepted",
                "operation_id": operation_id,
                "task_id": task_id,
            }

        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            error_msg = data.get("error", "Unknown error from Agent Generator")
            error_type = data.get("error_type", "unknown")
            logger.error(
                f"Agent Generator patch generation failed: {error_msg} "
                f"(type: {error_type})"
            )
            return _create_error_response(error_msg, error_type)

        # Check if it's clarifying questions
        if data.get("type") == "clarifying_questions":
            return {
                "type": "clarifying_questions",
                "questions": data.get("questions", []),
            }

        # Check if it's an error passed through
        if data.get("type") == "error":
            return _create_error_response(
                data.get("error", "Unknown error"),
                data.get("error_type", "unknown"),
            )

        # Otherwise return the updated agent JSON
        return data.get("agent_json")

    except httpx.HTTPStatusError as e:
        error_type, error_msg = _classify_http_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except httpx.RequestError as e:
        error_type, error_msg = _classify_request_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except Exception as e:
        error_msg = f"Unexpected error calling Agent Generator: {e}"
        logger.error(error_msg)
        return _create_error_response(error_msg, "unexpected_error")


async def customize_template_external(
    template_agent: dict[str, Any],
    modification_request: str,
    context: str = "",
) -> dict[str, Any] | None:
    """Call the external service to customize a template/marketplace agent.

    Args:
        template_agent: The template agent JSON to customize
        modification_request: Natural language description of customizations
        context: Additional context (e.g., answers to previous questions)

    Returns:
        Customized agent JSON, clarifying questions dict, or error dict on error
    """
    client = _get_client()

    request = modification_request
    if context:
        request = f"{modification_request}\n\nAdditional context from user:\n{context}"

    payload: dict[str, Any] = {
        "template_agent_json": template_agent,
        "modification_request": request,
    }

    try:
        response = await client.post("/api/template-modification", json=payload)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            error_msg = data.get("error", "Unknown error from Agent Generator")
            error_type = data.get("error_type", "unknown")
            logger.error(
                f"Agent Generator template customization failed: {error_msg} "
                f"(type: {error_type})"
            )
            return _create_error_response(error_msg, error_type)

        # Check if it's clarifying questions
        if data.get("type") == "clarifying_questions":
            return {
                "type": "clarifying_questions",
                "questions": data.get("questions", []),
            }

        # Check if it's an error passed through
        if data.get("type") == "error":
            return _create_error_response(
                data.get("error", "Unknown error"),
                data.get("error_type", "unknown"),
            )

        # Otherwise return the customized agent JSON
        return data.get("agent_json")

    except httpx.HTTPStatusError as e:
        error_type, error_msg = _classify_http_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except httpx.RequestError as e:
        error_type, error_msg = _classify_request_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except Exception as e:
        error_msg = f"Unexpected error calling Agent Generator: {e}"
        logger.error(error_msg)
        return _create_error_response(error_msg, "unexpected_error")


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
