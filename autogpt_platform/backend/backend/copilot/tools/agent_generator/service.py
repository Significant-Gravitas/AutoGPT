"""External Agent Generator service client.

This module provides a client for communicating with the external Agent Generator
microservice. All generation endpoints use async polling: submit a job (202),
then poll GET /api/jobs/{job_id} every few seconds until the result is ready.
"""

import asyncio
import logging
import time
from typing import Any

import httpx

from backend.util.settings import Settings

from .dummy import (
    customize_template_dummy,
    decompose_goal_dummy,
    generate_agent_dummy,
    generate_agent_patch_dummy,
    get_blocks_dummy,
    health_check_dummy,
)

logger = logging.getLogger(__name__)

_dummy_mode_warned = False

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 10.0
MAX_POLL_TIME_SECONDS = 1800.0  # 30 minutes
MAX_CONSECUTIVE_POLL_ERRORS = 5


def _create_error_response(
    error_message: str,
    error_type: str = "unknown",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized error response dict."""
    response: dict[str, Any] = {
        "type": "error",
        "error": error_message,
        "error_type": error_type,
    }
    if details:
        response["details"] = details
    return response


def _classify_http_error(e: httpx.HTTPStatusError) -> tuple[str, str]:
    """Classify an HTTP error into error_type and message."""
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
    """Classify a request error into error_type and message."""
    error_str = str(e).lower()
    if "timeout" in error_str or "timed out" in error_str:
        return "timeout", f"Agent Generator request timed out: {e}"
    elif "connect" in error_str:
        return "connection_error", f"Could not connect to Agent Generator: {e}"
    else:
        return "request_error", f"Request error calling Agent Generator: {e}"


# ---------------------------------------------------------------------------
# Client / settings singletons
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None
_settings: Settings | None = None


def _get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _is_dummy_mode() -> bool:
    """Check if dummy mode is enabled for testing."""
    global _dummy_mode_warned
    settings = _get_settings()
    is_dummy = bool(settings.config.agentgenerator_use_dummy)
    if is_dummy and not _dummy_mode_warned:
        logger.warning(
            "Agent Generator running in DUMMY MODE - returning mock responses. "
            "Do not use in production!"
        )
        _dummy_mode_warned = True
    return is_dummy


def is_external_service_configured() -> bool:
    """Check if external Agent Generator service is configured (or dummy mode)."""
    settings = _get_settings()
    return bool(settings.config.agentgenerator_host) or bool(
        settings.config.agentgenerator_use_dummy
    )


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
        timeout = httpx.Timeout(float(settings.config.agentgenerator_timeout))
        _client = httpx.AsyncClient(
            base_url=_get_base_url(),
            timeout=timeout,
        )
    return _client


# ---------------------------------------------------------------------------
# Core polling helper
# ---------------------------------------------------------------------------


async def _submit_and_poll(
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Submit a job to the agent-generator and poll until the result is ready.

    The endpoint is expected to return 202 with ``{"job_id": "..."}`` on success.
    We then poll ``GET /api/jobs/{job_id}`` every ``POLL_INTERVAL_SECONDS``
    until the job completes or fails.

    Returns:
        The *result* dict from a completed job, or an error dict.
    """
    client = _get_client()

    # 1. Submit ----------------------------------------------------------------
    try:
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        error_type, error_msg = _classify_http_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)
    except httpx.RequestError as e:
        error_type, error_msg = _classify_request_error(e)
        logger.error(error_msg)
        return _create_error_response(error_msg, error_type)

    data = response.json()
    job_id = data.get("job_id")
    if not job_id:
        return _create_error_response(
            "Agent Generator did not return a job_id", "invalid_response"
        )

    logger.info(f"Agent Generator job submitted: {job_id} via {endpoint}")

    # 2. Poll ------------------------------------------------------------------
    start = time.monotonic()
    consecutive_errors = 0
    while (time.monotonic() - start) < MAX_POLL_TIME_SECONDS:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

        try:
            poll_resp = await client.get(f"/api/jobs/{job_id}")
            poll_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return _create_error_response(
                    "Agent Generator job not found or expired", "job_not_found"
                )
            status_code = e.response.status_code
            if status_code in {429, 503, 504, 408}:
                consecutive_errors += 1
                logger.warning(
                    f"Transient HTTP {status_code} polling job {job_id} "
                    f"({consecutive_errors}/{MAX_CONSECUTIVE_POLL_ERRORS}): {e}"
                )
                if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                    error_type, error_msg = _classify_http_error(e)
                    logger.error(
                        f"Giving up on job {job_id} after "
                        f"{MAX_CONSECUTIVE_POLL_ERRORS} consecutive poll errors: {error_msg}"
                    )
                    return _create_error_response(error_msg, error_type)
                continue
            error_type, error_msg = _classify_http_error(e)
            logger.error(f"Poll error for job {job_id}: {error_msg}")
            return _create_error_response(error_msg, error_type)
        except httpx.RequestError as e:
            consecutive_errors += 1
            logger.warning(
                f"Transient poll error for job {job_id} "
                f"({consecutive_errors}/{MAX_CONSECUTIVE_POLL_ERRORS}): {e}"
            )
            if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                error_msg = (
                    f"Giving up on job {job_id} after "
                    f"{MAX_CONSECUTIVE_POLL_ERRORS} consecutive poll errors: {e}"
                )
                logger.error(error_msg)
                return _create_error_response(error_msg, "poll_error")
            continue

        consecutive_errors = 0
        poll_data = poll_resp.json()
        status = poll_data.get("status")

        if status == "completed":
            logger.info(f"Agent Generator job {job_id} completed")
            result = poll_data.get("result", {})
            if not isinstance(result, dict):
                return _create_error_response(
                    "Agent Generator returned invalid result payload",
                    "invalid_response",
                )
            return result
        elif status == "failed":
            error_msg = poll_data.get("error", "Job failed")
            logger.error(f"Agent Generator job {job_id} failed: {error_msg}")
            return _create_error_response(error_msg, "job_failed")
        elif status in {"running", "pending", "queued"}:
            continue
        else:
            return _create_error_response(
                f"Agent Generator returned unexpected job status: {status}",
                "invalid_response",
            )

    return _create_error_response("Agent generation timed out after polling", "timeout")


def _extract_agent_json(result: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate agent_json from a job result.

    Returns the agent_json dict, or an error response if missing/invalid.
    """
    agent_json = result.get("agent_json")
    if not isinstance(agent_json, dict):
        return _create_error_response(
            "Agent Generator returned no agent_json in result", "invalid_response"
        )
    return agent_json


# ---------------------------------------------------------------------------
# Public functions — same signatures as before, now using polling
# ---------------------------------------------------------------------------


async def decompose_goal_external(
    description: str,
    context: str = "",
    library_agents: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Call the external service to decompose a goal.

    Returns one of the following dicts (keyed by ``"type"``):

    * ``{"type": "instructions", "steps": [...]}``
    * ``{"type": "clarifying_questions", "questions": [...]}``
    * ``{"type": "unachievable_goal", "reason": ..., "suggested_goal": ...}``
    * ``{"type": "vague_goal", "suggested_goal": ...}``
    * ``{"type": "error", "error": ..., "error_type": ...}``
    """
    if _is_dummy_mode():
        return await decompose_goal_dummy(description, context, library_agents)

    if context:
        description = f"{description}\n\nAdditional context from user:\n{context}"

    payload: dict[str, Any] = {"description": description}
    if library_agents:
        payload["library_agents"] = library_agents

    try:
        result = await _submit_and_poll("/api/decompose-description", payload)
    except Exception as e:
        error_msg = f"Unexpected error calling Agent Generator: {e}"
        logger.error(error_msg)
        return _create_error_response(error_msg, "unexpected_error")

    # The result dict from the job is already in the expected format
    # (type, steps, questions, etc.) — just return it as-is.
    if result.get("type") == "error":
        return result

    response_type = result.get("type")
    if response_type == "instructions":
        return {"type": "instructions", "steps": result.get("steps", [])}
    elif response_type == "clarifying_questions":
        return {
            "type": "clarifying_questions",
            "questions": result.get("questions", []),
        }
    elif response_type == "unachievable_goal":
        return {
            "type": "unachievable_goal",
            "reason": result.get("reason"),
            "suggested_goal": result.get("suggested_goal"),
        }
    elif response_type == "vague_goal":
        return {
            "type": "vague_goal",
            "suggested_goal": result.get("suggested_goal"),
        }
    else:
        logger.error(f"Unknown response type from Agent Generator job: {response_type}")
        return _create_error_response(
            f"Unknown response type: {response_type}",
            "invalid_response",
        )


async def generate_agent_external(
    instructions: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Call the external service to generate an agent from instructions.

    Returns:
        Agent JSON dict or error dict {"type": "error", ...} on error.
    """
    if _is_dummy_mode():
        return await generate_agent_dummy(instructions, library_agents)

    payload: dict[str, Any] = {"instructions": instructions}
    if library_agents:
        payload["library_agents"] = library_agents

    try:
        result = await _submit_and_poll("/api/generate-agent", payload)
    except Exception as e:
        error_msg = f"Unexpected error calling Agent Generator: {e}"
        logger.error(error_msg)
        return _create_error_response(error_msg, "unexpected_error")

    if result.get("type") == "error":
        return result

    return _extract_agent_json(result)


async def generate_agent_patch_external(
    update_request: str,
    current_agent: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Call the external service to generate a patch for an existing agent.

    Returns:
        Updated agent JSON, clarifying questions dict, or error dict.
    """
    if _is_dummy_mode():
        return await generate_agent_patch_dummy(
            update_request, current_agent, library_agents
        )

    payload: dict[str, Any] = {
        "update_request": update_request,
        "current_agent_json": current_agent,
    }
    if library_agents:
        payload["library_agents"] = library_agents

    try:
        result = await _submit_and_poll("/api/update-agent", payload)
    except Exception as e:
        error_msg = f"Unexpected error calling Agent Generator: {e}"
        logger.error(error_msg)
        return _create_error_response(error_msg, "unexpected_error")

    if result.get("type") == "error":
        return result

    if result.get("type") == "clarifying_questions":
        return {
            "type": "clarifying_questions",
            "questions": result.get("questions", []),
        }

    return _extract_agent_json(result)


async def customize_template_external(
    template_agent: dict[str, Any],
    modification_request: str,
    context: str = "",
) -> dict[str, Any] | None:
    """Call the external service to customize a template/marketplace agent.

    Returns:
        Customized agent JSON, clarifying questions dict, or error dict.
    """
    if _is_dummy_mode():
        return await customize_template_dummy(
            template_agent, modification_request, context
        )

    request_text = modification_request
    if context:
        request_text = (
            f"{modification_request}\n\nAdditional context from user:\n{context}"
        )

    payload: dict[str, Any] = {
        "template_agent_json": template_agent,
        "modification_request": request_text,
    }

    try:
        result = await _submit_and_poll("/api/template-modification", payload)
    except Exception as e:
        error_msg = f"Unexpected error calling Agent Generator: {e}"
        logger.error(error_msg)
        return _create_error_response(error_msg, "unexpected_error")

    if result.get("type") == "error":
        return result

    if result.get("type") == "clarifying_questions":
        return {
            "type": "clarifying_questions",
            "questions": result.get("questions", []),
        }

    return _extract_agent_json(result)


# ---------------------------------------------------------------------------
# Non-generation endpoints (still synchronous — quick responses)
# ---------------------------------------------------------------------------


async def get_blocks_external() -> list[dict[str, Any]] | None:
    """Get available blocks from the external service."""
    if _is_dummy_mode():
        return await get_blocks_dummy()

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
    """Check if the external service is healthy."""
    if not is_external_service_configured():
        return False

    if _is_dummy_mode():
        return await health_check_dummy()

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
