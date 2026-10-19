"""Record turn input without passing image payloads to tracing integrations."""

import logging

from langfuse import get_client

logger = logging.getLogger(__name__)


def record_turn_input(prompt: str, file_ids: list[str] | None) -> None:
    """Keep the prompt on the parent trace even when SDK query hooks are bypassed."""
    trace_input: dict[str, str | list[str]] = {"prompt": prompt}
    if file_ids:
        trace_input["attachments"] = [f"workspace://{file_id}" for file_id in file_ids]
    try:
        client = get_client()
        client.update_current_trace(input=trace_input)
        client.update_current_span(input=trace_input)
    except Exception:
        logger.warning("Failed to record SDK turn input", exc_info=True)
