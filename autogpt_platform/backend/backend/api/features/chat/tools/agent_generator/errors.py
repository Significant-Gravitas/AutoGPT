"""Error handling utilities for agent generator."""


def get_user_message_for_error(
    error_type: str,
    operation: str = "process the request",
    llm_parse_message: str | None = None,
    validation_message: str | None = None,
    error_details: str | None = None,
) -> str:
    """Get a user-friendly error message based on error type.

    This function maps internal error types to user-friendly messages,
    providing a consistent experience across different agent operations.

    Args:
        error_type: The error type from the external service
            (e.g., "llm_parse_error", "timeout", "rate_limit")
        operation: Description of what operation failed, used in the default
            message (e.g., "analyze the goal", "generate the agent")
        llm_parse_message: Custom message for llm_parse_error type
        validation_message: Custom message for validation_error type
        error_details: Optional additional details about the error

    Returns:
        User-friendly error message suitable for display to the user
    """
    base_message = ""

    if error_type == "llm_parse_error":
        base_message = (
            llm_parse_message
            or "The AI had trouble processing this request. Please try again."
        )
    elif error_type == "validation_error":
        base_message = (
            validation_message
            or "The generated agent failed validation. "
            "This usually happens when the agent structure doesn't match "
            "what the platform expects. Please try simplifying your goal "
            "or breaking it into smaller parts."
        )
    elif error_type == "patch_error":
        base_message = (
            "Failed to apply the changes. The modification couldn't be "
            "validated. Please try a different approach or simplify the change."
        )
    elif error_type in ("timeout", "llm_timeout"):
        base_message = (
            "The request took too long to process. This can happen with "
            "complex agents. Please try again or simplify your goal."
        )
    elif error_type in ("rate_limit", "llm_rate_limit"):
        base_message = "The service is currently busy. Please try again in a moment."
    else:
        base_message = f"Failed to {operation}. Please try again."

    # Add error details if provided (for debugging, truncated)
    if error_details:
        # Truncate long error details
        details = (
            error_details[:200] + "..." if len(error_details) > 200 else error_details
        )
        base_message += f"\n\nTechnical details: {details}"

    return base_message
