"""Error handling utilities for agent generator."""


def get_user_message_for_error(
    error_type: str,
    operation: str = "process the request",
    llm_parse_message: str | None = None,
    validation_message: str | None = None,
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

    Returns:
        User-friendly error message suitable for display to the user
    """
    if error_type == "llm_parse_error":
        return (
            llm_parse_message
            or "The AI had trouble processing this request. Please try again."
        )
    elif error_type == "validation_error":
        return (
            validation_message
            or "The request failed validation. Please try rephrasing."
        )
    elif error_type == "patch_error":
        return "Failed to apply the changes. Please try a different approach."
    elif error_type in ("timeout", "llm_timeout"):
        return "The request took too long. Please try again."
    elif error_type in ("rate_limit", "llm_rate_limit"):
        return "The service is currently busy. Please try again in a moment."
    else:
        return f"Failed to {operation}. Please try again."
