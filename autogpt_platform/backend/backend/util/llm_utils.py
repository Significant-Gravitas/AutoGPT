"""
Utilities for LLM interactions and structured response processing.
"""

import json
import secrets
from json import JSONDecodeError
from typing import Any

from backend.blocks.llm import LlmModel, llm_call, trim_prompt
from backend.data.model import APIKeyCredentials


async def structured_llm_call(
    credentials: APIKeyCredentials,
    llm_model: LlmModel,
    prompt: list[dict],
    expected_format: dict[str, str],
    max_tokens: int | None = None,
    retry: int = 3,
) -> dict[str, Any]:
    """
    Make a structured LLM call that returns a validated JSON response.

    Args:
        credentials: API credentials for the LLM provider
        llm_model: The language model to use
        prompt: The conversation prompt as a list of message dictionaries
        expected_format: Expected JSON structure with field descriptions
        max_tokens: Maximum tokens to generate
        retry: Number of retry attempts for validation failures

    Returns:
        Parsed and validated JSON response

    Raises:
        RuntimeError: If all retry attempts fail
    """
    # Use a one-time unique tag to prevent collisions with user/LLM content
    output_tag_id = secrets.token_hex(8)
    output_tag_start = f'<json_output id="{output_tag_id}">'

    # Add format instructions to the prompt
    expected_output_format = json.dumps(expected_format, indent=2).replace("\n", "\n|")

    format_instruction = f"""You must respond with a JSON object in the following format:
    |{output_tag_start}
    |{expected_output_format}
    |</json_output>
    
    |The JSON must contain all the specified fields. Make sure your response is valid JSON."""

    enhanced_prompt = prompt + [
        {"role": "system", "content": trim_prompt(format_instruction)}
    ]

    def validate_response(parsed: dict) -> str | None:
        try:
            if not isinstance(parsed, dict):
                return f"Expected a dictionary, but got {type(parsed)}"
            miss_keys = set(expected_format.keys()) - set(parsed.keys())
            if miss_keys:
                return f"Missing keys: {miss_keys}"
            return None
        except Exception as e:
            return f"Validation error: {e}"

    error_feedback_message = ""

    for retry_count in range(retry):
        try:
            llm_response = await llm_call(
                credentials=credentials,
                llm_model=llm_model,
                prompt=enhanced_prompt,
                max_tokens=max_tokens,
                compress_prompt_to_fit=True,
            )
            response_text = llm_response.response

            # Extract JSON from response
            if output_tag_start not in response_text:
                error_feedback_message = f"Response does not contain the expected {output_tag_start}...</json_output> block."
                enhanced_prompt.append({"role": "assistant", "content": response_text})
                enhanced_prompt.append(
                    {"role": "user", "content": error_feedback_message}
                )
                continue

            json_output = (
                response_text.split(output_tag_start, 1)[1]
                .rsplit("</json_output>", 1)[0]
                .strip()
            )

            try:
                response_obj = json.loads(json_output)
            except JSONDecodeError as parse_error:
                error_feedback_message = f"Invalid JSON in response: {parse_error}"
                enhanced_prompt.append({"role": "assistant", "content": response_text})
                enhanced_prompt.append(
                    {"role": "user", "content": error_feedback_message}
                )
                continue

            # Validate the response
            validation_error = validate_response(response_obj)
            if validation_error is None:
                return response_obj

            # Add error feedback for next attempt
            error_feedback_message = f"Response validation failed: {validation_error}. Please provide a valid JSON response with all required fields."
            enhanced_prompt.append({"role": "assistant", "content": response_text})
            enhanced_prompt.append({"role": "user", "content": error_feedback_message})

        except Exception as e:
            error_feedback_message = f"Error calling LLM: {e}"

    raise RuntimeError(
        f"Failed to get valid structured response after {retry} attempts: {error_feedback_message}"
    )
