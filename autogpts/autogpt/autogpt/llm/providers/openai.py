from __future__ import annotations

import enum
import functools
import logging
import time
from typing import Callable, TypeVar
from unittest.mock import patch

import openai
import openai.api_resources.abstract.engine_api_resource as engine_api_resource
from colorama import Fore, Style
from openai.error import APIError, RateLimitError, ServiceUnavailableError, Timeout
from openai.openai_object import OpenAIObject

from autogpt.core.resource.model_providers import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.logs.helpers import request_user_double_check
from autogpt.models.command_registry import CommandRegistry

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable)


def meter_api(func: T) -> T:
    """Adds ApiManager metering to functions which make OpenAI API calls"""
    from autogpt.llm.api_manager import ApiManager

    api_manager = ApiManager()

    openai_obj_processor = openai.util.convert_to_openai_object

    def update_usage_with_response(response: OpenAIObject):
        try:
            usage = response.usage
            logger.debug(f"Reported usage from call to model {response.model}: {usage}")
            api_manager.update_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens if "completion_tokens" in usage else 0,
                response.model,
            )
        except Exception as err:
            logger.warn(f"Failed to update API costs: {err.__class__.__name__}: {err}")

    def metering_wrapper(*args, **kwargs):
        openai_obj = openai_obj_processor(*args, **kwargs)
        if isinstance(openai_obj, OpenAIObject) and "usage" in openai_obj:
            update_usage_with_response(openai_obj)
        return openai_obj

    @functools.wraps(func)
    def metered_func(*args, **kwargs):
        with patch.object(
            engine_api_resource.util,
            "convert_to_openai_object",
            side_effect=metering_wrapper,
        ):
            return func(*args, **kwargs)

    return metered_func


def retry_api(
    max_retries: int = 10,
    backoff_base: float = 2.0,
    warn_user: bool = True,
):
    """Retry an OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """
    error_messages = {
        ServiceUnavailableError: "The OpenAI API engine is currently overloaded",
        RateLimitError: "Reached rate limit",
    }
    api_key_error_msg = (
        f"Please double check that you have setup a "
        f"{Style.BRIGHT}PAID{Style.NORMAL} OpenAI API Account. You can "
        f"read more here: {Fore.CYAN}https://docs.agpt.co/setup/#getting-an-api-key{Fore.RESET}"
    )
    backoff_msg = "Waiting {backoff} seconds..."

    def _wrapper(func: T) -> T:
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            user_warned = not warn_user
            max_attempts = max_retries + 1  # +1 for the first attempt
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except (RateLimitError, ServiceUnavailableError) as e:
                    if attempt >= max_attempts or (
                        # User's API quota exceeded
                        isinstance(e, RateLimitError)
                        and (err := getattr(e, "error", {}))
                        and err.get("code") == "insufficient_quota"
                    ):
                        raise

                    error_msg = error_messages[type(e)]
                    logger.warn(error_msg)
                    if not user_warned:
                        request_user_double_check(api_key_error_msg)
                        logger.debug(f"Status: {e.http_status}")
                        logger.debug(f"Response body: {e.json_body}")
                        logger.debug(f"Response headers: {e.headers}")
                        user_warned = True

                except (APIError, Timeout) as e:
                    if (e.http_status not in [429, 502]) or (attempt == max_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                logger.warn(backoff_msg.format(backoff=backoff))
                time.sleep(backoff)

        return _wrapped

    return _wrapper


def format_openai_function_for_prompt(func: CompletionModelFunction) -> str:
    """Returns the function formatted similarly to the way OpenAI does it internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    Example:
    ```ts
    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;
    ```
    """

    def param_signature(name: str, spec: JSONSchema) -> str:
        # TODO: enum type support
        type_dec = (
            spec.type if not spec.enum else " | ".join(repr(e) for e in spec.enum)
        )
        return (
            f"// {spec.description}\n" if spec.description else ""
        ) + f"{name}{'' if spec.required else '?'}: {type_dec},"

    return "\n".join(
        [
            f"// {func.description}",
            f"type {func.name} = (_ :{{",
            *[param_signature(name, p) for name, p in func.parameters.items()],
            "}) => any;",
        ]
    )


def get_openai_command_specs(
    command_registry: CommandRegistry,
) -> list[CompletionModelFunction]:
    """Get OpenAI-consumable function specs for the agent's available commands.
    see https://platform.openai.com/docs/guides/gpt/function-calling
    """
    return [
        CompletionModelFunction(
            name=command.name,
            description=command.description,
            parameters={
                param.name: JSONSchema(
                    type=param.type if type(param.type) == JSONSchema.Type else None,
                    enum=[v.value for v in type(param.type)]
                    if type(param.type) == enum.Enum
                    else None,
                    required=param.required,
                    description=param.description,
                )
                for param in command.parameters
            },
        )
        for command in command_registry.commands.values()
    ]


def count_openai_functions_tokens(
    functions: list[CompletionModelFunction], for_model: str
) -> int:
    """Returns the number of tokens taken up by a set of function definitions

    Reference: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18
    """
    from autogpt.llm.utils import (
        count_string_tokens,  # FIXME: maybe move to OpenAIProvider?
    )

    return count_string_tokens(
        f"# Tools\n\n## functions\n\n{format_function_specs_as_typescript_ns(functions)}",
        for_model,
    )


def format_function_specs_as_typescript_ns(
    functions: list[CompletionModelFunction],
) -> str:
    """Returns a function signature block in the format used by OpenAI internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    For use with `count_string_tokens` to determine token usage of provided functions.

    Example:
    ```ts
    namespace functions {

    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;

    } // namespace functions
    ```
    """

    return (
        "namespace functions {\n\n"
        + "\n\n".join(format_openai_function_for_prompt(f) for f in functions)
        + "\n\n} // namespace functions"
    )
