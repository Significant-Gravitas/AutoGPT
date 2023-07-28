from __future__ import annotations

from typing import List, Literal, Optional

from colorama import Fore

from autogpt.config import Config

from ..api_manager import ApiManager
from ..base import (
    ChatModelResponse,
    ChatSequence,
    FunctionCallDict,
    Message,
    ResponseMessageDict,
)
from ..providers import openai as iopenai
from ..providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAIFunctionCall,
    OpenAIFunctionSpec,
    count_openai_functions_tokens,
)
from .token_counter import *


def call_ai_function(
    function: str,
    args: list,
    description: str,
    config: Config,
    model: Optional[str] = None,
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    if model is None:
        model = config.smart_llm
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    arg_str: str = ", ".join(args)

    prompt = ChatSequence.for_model(
        model,
        [
            Message(
                "system",
                f"You are now the following python function: ```# {description}"
                f"\n{function}```\n\nOnly respond with your `return` value.",
            ),
            Message("user", arg_str),
        ],
    )
    return create_chat_completion(prompt=prompt, temperature=0, config=config).content


def create_text_completion(
    prompt: str,
    config: Config,
    model: Optional[str],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> str:
    if model is None:
        model = config.fast_llm
    if temperature is None:
        temperature = config.temperature

    kwargs = {"model": model}
    kwargs.update(config.get_openai_credentials(model))

    response = iopenai.create_text_completion(
        prompt=prompt,
        **kwargs,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    logger.debug(f"Response: {response}")

    return response.choices[0].text


# Overly simple abstraction until we create something better
def create_chat_completion(
    prompt: ChatSequence,
    config: Config,
    functions: Optional[List[OpenAIFunctionSpec]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatModelResponse:
    """Create a chat completion using the OpenAI API

    Args:
        messages (List[Message]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """

    if model is None:
        model = prompt.model.name
    if temperature is None:
        temperature = config.temperature
    if max_tokens is None:
        prompt_tlength = prompt.token_length
        max_tokens = OPEN_AI_CHAT_MODELS[model].max_tokens - prompt_tlength
        logger.debug(f"Prompt length: {prompt_tlength} tokens")
        if functions:
            functions_tlength = count_openai_functions_tokens(functions, model)
            max_tokens -= functions_tlength
            logger.debug(f"Functions take up {functions_tlength} tokens in API call")

    logger.debug(
        f"{Fore.GREEN}Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}{Fore.RESET}"
    )
    chat_completion_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for plugin in config.plugins:
        if plugin.can_handle_chat_completion(
            messages=prompt.raw(),
            **chat_completion_kwargs,
        ):
            message = plugin.handle_chat_completion(
                messages=prompt.raw(),
                **chat_completion_kwargs,
            )
            if message is not None:
                return message

    chat_completion_kwargs.update(config.get_openai_credentials(model))

    if functions:
        chat_completion_kwargs["functions"] = [
            function.schema for function in functions
        ]

    response = iopenai.create_chat_completion(
        messages=prompt.raw(),
        **chat_completion_kwargs,
    )
    logger.debug(f"Response: {response}")

    if hasattr(response, "error"):
        logger.error(response.error)
        raise RuntimeError(response.error)

    first_message: ResponseMessageDict = response.choices[0].message
    content: str | None = first_message.get("content")
    function_call: FunctionCallDict | None = first_message.get("function_call")

    for plugin in config.plugins:
        if not plugin.can_handle_on_response():
            continue
        # TODO: function call support in plugin.on_response()
        content = plugin.on_response(content)

    return ChatModelResponse(
        model_info=OPEN_AI_CHAT_MODELS[model],
        content=content,
        function_call=OpenAIFunctionCall(
            name=function_call["name"], arguments=function_call["arguments"]
        )
        if function_call
        else None,
    )
