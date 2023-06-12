from __future__ import annotations

from typing import List, Literal, Optional

from colorama import Fore

from autogpt.config import Config
from autogpt.logs import logger

from ..api_manager import ApiManager
from ..base import ChatSequence, Message
from ..providers import openai as iopenai
from .token_counter import *


def call_ai_function(
    function: str,
    args: list,
    description: str,
    model: Optional[str] = None,
    config: Optional[Config] = None,
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
        model = config.smart_llm_model
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
    return create_chat_completion(prompt=prompt, temperature=0)


def create_text_completion(
    prompt: str,
    model: Optional[str],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> str:
    cfg = Config()
    if model is None:
        model = cfg.fast_llm_model
    if temperature is None:
        temperature = cfg.temperature

    if cfg.use_azure:
        kwargs = {"deployment_id": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    response = iopenai.create_text_completion(
        prompt=prompt,
        **kwargs,
        temperature=temperature,
        max_tokens=max_output_tokens,
        api_key=cfg.openai_api_key,
    )
    logger.debug(f"Response: {response}")

    return response.choices[0].text


# Overly simple abstraction until we create something better
def create_chat_completion(
    prompt: ChatSequence,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (List[Message]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    cfg = Config()
    if model is None:
        model = prompt.model.name
    if temperature is None:
        temperature = cfg.temperature

    logger.debug(
        f"{Fore.GREEN}Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}{Fore.RESET}"
    )
    chat_completion_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for plugin in cfg.plugins:
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

    chat_completion_kwargs["api_key"] = cfg.openai_api_key
    if cfg.use_azure:
        chat_completion_kwargs["deployment_id"] = cfg.get_azure_deployment_id_for_model(
            model
        )

    response = iopenai.create_chat_completion(
        messages=prompt.raw(),
        **chat_completion_kwargs,
    )
    logger.debug(f"Response: {response}")

    resp = ""
    if not hasattr(response, "error"):
        resp = response.choices[0].message["content"]
    else:
        logger.error(response.error)
        raise RuntimeError(response.error)

    for plugin in cfg.plugins:
        if not plugin.can_handle_on_response():
            continue
        resp = plugin.on_response(resp)

    return resp


def check_model(
    model_name: str, model_type: Literal["smart_llm_model", "fast_llm_model"]
) -> str:
    """Check if model is available for use. If not, return gpt-3.5-turbo."""
    api_manager = ApiManager()
    models = api_manager.get_models()

    if any(model_name in m["id"] for m in models):
        return model_name

    logger.typewriter_log(
        "WARNING: ",
        Fore.YELLOW,
        f"You do not have access to {model_name}. Setting {model_type} to "
        f"gpt-3.5-turbo.",
    )
    return "gpt-3.5-turbo"
