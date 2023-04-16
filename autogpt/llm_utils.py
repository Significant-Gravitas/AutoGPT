from __future__ import annotations

import openai
from colorama import Fore

from langchain.schema import BaseMessage, SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from autogpt.config import Config
from autogpt.logs import logger

CFG = Config()

openai.api_key = CFG.openai_api_key


def call_ai_function(
    function: str, args: list, description: str, model: str | None = None
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
        model = CFG.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args = ", ".join(args)
    messages = [
        SystemMessage(content=f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value."),
        HumanMessage(content=args)
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)


def create_chat_completion(
    messages: list[BaseMessage],
    model: str | None = None,
    temperature: float = CFG.temperature,
    max_tokens: int | None = None,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    num_retries = 10
    warned_user = False
    if CFG.debug_mode:
        print(
            Fore.GREEN
            + f"Creating chat completion with model {model}, temperature {temperature},"
            f" max_tokens {max_tokens}" + Fore.RESET
        )

    chat = (
        AzureChatOpenAI(
            deployment_name=CFG.get_azure_deployment_id_for_model(model),
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=num_retries
        ) if CFG.use_azure else ChatOpenAI(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=num_retries
        )
    )

    response = chat(messages)

    return response.content


def create_embedding_with_ada(text) -> list:
    """Create an embedding with text-ada-002 using the OpenAI SDK"""
    num_retries = 10

    model_ada = "text-embedding-ada-002"
    model = CFG.get_azure_deployment_id_for_model(model_ada) if CFG.use_azure else model_ada

    embeddings = OpenAIEmbeddings(
        model=model,
        max_retries=num_retries
    )

    return embeddings.embed_query(text)


def get_num_tokens_from_messages(model, messages):
    return ChatOpenAI(model_name=model, openai_api_key=CFG.openai_api_key).get_num_tokens_from_messages(messages)
