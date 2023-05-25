from __future__ import annotations

from itertools import islice
from typing import List, Literal, Optional

import numpy as np
import tiktoken
from colorama import Fore

from autogpt.config import Config
from autogpt.llm.api_manager import ApiManager
from autogpt.llm.base import Message
from autogpt.llm.providers import openai
from autogpt.logs import logger


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
    cfg = Config()
    if model is None:
        model = cfg.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args: str = ", ".join(args)
    messages: List[Message] = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)


# Overly simple abstraction until we create something better
# simple retry mechanism when getting a rate error or a bad gateway
def create_chat_completion(
    messages: List[Message],  # type: ignore
    model: Optional[str] = None,
    temperature: float = None,
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
            messages=messages,
            **chat_completion_kwargs,
        ):
            message = plugin.handle_chat_completion(
                messages=messages,
                **chat_completion_kwargs,
            )
            if message is not None:
                return message
    api_manager = ApiManager()

    chat_completion_kwargs["api_key"] = cfg.openai_api_key
    if cfg.use_azure:
        chat_completion_kwargs["deployment_id"] = cfg.get_azure_deployment_id_for_model(
            model
        )

    response = openai.create_chat_completion(
        messages=messages,
        **chat_completion_kwargs,
    )
    if not hasattr(response, "error"):
        logger.debug(f"Response: {response}")
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        api_manager.update_cost(prompt_tokens, completion_tokens, model)

        resp = response.choices[0].message["content"]
    for plugin in cfg.plugins:
        if not plugin.can_handle_on_response():
            continue
        resp = plugin.on_response(resp)
    return resp


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def chunked_tokens(text, tokenizer_name, chunk_length):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    tokens = tokenizer.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


def get_ada_embedding(text: str) -> List[float]:
    """Get an embedding from the ada model.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding.
    """
    cfg = Config()
    model = cfg.embedding_model
    text = text.replace("\n", " ")

    if cfg.use_azure:
        kwargs = {"engine": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    embedding = create_embedding(text, **kwargs)
    return embedding


def create_embedding(
    text: str,
    *_,
    **kwargs,
) -> openai.Embedding:
    """Create an embedding using the OpenAI API

    Args:
        text (str): The text to embed.
        kwargs: Other arguments to pass to the OpenAI API embedding creation call.

    Returns:
        openai.Embedding: The embedding object.
    """
    cfg = Config()
    chunk_embeddings = []
    chunk_lengths = []
    for chunk in chunked_tokens(
        text,
        tokenizer_name=cfg.embedding_tokenizer,
        chunk_length=cfg.embedding_token_limit,
    ):
        embedding = openai.create_embedding(
            text=chunk,
            api_key=cfg.openai_api_key,
            **kwargs,
        )
        api_manager = ApiManager()
        api_manager.update_cost(
            prompt_tokens=embedding.usage.prompt_tokens,
            completion_tokens=0,
            model=cfg.embedding_model,
        )
        chunk_embeddings.append(embedding["data"][0]["embedding"])
        chunk_lengths.append(len(chunk))

    # do weighted avg
    chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lengths)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(
        chunk_embeddings
    )  # normalize the length to one
    chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings


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
