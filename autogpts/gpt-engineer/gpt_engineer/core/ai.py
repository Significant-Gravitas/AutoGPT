"""
This module provides an interface to interact with AI models.
It leverages the OpenAI GPT models and allows for integration with Azure-based instances of the same.
The AI class encapsulates the chat functionalities, allowing to start, advance, and manage a conversation with the model.

Key Features:
- Integration with Azure-based OpenAI instances through the LangChain AzureChatOpenAI class.
- Token usage logging to monitor the number of tokens consumed during a conversation.
- Seamless fallback to default models in case the desired model is unavailable.
- Serialization and deserialization of chat messages for easier transmission and storage.

Classes:
- AI: Main class providing chat functionalities.
- TokenUsage: Data class for logging token usage details.

Dependencies:
- langchain: For chat models and message schemas.
- openai: For the core GPT models interaction.
- tiktoken: For token counting.
- backoff: For handling rate limits and retries.
- dataclasses, json, and logging for internal functionalities.
- typing: For type hints.

For more specific details, refer to the docstrings within each class and function.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import List, Optional, Union

import backoff
import openai
import tiktoken

from langchain.callbacks.openai_info import MODEL_COST_PER_1K_TOKENS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

# Type hint for a chat message
Message = Union[AIMessage, HumanMessage, SystemMessage]

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    step_name: str
    in_step_prompt_tokens: int
    in_step_completion_tokens: int
    in_step_total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class AI:
    """
    A class to interface with a language model for chat-based interactions.

    This class provides methods to initiate and maintain conversations using
    a specified language model. It handles token counting, message creation,
    serialization and deserialization of chat messages, and interfaces with
    the language model to get AI-generated responses.

    Attributes
    ----------
    temperature : float
        The temperature setting for the model, affecting the randomness of the output.
    azure_endpoint : str
        The Azure endpoint URL, if applicable.
    model_name : str
        The name of the model being used.
    llm : Any
        The chat model instance.
    tokenizer : Any
        The tokenizer associated with the model.
    cumulative_prompt_tokens : int
        The running count of prompt tokens used.
    cumulative_completion_tokens : int
        The running count of completion tokens used.
    cumulative_total_tokens : int
        The running total of tokens used.
    token_usage_log : List[TokenUsage]
        A log of token usage details per step in the conversation.

    Methods
    -------
    start(system, user, step_name) -> List[Message]:
        Start the conversation with a system and user message.
    fsystem(msg) -> SystemMessage:
        Create a system message.
    fuser(msg) -> HumanMessage:
        Create a user message.
    fassistant(msg) -> AIMessage:
        Create an AI message.
    next(messages, prompt, step_name) -> List[Message]:
        Advance the conversation by interacting with the language model.
    backoff_inference(messages, callbacks) -> Any:
        Interact with the model using an exponential backoff strategy in case of rate limits.
    serialize_messages(messages) -> str:
        Serialize a list of messages to a JSON string.
    deserialize_messages(jsondictstr) -> List[Message]:
        Deserialize a JSON string into a list of messages.
    update_token_usage_log(messages, answer, step_name) -> None:
        Log the token usage details for the current step.
    format_token_usage_log() -> str:
        Format the token usage log as a CSV string.
    usage_cost() -> float:
        Calculate the total cost based on token usage and model pricing.
    num_tokens(txt) -> int:
        Count the number of tokens in a given text.
    num_tokens_from_messages(messages) -> int:
        Count the total number of tokens in a list of messages.
    """

    def __init__(self, model_name="gpt-4", temperature=0.1, azure_endpoint=""):
        """
        Initialize the AI class.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use, by default "gpt-4".
        temperature : float, optional
            The temperature to use for the model, by default 0.1.
        """
        self.temperature = temperature
        self.azure_endpoint = azure_endpoint
        self.model_name = (
            fallback_model(model_name) if azure_endpoint == "" else model_name
        )
        self.llm = create_chat_model(self, self.model_name, self.temperature)
        self.tokenizer = get_tokenizer(self.model_name)
        logger.debug(f"Using model {self.model_name} with llm {self.llm}")

        # initialize token usage log
        self.cumulative_prompt_tokens = 0
        self.cumulative_completion_tokens = 0
        self.cumulative_total_tokens = 0
        self.token_usage_log = []

    def start(self, system: str, user: str, step_name: str) -> List[Message]:
        """
        Start the conversation with a system message and a user message.

        Parameters
        ----------
        system : str
            The content of the system message.
        user : str
            The content of the user message.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The list of messages in the conversation.
        """
        messages: List[Message] = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        return self.next(messages, step_name=step_name)

    def fsystem(self, msg: str) -> SystemMessage:
        """
        Create a system message.

        Parameters
        ----------
        msg : str
            The content of the message.

        Returns
        -------
        SystemMessage
            The created system message.
        """
        return SystemMessage(content=msg)

    def fuser(self, msg: str) -> HumanMessage:
        """
        Create a user message.

        Parameters
        ----------
        msg : str
            The content of the message.

        Returns
        -------
        HumanMessage
            The created user message.
        """
        return HumanMessage(content=msg)

    def fassistant(self, msg: str) -> AIMessage:
        """
        Create an AI message.

        Parameters
        ----------
        msg : str
            The content of the message.

        Returns
        -------
        AIMessage
            The created AI message.
        """
        return AIMessage(content=msg)

    def next(
        self,
        messages: List[Message],
        prompt: Optional[str] = None,
        *,
        step_name: str,
    ) -> List[Message]:
        """
        Advances the conversation by sending message history
        to LLM and updating with the response.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        prompt : Optional[str], optional
            The prompt to use, by default None.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The updated list of messages in the conversation.
        """
        """
        Advances the conversation by sending message history
        to LLM and updating with the response.
        """
        if prompt:
            messages.append(self.fuser(prompt))

        logger.debug(f"Creating a new chat completion: {messages}")

        callbacks = [StreamingStdOutCallbackHandler()]
        response = self.backoff_inference(messages, callbacks)

        self.update_token_usage_log(
            messages=messages, answer=response.content, step_name=step_name
        )
        messages.append(response)
        logger.debug(f"Chat completion finished: {messages}")

        return messages

    @backoff.on_exception(
        backoff.expo, openai.error.RateLimitError, max_tries=7, max_time=45
    )
    def backoff_inference(self, messages, callbacks):
        """
        Perform inference using the language model while implementing an exponential backoff strategy.

        This function will retry the inference in case of a rate limit error from the OpenAI API.
        It uses an exponential backoff strategy, meaning the wait time between retries increases
        exponentially. The function will attempt to retry up to 7 times within a span of 45 seconds.

        Parameters
        ----------
        messages : List[Message]
            A list of chat messages which will be passed to the language model for processing.

        callbacks : List[Callable]
            A list of callback functions that are triggered after each inference. These functions
            can be used for logging, monitoring, or other auxiliary tasks.

        Returns
        -------
        Any
            The output from the language model after processing the provided messages.

        Raises
        ------
        openai.error.RateLimitError
            If the number of retries exceeds the maximum or if the rate limit persists beyond the
            allotted time, the function will ultimately raise a RateLimitError.

        Example
        -------
        >>> messages = [SystemMessage(content="Hello"), HumanMessage(content="How's the weather?")]
        >>> callbacks = [some_logging_callback]
        >>> response = backoff_inference(messages, callbacks)
        """
        return self.llm(messages, callbacks=callbacks)  # type: ignore

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        """
        Serialize a list of messages to a JSON string.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to serialize.

        Returns
        -------
        str
            The serialized messages as a JSON string.
        """
        return json.dumps(messages_to_dict(messages))

    @staticmethod
    def deserialize_messages(jsondictstr: str) -> List[Message]:
        """
        Deserialize a JSON string to a list of messages.

        Parameters
        ----------
        jsondictstr : str
            The JSON string to deserialize.

        Returns
        -------
        List[Message]
            The deserialized list of messages.
        """
        data = json.loads(jsondictstr)
        # Modify implicit is_chunk property to ALWAYS false
        # since Langchain's Message schema is stricter
        prevalidated_data = [
            {**item, "data": {**item["data"], "is_chunk": False}} for item in data
        ]
        return list(messages_from_dict(prevalidated_data))  # type: ignore

    def update_token_usage_log(
        self, messages: List[Message], answer: str, step_name: str
    ) -> None:
        """
        Update the token usage log with the number of tokens used in the current step.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        answer : str
            The answer from the AI.
        step_name : str
            The name of the step.
        """
        prompt_tokens = self.num_tokens_from_messages(messages)
        completion_tokens = self.num_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens

        self.cumulative_prompt_tokens += prompt_tokens
        self.cumulative_completion_tokens += completion_tokens
        self.cumulative_total_tokens += total_tokens

        self.token_usage_log.append(
            TokenUsage(
                step_name=step_name,
                in_step_prompt_tokens=prompt_tokens,
                in_step_completion_tokens=completion_tokens,
                in_step_total_tokens=total_tokens,
                total_prompt_tokens=self.cumulative_prompt_tokens,
                total_completion_tokens=self.cumulative_completion_tokens,
                total_tokens=self.cumulative_total_tokens,
            )
        )

    def format_token_usage_log(self) -> str:
        """
        Format the token usage log as a CSV string.

        Returns
        -------
        str
            The token usage log formatted as a CSV string.
        """
        result = "step_name,"
        result += "prompt_tokens_in_step,completion_tokens_in_step,total_tokens_in_step"
        result += ",total_prompt_tokens,total_completion_tokens,total_tokens\n"
        for log in self.token_usage_log:
            result += log.step_name + ","
            result += str(log.in_step_prompt_tokens) + ","
            result += str(log.in_step_completion_tokens) + ","
            result += str(log.in_step_total_tokens) + ","
            result += str(log.total_prompt_tokens) + ","
            result += str(log.total_completion_tokens) + ","
            result += str(log.total_tokens) + "\n"
        return result

    def usage_cost(self) -> float:
        """
        Return the total cost in USD of the api usage.

        Returns
        -------
        float
            Cost in USD.
        """
        prompt_price = MODEL_COST_PER_1K_TOKENS[self.model_name]
        completion_price = MODEL_COST_PER_1K_TOKENS[self.model_name + "-completion"]

        result = 0
        for log in self.token_usage_log:
            result += log.total_prompt_tokens / 1000 * prompt_price
            result += log.total_completion_tokens / 1000 * completion_price
        return result

    def num_tokens(self, txt: str) -> int:
        """
        Get the number of tokens in a text.

        Parameters
        ----------
        txt : str
            The text to count the tokens in.

        Returns
        -------
        int
            The number of tokens in the text.
        """
        return len(self.tokenizer.encode(txt))

    def num_tokens_from_messages(self, messages: List[Message]) -> int:
        """
        Get the total number of tokens used by a list of messages.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to count the tokens in.

        Returns
        -------
        int
            The total number of tokens used by the messages.
        """
        """Returns the number of tokens used by a list of messages."""
        n_tokens = 0
        for message in messages:
            n_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            n_tokens += self.num_tokens(message.content)
        n_tokens += 2  # every reply is primed with <im_start>assistant
        return n_tokens


def fallback_model(model: str) -> str:
    """
    Retrieve the specified model, or fallback to "gpt-3.5-turbo" if the model is not available.

    Parameters
    ----------
    model : str
        The name of the model to retrieve.

    Returns
    -------
    str
        The name of the retrieved model, or "gpt-3.5-turbo" if the specified model is not available.
    """
    try:
        openai.Model.retrieve(model)
        return model
    except openai.InvalidRequestError:
        print(
            f"Model {model} not available for provided API key. Reverting "
            "to gpt-3.5-turbo. Sign up for the GPT-4 wait list here: "
            "https://openai.com/waitlist/gpt-4-api\n"
        )
        return "gpt-3.5-turbo"


def create_chat_model(self, model: str, temperature) -> BaseChatModel:
    """
    Create a chat model with the specified model name and temperature.

    Parameters
    ----------
    model : str
        The name of the model to create.
    temperature : float
        The temperature to use for the model.

    Returns
    -------
    BaseChatModel
        The created chat model.
    """
    if self.azure_endpoint:
        return AzureChatOpenAI(
            openai_api_base=self.azure_endpoint,
            openai_api_version="2023-05-15",  # might need to be flexible in the future
            deployment_name=model,
            openai_api_type="azure",
            streaming=True,
        )
    # Fetch available models from OpenAI API
    supported = [model["id"] for model in openai.Model.list()["data"]]
    if model not in supported:
        raise ValueError(
            f"Model {model} is not supported, supported models are: {supported}"
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True,
        client=openai.ChatCompletion,
    )


def get_tokenizer(model: str):
    """
    Get the tokenizer for the specified model.

    Parameters
    ----------
    model : str
        The name of the model to get the tokenizer for.

    Returns
    -------
    Tokenizer
        The tokenizer for the specified model.
    """
    if "gpt-4" in model or "gpt-3.5" in model:
        return tiktoken.encoding_for_model(model)

    logger.debug(
        f"No encoder implemented for model {model}."
        "Defaulting to tiktoken cl100k_base encoder."
        "Use results only as estimates."
    )
    return tiktoken.get_encoding("cl100k_base")


def serialize_messages(messages: List[Message]) -> str:
    """
    Serialize a list of chat messages into a JSON-formatted string.

    This function acts as a wrapper around the `AI.serialize_messages` method,
    providing a more straightforward access to message serialization.

    Parameters
    ----------
    messages : List[Message]
        A list of chat messages to be serialized. Each message should be an
        instance of the `Message` type (which includes `AIMessage`, `HumanMessage`,
        and `SystemMessage`).

    Returns
    -------
    str
        A JSON-formatted string representation of the input messages.

    Example
    -------
    >>> msgs = [SystemMessage(content="Hello"), HumanMessage(content="Hi, AI!")]
    >>> serialize_messages(msgs)
    '[{"type": "system", "content": "Hello"}, {"type": "human", "content": "Hi, AI!"}]'
    """
    return AI.serialize_messages(messages)
