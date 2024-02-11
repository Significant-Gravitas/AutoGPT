import os
from typing import Any, Callable, Dict, ParamSpec, Tuple, TypeVar

import tiktoken
from openai import AsyncOpenAI
from openai.resources import AsyncCompletions

from AFAAS.core.adapters.openai.common import (
    OPEN_AI_CHAT_MODELS,
    OPEN_AI_DEFAULT_CHAT_CONFIGS,
    OPEN_AI_MODELS,
    OpenAIChatMessage,
    OpenAIModelName,
    OpenAIPromptConfiguration,
    OpenAISettings,
    _OpenAIRetryHandler,
)
from AFAAS.core.adapters.openai.embeddings import _create_embedding

aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

from AFAAS.configs.schema import Configurable
from AFAAS.interfaces.adapters.chatmodel import (
    AbstractChatModelProvider,
    AbstractChatModelResponse,
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
)
from AFAAS.interfaces.adapters.language_model import Embedding, ModelTokenizer
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class AFAASChatOpenAI(Configurable[OpenAISettings], AbstractChatModelProvider):
    """A provider for OpenAI's API.

    Provides methods to communicate with OpenAI's API and generate responses.

    Attributes:
        default_settings: The default settings for the OpenAI provider.
    """

    def __init__(
        self,
        # agent_systems: list[Configurable],
        settings: OpenAISettings = OpenAISettings(),
    ):
        """
        Initialize the OpenAIProvider.

        Args:
            settings (OpenAISettings, optional): Specific settings for the OpenAI provider. Uses default settings if none provided.
        """
        super().__init__(settings)
        self._credentials = settings.credentials
        self._budget = settings.budget
        self._chat = settings.chat

        retry_handler = _OpenAIRetryHandler(
            num_retries=self._settings.configuration.retries_per_request,
        )

        self._create_chat_completion = retry_handler(_create_chat_completion)
        self._create_embedding = retry_handler(_create_embedding)

        self._func_call_fails_count = 0

    def get_token_limit(self, model_name: str) -> int:
        """
        Get the token limit for a given model.

        Args:
            model_name (str): The name of the model.

        Returns:
            int: The maximum number of tokens allowed for the given model.

        Example:
            >>> provider = OpenAIProvider()
            >>> provider.get_token_limit("gpt-3.5-turbo")
            4096
        """
        return OPEN_AI_MODELS[model_name].max_tokens

    def get_remaining_budget(self) -> float:
        """
        Get the remaining budget.

        Returns:
            float: Remaining budget value.

        Example:
            >>> provider = OpenAIProvider(...)
            >>> remaining_budget = provider.get_remaining_budget()
            >>> print(remaining_budget)
            inf
        """
        """Get the remaining budget."""
        return self._budget.remaining_budget

    @classmethod
    def get_tokenizer(cls, model_name: OpenAIModelName) -> ModelTokenizer:
        """
        Get the tokenizer for a given model.

        Args:
            model_name (OpenAIModelName): Enum value representing the model.

        Returns:
            ModelTokenizer: Tokenizer for the specified model.

        Example:
            >>> tokenizer = OpenAIProvider.get_tokenizer(OpenAIModelName.GPT3)
            >>> type(tokenizer)
            <class 'ModelTokenizer'>
        """
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: OpenAIModelName) -> int:
        """
        Count the number of tokens in a given text for a specific model.

        Args:
            text (str): Input text.
            model_name (OpenAIModelName): Enum value representing the model.

        Returns:
            int: Number of tokens in the text.

        Example:
            >>> token_count = OpenAIProvider.count_tokens("Hello, world!", OpenAIModelName.GPT3)
            >>> print(token_count)
            3
        """
        encoding = cls.get_tokenizer(model_name)
        return len(encoding.encode(text))

    @classmethod
    def count_message_tokens(
        cls,
        messages: OpenAIChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:
        """
        Count the number of tokens in a given set of messages for a specific model.

        Args:
            messages (Union[ChatMessage, List[ChatMessage]]): Input messages.
            model_name (OpenAIModelName): Enum value representing the model.

        Returns:
            int: Number of tokens in the messages.

        Example:
            >>> messages = [ChatMessage(role="user", content="Hello?")]
            >>> token_count = OpenAIProvider.count_message_tokens(messages, OpenAIModelName.GPT3)
            >>> print(token_count)
            5
        """
        if isinstance(messages, OpenAIChatMessage):
            messages = [messages]

        if model_name.startswith("gpt-3.5-turbo"):
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
            encoding_model = "gpt-3.5-turbo"
        elif model_name.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
            encoding_model = "gpt-4"
        else:
            raise NotImplementedError(
                f"count_message_tokens() is not implemented for model {model_name}.\n"
                " See https://github.com/openai/openai-python/blob/main/chatml.md for"
                " information on how messages are converted to tokens."
            )
        try:
            encoding = tiktoken.encoding_for_model(encoding_model)
        except KeyError:
            LOG.warning(
                f"Model {model_name} not found. Defaulting to cl100k_base encoding."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.dict().items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    async def create_chat_completion(
        self,
        chat_messages: list[ChatMessage],
        tools: list[CompletionModelFunction],
        llm_model_name: OpenAIModelName,
        tool_choice: str,
        default_tool_choice: str,  # This one would be called after 3 failed attemps(cf : try/catch block)
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        **kwargs,
    ) -> AbstractChatModelResponse[_T]:
        """Create a completion using the OpenAI API.

        Args:
            model_prompt (list): A list of chat messages.
            functions (list): A list of completion model functions.
            model_name (str): The name of the model.
            tool_choice (str): The function call string.
            default_tool_choice (str): The default function call to use after 3 failed attempts.
            completion_parser (Callable): A parser to process the chat response.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatModelResponse: Response from the chat completion.

        Example:
            >>> provider = OpenAIProvider(...)
            >>> messages = [ChatMessage(role="user", content="Tell me a joke.")]
            >>> response = await provider.create_chat_completion(messages, ...)
            >>> print(response.content)
            "Why did the chicken cross the road? To get to the other side!"
        """

        # ##############################################################################
        # ### Step 1: Prepare arguments for API call
        # ##############################################################################
        completion_kwargs = self._initialize_completion_args(
            model_name=llm_model_name,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        # ##############################################################################
        # ### Step 2: Execute main chat completion and extract details
        # ##############################################################################
        response = await self._get_chat_response(
            model_prompt=chat_messages, **completion_kwargs
        )
        response_message, response_args = self._extract_response_details(
            response=response, model_name=llm_model_name
        )

        # ##############################################################################
        # ### Step 3: Handle missing function call and retry if necessary
        # ##############################################################################
        if self._should_retry_function_call(
            tools=tools, response_message=response_message
        ):
            if (
                self._func_call_fails_count
                <= self._settings.configuration.maximum_retry
            ):
                return await self._retry_chat_completion(
                    model_prompt=chat_messages,
                    tools=tools,
                    completion_kwargs=completion_kwargs,
                    model_name=llm_model_name,
                    completion_parser=completion_parser,
                    default_tool_choice=default_tool_choice,
                    response=response,
                    response_args=response_args,
                )

            # FIXME, TODO, NOTE: Organize application save feedback loop to improve the prompts, as it is not normal that function are not called
            response_message["tool_calls"] = None
            response.choices[0].message["tool_calls"] = None
            # self._handle_failed_retry(response_message)

        # ##############################################################################
        # ### Step 4: Reset failure count and integrate improvements
        # ##############################################################################
        self._func_call_fails_count = 0

        # ##############################################################################
        # ### Step 5: Self feedback
        # ##############################################################################

        # Create an option to deactivate feedbacks
        # Option : Maximum number of feedbacks allowed

        # Prerequisite : Read OpenAI API (Chat Model) tool_choice section

        # User : 1 shirt take 5 minutes to dry , how long take 10 shirt to dry
        # Assistant : It takes 50 minutes

        # System : "The user question was ....
        # The Assistant Response was ..."
        # Is it ok ?
        # If not provide a feedback

        # => T shirt can be dried at the same time

        # ##############################################################################
        # ### Step 6: Formulate the response
        # ##############################################################################
        return self._formulate_final_response(
            response_message=response_message,
            completion_parser=completion_parser,
            response_args=response_args,
        )

    def _initialize_completion_args(
        self,
        model_name: str,
        tools: list[CompletionModelFunction],
        tool_choice: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        completion_kwargs = self._get_completion_kwargs(model_name, tools, **kwargs)
        completion_kwargs["tool_choice"] = tool_choice
        return completion_kwargs

    async def _get_chat_response(
        self, model_prompt: list[ChatMessage], **completion_kwargs: Any
    ) -> AsyncCompletions:
        return await self._create_chat_completion(
            messages=model_prompt, **completion_kwargs
        )

    def _extract_response_details(
        self, response: AsyncCompletions, model_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        response_args = {
            "llm_model_info": OPEN_AI_CHAT_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        response_message = response.choices[0].message.model_dump()
        return response_message, response_args

    def _should_retry_function_call(
        self, tools: list[CompletionModelFunction], response_message: Dict[str, Any]
    ) -> bool:
        if tools is not None and "tool_calls" not in response_message:
            LOG.error(
                f"Attempt number {self._func_call_fails_count + 1} : Function Call was expected"
            )
            return True
        return False

    async def _retry_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        tools: list[CompletionModelFunction],
        completion_kwargs: Dict[str, Any],
        model_name: str,
        completion_parser: Callable[[AssistantChatMessageDict], _T],
        default_tool_choice: str,
        response: AsyncCompletions,
        response_args: Dict[str, Any],
    ) -> AbstractChatModelResponse[_T]:
        completion_kwargs = self._update_function_call_for_retry(
            completion_kwargs=completion_kwargs,
            default_tool_choice=default_tool_choice,
        )
        completion_kwargs["tools"] = tools
        response.update(response_args)
        self._budget.update_usage_and_cost(model_response=response)
        return await self.create_chat_completion(
            chat_messages=model_prompt,
            llm_model_name=model_name,
            completion_parser=completion_parser,
            **completion_kwargs,
        )

    def _update_function_call_for_retry(
        self, completion_kwargs: Dict[str, Any], default_tool_choice: str
    ) -> Dict[str, Any]:
        if (
            self._func_call_fails_count
            >= self._settings.configuration.maximum_retry_before_default_function
        ):
            completion_kwargs["tool_calls"] = default_tool_choice
        else:
            completion_kwargs["tool_calls"] = completion_kwargs.get(
                "tool_calls", "auto"
            )
        completion_kwargs["default_tool_choice"] = completion_kwargs.get(
            "default_tool_choice", default_tool_choice
        )
        self._func_call_fails_count += 1
        return completion_kwargs

    # def _handle_failed_retry(self, response_message: Dict[str, Any], response: openai.Completion) -> None:

    def _formulate_final_response(
        self,
        response_message: Dict[str, Any],
        completion_parser: Callable[[AssistantChatMessageDict], _T],
        response_args: Dict[str, Any],
    ) -> AbstractChatModelResponse[_T]:
        response = AbstractChatModelResponse(
            response=response_message,
            parsed_result=completion_parser(response_message),
            **response_args,
        )
        self._budget.update_usage_and_cost(model_response=response)
        return response

    async def create_language_completion(self, **kwargs):
        LOG.warning(
            "create_language_completion is deprecated, use create_chat_completion"
        )
        return await self.create_chat_completion(**kwargs)

    def _get_completion_kwargs(
        self,
        model_name: OpenAIModelName,
        functions: list[CompletionModelFunction],
        **kwargs,
    ) -> dict:
        """Get kwargs for completion API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            dict: Dictionary containing the kwargs.

        Example:
            >>> provider = OpenAIProvider(...)
            >>> completion_kwargs = provider._get_completion_kwargs(OpenAIModelName.GPT3, ...)
            >>> print(completion_kwargs)
            {'model': 'gpt-3.5-turbo-0613', ...}

        """
        completion_kwargs = {
            "model": model_name,
            **kwargs,
            # **self._credentials.unmasked(),
        }
        if functions:
            completion_kwargs["tools"] = [
                {"type": "function", "function": f.schema} for f in functions
            ]

        return completion_kwargs

    def __repr__(self):
        """
        String representation of the class.

        Returns:
            str: String representation.

        Example:
            >>> provider = OpenAIProvider(...)
            >>> print(provider)
            <OpenAIProvider: api_key=XXXXXXX, budget=inf>
        """
        return "OpenAIProvider()"

    def has_oa_tool_calls_api(self, model_name: str) -> bool:
        # print(self._providers[model_name])
        return OPEN_AI_CHAT_MODELS[model_name].has_function_call_api

    def get_default_config(self) -> OpenAIPromptConfiguration:
        return OPEN_AI_DEFAULT_CHAT_CONFIGS.SMART_MODEL_32K


async def _create_chat_completion(
    messages: list[ChatMessage], *_, **kwargs
) -> AsyncCompletions:
    """Create a chat completion using the OpenAI API.

    Args:
        messages: The prompt to use.

    Returns:
        The completion.

    """
    raw_messages = [
        message.dict(include={"role", "content", "tool_calls", "name"})
        for message in messages
    ]

    if not "tools" in kwargs or kwargs["tools"] is None or len(kwargs["tools"]) == 0:
        if "tools" in kwargs:
            del kwargs["tools"]
        kwargs.pop("tool_choice", None)

    else:
        # kwargs["tools"] = [function for function in kwargs["tools"]]
        if len(kwargs["tools"]) == 0:
            del kwargs["tools"]
            kwargs.pop("tool_choice", None)
        elif len(kwargs["tools"]) == 1:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": kwargs["tools"][0]["function"]["name"]},
            }
        elif kwargs["tool_choice"] != "auto":
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": kwargs["tool_choice"]},
            }

    LOG.trace(raw_messages[0]["content"])
    LOG.trace(kwargs)
    return_value = await aclient.chat.completions.create(
        messages=raw_messages, **kwargs
    )
    return return_value
