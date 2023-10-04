import enum
import functools
import logging
import math
import time
from typing import Callable, ParamSpec, TypeVar

import openai
import tiktoken
from openai.error import APIError, RateLimitError

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    UserConfigurable,
)
from autogpt.core.resource.model_providers.schema import (
    Embedding,
    EmbeddingModelInfo,
    EmbeddingModelProvider,
    EmbeddingModelResponse,
    BaseModelProviderBudget,
    BaseModelProviderCredentials,
    ModelProviderName,
    ModelProviderService,
    BaseModelProviderSettings,
    BaseModelProviderUsage,
    ModelTokenizer,
)

from autogpt.core.resource.model_providers.chat_schema import (
    AssistantChatMessageDict,
    ChatMessage,
    ChatModelInfo,
    BaseChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    CompletionModelFunction,
    ChatMessage,
    BaseChatModelProvider,
    ChatModelInfo,
    ChatModelResponse,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class OpenAIModelName(str, enum.Enum):
    """
    Enumeration of OpenAI model names.

    Attributes:
        Each enumeration represents a distinct OpenAI model name.
    """

    ADA = "text-embedding-ada-002"
    GPT3 = "gpt-3.5-turbo-0613"
    GPT3_16k = "gpt-3.5-turbo-16k-0613"
    GPT3_FINE_TUNED = "gpt-3.5-turbo" + ""
    # GPT4 = "gpt-4-0613" # TODO for tests
    GPT4 = "gpt-3.5-turbo-0613"
    GPT4_32k = "gpt-4-32k-0613"


OPEN_AI_EMBEDDING_MODELS = {
    OpenAIModelName.ADA: EmbeddingModelInfo(
        name=OpenAIModelName.ADA,
        service=ModelProviderService.EMBEDDING,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0001 / 1000,
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
}


OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=OpenAIModelName.GPT3,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_16k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_FINE_TUNED,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0120 / 1000,
            completion_token_cost=0.0160 / 1000,
            max_tokens=4096,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_32k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=32768,
            has_function_call_api=True,
        ),
    ]
}


OPEN_AI_MODELS = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAIConfiguration(SystemConfiguration):
    """Configuration for OpenAI.

    Attributes:
        retries_per_request: The number of retries per request.
        maximum_retry: The maximum number of retries allowed.
        maximum_retry_before_default_function: The maximum number of retries before a default function is used.
    """

    retries_per_request: int = UserConfigurable()
    maximum_retry = 1
    maximum_retry_before_default_function = 1


class OpenAIModelProviderBudget(BaseModelProviderBudget):
    """Budget configuration for the OpenAI Model Provider.

    Attributes:
        graceful_shutdown_threshold: The threshold for graceful shutdown.
        warning_threshold: The warning threshold for budget.
    """

    graceful_shutdown_threshold: float = UserConfigurable()
    warning_threshold: float = UserConfigurable()


class OpenAISettings(BaseModelProviderSettings):
    """Settings for the OpenAI provider.

    Attributes:
        configuration: Configuration settings for OpenAI.
        credentials: The credentials for the model provider.
        budget: Budget settings for the model provider.
    """

    configuration: OpenAIConfiguration
    credentials: BaseModelProviderCredentials
    budget: OpenAIModelProviderBudget


class OpenAIProvider(
    Configurable[OpenAISettings], BaseChatModelProvider, EmbeddingModelProvider
):
    """A provider for OpenAI's API.

    Provides methods to communicate with OpenAI's API and generate responses.

    Attributes:
        default_settings: The default settings for the OpenAI provider.
    """

    default_settings = OpenAISettings(
        name="openai_provider",
        description="Provides access to OpenAI's API.",
        configuration=OpenAIConfiguration(
            retries_per_request=10,
        ),
        credentials=BaseModelProviderCredentials(),
        budget=OpenAIModelProviderBudget(
            total_budget=math.inf,
            total_cost=0.0,
            remaining_budget=math.inf,
            usage=BaseModelProviderUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            graceful_shutdown_threshold=0.005,
            warning_threshold=0.01,
        ),
    )

    def __init__(
        self,
        settings: OpenAISettings,
        logger: logging.Logger,
    ):
        """
        Initialize the OpenAIProvider.

        Args:
            settings (OpenAISettings, optional): Specific settings for the OpenAI provider. Uses default settings if none provided.
        """
        super().__init__(settings, logger)
        self._credentials = settings.credentials
        self._budget = settings.budget

        retry_handler = _OpenAIRetryHandler(
            logger=self._logger,
            num_retries=self._configuration.retries_per_request,
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
        messages: ChatMessage | list[ChatMessage],
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
        if isinstance(messages, ChatMessage):
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
            cls._logger.warn(
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
        model_prompt: list[ChatMessage],
        functions: list[CompletionModelFunction],
        model_name: OpenAIModelName,
        function_call: str,
        default_function_call: str,  # This one would be called after 3 failed attemps(cf : try/catch block)
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the OpenAI API.

        Args:
            model_prompt (list): A list of chat messages.
            functions (list): A list of completion model functions.
            model_name (str): The name of the model.
            function_call (str): The function call string.
            default_function_call (str): The default function call to use after 3 failed attempts.
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

        completion_kwargs = self._get_completion_kwargs(model_name, functions, **kwargs)

        completion_kwargs["function_call"] = function_call

        try:
            response = await self._create_chat_completion(
                messages=model_prompt,
                **completion_kwargs,
            )

            response_args = {
                "model_info": OPEN_AI_CHAT_MODELS[model_name],
                "prompt_tokens_used": response.usage.prompt_tokens,
                "completion_tokens_used": response.usage.completion_tokens,
            }

            response_message = response.choices[0].message.to_dict_recursive()

            if functions is not None and not "function_call" in response_message:
                self._logger.error(
                    f"Attempt number {self._func_call_fails_count + 1} : Function Call was expected  "
                )
                raise Exception("function_call was expected")
        except:
            if self._func_call_fails_count <= self._configuration.maximum_retry:
                if (
                    self._func_call_fails_count
                    >= self._configuration.maximum_retry_before_default_function
                ):
                    completion_kwargs["function_call"] = default_function_call
                else:
                    if "function_call" not in completion_kwargs:
                        completion_kwargs["function_call"] = "auto"

                if "default_function_call" not in completion_kwargs:
                    completion_kwargs["default_function_call"] = default_function_call

                self._func_call_fails_count += 1

                # completion_kwarg is a dict but function is a list[CompletionModelFunction] which has to be restored
                # alternative is to pass the **kwarg not the completion kwarg
                completion_kwargs["functions"] = functions

                return await self.create_language_completion(
                    model_prompt=model_prompt,
                    # functions = functions,
                    model_name=model_name,
                    completion_parser=completion_parser,
                    **completion_kwargs,
                )

            else:
                # FIXME : Provide self improvement mechanism
                # TODO : Provide self improvement mechanism
                # NOTE : Provide self improvement mechanism

                # NOTE : In any case leave these notes as this block may also serve to automaticaly generate bug report on a bug tracking platform (if a users allows to share this kind of data)
                pass

            response_message["function_call"] = None
            response.choices[0].message["function_call"] = None

        self._func_call_fails_count = 0

        # New
        response = ChatModelResponse(
            response=response_message,
            parsed_result=completion_parser(response_message),
            **response_args,
        )

        # Old
        # parsed_response = completion_parser(
        #          response_message
        #      )
        # response = ChatModelResponse(
        #     content=parsed_response,
        #     **response_args,
        # )
        self._budget.update_usage_and_cost(response)
        return response

    async def create_language_completion(self, **kwargs):
        self._logger.warning(
            "create_language_completion is deprecated, use create_chat_completion"
        )
        return await self.create_chat_completion(**kwargs)

    async def create_embedding(
        self,
        text: str,
        model_name: OpenAIModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Create an embedding using the OpenAI API.

        Args:
            text (str): The text to embed.
            model_name (str): The name of the embedding model.
            embedding_parser (Callable): A parser to process the embedding.
            **kwargs: Additional keyword arguments.

        Returns:
            EmbeddingModelResponse: Response containing the embedding.

        Example:
            >>> provider = OpenAIProvider(...)
            >>> embedding_response = await provider.create_embedding("Hello, world!", ...)
            >>> print(embedding_response.embedding)
            [0.123, -0.456, ...]
        """
        embedding_kwargs = self._get_embedding_kwargs(model_name, **kwargs)
        response = await self._create_embedding(text=text, **embedding_kwargs)

        response_args = {
            "model_info": OPEN_AI_EMBEDDING_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        response = EmbeddingModelResponse(
            **response_args,
            embedding=embedding_parser(response.embeddings[0]),
        )
        self._budget.update_usage_and_cost(response)
        return response

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
            **self._credentials.unmasked(),
        }
        if functions:
            completion_kwargs["functions"] = [f.schema for f in functions]

        return completion_kwargs

    def _get_embedding_kwargs(
        self,
        model_name: OpenAIModelName,
        **kwargs,
    ) -> dict:
        """Get kwargs for embedding API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            dict: Dictionary containing the kwargs.

        Example:
            >>> provider = OpenAIProvider(...)
            >>> embedding_kwargs = provider._get_embedding_kwargs(OpenAIModelName.ADA, ...)
            >>> print(embedding_kwargs)
            {'model': 'text-embedding-ada-002', ...}
        """
        embedding_kwargs = {
            "model": model_name,
            **kwargs,
            **self._credentials.unmasked(),
        }

        return embedding_kwargs

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

    def has_function_call_api(self, model_name: str) -> bool:
        # print(self._providers[model_name])
        return OPEN_AI_CHAT_MODELS[model_name].has_function_call_api


async def _create_embedding(text: str, *_, **kwargs) -> openai.Embedding:
    """Embed text using the OpenAI API.

    Args:
        text str: The text to embed.
        model_name str: The name of the model to use.

    Returns:
        str: The embedding.
    """
    return await openai.Embedding.acreate(
        input=[text],
        **kwargs,
    )


async def _create_chat_completion(
    messages: list[ChatMessage], *_, **kwargs
) -> openai.Completion:
    """Create a chat completion using the OpenAI API.

    Args:
        messages: The prompt to use.

    Returns:
        The completion.

    """
    raw_messages = [
        message.dict(include={"role", "content", "function_call", "name"})
        for message in messages
    ]
    if "functions" in kwargs:
        # wargs["functions"] = [function.dict() for function in kwargs["functions"]]
        kwargs["functions"] = [function for function in kwargs["functions"]]

    if kwargs["function_call"] != "auto":
        kwargs["function_call"] = {"name": kwargs["function_call"]}

    return_value = await openai.ChatCompletion.acreate(
        messages=raw_messages,
        **kwargs,
    )
    return return_value


class _OpenAIRetryHandler:
    """Retry Handler for OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """

    _retry_limit_msg = "Error: Reached rate limit, passing..."
    _api_key_error_msg = (
        "Please double check that you have setup a PAID OpenAI API Account. You can "
        "read more here: https://docs.agpt.co/setup/#getting-an-api-key"
    )
    _backoff_msg = "Error: API Bad gateway. Waiting {backoff} seconds..."

    def __init__(
        self,
        logger: logging.Logger,
        num_retries: int = 10,
        backoff_base: float = 2.0,
        warn_user: bool = True,
    ):
        self._logger = logger
        self._num_retries = num_retries
        self._backoff_base = backoff_base
        self._warn_user = warn_user

    def _log_rate_limit_error(self) -> None:
        self._logger.debug(self._retry_limit_msg)
        if self._warn_user:
            self._logger.warning(self._api_key_error_msg)
            self._warn_user = False

    def _backoff(self, attempt: int) -> None:
        backoff = self._backoff_base ** (attempt + 2)
        self._logger.debug(self._backoff_msg.format(backoff=backoff))
        time.sleep(backoff)

    def __call__(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(func)
        async def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            num_attempts = self._num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except RateLimitError:
                    if attempt == num_attempts:
                        raise
                    self._log_rate_limit_error()

                except APIError as e:
                    if (e.http_status != 502) or (attempt == num_attempts):
                        raise
                except Exception as e:
                    self._logger.warning(e)
                self._backoff(attempt)

        return _wrapped
