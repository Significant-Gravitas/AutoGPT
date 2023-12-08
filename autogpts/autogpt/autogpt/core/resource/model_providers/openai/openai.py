import enum
import functools
import logging
import math
import os
import time
from typing import Any, Callable, Dict, ParamSpec, Tuple, TypeVar

from openai import AsyncOpenAI
from openai.resources import AsyncCompletions, AsyncEmbeddings

aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
import tiktoken
from openai import APIError, RateLimitError, completions  # , OpenAI, Em

from  AFAAS.core.lib.sdk.logger import AFAASLogger
from AFAAS.core.configuration import (Configurable,
                                                         SystemConfiguration,
                                                         UserConfigurable)
from AFAAS.core.resource.model_providers.chat_schema import (
    AssistantChatMessageDict, AssistantToolCallDict, BaseChatModelProvider,
    ChatMessage, ChatModelInfo, ChatModelResponse, CompletionModelFunction)
from AFAAS.core.resource.model_providers.schema import (
    BaseModelProviderBudget, BaseModelProviderCredentials,
    BaseModelProviderSettings, BaseModelProviderUsage, Embedding,
    EmbeddingModelInfo, EmbeddingModelProvider, EmbeddingModelResponse,
    ModelProviderName, ModelProviderService, ModelTokenizer)
from AFAAS.core.utils.json_schema import JSONSchema

LOG = AFAASLogger(__name__)

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

    # ADA = "text-embedding-ada-002"
    # GPT3 = "gpt-3.5-turbo-instruct"
    # GPT3_16k = "gpt-3.5-turbo-1106"
    # GPT3_FINE_TUNED = "gpt-3.5-turbo-1106" + ""
    # # GPT4 = "gpt-4-0613" # TODO for tests
    # GPT4 = "gpt-3.5-turbo-1106"
    # GPT4_32k = "gpt-4-1106-preview"

    ADA = "text-embedding-ada-002"
    GPT3 = "gpt-3.5-turbo-1106"
    GPT3_16k = "gpt-3.5-turbo-1106"
    GPT3_FINE_TUNED = "gpt-3.5-turbo-1106" + ""
    # GPT4 = "gpt-3.5-turbo-1106" # TODO for tests
    GPT4 = "gpt-3.5-turbo-1106"
    GPT4_32k = "gpt-3.5-turbo-1106"


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

    retries_per_request: int = UserConfigurable(default=10)
    maximum_retry: int = 1
    maximum_retry_before_default_function: int = 1


class OpenAIModelProviderBudget(BaseModelProviderBudget):
    """Budget configuration for the OpenAI Model Provider.

    Attributes:
        graceful_shutdown_threshold: The threshold for graceful shutdown.
        warning_threshold: The warning threshold for budget.
    """

    graceful_shutdown_threshold: float = UserConfigurable(default=0.005)
    warning_threshold: float = UserConfigurable(default=0.01)

    total_budget: float = math.inf
    total_cost: float = 0.0
    remaining_budget: float = math.inf
    usage: BaseModelProviderUsage = BaseModelProviderUsage()


class OpenAISettings(BaseModelProviderSettings):
    """Settings for the OpenAI provider.

    Attributes:
        configuration: Configuration settings for OpenAI.
        credentials: The credentials for the model provider.
        budget: Budget settings for the model provider.
    """

    configuration: OpenAIConfiguration = OpenAIConfiguration()
    credentials: BaseModelProviderCredentials = BaseModelProviderCredentials()
    budget: OpenAIModelProviderBudget = OpenAIModelProviderBudget()

    name = "chat_model_provider"
    description = "Provides access to OpenAI's API."


class OpenAIProvider(
    Configurable[OpenAISettings], BaseChatModelProvider, EmbeddingModelProvider
):
    """A provider for OpenAI's API.

    Provides methods to communicate with OpenAI's API and generate responses.

    Attributes:
        default_settings: The default settings for the OpenAI provider.
    """

    default_settings = OpenAISettings()

    def __init__(
        self,
        settings: OpenAISettings,
        logger: logging.Logger,
        agent_systems: list[Configurable],
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
            LOG.warn(
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
        model_name: OpenAIModelName,
        tool_choice: str,
        default_tool_choice: str,  # This one would be called after 3 failed attemps(cf : try/catch block)
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
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
            model_name=model_name,
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
            response=response, model_name=model_name
        )

        # ##############################################################################
        # ### Step 3: Handle missing function call and retry if necessary
        # ##############################################################################
        if self._should_retry_function_call(
            tools=tools, response_message=response_message
        ):
            if self._func_call_fails_count <= self._configuration.maximum_retry:
                return await self._retry_chat_completion(
                    model_prompt=chat_messages,
                    tools=tools,
                    completion_kwargs=completion_kwargs,
                    model_name=model_name,
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
            "model_info": OPEN_AI_CHAT_MODELS[model_name],
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
    ) -> ChatModelResponse[_T]:
        completion_kwargs = self._update_function_call_for_retry(
            completion_kwargs=completion_kwargs,
            default_tool_choice=default_tool_choice,
        )
        completion_kwargs["tools"] = tools
        response.update(response_args)
        self._budget.update_usage_and_cost(model_response=response)
        return await self.create_chat_completion(
            chat_messages=model_prompt,
            model_name=model_name,
            completion_parser=completion_parser,
            **completion_kwargs,
        )

    def _update_function_call_for_retry(
        self, completion_kwargs: Dict[str, Any], default_tool_choice: str
    ) -> Dict[str, Any]:
        if (
            self._func_call_fails_count
            >= self._configuration.maximum_retry_before_default_function
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
    ) -> ChatModelResponse[_T]:
        response = ChatModelResponse(
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
            # **self._credentials.unmasked(),
        }
        if functions:
            completion_kwargs["tools"] = [
                {"type": "function", "function": f.schema} for f in functions
            ]
        else:
            # Provide compatibility with older models
            _functions_compat_fix_kwargs(functions, completion_kwargs)

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

    def has_oa_tool_calls_api(self, model_name: str) -> bool:
        # print(self._providers[model_name])
        return OPEN_AI_CHAT_MODELS[model_name].has_function_call_api


async def _create_embedding(text: str, *_, **kwargs) -> AsyncEmbeddings:
    """Embed text using the OpenAI API.

    Args:
        text str: The text to embed.
        model_name str: The name of the model to use.

    Returns:
        str: The embedding.
    """
    return await aclient.embeddings.create(input=[text], **kwargs)


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

    if "tools" in kwargs:
        # wargs["tools"] = [function.dict() for function in kwargs["tools"]]
        kwargs["tools"] = [function for function in kwargs["tools"]]
        if len(kwargs["tools"]) == 1:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": kwargs["tools"][0]["function"]["name"]},
            }
        elif kwargs["tool_choice"] != "auto":
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": kwargs["tool_choice"]},
            }

    LOG.debug(raw_messages[0]["content"])
    LOG.debug(kwargs)
    return_value = await aclient.chat.completions.create(
        messages=raw_messages, **kwargs
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
        LOG.debug(self._retry_limit_msg)
        if self._warn_user:
            LOG.warning(self._api_key_error_msg)
            self._warn_user = False

    def _backoff(self, attempt: int) -> None:
        backoff = self._backoff_base ** (attempt + 2)
        LOG.debug(self._backoff_msg.format(backoff=backoff))
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
                    LOG.warning(e)
                self._backoff(attempt)

        return _wrapped


def format_function_specs_as_typescript_ns(
    functions: list[CompletionModelFunction],
) -> str:
    """Returns a function signature block in the format used by OpenAI internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    For use with `count_tokens` to determine token usage of provided functions.

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
        return (
            f"// {spec.description}\n" if spec.description else ""
        ) + f"{name}{'' if spec.required else '?'}: {spec.typescript_type},"

    return "\n".join(
        [
            f"// {func.description}",
            f"type {func.name} = (_ :{{",
            *[param_signature(name, p) for name, p in func.parameters.items()],
            "}) => any;",
        ]
    )


def count_openai_functions_tokens(
    functions: list[CompletionModelFunction], count_tokens: Callable[[str], int]
) -> int:
    """Returns the number of tokens taken up by a set of function definitions

    Reference: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18
    """
    return count_tokens(
        f"# Tools\n\n## functions\n\n{format_function_specs_as_typescript_ns(functions)}"
    )


def _functions_compat_fix_kwargs(
    functions: list[CompletionModelFunction],
    completion_kwargs: dict,
):
    function_definitions = format_function_specs_as_typescript_ns(functions)
    function_call_schema = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "name": JSONSchema(
                description="The name of the function to call",
                enum=[f.name for f in functions],
                required=True,
            ),
            "arguments": JSONSchema(
                description="The arguments for the function call",
                type=JSONSchema.Type.OBJECT,
                required=True,
            ),
        },
    )
    tool_calls_schema = JSONSchema(
        type=JSONSchema.Type.ARRAY,
        items=JSONSchema(
            type=JSONSchema.Type.OBJECT,
            properties={
                "type": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    enum=["function"],
                ),
                "function": function_call_schema,
            },
        ),
    )

    completion_kwargs["messages"] = [
        ChatMessage.system(
            "# tool usage instructions\n\n"
            "Specify a '```tool_calls' block in your response,"
            " with a valid JSON object that adheres to the following schema:\n\n"
            f"{tool_calls_schema.to_dict()}\n\n"
            "Specify any tools that you need to use through this JSON object.\n\n"
            "Put the tool_calls block at the end of your response"
            " and include its fences if it is not the only content.\n\n"
            "## functions\n\n"
            "For the function call itself, use one of the following"
            f" functions:\n\n{function_definitions}"
        ),
    ]


def _tool_calls_compat_extract_calls(response: str) -> list[AssistantToolCallDict]:
    import json
    import re

    logging.debug(f"Trying to extract tool calls from response:\n{response}")

    if response[0] == "[":
        tool_calls: list[AssistantToolCallDict] = json.loads(response)
    else:
        block = re.search(r"```(?:tool_calls)?\n(.*)\n```\s*$", response, re.DOTALL)
        if not block:
            raise ValueError("Could not find tool calls block in response")
        tool_calls: list[AssistantToolCallDict] = json.loads(block.group(1))

    for t in tool_calls:
        t["function"]["arguments"] = str(t["function"]["arguments"])  # HACK
    return tool_calls
