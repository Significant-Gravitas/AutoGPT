from __future__ import annotations

"""
This module defines the structure and representation of chat interactions, properties, and function specifications for a language model within a chat-based system. It includes classes for representing chat messages, function parameters, and function specifications, which are crucial for building prompts and handling interactions in a structured and type-safe manner.

The classes in this module are typically utilized within the AutoGPT framework, particularly in classes inheriting from `PromptStrategy`, for constructing system prompts and processing user and assistant interactions. The `CompletionModelFunction` class is instrumental in defining function specifications that can be invoked by the language model. These definitions are used in `OneShotAgentPromptStrategy` to detail available commands for the language model.

Key Classes:
    - `Role`: An enumeration class representing the role of an entity in a chat.
    - `ChatMessage`: Represents a message in a chat interaction, encapsulating the sender's role and message content.
    - `Property`: Defines the basic structure for a single property within a function parameter specification.
    - `FunctionParameters`: Provides a structured representation of a function's parameters.
    - `CompletionModelFunction`: Encapsulates a function specification that can be invoked by the language model.
    - `AssistantFunctionCall`: Represents a function call by the assistant, encapsulating the function name and arguments.
    - `AssistantChatMessage`: Extends `ChatMessage` to include optional function call information alongside message content.
    - `AssistantChatMessageDict`: A typed dictionary for representing an `AssistantChatMessage` as a dictionary.
     - `AssistantFunctionCall`: Encapsulates a function call made by the assistant within a chat interaction.
    - `AssistantFunctionCallDict`: A typed dictionary for representing an `AssistantFunctionCall` as a dictionary.
    - `ChatPrompt`: Encapsulates the structure of a chat prompt used within a chat interaction with the language model.
    - `ChatModelResponse`: Standard response structure for a response from a language model, encapsulating the response and parsed result.
    - `Property`: Defines the basic structure for a single property within a function parameter specification.
    - `FunctionParameters`: Provides a structured representation of a function's parameters.
    - `CompletionModelFunction`: Encapsulates a function specification that can be invoked by the language model.


Module Usage:
    This module is used to define and structure the various elements involved in a chat interaction within the AutoGPT framework. It is imported and utilized in constructing and processing prompts, messages, and function calls within the chat-based interaction model.

Examples:
    >>> from AFAAS.lib.utils.json_schema import JSONSchema
    >>> from AFAAS.core.agents.exampleagent.strategies.mystrategy import MyStrategy

    >>> # Defining a function specification
    >>> func_spec = {
    ...     "name": "sum_numbers",
    ...     "description": "Sums two numbers.",
    ...     "parameters": {
    ...         "a": {"type": "number", "description": "First number"},
    ...         "b": {"type": "number", "description": "Second number"}
    ...     }
    ... }
    >>> cmf = CompletionModelFunction.parse(func_spec)
    >>> print(cmf.fmt_line())
    sum_numbers: Sums two numbers. Params: (a: number, b: number)

    >>> # Creating a chat message
    >>> msg = ChatMessage.user("Calculate the sum of 5 and 3.")
    >>> print(msg.role, msg.content)
    Role.USER Calculate the sum of 5 and 3.
"""

import abc
import enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field

from AFAAS.interfaces.adapters.language_model import (
    AbstractLanguageModelProvider,
    BaseModelInfo,
    BaseModelResponse,
    ModelProviderService,
)
from AFAAS.lib.utils.json_schema import JSONSchema


class AbstractRoleLabels(abc.ABC, BaseModel):
    USER: str
    SYSTEM: str
    ASSISTANT: str
    FUNCTION: Optional[str] = None


class AbstractChatMessage(abc.ABC, BaseModel):
    _role_labels: ClassVar[AbstractRoleLabels]
    role: str
    content: str

    @classmethod
    def assistant(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.ASSISTANT, content=content)

    @classmethod
    def user(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.USER, content=content)

    @classmethod
    def system(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.SYSTEM, content=content)

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["role"] = self.role
        return d


class Role(str, enum.Enum):
    """
    An enumeration class representing the roles of different entities in a chat conversation.
    The `Role` class is fundamental to role-based messaging within the module,
    serving as a key attribute in `ChatMessage` and `AssistantChatMessage` classes.

    Attributes:
        USER (str): Represents the user in the conversation.
        SYSTEM (str): Represents system-specific instructions or information.
        ASSISTANT (str): Represents the assistant's responses or actions.
        FUNCTION (str): Represents the return value of function calls within a conversation.

    Example:
    1.
    role = Role.USER
    print(role)  # Output: Role.USER

    2.
    def get_role(role_str: str) -> Role:
        return Role(role_str)

    print(get_role("assistant"))  # Output: Role.ASSISTANT
    """

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

    FUNCTION = "function"
    """May be used for the return value of function calls"""


from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage, AIMessage


class AFAASChatMessage(BaseModel):
    role: Role
    content: str

    @staticmethod
    def assistant(content: str) -> "AFAASChatMessage":
        return AFAASChatMessage(role=Role.ASSISTANT, content=content)

    @staticmethod
    def user(content: str) -> "AFAASChatMessage":
        return AFAASChatMessage(role=Role.USER, content=content)

    @staticmethod
    def system(content: str) -> "AFAASChatMessage":
        return AFAASChatMessage(role=Role.SYSTEM, content=content)

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["role"] = self.role.value
        return d


class ChatMessageDict(TypedDict):
    """
    `ChatMessageDict` serves as a typed dictionary representation of a chat message,
    providing a structured format for `ChatMessage` instances when transformed into a dictionary.
    It is utilized within the `ChatPrompt` class, specifically in the `raw()` method,
    to return a list of dictionary representations of chat messages.


    Attributes:
    - role (str): The role of the entity sending the message. (e.g., "user", "system", "assistant", "function")
    - content (str): The content of the message.

    Example:
    1.
    message = ChatMessageDict(role="user", content="Hello, World!")
    print(message)  # Output: {'role': 'user', 'content': 'Hello, World!'}

    2.
    def print_message(message: ChatMessageDict) -> None:
        print(f"{message['role'].title()}: {message['content']}")

    print_message({'role': 'assistant', 'content': 'How can I help?'})  # Output: Assistant: How can I help?
    """

    role: str
    content: str


# Basic structure for a single property
class Property(BaseModel):
    """
    `Property` represents a single property within a function's parameters or a model's schema.
    It is used within the `FunctionParameters` class to structure the parameters of a `CompletionModelFunction`.


    Attributes:
    - type (str): The type of the property.
    - description (str): A description of the property.
    - items (Optional[Union["Property", Dict]]): If the property is of type array, this attribute defines the schema of the items in the array.
    - properties (Optional[dict]): If the property is of type object, this attribute defines the schema of the properties of the object.

    Example:
    1.
    prop = Property(type="string", description="A simple string property")
    print(prop.dict())  # Output: {'type': 'string', 'description': 'A simple string property', 'items': None, 'properties': None}

    2.
    nested_prop = Property(
        type="object",
        description="A nested object property",
        properties={"name": {"type": "string", "description": "The name of the item"}}
    )
    print(nested_prop.dict())  # Output: {'type': 'object', 'description': 'A nested object property', 'items': None, 'properties': {'name': {'type': 'string', 'description': 'The name of the item'}}}
    """

    type: str
    description: str
    items: Optional[Union["Property", Dict]]
    properties: Optional[dict]  # Allows nested properties


# Defines a function's parameters
class FunctionParameters(BaseModel):
    """
    `FunctionParameters` provides a structured representation of the parameters required for a function call within a language model. It captures the type, properties, and required fields to ensure a valid function call. This class is often utilized within `CompletionModelFunction` to define function specifications for language model interactions.

    Attributes:
        type (str): Specifies the data type of the function parameters, usually 'object'.
        properties (Dict[str, Property]): A dictionary mapping parameter names to `Property` objects, detailing the individual properties of the parameters.
        required (List[str]): A list of parameter names that are required for the function call.

    Methods:
        None

    Examples:
        >>> param_specs = {
        ...     "text": {"type": "string", "description": "Text to be processed"},
        ...     "num": {"type": "integer", "description": "A number parameter"}
        ... }
        >>> func_params = FunctionParameters(type="object", properties=param_specs, required=["text"])
        >>> print(func_params.type, func_params.required)
        object ['text']

        >>> param_specs_2 = {
        ...     "query": {"type": "string", "description": "Query text"},
        ...     "limit": {"type": "integer", "description": "Limit on responses"}
        ... }
        >>> func_params_2 = FunctionParameters(type="object", properties=param_specs_2, required=["query", "limit"])
        >>> print(func_params_2.type, func_params_2.required)
        object ['query', 'limit']
    """

    type: str
    properties: Dict[str, Property]
    required: list[str]


class AssistantFunctionCall(BaseModel):
    """
    `AssistantFunctionCall` encapsulates a function call made by the assistant within a chat interaction.
    This class is utilized within `AssistantChatMessage` to represent function call information alongside
    the assistant's message content.

    Attributes:
        name (str): The name of the function being called.
        arguments (str): The arguments passed to the function in string format.

    Examples:
        >>> afc = AssistantFunctionCall(name="calculate_sum", arguments="5, 3")
        >>> print(afc.name, afc.arguments)
        calculate_sum 5, 3
    """

    name: str
    arguments: str


class AssistantFunctionCallDict(TypedDict):
    """
    A Typed Dictionary for representing an `AssistantFunctionCall` as a dictionary.
    This representation is used within `AssistantChatMessageDict` to provide a structured
    format for function call information.

    Attributes:
        name (str): The name of the function being called.
        arguments (str): The arguments passed to the function in string format.

    Example:
        >>> afc_dict = AssistantFunctionCallDict(name="calculate_sum", arguments="5, 3")
        >>> print(afc_dict)
        {'name': 'calculate_sum', 'arguments': '5, 3'}
    """

    name: str
    arguments: str


class AssistantToolCall(BaseModel):
    # id: str
    type: Literal["function"]
    function: AssistantFunctionCall


class AssistantToolCallDict(TypedDict):
    # id: str
    type: Literal["function"]
    function: AssistantFunctionCallDict


class AssistantChatMessage(AFAASChatMessage):

    role: Role.ASSISTANT
    content: Optional[str]
    tool_calls: Optional[list[AssistantToolCall]]


class AssistantChatMessageDict(TypedDict, total=False):
    """
    A Typed Dictionary for representing an `AssistantChatMessage` as a dictionary.
    This representation provides a structured format for the assistant's message and function call information.

    Attributes:
        role (str): The role of the assistant.
        content (str): The textual content of the message.
        function_call (AssistantFunctionCallDict): A dictionary representation of the function call made by the assistant.

    Example:
        >>> acm_dict = AssistantChatMessageDict(role="assistant", content="The sum is 8", function_call={'name': 'calculate_sum', 'arguments': '5, 3'})
        >>> print(acm_dict)
        {'role': 'assistant', 'content': 'The sum is 8', 'function_call': {'name': 'calculate_sum', 'arguments': '5, 3'}}
    """

    role: str
    content: str
    tool_calls: list[AssistantToolCallDict]


class CompletionModelFunction(BaseModel):
    """
    `CompletionModelFunction` encapsulates a function specification that can be invoked by the language model within a chat-based interaction. The instances of this class are used to communicate the available commands to the language model, as seen in the `OneShotAgentPromptStrategy` where they are provided as input to build the system prompt section detailing available commands. The class allows for a structured representation of a function's name, description, and parameters, facilitating a clear and unambiguous definition of what functions the language model can call and with what arguments.

    Attributes:
        name (str): The name of the function, used to uniquely identify and call the function.
        description (str): A textual description providing an overview of the function's purpose and behavior.
        parameters (dict[str, "JSONSchema"]): A dictionary mapping parameter names to JSONSchema objects, describing the type and other properties of each parameter.

    Methods:
        schema() -> dict[str, str | dict | list]: Returns an OpenAI-consumable function specification as a dictionary.
        parse(schema: dict) -> "CompletionModelFunction": Classmethod to parse a dictionary representation of a function specification into a `CompletionModelFunction` instance.
        _remove_none_entries(data: Dict[str, Any]) -> Dict[str, Any]: Helper method to remove entries with None values from a dictionary.
        dict(*args, **kwargs) -> Dict[str, Any]: Returns a dictionary representation of the `CompletionModelFunction` object, excluding entries with None values.
        fmt_line() -> str: Returns a string representation of the function signature including the function name, description, and formatted parameter list.

    Examples:
        >>> func_spec = {
        ...     "name": "sum_numbers",
        ...     "description": "Sums two numbers.",
        ...     "parameters": {
        ...         "a": {"type": "number", "description": "First number"},
        ...         "b": {"type": "number", "description": "Second number"}
        ...     }
        ... }
        >>> cmf = CompletionModelFunction.parse(func_spec)
        >>> print(cmf.fmt_line())
        sum_numbers: Sums two numbers. Params: (a: number, b: number)

        >>> func_schema = cmf.schema()
        >>> print(func_schema["name"], func_schema["description"])
        sum_numbers Sums two numbers.
    """

    name: str
    description: str
    parameters: dict[str, "JSONSchema"]

    @property
    def schema(self) -> dict[str, str | dict | list]:
        """Returns an OpenAI-consumable function specification"""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: param.to_dict() for name, param in self.parameters.items()
                },
                "required": [
                    name for name, param in self.parameters.items() if param.required
                ],
            },
        }

    @staticmethod
    def parse(schema: dict) -> "CompletionModelFunction":
        return CompletionModelFunction(
            name=schema["name"],
            description=schema["description"],
            parameters=JSONSchema.parse_properties(schema["parameters"]),
        )

    def _remove_none_entries(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_data = {}
        for key, value in data.items():
            if value is not None:
                if isinstance(value, dict):
                    cleaned_data[key] = self._remove_none_entries(value)
                else:
                    cleaned_data[key] = value
        return cleaned_data

    def dict(self, *args, **kwargs):
        # Call the parent class's dict() method to get the original dictionary
        data = super().dict(*args, **kwargs)

        # Remove entries with None values recursively
        cleaned_data = self._remove_none_entries(data)

        return cleaned_data

    def fmt_line(self) -> str:
        params = ", ".join(
            f"{name}: {p.type.value}" for name, p in self.parameters.items()
        )
        return f"{self.name}: {self.description}. Params: ({params})"


class ChatPrompt(BaseModel):
    """
    `ChatPrompt` encapsulates the structure of a chat prompt used within a chat interaction with the language model. It holds a sequence of chat messages, a list of available function specifications (`CompletionModelFunction` instances), a designated function call, and a default function call. This class is instrumental in structuring the interaction and providing the language model with necessary context, available commands, and specific instructions for the ongoing interaction.

    Attributes:
        messages (list[ChatMessage]): A list of `ChatMessage` instances representing the conversation history.
        functions (list[CompletionModelFunction], optional): A list of `CompletionModelFunction` instances representing the available functions the language model can call. Defaults to an empty list.
        function_call (str): A string representing a designated function (example : `myfunction`) to be called within the interaction. If you have only one function and you want to force the LLM to call this function, we recommand you to put the name of your function (example `function_call="myfunction"`), to let a LLM select the most apropriate of function within a list of function  `function_call="auto"`
        default_tool_choice (str): This is a safeguard mechanism especialy usefull when using `function_call="auto"`, after 2 fails you can force a function of your choice.

    Methods:
        raw() -> list[ChatMessageDict]: Returns a list of dictionary representations of the messages in the chat prompt.
        __str__() -> str: Returns a string representation of the chat prompt, formatting each message as "ROLE: content".

    Examples:
        >>> chat_msgs = [ChatMessage.user("Hello!"), ChatMessage.assistant("Hi there!")]
        >>> func_specs = [CompletionModelFunction.parse({
        ...     "name": "greet",
        ...     "description": "Greets the user.",
        ...     "parameters": {}
        ... })]
        >>> chat_prompt = ChatPrompt(
        ...     messages=chat_msgs,
        ...     functions=func_specs,
        ...     function_call="greet",
        ...     default_function_call="greet"
        ... )
        >>> print(chat_prompt)
        USER: Hello!

        ASSISTANT: Hi there!
        >>> raw_msgs = chat_prompt.raw()
        >>> print(raw_msgs)
        [{'role': 'user', 'content': 'Hello!'}, {'role': 'assistant', 'content': 'Hi there!'}]
    """

    messages: list[ChatMessage]
    tools: list[CompletionModelFunction] = Field(default_factory=list)
    tool_choice: str
    default_tool_choice: str

    def raw(self) -> list[ChatMessageDict]:
        return [m.dict() for m in self.messages]

    def __str__(self):
        return "\n\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in self.messages
        )


_T = TypeVar("_T")


class AbstractChatModelResponse(BaseModelResponse, Generic[_T]):
    """
    `ChatModelResponse` extends `BaseModelResponse` to provide a structured representation of a language model's response in a chat-based interaction. It encapsulates the standard response from the language model, along with a parsed result of a generic type `_T`, and additional content as a dictionary.

    This class is instrumental in handling and processing the responses from the language model, allowing for typed parsing of the result and additional content.

    Attributes:
        response (AssistantChatMessageDict): The standard dictionary representation of the assistant's chat message, encapsulating the role, content, and potential function call information.
        parsed_result (_T, optional): A parsed result of generic type `_T`, facilitating typed handling of the language model's response. Defaults to None.
        content (dict, optional): Additional content or data accompanying the language model's response. Defaults to None.

    Example:
        Suppose there's a function `parse_response` that processes the language model's response to extract certain information.

        >>> def parse_response(response: AssistantChatMessageDict) -> str:
        ...     # Assume it extracts and returns some text from the response
        ...     return extracted_text

        >>> lm_response = {
        ...     "role": "assistant",
        ...     "content": "The sum is 8.",
        ...     "tool_calls": None
        ... }
        >>> chat_model_response = ChatModelResponse(response=lm_response, parsed_result=parse_response(lm_response))
        >>> print(chat_model_response.parsed_result)
        The sum is 8.
    """

    response: AssistantChatMessageDict
    parsed_result: _T = None
    """Standard response struct for a response from a language model."""

    content: dict = None
    chat_messages: list[ChatMessage] = []
    system_prompt: str = None


###############
# Chat Models #
###############


class ChatModelInfo(BaseModelInfo):
    """Struct for language model information."""

    llm_service = ModelProviderService.CHAT
    max_tokens: int
    has_function_call_api: bool = False


class ChatCompletionKwargs(BaseModel):
    llm_model_name: str
    """The name of the language model"""
    tools: Optional[list[CompletionModelFunction]] = None
    """List of available tools"""
    tool_choice: Optional[str] = None
    """Force the use of one tool"""
    default_tool_choice: Optional[str] = None
    """This tool would be called after 3 failed attemps(cf : try/catch block)"""

class ChatModelWrapper:

    llm_adapter : AbstractChatModelProvider


    def __init__(self, llm_model: BaseChatModel) -> None:

        self.llm_adapter = llm_model

        self.retry_per_request = llm_model._settings.configuration.retries_per_request
        self.maximum_retry = llm_model._settings.configuration.maximum_retry
        self.maximum_retry_before_default_function = llm_model._settings.configuration.maximum_retry_before_default_function

        retry_handler = _RetryHandler(
            num_retries=self.retry_per_request,
        )
        self._create_chat_completion = retry_handler(self._chat)
        self._func_call_fails_count = 0


    async def create_chat_completion(
        self,
        chat_messages: list[ChatMessage],
        completion_kwargs: ChatCompletionKwargs,
        completion_parser: Callable[[AssistantChatMessageDict], _T], 
        # Function to parse the response, usualy injectect by an AbstractPromptStrategy
        **kwargs,
    ) -> AbstractChatModelResponse[_T]:
        if isinstance(messages, ChatMessage):
            messages = [messages]
        elif not isinstance(messages, list):
            raise TypeError(
                f"Expected ChatMessage or list[ChatMessage], but got {type(messages)}"
            )

        # ##############################################################################
        # ### Prepare arguments for API call using CompletionKwargs
        # ##############################################################################
        llm_kwargs = self._make_chat_kwargs(completion_kwargs=completion_kwargs, **kwargs)

        # ##############################################################################
        # ### Step 2: Execute main chat completion and extract details
        # ##############################################################################

        response = await self._create_chat_completion(
            messages=chat_messages, 
            llm_kwargs = llm_kwargs,
            **kwargs
        )
        response_message, response_args = self.llm_adapter._extract_response_details(
            response=response, 
            model_name=completion_kwargs.llm_model_name
        )

        # ##############################################################################
        # ### Step 3: Handle missing function call and retry if necessary
        # ##############################################################################
        if self.llm_adapter._should_retry_function_call(
            tools=completion_kwargs.tools, response_message=response_message
        ):
            LOG.error(
                f"Attempt number {self._func_call_fails_count + 1} : Function Call was expected"
            )
            if (
                self._func_call_fails_count
                <= self.maximum_retry
            ):
                return await self._retry_chat_completion(
                    model_prompt=chat_messages,
                    completion_kwargs=completion_kwargs,
                    completion_parser=completion_parser,
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
        return self.llm_adapter._formulate_final_response(
            response_message=response_message,
            completion_parser=completion_parser,
            response_args=response_args,
        )


    async def _retry_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        completion_kwargs: ChatCompletionKwargs,
        completion_parser: Callable[[AssistantChatMessageDict], _T],
        response: AsyncCompletions,
        response_args: Dict[str, Any],
        **kwargs
    ) -> AbstractChatModelResponse[_T]:
        self._func_call_fails_count += 1
        self.llm_adapter._budget.update_usage_and_cost(model_response=response)
        return await self.create_chat_completion(
            chat_messages=model_prompt,
            completion_parser=completion_parser,
            completion_kwargs= completion_kwargs,
            **kwargs
        )

    def _make_chat_kwargs(self, completion_kwargs : ChatCompletionKwargs , **kwargs) -> dict:   

        built_kwargs = {}
        built_kwargs.update(self.llm_adapter.make_model_arg(model_name=completion_kwargs.llm_model_name))

        if completion_kwargs.tools is None or len(completion_kwargs.tools) == 0:
            #if their is no tool we do nothing 
            return built_kwargs

        else:
            built_kwargs.update(self.llm_adapter.make_tools_arg(tools=completion_kwargs.tools))

            if len(completion_kwargs.tools) == 1:
                built_kwargs.update(self.llm_adapter.make_tool_choice_arg(name= completion_kwargs.tools[0].name))
                #built_kwargs.update(self.llm_adapter.make_tool_choice_arg(name= completion_kwargs.tools[0]["function"]["name"]))
            elif completion_kwargs.tool_choice!= "auto":
                if (
                    self._func_call_fails_count
                    >= self.maximum_retry_before_default_function
                ):
                    built_kwargs.update(self.llm_adapter.make_tool_choice_arg(name=completion_kwargs.default_tool_choice))
                else:
                    built_kwargs.update(self.llm_adapter.make_tool_choice_arg(name=completion_kwargs.tool_choice))
        return built_kwargs

    def count_message_tokens(
        self,
        messages: AFAASChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int: 
        return self.llm_adapter.count_message_tokens(messages, model_name)

    async def _chat(
        self, 
        messages: list[ChatMessage],
        llm_kwargs : dict, 
        *_, 
        **kwargs
    ) -> AsyncCompletions:

        raw_messages = [
            message.dict(include={"role", "content", "tool_calls", "name"})
            for message in messages
        ]

        #llm_kwargs = self._make_chat_kwargs(**kwargs)
        LOG.trace(raw_messages[0]["content"])
        LOG.trace(llm_kwargs)
        return_value = await self.llm_adapter.chat(
            messages=raw_messages, **llm_kwargs
        )

        return return_value

    def has_oa_tool_calls_api(self, model_name: str) -> bool:
        self.llm_adapter.has_oa_tool_calls_api(model_name)

    def get_default_config(self) -> AbstractPromptConfiguration:
        return self.llm_adapter.get_default_config()



class AbstractChatModelProvider(AbstractLanguageModelProvider): 

    llm_model : Optional[BaseChatModel] = None

    @abc.abstractmethod
    def count_message_tokens(
        self,
        messages: AFAASChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int: ...

    async def create_language_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: str,
        completion_parser: Callable[[dict], dict],
        tools: list[CompletionModelFunction],
        tool_choice: str,
        **kwargs,
    ) -> AbstractChatModelResponse: ...

    @abc.abstractmethod
    async def create_chat_completion(
        self,
        model_name: str,
        chat_messages: list[ChatMessage],
        tools: list[CompletionModelFunction] = [],
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        **kwargs,
    ) -> AbstractChatModelResponse[_T]: ...
