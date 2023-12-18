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
from typing import (Any, Callable, Dict, Generic, List, Literal, Optional,
                    TypedDict, TypeVar, Union, ClassVar)

from pydantic import BaseModel, Field

from AFAAS.interfaces.adapters.language_model import (
    AbstractLanguageModelProvider, BaseModelInfo, BaseModelResponse,
    ModelProviderService)
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


class ChatMessage(BaseModel):
    """
    We invite you to Read OpenAI API function_call documentation for further understanding. `ChatMessage` is the representation of a chat interaction with a large language model. During chat interaction different persons have send messages (system, user, assistant...).

    `ChatMessage` encapsulates a message within a chat interaction with a language model. This object captures the sender's role (`Role`) and the message content (`content`).

    It is typically utilized in classes inheriting from `PromptStrategy`, especially within the `build_prompt` method.
    Instances of `ChatMessage` constitute the conversation history
    in the `ChatPrompt` class, aiding in structuring the interaction with the language model.

    Attributes:
        role (Role): The role of the entity sending the message. It can be 'user', 'system', or 'assistant'.
        content (str): The textual content of the message.

    Methods:
        assistant(content: str) -> "ChatMessage": Constructs a ChatMessage object with role 'ASSISTANT'.
        user(content: str) -> "ChatMessage": Constructs a ChatMessage object with role 'USER'.
        system(content: str) -> "ChatMessage": Constructs a ChatMessage object with role 'SYSTEM'.
        dict(**kwargs): Returns a dictionary representation of the ChatMessage object, with 'role' represented as a string.

    Examples:
        >>> msg1 = ChatMessage.assistant("Hello there!")
        >>> print(msg1.role, msg1.content)
        Role.ASSISTANT Hello there!

        >>> msg2 = ChatMessage.user("Good morning!")
        >>> print(msg2.role, msg2.content)
        Role.USER Good morning!
    """

    role: Role
    content: str

    @staticmethod
    def assistant(content: str) -> "ChatMessage":
        return ChatMessage(role=Role.ASSISTANT, content=content)

    @staticmethod
    def user(content: str) -> "ChatMessage":
        return ChatMessage(role=Role.USER, content=content)

    @staticmethod
    def system(content: str) -> "ChatMessage":
        return ChatMessage(role=Role.SYSTEM, content=content)

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


class AssistantChatMessage(ChatMessage):
    """
    `AssistantChatMessage` extends `ChatMessage` to include optional function call information
    alongside the message content. This class provides a structured representation of the assistant's
    responses and actions within a chat interaction.

    Attributes:
        role (Role.ASSISTANT): The role of the assistant, inherited from `ChatMessage`.
        content (Optional[str]): The textual content of the message.
        function_call (Optional[AssistantFunctionCall]): An `AssistantFunctionCall` instance representing a function call made by the assistant.

    Examples:
        >>> acm = AssistantChatMessage(content="The sum is 8", function_call=AssistantFunctionCall(name="calculate_sum", arguments="5, 3"))
        >>> print(acm.role, acm.content, acm.function_call.name)
        Role.ASSISTANT The sum is 8 calculate_sum
    """

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


class ChatModelResponse(BaseModelResponse, Generic[_T]):
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

class BaseChatModelProvider(AbstractLanguageModelProvider):
    @abc.abstractmethod
    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int:
        ...

    async def create_language_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: str,
        completion_parser: Callable[[dict], dict],
        tools: list[CompletionModelFunction],
        tool_choice: str,
        **kwargs,
    ) -> ChatModelResponse:
        ...

    @abc.abstractmethod
    async def create_chat_completion(
        self,
        model_name: str,
        chat_messages: list[ChatMessage],
        tools: list[CompletionModelFunction] = [],
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
        ...
