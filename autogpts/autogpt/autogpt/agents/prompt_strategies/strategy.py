from __future__ import annotations

import abc
import os
import re
import sys
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import ChatMessage, AIMessage


from autogpt.core.configuration import SystemConfiguration

if TYPE_CHECKING:
    from autogpt.interfaces.task.task import AbstractTask

from autogpt.core.configuration import SystemConfiguration
from autogpt.interfaces.adapters.language_model import AbstractLanguageModelProvider, AbstractPromptConfiguration
from autogpt.interfaces.adapters.chatmodel import  (
    ChatPrompt,
    AbstractChatModelProvider,
    AbstractChatModelResponse,
    AssistantChatMessage,
    CompletionModelFunction,
)
from autogpt.interfaces.utils import (
    indent,
    json_loads,
    to_dotted_list,
    to_md_quotation,
    to_numbered_list,
    to_string_list,
)
from autogpt.core.utils.json_schema import JSONSchema

import logging

LOG = logging.getLogger(__name__)
RESPONSE_SCHEMA = JSONSchema(
    type=JSONSchema.Type.OBJECT,
    properties={
        "thoughts": JSONSchema(
            type=JSONSchema.Type.OBJECT,
            required=True,
            properties={
                "limits": JSONSchema(
                    description="Express your limitations (Context Limitation, Token Limitation, Cognitive Limitation) if you were an autonomous program hosted on a server and relying on a Large Language Model to take decision",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "overcome_limit": JSONSchema(
                    description="How you woud overcome this limit (if any)",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "reasoning": JSONSchema(
                    description="Your process of thoughts",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "criticism": JSONSchema(
                    description="Constructive self-criticism of your process of thoughts",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "plan": JSONSchema(
                    description="Short markdown-style bullet list that conveys your plan",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "self_feedback": JSONSchema(
                    description="if you were to do it again what would you told yourself",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
            },
        )
    },
)


class DefaultParsedResponse(dict):
    id: str
    type: str
    command_name: str
    command_args: dict
    assistant_reply_dict: dict


class PromptStrategiesConfiguration(SystemConfiguration):
    temperature: float
    # top_p: Optional[float] = None
    # max_tokens: Optional[int] = None
    # frequency_penalty: Optional[float] = None  # Avoid repeting oneselfif coding 0.3
    # presence_penalty: Optional[float] = None  # Avoid certain subjects


class AbstractPromptStrategy( abc.ABC):
    STRATEGY_NAME: str
    default_configuration: PromptStrategiesConfiguration

    @abc.abstractmethod
    async def build_message(self, *_, **kwargs) -> ChatPrompt: ...

    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessage): ...

    @abc.abstractmethod
    def set_tools(self, **kwargs): ...

    @abc.abstractmethod
    def get_llm_provider(self) -> AbstractChatModelProvider:
        """ Get the Provider : Gemini, OpenAI, Llama, ... """
        return self._agent.prompt_manager.config.default

    @abc.abstractmethod
    def get_prompt_config(self) -> AbstractPromptConfiguration:
        """ Get the Prompt Configuration : Model version (eg: gpt-3.5, gpt-4...), temperature, top_k, top_p, ... """
        return self.get_llm_provider().get_default_config()

    # TODO : This implementation is shit :)
    def get_tools(self) -> list[CompletionModelFunction]:
        """
        Returns a list of functions related to refining user context.

        Returns:
            list: A list of CompletionModelFunction objects detailing each function's purpose and parameters.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> functions = strategy.get_functions()
            >>> print(functions[0].name)
            'refine_requirements'
        """
        return self._tools

    def get_tools_names(self) -> list[str]:
        """
        Returns a list of names of functions related to refining user context.

        Returns:
            list: A list of strings, each representing the name of a function.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> function_names = strategy.get_tools_names()
            >>> print(function_names)
            ['refine_requirements']
        """
        return [item.name for item in self._tools]

    # This can be expanded to support multiple types of (inter)actions within an agent
    @abc.abstractmethod
    def response_format_instruction(
        self,
    ) -> str:
        language_model_provider = self.get_llm_provider()
        model_name = self.get_prompt_config().llm_model_name

        #response_schema = RESPONSE_SCHEMA.copy(deep=True)
        response_schema = RESPONSE_SCHEMA

        # Unindent for performance
        response_format: str = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface("Response"),
        )

        return (
            f"Respond strictly with JSON. The JSON should be compatible with the TypeScript type `Response` from the following:\n"
            f"{response_format}"
        )

    ###
    ### parse_response_content
    ###
    def default_parse_response_content(
        self,
        response_content: AssistantChatMessage,
    ) -> list[DefaultParsedResponse]:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        assistant_return_list: list[DefaultParsedResponse] = []

        assistant_reply_dict = response_content.content

        if (isinstance(response_content, AIMessage)):
            tool_calls = response_content.additional_kwargs['tool_calls']
        else:
            tool_calls = response_content["tool_calls"]

        if tool_calls : 
            for tool in tool_calls:
                try:
                    command_args = json_loads(tool["function"]["arguments"])
                except Exception:
                    LOG.warning(command_args)

                ###
                ### NEW
                ###
                command_name = tool["function"]["name"]

                assistant_return_list.append(
                    DefaultParsedResponse(
                        id=tool["id"],
                        type=tool["type"],
                        command_name=command_name,
                        command_args=command_args,
                        assistant_reply_dict=assistant_reply_dict,
                    )
                )

        return assistant_return_list

    @staticmethod
    def get_autocorrection_response(response: AbstractChatModelResponse):
        return response.parsed_result[0]["command_args"]["note_to_agent"]

    async def _build_jinja_message(
        self, task: AbstractTask, template_name: str, template_params: dict
    ) -> str:
        """Build a message using jinja2 template engine"""

        # Get the module of the calling (child) class
        class_module = sys.modules[self.__class__.__module__]
        child_directory = os.path.dirname(os.path.abspath(class_module.__file__))
        # Check if template exists in child class directory, else use parent directory
        if os.path.exists(os.path.join(child_directory, template_name)):
            directory_to_use = child_directory
        else:
            directory_to_use = os.path.dirname(os.path.abspath(__file__))

        file_loader = FileSystemLoader(directory_to_use)
        env = Environment(
            loader=file_loader,
            autoescape=select_autoescape(["html", "xml"]),
            extensions=["jinja2.ext.loopcontrols"],
        )
        template = env.get_template(template_name)

        template_params.update(
            {
                "to_md_quotation": to_md_quotation,
                "to_dotted_list": to_dotted_list,
                "to_numbered_list": to_numbered_list,
                "to_string_list": to_string_list,
                "indent": indent,
                "task": task,
                "strategy": self,
            }
        )
        if hasattr(task, "task_parent"):
            template_params.update({"task_parent": await task.task_parent()})

        return template.render(template_params)





    def build_chat_prompt(self, messages: list[ChatMessage] , tool_choice : str = "auto") -> ChatPrompt:
        ChatMessage

        # messagev2 = []
        # if isinstance(messages, list):
        #     for message in messages:
        #         messagev2.append(convert_v1_instance_to_v2_dynamic(message))
        # else : 
        #     messagev2.append(convert_v1_instance_to_v2_dynamic(messages))

        # print("""////////////////////////////////////\n\n"""*3)
        # print(messages[0])
        # print(messagev2[0])
        # exit()
        strategy_tools = self.get_tools()
        prompt = ChatPrompt(
            messages = messages ,
            tools= strategy_tools ,
            tool_choice = tool_choice ,
            default_tool_choice =self.default_tool_choice ,
            tokens_used = 0 ,
        )

        return prompt




# from pydantic import BaseModel
# def convert_v1_instance_to_v2_dynamic(obj_v1: BaseModel) -> BaseModel:

#         from pydantic import  create_model
#         from typing import Type
#         """
#         Converts an instance of a Pydantic v1 model to a dynamically created Pydantic v2 model instance.

#         Parameters:
#         - obj_v1: The instance of the Pydantic version 1 model.

#         Returns:
#         - An instance of a dynamically created Pydantic version 2 model that mirrors the structure of obj_v1.
#         """
#         # Extract field definitions from the v1 instance
#         fields = {name: (field.outer_type_, ...) for name, field in obj_v1.__fields__.items()}

#         # Dynamically create a new Pydantic model class
#         DynamicModelV2 = create_model('DynamicModelV2', **fields)

#         # Convert the v1 instance to a dictionary and use it to create an instance of the new model
#         obj_v2 = DynamicModelV2.parse_obj(obj_v1.dict())

#         return obj_v2
