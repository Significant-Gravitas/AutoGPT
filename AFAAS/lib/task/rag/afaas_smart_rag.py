from __future__ import annotations
from langchain.tools import DuckDuckGoSearchRun

import enum
import os
import uuid
from typing import Optional, TYPE_CHECKING, Callable
from jinja2 import Environment, FileSystemLoader, select_autoescape

if  TYPE_CHECKING: 
    from AFAAS.interfaces.task import AbstractTask

from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy, DefaultParsedResponse, PromptStrategiesConfiguration)
from AFAAS.interfaces.prompts.schema import \
     PromptStrategyLanguageModelClassification
from AFAAS.interfaces.adapters import (
    AssistantChatMessageDict, ChatMessage, ChatPrompt, CompletionModelFunction)
from AFAAS.lib.utils.json_schema import JSONSchema
from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name = __name__)

from AFAAS.interfaces.prompts.utils import (to_dotted_list, to_md_quotation,
                    to_numbered_list, to_string_list, indent)     
class AFAAS_SMART_RAGStrategyFunctionNames(str, enum.Enum):
    MAKE_SMART_RAG: str = "afaas_smart_rag"


class AFAAS_SMART_RAGStrategyConfiguration(PromptStrategiesConfiguration):
    """
    A Pydantic model that represents the default configurations for the refine user context strategy.
    """
    model_classification:  PromptStrategyLanguageModelClassification = (
         PromptStrategyLanguageModelClassification.FAST_MODEL_4K
    )
    default_tool_choice: AFAAS_SMART_RAGStrategyFunctionNames = (
        AFAAS_SMART_RAGStrategyFunctionNames.MAKE_SMART_RAG
    )
    task_context_length: int = 300
    temperature : float = 0.4


class AFAAS_SMART_RAG_Strategy(AbstractPromptStrategy):
    default_configuration = AFAAS_SMART_RAGStrategyConfiguration()
    STRATEGY_NAME = "afaas_smart_rag"

    def __init__(
        self,
        model_classification:  PromptStrategyLanguageModelClassification,
        default_tool_choice: AFAAS_SMART_RAGStrategyFunctionNames,
        temperature : float , #if coding 0.05
        top_p: Optional[float] ,
        max_tokens : Optional[int] ,
        frequency_penalty: Optional[float], # Avoid repeting oneselfif coding 0.3
        presence_penalty : Optional[float], # Avoid certain subjects
        count=0,
        exit_token: str = str(uuid.uuid4()),
        task_context_length: int = 300,
    ):
        self._model_classification = model_classification
        self._count = count
        self._config = self.default_configuration
        self.default_tool_choice = default_tool_choice
        self.task_context_length = task_context_length

    def set_tools(self, 
                    task : AbstractTask, 
                    task_history : list[AbstractTask],
                    task_sibblings : list[AbstractTask],
              **kwargs):
        self.afaas_smart_rag : CompletionModelFunction = CompletionModelFunction(
            name=AFAAS_SMART_RAGStrategyFunctionNames.MAKE_SMART_RAG.value,
            description="Provide accurate information to perform a task",
            parameters={
                "uml_diagrams": JSONSchema(
                    type=JSONSchema.Type.ARRAY,
                    items=JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description= f"A list of the task identified by their IF with UML diagrams relevant to the task",
                        required=True,
                        enum=[task.task_id for task in task_history + task_sibblings]
                    )
                ),
                "resume": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description= f"Information related to past tasks that can be relevant to the execution of the task {task.task_goal}task but the \"situational context\" and the \"task goal\". This note should be {str(self.task_context_length * 0.8)} to {str(self.task_context_length *  1.25)}  words long.",
                    required=True,
                ),
                "long_description" : JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description= f"Description of the tasks (minimum 80 words long).",
                    required=True,
                ),
            }
        )   


        self._tools = [
            self.afaas_smart_rag,
        ]


    def build_message(
        self, task : AbstractTask , **kwargs
    ) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + task.debug_dump_str())
        self._task : AbstractTask = task
        self._model_name = kwargs.get("model_name")
        smart_rag_param = {
            "task_goal" : task.task_goal,
            "additional_context_description": str(task.task_context),
            'task_history' : kwargs.get('task_history', None),
            'task_sibblings' : kwargs.get('task_sibblings', None),
            'task_path' : kwargs.get('task_path', None),
            'related_tasks' : kwargs.get('related_tasks', None),
        }

        messages = []
        messages.append(
                        ChatMessage.system(
                            self._build_jinja_message(task = task, 
                                            template_name= f'{self.STRATEGY_NAME}.jinja', template_params = smart_rag_param)
                            )
                    )
        messages.append(
            ChatMessage.system(
                self.response_format_instruction(
                    model_name=self._model_name,
                    )
                )
            )

        return self.build_chat_prompt(messages=messages)

    def _build_jinja_message(self, task : AbstractTask, template_name : str, template_params : dict) -> str:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_loader = FileSystemLoader(current_directory)
        env = Environment(loader=file_loader,
            autoescape=select_autoescape(['html', 'xml']),
            extensions=["jinja2.ext.loopcontrols"]
            )
        template = env.get_template(template_name)

        template_params.update({"to_md_quotation": to_md_quotation,
                                 "to_dotted_list": to_dotted_list, 
                                 "to_numbered_list": to_numbered_list,
                                    "to_string_list": to_string_list,
                                    "indent": indent, 
                                "task" : self._task})
        return template.render(template_params)
        

    def build_chat_prompt(self, messages: list[ChatMessage])-> ChatPrompt:

        strategy_tools = self.get_tools()
        prompt = ChatPrompt(
            messages=messages,
            tools=strategy_tools,
            tool_choice="auto",
            default_tool_choice=self.default_tool_choice,
            tokens_used=0,
        )

        return prompt

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    )   -> DefaultParsedResponse:  
        return self.default_parse_response_content(response_content ) 
        # parsed_response : DefaultParsedResponse = self.default_parse_response_content(response_content ) 
        # parsed_response.command_name
        # self._task.task_context = response_content.get("task_context", None)
        # self._task.task_context = response_content.get("task_context", None)

    def response_format_instruction(self, model_name: str) -> str:
        model_provider = self._agent._chat_model_provider
        return super().response_format_instruction(language_model_provider=model_provider, model_name = model_name)
