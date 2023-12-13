from __future__ import annotations

import enum
import os
import uuid
from typing import Optional,TYPE_CHECKING
from jinja2 import Environment, FileSystemLoader

if  TYPE_CHECKING: 
    from AFAAS.core.lib.task import Task

from AFAAS.core.prompting.base import (
    BasePromptStrategy, DefaultParsedResponse, PromptStrategiesConfiguration)
from AFAAS.core.prompting.schema import \
    LanguageModelClassification
from AFAAS.core.resource.model_providers import (
    AssistantChatMessageDict, ChatMessage, ChatPrompt, CompletionModelFunction)
from AFAAS.core.utils.json_schema import JSONSchema
from AFAAS.core.lib.sdk.logger import AFAASLogger 

LOG = AFAASLogger(name= __name__)


class AFAAS_SMART_RAGStrategyFunctionNames(str, enum.Enum):
    MAKE_SMART_RAG: str = "afaas_smart_rag"


class AFAAS_SMART_RAGStrategyConfiguration(PromptStrategiesConfiguration):
    """
    A Pydantic model that represents the default configurations for the refine user context strategy.
    """
    model_classification: LanguageModelClassification = (
        LanguageModelClassification.FAST_MODEL_4K
    )
    default_tool_choice: AFAAS_SMART_RAGStrategyFunctionNames = (
        AFAAS_SMART_RAGStrategyFunctionNames.MAKE_SMART_RAG
    )
    note_to_agent_length : int = 250
    temperature : float = 0.4


class AFAAS_SMART_RAG_Strategy(BasePromptStrategy):
    default_configuration = AFAAS_SMART_RAGStrategyConfiguration()
    STRATEGY_NAME = "afaas_smart_rag"

    def __init__(
        self,
        logger: AFAASLogger,
        model_classification: LanguageModelClassification,
        default_tool_choice: AFAAS_SMART_RAGStrategyFunctionNames,
        temperature : float , #if coding 0.05
        top_p: Optional[float] ,
        max_tokens : Optional[int] ,
        frequency_penalty: Optional[float], # Avoid repeting oneselfif coding 0.3
        presence_penalty : Optional[float], # Avoid certain subjects
        count=0,
        exit_token: str = str(uuid.uuid4()),
        use_message: bool = False,
        note_to_agent_length : int = 250
    ):
        self._model_classification = model_classification
        self._count = count
        self._config = self.default_configuration
        self.note_to_agent_length = note_to_agent_length



    def set_tools(self,**kwargs):
        self.afaas_smart_rag : CompletionModelFunction = CompletionModelFunction(
            name=AFAAS_SMART_RAGStrategyFunctionNames.MAKE_SMART_RAG.value,
            description="Provide accurate information to perform a task",
            parameters={
                "task_history": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description= f"Detailed history of the task",
                    required=True,
                ),
                "task_sibblings": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description= f"List of sibblings of the task",
                    required=True,
                ),
                "task_path": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description= f"Path to the task",
                    required=True,
                ),
                "related_tasks": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description= f"Token to be used for a similarity search in a Vector DB",
                    required=True,
                ),
            }
            # task=self,
            # task_history= list(history_and_predecessors).sort(key=lambda task: task.modified_at),
            # task_sibblings=task_sibblings,
            # task_path=task_path,
            # related_tasks = None
            # },
        )   
        

        self._tools = [
            self.afaas_smart_rag,
        ]


    def build_prompt(
        self, task : Task ,**kwargs
    ) -> ChatPrompt:

        # Get the directory containing the currently executing script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        file_loader = FileSystemLoader(current_directory)
        env = Environment(loader=file_loader)
        template = env.get_template('10_routing.jinja')
        
        self.logger().notice("Building prompt for task : " + str(task) )
        routing_param = {
            "step" : 'ROUTING',
            "task" : task, 
            "task_goal" : task.task_goal,
            "additional_context_description": str(task.task_context),
        }
        content = template.render(routing_param)
        messages = [ChatMessage.system(content)]
        strategy_tools = self.get_tools()

        #
        # Step 5 :
        #
        prompt = ChatPrompt(
            messages=messages,
            tools=strategy_tools,
            tool_choice = "auto", 
            default_tool_choice=AFAAS_SMART_RAGStrategyFunctionNames.MAKE_SMART_RAG,
            # TODO
            tokens_used=0,
        )
        
        return prompt

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    )   -> DefaultParsedResponse:
        return self.default_parse_response_content(response_content )    
    
    
    def response_format_instruction(self, model_name: str) -> str:
        model_provider = self._agent._chat_model_provider
        return super().response_format_instruction(language_model_provider=model_provider, model_name = model_name)