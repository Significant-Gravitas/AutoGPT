from __future__ import annotations

import enum
import uuid

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AssistantChatMessageDict,
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptConfiguration,
    AbstractPromptStrategy,
    PromptStrategiesConfiguration,
)
from AFAAS.interfaces.prompts.utils.utils import (
    json_loads,
    to_numbered_list,
    to_string_list,
)
from AFAAS.lib.message_agent_user import Questions
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema
from AFAAS.interfaces.task.task import AbstractTask

LOG = AFAASLogger(name=__name__)


class RefineUserContextFunctionNames(str, enum.Enum):


    REFINE_REQUIREMENTS: str = "refine_requirements"
    REQUEST_SECOND_CONFIRMATION: str = "request_second_confirmation"
    VALIDATE_REQUIREMENTS: str = "validate_requirements"


class RefineUserContextStrategyConfiguration(PromptStrategiesConfiguration):
    """
    A Pydantic model that represents the default configurations for the refine user context strategy.
    """

    default_tool_choice: RefineUserContextFunctionNames = (
        RefineUserContextFunctionNames.REFINE_REQUIREMENTS
    )
    context_min_tokens: int = 250
    context_max_tokens: int = 500
    use_message: bool = False
    temperature: float = 0.9


class RefineUserContextStrategy(AbstractPromptStrategy):
    default_configuration = RefineUserContextStrategyConfiguration()
    STRATEGY_NAME = "refine_user_context"

    ###
    ### PROMPTS
    ###


    def __init__(
        self,
        default_tool_choice: RefineUserContextFunctionNames,
        context_min_tokens: int,
        context_max_tokens: int,
        temperature: float,  # if coding 0.05
        count=0,
        user_last_goal="",
        exit_token: str = str(uuid.uuid4()),
        use_message: bool = False,
    ):
        # NOTE : Make a list of Questions ?
        self.context_min_tokens: int = context_min_tokens
        self.context_max_tokens: int = context_max_tokens
        self.use_message: bool = use_message

        self.question_history_full: list[Questions] = []
        self.question_history_label_full: list[str] = []
        self._last_questions: list[Questions] = []
        self._last_questions_label: list[str] = []
        self._user_last_goal = user_last_goal
        self._count = count
        self._config = self.default_configuration
        self.exit_token: str = exit_token
        self.default_tool_choice = default_tool_choice

    def set_tools(self, **kwargs):
        ###
        ### FUNCTIONS
        ###
        self.function_refine_user_context: CompletionModelFunction = (
            CompletionModelFunction(
                name=RefineUserContextFunctionNames.REFINE_REQUIREMENTS,
                description="Refines and clarifies user requirements by reformulating them and generating pertinent questions to extract more detailed and explicit information from the user, all while adhering to the COCE Framework.",
                parameters={
                    "reformulated_goal": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"Users requirements as interpreted from their initial expressed requirements and their most recent answer, making sure it adheres to the COCE Framework and remains true to the user's intent. It should be formatted using Markdown and expressed in less than {self.context_max_tokens} words.",
                        required=True,
                    ),
                    "questions": JSONSchema(
                        type=JSONSchema.Type.ARRAY,
                        minItems=1,
                        maxItems=5,
                        items=JSONSchema(
                            type=JSONSchema.Type.STRING,
                        ),
                        description="Five questions designed to extract more detailed and explicit information from the user, guiding them towards a clearer expression of their requirements while staying within the COCE Framework.",
                        required=True,
                    ),
                },
            )
        )

        self.function_request_second_confirmation: CompletionModelFunction = {
            "name": RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION,
            "description": "Double-check the user's intent to end the iterative requirement refining process. It poses a simple yes/no question to ensure that the user truly wants to conclude refining and proceed to the validation step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "string",
                        "description": "A question aimed at reconfirming the user's intention to finalize their requirements. The question should be phrased in a manner that lets the user easily signify continuation ('no') or conclusion ('yes') of the refining process.",
                    }
                },
                "required": ["questions"],
            },
        }

        self.function_request_second_confirmation: CompletionModelFunction = (
            CompletionModelFunction(
                name=RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION,
                description="Double-check the user's intent to end the iterative requirement refining process. It poses a simple yes/no question to ensure that the user truly wants to conclude refining and proceed to the validation step.",
                parameters={
                    "questions": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description="A question aimed at reconfirming the user's intention to finalize their requirements. The question should be phrased in a manner that lets the user easily signify continuation ('no') or conclusion ('yes') of the refining process.",
                        required=True,
                    ),
                },
            )
        )

        self.function_validate_goal: CompletionModelFunction = CompletionModelFunction(
            name=RefineUserContextFunctionNames.VALIDATE_REQUIREMENTS,
            description="Seals the iterative process of refining requirements. It gets activated when the user communicates satisfaction with the requirements, signaling readiness to finalize the current list of goals.",
            parameters={
                "goal_list": JSONSchema(
                    type=JSONSchema.Type.ARRAY,
                    minItems=1,
                    maxItems=5,
                    items=JSONSchema(
                        type=JSONSchema.Type.STRING,
                    ),
                    description="List of user requirements that emerged from prior interactions. Each entry in the list stands for a distinct and atomic requirement or aim expressed by the user.",
                    required=True,
                )
            },
        )

        self._tools = [
            self.function_refine_user_context,
            self.function_request_second_confirmation,
            self.function_validate_goal,
        ]

    async def build_message(
        self, 
        interupt_refinement_process: bool, 
        task : AbstractTask,
        user_objectives: str = "",
        **kwargs
    ) -> ChatPrompt:
        #
        # STEP 1 : List all functions available
        #
        strategy_tools = self.get_tools()

        messages = []
        smart_rag_param = {
            "os_info" : kwargs["os_info"],
            "tools_list" : self.get_tools_names(),
            "tools_names" : to_string_list(self.get_tools_names()),
            "user_last_goal" : self._user_last_goal,
            "questions_history_full" : to_numbered_list(
                self.question_history_label_full
            ),
            "last_questions" : self._last_questions_label,
            "user_response" : user_objectives,
            "count" : self._count + 1,
        }
        messages.append(
            ChatMessage.system(
                await self._build_jinja_message(
                    task=task,
                    template_name=f"{self.STRATEGY_NAME}.jinja",
                    template_params=smart_rag_param,
                )
            )
        )

        # Step 4 : Hallucination safegard
        #
        tool_choice = "auto"
        if self._count == 0:
            tool_choice = RefineUserContextFunctionNames.REFINE_REQUIREMENTS
            messages.append(
                ChatMessage.system(
                    content="""Before finalizing your response, ensure that:
1. The reformulated goal adheres strictly to the user's provided information, with no assumptions or hallucinations.
2. You have prepared 5 relevant questions based on the user's requirements."""
                )
            )
        elif interupt_refinement_process == True:
            tool_choice = RefineUserContextFunctionNames.VALIDATE_REQUIREMENTS
        #messages.append(ChatMessage.system(self.response_format_instruction()))
        prompt_v2 = self.build_chat_prompt(messages=messages , tool_choice=tool_choice)

        return prompt_v2

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> dict:
        try:
            parsed_response = json_loads(
                response_content["tool_calls"][0]["function"]["arguments"]
            )
        except Exception:
            LOG.error(parsed_response)
            raise Exception(f"Error parsing response content {response_content}")


        save_questions = False
        questions_with_uuid = []
        if (
            response_content["tool_calls"][0]["function"]["name"]
            == RefineUserContextFunctionNames.REFINE_REQUIREMENTS
        ):
            question = []
            # questions_with_uuid = [{"id": "Q" + str(uuid.uuid4()), "question": q} for q in parsed_response["questions"]]
            for q_text in parsed_response["questions"]:
                question_id = Questions.generate_uuid()
                question = Questions(
                    question_id=question_id,
                    message=q_text,
                    type=None,
                    state=None,
                    items=[],
                )
            questions_with_uuid.append(question)
            save_questions = True

            # Saving the last goal
            self._user_last_goal = parsed_response["reformulated_goal"]

        elif (
            response_content["tool_calls"][0]["function"]["name"]
            == RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION
        ):
            # questions_with_uuid = [{"id": "Q" + str(uuid.uuid4()), "question":  parsed_response["questions"]}]
            question_id = Questions.generate_uuid()
            question_text = parsed_response["questions"]
            question = Questions(
                question_id=question_id,
                message=question_text,
                type=None,
                state=None,
                items=[],
            )
            questions_with_uuid.append(question)
            save_questions = True

        # Saving the questions
        if save_questions:
            self.question_history_full.extend(questions_with_uuid)
            self.question_history_label_full.extend(parsed_response["questions"])
            self._last_questions = questions_with_uuid
            self._last_questions_label = parsed_response["questions"]

        parsed_response["name"] = response_content["tool_calls"][0]["function"]["name"]
        LOG.trace(parsed_response)
        self._count += 1
        return parsed_response

    def save(self):
        pass

    def response_format_instruction(
        self, language_model_provider: AbstractLanguageModelProvider, model_name: str
    ) -> str:
        pass

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()
