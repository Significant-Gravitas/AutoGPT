"""
RefineUserContextStrategy Module

This module provides strategies and configurations to assist the AI in refining and clarifying user requirements
through an iterative process, based on the COCE Framework.

Classes:
---------
RefineUserContextFunctionNames: Enum
    Enum class that lists function names used in refining user context.

RefineUserContextConfiguration: BaseModel
    Pydantic model that represents the default configurations for the refine user context strategy.

RefineUserContextStrategy: BasePromptStrategy
    Strategy that guides the AI in refining and clarifying user requirements based on the COCE Framework.

Examples:
---------
To initialize and use the `RefineUserContextStrategy`:

>>> strategy = RefineUserContextStrategy(logger, model_classification=LanguageModelClassification.FAST_MODEL_4K, default_function_call=RefineUserContextFunctionNames.REFINE_REQUIREMENTS, strategy_name="refine_user_context", context_min_tokens=250, context_max_tokens=300)
>>> prompt = strategy.build_prompt(exit_refinement_process=False, user_objective="Build a web app")
"""
from logging import Logger
import uuid
import enum
from typing import Optional
from pydantic import BaseModel

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

from autogpt.core.utils.json_schema import JSONSchema

from autogpt.core.prompting.utils.utils import json_loads, to_numbered_list, to_string_list
from autogpt.core.prompting.base import (
    BasePromptStrategy,
    PromptStrategiesConfiguration,
)
from autogpt.core.prompting.schema import (
    LanguageModelClassification,
)

from autogpt.core.resource.model_providers import (
    CompletionModelFunction,
    ChatMessage,
    AssistantChatMessageDict,
    ChatPrompt,
)

from autogpt.core.agents.simple.lib.models.user_response import Questions, AgentUserResponse


class RefineUserContextFunctionNames(str, enum.Enum):
    """
    An enumeration that lists the function names used in refining user context.

    Attributes:
    -----------
    REFINE_REQUIREMENTS: str
        Function to refine requirements.
    REQUEST_SECOND_CONFIRMATION: str
        Function to request a second confirmation.
    VALIDATE_REQUIREMENTS: str
        Function to validate requirements.
    """
    REFINE_REQUIREMENTS: str = "refine_requirements"
    REQUEST_SECOND_CONFIRMATION: str = "request_second_confirmation"
    VALIDATE_REQUIREMENTS: str = "validate_requirements"


class RefineUserContextConfiguration(PromptStrategiesConfiguration):
    """
    A Pydantic model that represents the default configurations for the refine user context strategy.
    """
    model_classification: LanguageModelClassification = (
        LanguageModelClassification.FAST_MODEL_4K
    )
    default_function_call: RefineUserContextFunctionNames = (
        RefineUserContextFunctionNames.REFINE_REQUIREMENTS
    )
    strategy_name: str = "refine_user_context"
    context_min_tokens: int = 250
    context_max_tokens: int = 500
    use_message: bool = False


class RefineUserContextStrategy(BasePromptStrategy):
    """
    A strategy that guides the AI in refining and clarifying user requirements based on the COCE Framework.

    Attributes:
    -----------
    default_configuration : RefineUserContextConfiguration
        The default configuration used for the strategy.
    STRATEGY_NAME : str
        Name of the strategy.
    CONTEXT_MIN_TOKENS : int
        Minimum number of tokens in the context.
    CONTEXT_MAX_TOKENS : int
        Maximum number of tokens in the context.
    DEFAULT_SYSTEM_PROMPT_TEMPLATE : str
        The default system prompt template.
    SYSTEM_PROMPT_MESSAGES : str
        Messages in the system prompt.
    SYSTEM_PROMPT_FOOTER : str
        Footer information for the system prompt.
    SYSTEM_PROMPT_NOTICE : str
        Notice information for the system prompt.

    Methods:
    --------
    build_prompt(exit_refinement_process: bool, user_objective: str, **kwargs) -> ChatPrompt:
        Build a chat prompt based on the user's objective and whether the refinement process should exit.
    """
    default_configuration = RefineUserContextConfiguration()
    STRATEGY_NAME = "refine_user_context"
    CONTEXT_MIN_TOKENS = 250
    CONTEXT_MAX_TOKENS = 300

    ###
    ### PROMPTS
    ###

    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        "## Instructions :\n\n"
        "You are an AI running on {os_info}. Your are tasked with assisting a user in formulating user requirements through an iterative process."
        """
## Iterative Process Flow:

1. Step 1 **User Provides Requirements**: You will receive user's requirements, reformulate them, and ask three questions to assist users in providing more information. 
2. Step 2 **User Answers Clarification Questions**: You will receive the reformulated user requirements from Step 1, questions from Step 1, and user responses. You will reformulate user requirements by merging the previous requirements with the answers.

**Loop Control**: Continue this iterative process until users express a willingness to finalize the process."""
        "\n\n"
        "## Guidance :\n\n"
        "To ensure the user's requirements adhere to our standards, follow the **COCE Framework**:\n\n"
        " - **Comprehensible**: Your AI needs to be able to understand the goal, even within its limited context window. Minimize ambiguity, and use specific terminologies or semantics that the AI can comprehend.\n"
        " - **Outcome-driven**: Focus on the end results or macro-goals that the AI should achieve, rather than measurable micro-goals or the steps that need to be taken to get there.\n"
        " - **Context-aware**: The goal should be aware of and clearly define the context in which the AI is expected to function. This is especially important if the AI has a limited understanding of the world or the domain in which it operates.\n"
        " - **Explicitness**: The goal must explicitly state what the AI needs to do. There should be no hidden assumptions or implied requirements. Everything that the AI needs to know to complete the goal should be explicitly stated.\n\n"
        "Your primary role is to assist the user in adhering to these principles and guide them through the process of formulating requirements that meet these criteria. Please use your capabilities to ensure that the user's goal aligns with these principles by generating questions that guide him closer to our expectations in term of user requirement expression. \n\n"
    )

    SYSTEM_PROMPT_MESSAGES = "{generated_message_history}"

    SYSTEM_PROMPT_FOOTER = """\n\n## Key points:

1. **Utilize User's Input**: ALWAYS use the user's prior requirements and their recent responses to inform your responses.
2. **No Assumptions**: NEVER make assumptions. Always align with the user's original expressions.
3. **Adhere to COCE**: Your reformulations, questions, and interactions should aim to get the user's requirements to fit the COCE Framework.
4. **Use Functions**: You MUST use the provided functions in your interactions: {function_names}.

It's crucial to use the user's input, make no assumptions, align with COCE, and use the provided functions. Prioritize the user's input and remain faithful to the COCE principles."""

    #     NEW_SYSTEM_PROMPT = ("###  User's Requirements:\n"
    # #"So far the user have expressed this requirements :\n"
    # "\"{user_response}\"\n\n"
    # )
    NEW_SYSTEM_PROMPT = ""
    NEW_ASSISTANT_PROMPT = "What are your requirements ?\n"
    # NEW_USER_PROMPT = "These are my requirement : \"{user_response}\"\n"
    NEW_USER_PROMPT = "{user_response}"

    #     REFINED_ONCE_SYSTEM_PROMPT = ("### User's Journey:\n"
    # #"So far the user have expressed these requirements :\n"
    # "\"{user_response}\"\n\n"
    # "- **Your questions**:\n"
    # "{last_questions}\n"
    # "- **User's Response**: \n"
    # "{user_response}\n\n"
    # )

    #     REFINED_SYSTEM_PROMPT = ("### User's Journey:\n"
    # #"So far the user have expressed these requirements :\n"
    # "'{user_last_goal}'\n\n"
    # "- **Your full history of Questions**:\n"
    # "'{questions_history_full}\n"
    # "- **Your Last Questions**:\n"
    # "{last_questions}\n"
    # "- **User's Last Response**:\n"
    # "{user_response}\n\n"
    # )

    #     REFINED_SYSTEM_PROMPT = ("### User's Journey:\n"
    # "So far the user have expressed these requirements :\n"
    # "'{user_last_goal}'\n")
    REFINED_SYSTEM_PROMPT = ""
    REFINED_USER_PROMPT_A = "{user_last_goal}"
    REFINED_ASSISTANT_PROMPT = "{last_questions}"
    REFINED_USER_PROMPT_B = "{user_response}"

    #     SYSTEM_PROMPT_FOOTER = ("Guide the user closer to a refined goal by referencing the above journey and adhering to the **COCE Framework**. Provide actionable insights and questions.\n\n"
    # "### Note:\n"
    # "This is an iterative process. Your generated `reformulated_goal` will serve as the user's next goal and the questions will gather further details from the user.\n"
    # )

    # SYSTEM_PROMPT_NOTICE = "**IMPORTANT** : YOU MUST USE FUNCTIONS AT YOUR DISPOSAL"
    SYSTEM_PROMPT_NOTICE = ""

    ###
    ### FUNCTIONS
    ###

    FUNCTION_REFINE_USER_CONTEXT = CompletionModelFunction(
        name=RefineUserContextFunctionNames.REFINE_REQUIREMENTS,
        description="Refines and clarifies user requirements by reformulating them and generating pertinent questions to extract more detailed and explicit information from the user, all while adhering to the COCE Framework.",
        parameters={
            "reformulated_goal": JSONSchema(
                type=JSONSchema.Type.STRING,
                ##"description": f"Users requirements as interpreted from their initial expressed requirements and their most recent answer, making sure it adheres to the COCE Framework and remains true to the user's intent. It should be formatted using Markdown and expressed in {CONTEXT_MIN_TOKENS} to {CONTEXT_MAX_TOKENS} words."
                description=f"Users requirements as interpreted from their initial expressed requirements and their most recent answer, making sure it adheres to the COCE Framework and remains true to the user's intent. It should be formatted using Markdown and expressed in less than {CONTEXT_MAX_TOKENS} words.",
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

    FUNCTION_REQUEST_SECOND_CONFIRMATION = {
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

    FUNCTION_VALIDATE_GOAL = CompletionModelFunction(
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

    def __init__(
        self,
        logger: Logger,
        model_classification: LanguageModelClassification,
        default_function_call: RefineUserContextFunctionNames,
        strategy_name: str,
        context_min_tokens: int,
        context_max_tokens: int,
        count=0,
        user_last_goal="",
        exit_token: str = str(uuid.uuid4()),
        use_message: bool = False,
    ):
        """
        Initialize the RefineUserContextStrategy.

        Parameters:
        -----------
        logger: Logger
            The logger object.
        model_classification: LanguageModelClassification
            Classification of the language model.
        default_function_call: RefineUserContextFunctionNames
            Default function call for the strategy.
        strategy_name: str
            Name of the strategy.
        context_min_tokens: int
            Minimum number of tokens in the context.
        context_max_tokens: int
            Maximum number of tokens in the context.
        count: int, optional (default = 0)
            The count for the iterative process.
        user_last_goal: str, optional (default = "")
            Last goal provided by the user.
        exit_token: str, optional
            Token to indicate exit from the process.
        use_message: bool, optional (default = False)
            Flag to determine whether to use messages.
        """
        self._logger = logger
        self._model_classification = model_classification
        # NOTE : Make a list of Questions ?
        self.question_history_full: list[Questions] = []
        self.question_history_label_full: list[str] = []
        self._last_questions: list[Questions] = []
        self._last_questions_label: list[str] = []
        self._user_last_goal = user_last_goal
        self._count = count
        self._config = self.default_configuration
        self.exit_token: str = exit_token

        function_refine_user_context = (
            RefineUserContextStrategy.FUNCTION_REFINE_USER_CONTEXT
        )
        # function_request_second_confirmation= CompletionModelFunction(
        #     **RefineUserContextStrategy.FUNCTION_REQUEST_SECOND_CONFIRMATION,
        # )
        function_validate_requirements = (
            RefineUserContextStrategy.FUNCTION_VALIDATE_GOAL
        )
        self._functions = [
            function_refine_user_context,
            # function_request_second_confirmation ,
            function_validate_requirements,
        ]

    def build_prompt(
        self, exit_refinement_process: bool, user_objective: str = "", **kwargs
    ) -> ChatPrompt:
        """
        Build a chat prompt based on the user's objective and whether the refinement process should exit.

        Parameters:
        -----------
        exit_refinement_process: bool
            Flag indicating whether to exit the refinement process.
        user_objective: str, optional
            Objective provided by the user.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> user_input = "I need a system that can monitor temperatures."
            >>> prompt = strategy.build_prompt(exit_refinement_process=False, user_objective=user_input)
            >>> print(prompt.messages[0].content)

        Returns:
        --------
        ChatPrompt
            The chat prompt generated for the strategy.
        """
        #
        # STEP 1 : List all functions available
        #
        strategy_functions = self.get_functions()

        if self._config.use_message == True:
            #
            # Step 2 A : Build the prompts using messages
            #
            if self._count == 0:
                # NEW
                first_system_prompt = self.DEFAULT_SYSTEM_PROMPT_TEMPLATE
                first_system_prompt += self.SYSTEM_PROMPT_FOOTER
                first_user_prompt = self.NEW_USER_PROMPT
            else:
                # REFINED
                first_system_prompt = self.DEFAULT_SYSTEM_PROMPT_TEMPLATE
                first_system_prompt += self.SYSTEM_PROMPT_FOOTER
                first_user_prompt = self.REFINED_USER_PROMPT_A
                first_assistant_prompt = self.REFINED_ASSISTANT_PROMPT
                second_user_prompt = self.REFINED_USER_PROMPT_B
        else:
            #
            # Step 2 B : Build a single system prompt
            #
            generated_message_history = "## Current Iteration:\n\n"
            generated_message_history += (
                f"This is the iteration number: {self._count+ 1}\n\n"
            )

            if self._count == 0:
                generated_message_history += (
                    f'The user requirement are:\n\n"{user_objective}"'
                )
            else:
                generated_message_history += (
                    f'The user requirement are:\n"{self._user_last_goal}"\n'
                )
                generated_message_history += "- **Your Last Questions**:\n"
                generated_message_history += (
                    f"{to_numbered_list(self._last_questions_label)}\n"
                )
                generated_message_history += "- **User's Last Response**:\n"
                generated_message_history += f"{user_objective}"

            first_system_prompt = self.DEFAULT_SYSTEM_PROMPT_TEMPLATE
            first_system_prompt += generated_message_history
            first_system_prompt += self.SYSTEM_PROMPT_FOOTER

        #
        # Step 3 : Build the first system message
        #
        first_system_message = ChatMessage.system(
            content=first_system_prompt.format(
                # generated_message_history = generated_message_history,
                os_info=kwargs["os_info"],
                functions_list=to_numbered_list(self.get_functions_names()),
                function_names=to_string_list(self.get_functions_names()),
                user_last_goal=self._user_last_goal,
                questions_history_full=to_numbered_list(
                    self.question_history_label_full
                ),
                last_questions=to_numbered_list(self._last_questions_label),
                user_response=user_objective,
                count=self._count + 1,
            )
        )

        if self._config.use_message == True:
            if self._count == 0:
                first_user_message = ChatMessage.user(
                    content=first_user_prompt.format(user_response=user_objective)
                )
                messages = [first_system_message, first_user_message]
            else:
                first_user_message = ChatMessage.user(
                    content=first_user_prompt.format(
                        user_last_goal=self._user_last_goal
                    )
                )
                first_assistant_message = ChatMessage.assistant(
                    content=first_assistant_prompt.format(
                    last_questions=to_numbered_list(self._last_questions_label)
                    )
                )
                second_user_message = ChatMessage.user(
                    content=second_user_prompt.format(user_response=user_objective)
                )
                messages = [
                    first_system_message,
                    first_user_message,
                    first_assistant_message,
                    second_user_message,
                ]
        else:
            messages = [first_system_message]

        # Step 4 : Hallucination safegard
        #
        function_call = "auto"
        if self._count == 0:
            function_call = RefineUserContextFunctionNames.REFINE_REQUIREMENTS
            messages.append(
                ChatMessage.system(
                    content="""Before finalizing your response, ensure that:
1. The reformulated goal adheres strictly to the user's provided information, with no assumptions or hallucinations.
2. You have prepared 5 relevant questions based on the user's requirements."""
                )
            )
        elif exit_refinement_process == True:
            function_call = RefineUserContextFunctionNames.VALIDATE_REQUIREMENTS

        #
        # Step 5 :
        #
        prompt = ChatPrompt(
            messages=messages,
            functions=strategy_functions,
            function_call=function_call,
            default_function_call=RefineUserContextFunctionNames.REFINE_REQUIREMENTS,
            # TODO
            tokens_used=0,
        )
        # self._logger.debug('Executing prompt : ' + str(prompt))
        return prompt

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> dict:
        """
        Parse the actual text response from the objective model.

        Args:
            response_content (AssistantChatMessageDict): The raw response content from the objective model.

        Returns:
            dict: The parsed response containing questions, goal refinements, and other related data.

        Raises:
            Exception: If the response_content can't be parsed properly.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> raw_response = {
            >>>     "function_call": {
            >>>         "name": "REFINE_REQUIREMENTS",
            >>>         "arguments": '{"questions": ["What temperature range?", "Any specific brand?"], "reformulated_goal": "Monitor temperatures in the specified range with preferred brand"}'
            >>>     }
            >>> }
            >>> parsed = strategy.parse_response_content(raw_response)
            >>> print(parsed['questions'])
            ['What temperature range?', 'Any specific brand?']
            >>> print(parsed['reformulated_goal'])
            'Monitor temperatures in the specified range with preferred brand'
        """
        try:
            parsed_response = json_loads(response_content["function_call"]["arguments"])
        except Exception:
            self._logger.warning(parsed_response)

        #
        # Give id to questions
        # TODO : Type Questions in a Class ?
        #
        save_questions = False
        if (
            response_content["function_call"]["name"]
            == RefineUserContextFunctionNames.REFINE_REQUIREMENTS
        ):
            questions_with_uuid = []
            question = []
            # questions_with_uuid = [{"id": "Q" + str(uuid.uuid4()), "question": q} for q in parsed_response["questions"]]
            for q_text in parsed_response["questions"]:
                question_id = Questions.generate_new_id()
                question = Questions(
                    id=question_id, message=q_text, type=None, state=None, items=[]
                )
            questions_with_uuid.append(question)
            save_questions = True

            # Saving the last goal
            self._user_last_goal = parsed_response["reformulated_goal"]

        elif (
            response_content["function_call"]["name"]
            == RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION
        ):
            # questions_with_uuid = [{"id": "Q" + str(uuid.uuid4()), "question":  parsed_response["questions"]}]
            question_id = Questions.generate_new_id()
            question_text = parsed_response["questions"]
            question = Questions(
                id=question_id, message=question_text, type=None, state=None, items=[]
            )
            questions_with_uuid.append(question)
            save_questions = True

        # Saving the questions
        if save_questions:
            self.question_history_full.extend(questions_with_uuid)
            self.question_history_label_full.extend(parsed_response["questions"])
            self._last_questions = questions_with_uuid
            self._last_questions_label = parsed_response["questions"]

        parsed_response["name"] = response_content["function_call"]["name"]
        self._logger.debug(parsed_response)
        self._count += 1
        return parsed_response

    def save(self):
        pass
