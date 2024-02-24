from pydantic import BaseModel
from autogpt.interfaces.adapters.chatmodel.chatmodel import LOG, _RetryHandler, AbstractChatModelProvider, AbstractChatModelResponse, CompletionModelFunction
from autogpt.interfaces.adapters.language_model import AbstractPromptConfiguration

from typing import (
    Callable,
    TypeVar,
    Optional
)
from autogpt.interfaces.adapters.chatmodel.chatmessage import AssistantChatMessage

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ChatMessage
from openai.resources import AsyncCompletions


from typing import Callable



_T = TypeVar("_T")


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
        completion_parser: Callable[[AssistantChatMessage], _T],
        # Function to parse the response, usualy injectect by an AbstractPromptStrategy
        **kwargs,
    ) -> AbstractChatModelResponse[_T]:
        if isinstance(chat_messages, ChatMessage):
            chat_messages = [chat_messages]
        elif not isinstance(chat_messages, list):
            raise TypeError(
                f"Expected ChatMessage or list[ChatMessage], but got {type(chat_messages)}"
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
        response_message = self.llm_adapter.extract_response_details(
            response=response,
            model_name=completion_kwargs.llm_model_name
        )

        # ##############################################################################
        # ### Step 3: Handle missing function call and retry if necessary
        # ##############################################################################
        # FIXME : Remove before commit
        if self.llm_adapter.should_retry_function_call(
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
                    response=response_message,
                )

            # FIXME, TODO, NOTE: Organize application save feedback loop to improve the prompts, as it is not normal that function are not called
            try :
                response_message.additional_kwargs['tool_calls'] = None
            except Exception as e:
                response_message['tool_calls'] = None
                LOG.warning(f"Following Exception occurred : {e}")

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
        return self.llm_adapter.formulate_final_response(
            response_message=response_message,
            completion_parser=completion_parser,
        )


    async def _retry_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        completion_kwargs: ChatCompletionKwargs,
        completion_parser: Callable[[AssistantChatMessage], _T],
        response: AsyncCompletions,
        **kwargs
    ) -> AbstractChatModelResponse[_T]:
        self._func_call_fails_count += 1

        self.llm_adapter._budget.update_usage_and_cost(model_response=response.base_response)
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
        messages: ChatMessage | list[ChatMessage],
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

        #llm_kwargs = self._make_chat_kwargs(**kwargs)
        LOG.trace(messages[0].content)
        LOG.trace(llm_kwargs)
        return_value = await self.llm_adapter.chat(
            messages=messages, **llm_kwargs
        )

        return return_value

    def get_default_config(self) -> AbstractPromptConfiguration:
        return self.llm_adapter.get_default_config()

