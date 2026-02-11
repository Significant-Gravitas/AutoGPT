from __future__ import annotations

import logging
from typing import Callable, Iterator, Optional

from pydantic import BaseModel

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import AfterExecute, AfterParse, MessageProvider
from forge.llm.prompting.utils import indent
from forge.llm.providers import ChatMessage, MultiProvider
from forge.llm.providers.multi import ModelName
from forge.llm.providers.openai import OpenAIModelName
from forge.llm.providers.schema import ToolResultMessage

from .model import ActionResult, AnyProposal, Episode, EpisodicActionHistory

logger = logging.getLogger(__name__)


class ActionHistoryConfiguration(BaseModel):
    llm_name: ModelName = OpenAIModelName.GPT3
    """Name of the llm model used to compress the history"""
    max_tokens: int = 1024
    """Maximum number of tokens to use up with generated history messages"""
    spacy_language_model: str = "en_core_web_sm"
    """Language model used for summary chunking using spacy"""
    full_message_count: int = 4
    """Number of latest non-summarized messages to include in the history"""
    enable_compression: bool = True
    """Enable LLM-based compression of action history. Disable for ReWOO/benchmarks."""


class ActionHistoryComponent(
    MessageProvider,
    AfterParse[AnyProposal],
    AfterExecute,
    ConfigurableComponent[ActionHistoryConfiguration],
):
    """Keeps track of the event history and provides a summary of the steps."""

    config_class = ActionHistoryConfiguration

    def __init__(
        self,
        event_history: EpisodicActionHistory[AnyProposal],
        count_tokens: Callable[[str], int],
        llm_provider: MultiProvider,
        config: Optional[ActionHistoryConfiguration] = None,
    ) -> None:
        ConfigurableComponent.__init__(self, config)
        self.event_history = event_history
        self.count_tokens = count_tokens
        self.llm_provider = llm_provider

    def get_messages(self) -> Iterator[ChatMessage]:
        messages: list[ChatMessage] = []
        step_summaries: list[str] = []
        tokens: int = 0
        n_episodes = len(self.event_history.episodes)

        # Include a summary for all except a few latest steps
        for i, episode in enumerate(reversed(self.event_history.episodes)):
            # Use full format for a few steps, summary or format for older steps
            if i < self.config.full_message_count:
                messages.insert(0, episode.action.raw_message)
                tokens += self.count_tokens(str(messages[0]))  # HACK
                if episode.result:
                    # Create result messages for ALL tool calls
                    # (required by Anthropic API)
                    result_messages = self._make_result_messages(
                        episode, episode.result
                    )
                    # Insert in reverse order so they appear in correct order
                    for j, result_message in enumerate(result_messages):
                        messages.insert(1 + j, result_message)
                        tokens += self.count_tokens(str(result_message))  # HACK
                continue
            elif episode.summary is None:
                step_content = indent(episode.format(), 2).strip()
            else:
                step_content = episode.summary

            step = f"* Step {n_episodes - i}: {step_content}"

            if self.config.max_tokens and self.count_tokens:
                step_tokens = self.count_tokens(step)
                if tokens + step_tokens > self.config.max_tokens:
                    break
                tokens += step_tokens

            step_summaries.insert(0, step)

        if step_summaries:
            step_summaries_fmt = "\n\n".join(step_summaries)
            yield ChatMessage.user(
                f"## Progress on your Task so far\n"
                "Here is a summary of the steps that you have executed so far, "
                "use this as your consideration for determining the next action!\n"
                f"{step_summaries_fmt}"
            )

        yield from messages

        # Include any pending user feedback as a prominent user message.
        # This ensures the agent pays attention to what the user said,
        # whether they approved a command with feedback or denied it.
        pending_feedback = self.event_history.pop_pending_feedback()
        if pending_feedback:
            feedback_text = "\n".join(f"- {feedback}" for feedback in pending_feedback)
            yield ChatMessage.user(
                f"[USER FEEDBACK] The user provided the following feedback. "
                f"Read it carefully and adjust your approach accordingly:\n"
                f"{feedback_text}"
            )

    def after_parse(self, result: AnyProposal) -> None:
        self.event_history.register_action(result)

    async def after_execute(self, result: ActionResult) -> None:
        self.event_history.register_result(result)
        # Note: Compression is now lazy - happens in prepare_messages() when needed

    async def prepare_messages(self) -> None:
        """Prepare messages by compressing older episodes if needed.

        Call this before get_messages() when building prompts. This enables
        lazy compression - only compress when we actually need the history.

        For strategies like ReWOO EXECUTING phase that skip prompt building,
        this won't be called, avoiding unnecessary LLM calls.
        """
        if self.config.enable_compression:
            await self.event_history.handle_compression(
                self.llm_provider,
                self.config.llm_name,
                self.config.spacy_language_model,
                self.config.full_message_count,
            )

    @staticmethod
    def _make_result_messages(
        episode: Episode, result: ActionResult
    ) -> list[ChatMessage]:
        """Create result messages for all tool calls in an episode.

        When multiple tools are called in parallel, we need to create a
        ToolResultMessage for EACH tool_call to satisfy API requirements
        (both Anthropic and OpenAI require tool_use to be followed by tool_result).

        Args:
            episode: The episode containing the action and its raw message
            result: The result of executing the action(s)

        Returns:
            List of ChatMessage objects (ToolResultMessage or user message)
        """
        tool_calls = (
            episode.action.raw_message.tool_calls
            if episode.action.raw_message.tool_calls
            else []
        )

        # Single tool call or no tool calls - use simple logic
        if len(tool_calls) <= 1:
            return [ActionHistoryComponent._make_single_result_message(episode, result)]

        # Multiple tool calls - create a result for each
        messages: list[ChatMessage] = []

        # Get outputs dict if parallel execution returned a dict
        outputs_dict: dict = {}
        errors_list: list[str] = []
        if result.status == "success" and isinstance(result.outputs, dict):
            outputs_dict = result.outputs
            errors_list = outputs_dict.pop("_errors", [])
        elif result.status == "error":
            # All tools failed - create error results for all
            for tool_call in tool_calls:
                messages.append(
                    ToolResultMessage(
                        content=f"{result.reason}\n\n{result.error or ''}".strip(),
                        is_error=True,
                        tool_call_id=tool_call.id,
                    )
                )
            return messages

        # Create result message for each tool call
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_id = tool_call.id

            # Check if this tool's result is in the outputs
            if tool_name in outputs_dict:
                output = outputs_dict[tool_name]
                messages.append(
                    ToolResultMessage(
                        content=str(output),
                        tool_call_id=tool_id,
                    )
                )
            else:
                # Check if there's an error for this tool
                error_msg = next(
                    (e for e in errors_list if e.startswith(f"{tool_name}:")), None
                )
                if error_msg:
                    messages.append(
                        ToolResultMessage(
                            content=error_msg,
                            is_error=True,
                            tool_call_id=tool_id,
                        )
                    )
                else:
                    # Fallback - tool not found in results
                    messages.append(
                        ToolResultMessage(
                            content="No result returned",
                            is_error=True,
                            tool_call_id=tool_id,
                        )
                    )

        return messages

    @staticmethod
    def _make_single_result_message(
        episode: Episode, result: ActionResult
    ) -> ChatMessage:
        """Create a result message for a single tool call."""
        if result.status == "success":
            return (
                ToolResultMessage(
                    content=str(result.outputs),
                    tool_call_id=episode.action.raw_message.tool_calls[0].id,
                )
                if episode.action.raw_message.tool_calls
                else ChatMessage.user(
                    f"{episode.action.use_tool.name} returned: "
                    + (
                        f"```\n{result.outputs}\n```"
                        if "\n" in str(result.outputs)
                        else f"`{result.outputs}`"
                    )
                )
            )
        elif result.status == "error":
            return (
                ToolResultMessage(
                    content=f"{result.reason}\n\n{result.error or ''}".strip(),
                    is_error=True,
                    tool_call_id=episode.action.raw_message.tool_calls[0].id,
                )
                if episode.action.raw_message.tool_calls
                else ChatMessage.user(
                    f"{episode.action.use_tool.name} raised an error: ```\n"
                    f"{result.reason}\n"
                    "```"
                )
            )
        else:
            # ActionInterruptedByHuman - user provided feedback instead of executing
            # Must return ToolResultMessage to satisfy API requirements (both Anthropic
            # and OpenAI require tool_use/function_call to be followed by tool_result)
            feedback_content = (
                f"Command not executed. User provided feedback: {result.feedback}"
            )
            return (
                ToolResultMessage(
                    content=feedback_content,
                    is_error=True,
                    tool_call_id=episode.action.raw_message.tool_calls[0].id,
                )
                if episode.action.raw_message.tool_calls
                else ChatMessage.user(feedback_content)
            )

    def _compile_progress(
        self,
        episode_history: list[Episode[AnyProposal]],
        max_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        if max_tokens and not count_tokens:
            raise ValueError("count_tokens is required if max_tokens is set")

        steps: list[str] = []
        tokens: int = 0
        n_episodes = len(episode_history)

        for i, episode in enumerate(reversed(episode_history)):
            # Use full format for a few latest steps, summary or format for older steps
            if i < self.config.full_message_count or episode.summary is None:
                step_content = indent(episode.format(), 2).strip()
            else:
                step_content = episode.summary

            step = f"* Step {n_episodes - i}: {step_content}"

            if max_tokens and count_tokens:
                step_tokens = count_tokens(step)
                if tokens + step_tokens > max_tokens:
                    break
                tokens += step_tokens

            steps.insert(0, step)

        return "\n\n".join(steps)
