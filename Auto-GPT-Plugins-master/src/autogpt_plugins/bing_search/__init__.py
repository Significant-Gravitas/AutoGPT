"""This is the Bing search engines plugin for Auto-GPT."""
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from .bing_search import _bing_search

PromptGenerator = TypeVar("PromptGenerator")


class Message(TypedDict):
    role: str
    content: str


class AutoGPTBingSearch(AutoGPTPluginTemplate):
    def __init__(self):
        super().__init__()
        self._name = "Bing-Search-Plugin"
        self._version = "0.1.0"
        self._description = (
            "This plugin performs Bing searches using the provided query."
        )
        self.load_commands = (
            os.getenv("SEARCH_ENGINE")
            and os.getenv("SEARCH_ENGINE").lower() == "bing"
            and os.getenv("BING_API_KEY")
        )

    def can_handle_post_prompt(self) -> bool:
        return True

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        if self.load_commands:
            # Add Bing Search command
            prompt.add_command(
                "Bing Search",
                "bing_search",
                {"query": "<query>"},
                _bing_search,
            )
        else:
            print(
                "Warning: Bing-Search-Plugin is not fully functional. "
                "Please set the SEARCH_ENGINE and BING_API_KEY environment variables."
            )
        return prompt

    def can_handle_pre_command(self) -> bool:
        return True

    def pre_command(
        self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        if command_name == "google" and self.load_commands:
            # this command does nothing but it is required to continue performing the post_command function
            return "bing_search", {"query": arguments["query"]}
        else:
            return command_name, arguments

    def can_handle_post_command(self) -> bool:
        return False

    def post_command(self, command_name: str, response: str) -> str:
        pass

    def can_handle_on_planning(self) -> bool:
        return False

    def on_planning(
        self, prompt: PromptGenerator, messages: List[Message]
    ) -> Optional[str]:
        pass

    def can_handle_on_response(self) -> bool:
        return False

    def on_response(self, response: str, *args, **kwargs) -> str:
        pass

    def can_handle_post_planning(self) -> bool:
        return False

    def post_planning(self, response: str) -> str:
        pass

    def can_handle_pre_instruction(self) -> bool:
        return False

    def pre_instruction(self, messages: List[Message]) -> List[Message]:
        pass

    def can_handle_on_instruction(self) -> bool:
        return False

    def on_instruction(self, messages: List[Message]) -> Optional[str]:
        pass

    def can_handle_post_instruction(self) -> bool:
        return False

    def post_instruction(self, response: str) -> str:
        pass

    def can_handle_pre_command(self) -> bool:
        return True

    def can_handle_chat_completion(
        self, messages: Dict[Any, Any], model: str, temperature: float, max_tokens: int
    ) -> bool:
        return False

    def handle_chat_completion(
        self, messages: List[Message], model: str, temperature: float, max_tokens: int
    ) -> str:
        pass
