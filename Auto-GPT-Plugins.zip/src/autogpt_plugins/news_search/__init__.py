"""This is the News search engine plugin for Auto-GPT."""
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from .news_search import NewsSearch

PromptGenerator = TypeVar("PromptGenerator")


class Message(TypedDict):
    role: str
    content: str


class AutoGPTNewsSearch(AutoGPTPluginTemplate):
    def __init__(self):
        super().__init__()
        self._name = "News-Search-Plugin"
        self._version = "0.1.0"
        self._description = (
            "This plugin searches the latest news using the provided query and the newsapi aggregator"
        )
        self.load_commands = (os.getenv("NEWSAPI_API_KEY")) # Wrapper, if more variables are needed in future
        self.news_search = NewsSearch(os.getenv("NEWSAPI_API_KEY"))

    def can_handle_post_prompt(self) -> bool:
        return True

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        if self.load_commands:
            # Add News Search command
            prompt.add_command(
                "News Search",
                "news_search",
                {"query": "<query>"},
                self.news_search.news_search,
            )
        else:
            print(
                "Warning: News-Search-Plugin is not fully functional. "
                "Please set the NEWSAPI_API_KEY environment variable."
            )
        return prompt

    def can_handle_pre_command(self) -> bool:
        return False

    def pre_command(
        self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        pass

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
        return False

    def can_handle_chat_completion(
        self, messages: Dict[Any, Any], model: str, temperature: float, max_tokens: int
    ) -> bool:
        return False

    def handle_chat_completion(
        self, messages: List[Message], model: str, temperature: float, max_tokens: int
    ) -> str:
        pass
