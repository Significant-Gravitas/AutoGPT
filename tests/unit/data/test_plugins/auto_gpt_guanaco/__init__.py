"""This is the Test plugin for Auto-GPT."""
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from auto_gpt_plugin_template import AutoGPTPluginTemplate

PromptGenerator = TypeVar("PromptGenerator")


class AutoGPTGuanaco(AutoGPTPluginTemplate):
    """
    This is plugin for Auto-GPT.
    """

    def __init__(self):
        super().__init__()
        self._name = "Auto-GPT-Guanaco"
        self._version = "0.1.0"
        self._description = "This is a Guanaco local model plugin."

    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.

        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model."""
        if len(response):
            print("OMG OMG It's Alive!")
        else:
            print("Is it alive?")

    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.

        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return False

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.

        Args:
            prompt (PromptGenerator): The prompt generator.

        Returns:
            PromptGenerator: The prompt generator.
        """

    def can_handle_on_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_planning method.

        Returns:
            bool: True if the plugin can handle the on_planning method."""
        return False

    def on_planning(
        self, prompt: PromptGenerator, messages: List[str]
    ) -> Optional[str]:
        """This method is called before the planning chat completeion is done.

        Args:
            prompt (PromptGenerator): The prompt generator.
            messages (List[str]): The list of messages.
        """

    def can_handle_post_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_planning method.

        Returns:
            bool: True if the plugin can handle the post_planning method."""
        return False

    def post_planning(self, response: str) -> str:
        """This method is called after the planning chat completeion is done.

        Args:
            response (str): The response.

        Returns:
            str: The resulting response.
        """

    def can_handle_pre_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_instruction method.

        Returns:
            bool: True if the plugin can handle the pre_instruction method."""
        return False

    def pre_instruction(self, messages: List[str]) -> List[str]:
        """This method is called before the instruction chat is done.

        Args:
            messages (List[str]): The list of context messages.

        Returns:
            List[str]: The resulting list of messages.
        """

    def can_handle_on_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_instruction method.

        Returns:
            bool: True if the plugin can handle the on_instruction method."""
        return False

    def on_instruction(self, messages: List[str]) -> Optional[str]:
        """This method is called when the instruction chat is done.

        Args:
            messages (List[str]): The list of context messages.

        Returns:
            Optional[str]: The resulting message.
        """

    def can_handle_post_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_instruction method.

        Returns:
            bool: True if the plugin can handle the post_instruction method."""
        return False

    def post_instruction(self, response: str) -> str:
        """This method is called after the instruction chat is done.

        Args:
            response (str): The response.

        Returns:
            str: The resulting response.
        """

    def can_handle_pre_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_command method.

        Returns:
            bool: True if the plugin can handle the pre_command method."""
        return False

    def pre_command(
        self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """This method is called before the command is executed.

        Args:
            command_name (str): The command name.
            arguments (Dict[str, Any]): The arguments.

        Returns:
            Tuple[str, Dict[str, Any]]: The command name and the arguments.
        """

    def can_handle_post_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_command method.

        Returns:
            bool: True if the plugin can handle the post_command method."""
        return False

    def post_command(self, command_name: str, response: str) -> str:
        """This method is called after the command is executed.

        Args:
            command_name (str): The command name.
            response (str): The response.

        Returns:
            str: The resulting response.
        """

    def can_handle_chat_completion(
        self,
        messages: list[Dict[Any, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        """This method is called to check that the plugin can
          handle the chat_completion method.

        Args:
            messages (Dict[Any, Any]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.

          Returns:
              bool: True if the plugin can handle the chat_completion method."""
        return False

    def handle_chat_completion(
        self,
        messages: list[Dict[Any, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """This method is called when the chat completion is done.

        Args:
            messages (Dict[Any, Any]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.

        Returns:
            str: The resulting response.
        """

    def can_handle_text_embedding(self, text: str) -> bool:
        """This method is called to check that the plugin can
          handle the text_embedding method.
        Args:
            text (str): The text to be convert to embedding.
          Returns:
              bool: True if the plugin can handle the text_embedding method."""
        return False

    def handle_text_embedding(self, text: str) -> list:
        """This method is called when the chat completion is done.
        Args:
            text (str): The text to be convert to embedding.
        Returns:
            list: The text embedding.
        """

    def can_handle_user_input(self, user_input: str) -> bool:
        """This method is called to check that the plugin can
        handle the user_input method.

        Args:
            user_input (str): The user input.

        Returns:
            bool: True if the plugin can handle the user_input method."""
        return False

    def user_input(self, user_input: str) -> str:
        """This method is called to request user input to the user.

        Args:
            user_input (str): The question or prompt to ask the user.

        Returns:
            str: The user input.
        """

    def can_handle_report(self) -> bool:
        """This method is called to check that the plugin can
        handle the report method.

        Returns:
            bool: True if the plugin can handle the report method."""
        return False

    def report(self, message: str) -> None:
        """This method is called to report a message to the user.

        Args:
            message (str): The message to report.
        """
