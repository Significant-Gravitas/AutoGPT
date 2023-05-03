"""Twitter API integrations using Tweepy."""
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar
from auto_gpt_plugin_template import AutoGPTPluginTemplate
import os
import tweepy

PromptGenerator = TypeVar("PromptGenerator")


class Message(TypedDict):
    role: str
    content: str


class AutoGPTTwitter(AutoGPTPluginTemplate):
    """
    Twitter API integrations using Tweepy
    """

    def __init__(self):
        super().__init__()
        self._name = "autogpt-twitter"
        self._version = "0.1.0"
        self._description = "Twitter API integrations using Tweepy."
        self.twitter_consumer_key = os.getenv("TW_CONSUMER_KEY")
        self.twitter_consumer_secret = os.getenv("TW_CONSUMER_SECRET")
        self.twitter_access_token = os.getenv("TW_ACCESS_TOKEN")
        self.twitter_access_token_secret = os.getenv("TW_ACCESS_TOKEN_SECRET")
        self.tweet_id = []
        self.tweets = []

        self.api = None

        if (
            self.twitter_consumer_key
            and self.twitter_consumer_secret
            and self.twitter_access_token
            and self.twitter_access_token_secret
        ) is not None:
            # Authenticating to twitter
            self.auth = tweepy.OAuth1UserHandler(
                self.twitter_consumer_key,
                self.twitter_consumer_secret,
                self.twitter_access_token,
                self.twitter_access_token_secret,
            )
            self.api = tweepy.API(self.auth)
            self.stream = tweepy.Stream(
                self.twitter_consumer_key,
                self.twitter_consumer_secret,
                self.twitter_access_token,
                self.twitter_access_token_secret,
            )
        else:
            print("Twitter credentials not found in .env file.")

    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.
        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model."""
        pass

    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.
        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return True

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        return None

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.
        Args:
            prompt (PromptGenerator): The prompt generator.
        Returns:
            PromptGenerator: The prompt generator.
        """
        if self.api:
            from .twitter import (
                get_mentions,
                post_reply,
                post_tweet,
                search_twitter_user,
            )

            prompt.add_command(
                "post_tweet", "Post Tweet", {"tweet_text": "<tweet_text>"}, post_tweet
            )
            prompt.add_command(
                "post_reply",
                "Post Twitter Reply",
                {"tweet_text": "<tweet_text>", "tweet_id": "<tweet_id>"},
                post_reply,
            )
            prompt.add_command("get_mentions", "Get Twitter Mentions", {}, get_mentions)
            prompt.add_command(
                "search_twitter_user",
                "Search Twitter",
                {
                    "target_user": "<target_user>",
                    "number_of_tweets": "<number_of_tweets",
                },
                search_twitter_user,
            )

        return prompt
