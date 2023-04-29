"""
This module summarizes what the agent has done so far and stores it in a summarized memory.
This is different and separate from our vectorized memory implementation found in the memory folder.
"""

from typing import Dict, List, Tuple

from autogpt.config import Config
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.singleton import Singleton


class SummarizedMemory(metaclass=Singleton):
    """
    This is not a memory module you configure!
    This is a history of what the agent has done so far!
    It is summarized and stored for all agents
    """
    prev_index = 0
    current_memory = ""

    @classmethod
    def get_newly_trimmed_messages(
            cls,
            full_message_history: List[Dict[str, str]],
            current_context: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        This function returns a list of dictionaries contained in full_message_history
        with an index higher than prev_index that are absent from current_context.

        Args:
            full_message_history (list): A list of dictionaries representing the full message history.
            current_context (list): A list of dictionaries representing the current context.
            prev_index (int): An integer representing the previous index.

        Returns:
            list: A list of dictionaries that are in full_message_history with an index higher than prev_index and absent from current_context.
        """

        # Select messages in full_message_history with an index higher than prev_index
        new_messages = [msg for i, msg in enumerate(full_message_history) if i > cls.prev_index]

        # Remove messages that are already present in current_context
        new_messages_not_in_context = [
            msg for msg in new_messages if msg not in current_context
        ]

        # Find the index of the last message processed
        new_index = cls.prev_index
        if new_messages_not_in_context:
            last_message = new_messages_not_in_context[-1]
            new_index = full_message_history.index(last_message)

        cls.prev_index = new_index

        return new_messages_not_in_context

    @classmethod
    def update_running_summary(cls, new_events: List[Dict], cfg: Config) -> str:
        """
        This function takes a list of dictionaries representing new events and combines them with the current summary,
        focusing on key and potentially important information to remember. The updated summary is returned in a message
        formatted in the 1st person past tense.

        Args:
            new_events (List[Dict]): A list of dictionaries containing the latest events to be added to the summary.
            cfg (Config): A Config object containing the configuration parameters.

        Returns:
            str: A message containing the updated summary of actions, formatted in the 1st person past tense.

        Example:
            new_events = [{"event": "entered the kitchen."}, {"event": "found a scrawled note with the number 7"}]
            update_running_summary(new_events)
            # Returns: "This reminds you of these events from your past: \nI entered the kitchen and found a scrawled note saying 7."
        """

        prompt = f'''Your task is to create a concise running summary of actions in the provided text, focusing on key and potentially important information to remember.

    You will receive the current summary and the latest development. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.

    Summary So Far:
    """
    {cls.current_memory}
    """

    Latest Development:
    """
    {new_events}
    """
    '''

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        current_memory = create_chat_completion(messages, cfg.fast_llm_model)

        message_to_return = {
            "role": "system",
            "content": f"This reminds you of these events from your past: \n{current_memory}",
        }

        return message_to_return
