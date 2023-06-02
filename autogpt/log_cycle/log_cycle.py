import json
import os
from typing import Any, Dict, Union

from autogpt.logs import logger

DEFAULT_PREFIX = "agent"
FULL_MESSAGE_HISTORY_FILE_NAME = "full_message_history.json"
CURRENT_CONTEXT_FILE_NAME = "current_context.json"
NEXT_ACTION_FILE_NAME = "next_action.json"
PROMPT_SUMMARY_FILE_NAME = "prompt_summary.json"
SUMMARY_FILE_NAME = "summary.txt"
SUPERVISOR_FEEDBACK_FILE_NAME = "supervisor_feedback.txt"
PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME = "prompt_supervisor_feedback.json"
USER_INPUT_FILE_NAME = "user_input.txt"


class LogCycleHandler:
    """
    A class for logging cycle data.
    """

    def __init__(self):
        self.log_count_within_cycle = 0

    @staticmethod
    def create_directory_if_not_exists(directory_path: str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

    def create_outer_directory(self, ai_name: str, created_at: str) -> str:
        log_directory = logger.get_log_directory()

        if os.environ.get("OVERWRITE_DEBUG") == "1":
            outer_folder_name = "auto_gpt"
        else:
            ai_name_short = ai_name[:15] if ai_name else DEFAULT_PREFIX
            outer_folder_name = f"{created_at}_{ai_name_short}"

        outer_folder_path = os.path.join(log_directory, "DEBUG", outer_folder_name)
        self.create_directory_if_not_exists(outer_folder_path)

        return outer_folder_path

    def create_inner_directory(self, outer_folder_path: str, cycle_count: int) -> str:
        nested_folder_name = str(cycle_count).zfill(3)
        nested_folder_path = os.path.join(outer_folder_path, nested_folder_name)
        self.create_directory_if_not_exists(nested_folder_path)

        return nested_folder_path

    def create_nested_directory(
        self, ai_name: str, created_at: str, cycle_count: int
    ) -> str:
        outer_folder_path = self.create_outer_directory(ai_name, created_at)
        nested_folder_path = self.create_inner_directory(outer_folder_path, cycle_count)

        return nested_folder_path

    def log_cycle(
        self,
        ai_name: str,
        created_at: str,
        cycle_count: int,
        data: Union[Dict[str, Any], Any],
        file_name: str,
    ) -> None:
        """
        Log cycle data to a JSON file.

        Args:
            data (Any): The data to be logged.
            file_name (str): The name of the file to save the logged data.
        """
        nested_folder_path = self.create_nested_directory(
            ai_name, created_at, cycle_count
        )

        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        log_file_path = os.path.join(
            nested_folder_path, f"{self.log_count_within_cycle}_{file_name}"
        )

        logger.log_json(json_data, log_file_path)
        self.log_count_within_cycle += 1
