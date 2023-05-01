import json
import os
from typing import Any

from autogpt.logs import logger


class LogCycleMixin:
    MESSAGES_PER_CYCLE = 3
    DEFAULT_PREFIX = "Agent"
    FULL_MESSAGE_HISTORY_FILE_NAME = "full_message_history.txt"
    PROMPT_SUMMARY_FILE_NAME = "prompt_summary.txt"
    SUMMARY_FILE_NAME = "summary.txt"
    PROMPT_NEXT_ACTION_FILE_NAME = "prompt_next_action.txt"
    NEXT_ACTION_FILE_NAME = "next_action.txt"

    @staticmethod
    def create_directory_if_not_exists(directory_path: str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

    def create_outer_directory(self) -> str:
        log_directory = (
            logger.get_log_directory()
        )  # Get the log directory from the Logger class

        if os.environ.get("OVERWRITE_DEBUG") == "1":
            outer_folder_name = "auto_gpt"
        else:
            ai_name_short = (
                self.config.ai_name[:15] if self.config.ai_name else self.DEFAULT_PREFIX
            )
            outer_folder_name = f"{self.created_at}_{ai_name_short}"

        # Update the outer folder path to be inside the log directory
        outer_folder_path = os.path.join(log_directory, "DEBUG", outer_folder_name)
        self.create_directory_if_not_exists(outer_folder_path)

        return outer_folder_path

    def create_inner_directory(self, outer_folder_path: str) -> str:
        nested_folder_name = str(self.cycle_count).zfill(3)
        nested_folder_path = os.path.join(outer_folder_path, nested_folder_name)
        self.create_directory_if_not_exists(nested_folder_path)

        return nested_folder_path

    def create_nested_directory(self) -> str:
        outer_folder_path = self.create_outer_directory()
        nested_folder_path = self.create_inner_directory(outer_folder_path)

        return nested_folder_path

    def log_cycle(self, data: Any, file_name: str) -> None:
        nested_folder_path = self.create_nested_directory()

        # Use the imported logger to log the JSON data
        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        log_file_path = os.path.join(
            nested_folder_path, f"{self.log_count_within_cycle}_{file_name}"
        )

        logger.log_json(json_data, log_file_path)
        self.log_count_within_cycle += 1
