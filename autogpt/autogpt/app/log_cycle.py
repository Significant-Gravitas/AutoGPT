import json
import os
from pathlib import Path
from typing import Any, Dict, Union

from forge.logging.config import LOG_DIR

DEFAULT_PREFIX = "agent"
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

    def create_outer_directory(self, ai_name: str, created_at: str) -> Path:
        if os.environ.get("OVERWRITE_DEBUG") == "1":
            outer_folder_name = "auto_gpt"
        else:
            ai_name_short = self.get_agent_short_name(ai_name)
            outer_folder_name = f"{created_at}_{ai_name_short}"

        outer_folder_path = LOG_DIR / "DEBUG" / outer_folder_name
        if not outer_folder_path.exists():
            outer_folder_path.mkdir(parents=True)

        return outer_folder_path

    def get_agent_short_name(self, ai_name: str) -> str:
        return ai_name[:15].rstrip() if ai_name else DEFAULT_PREFIX

    def create_inner_directory(self, outer_folder_path: Path, cycle_count: int) -> Path:
        nested_folder_name = str(cycle_count).zfill(3)
        nested_folder_path = outer_folder_path / nested_folder_name
        if not nested_folder_path.exists():
            nested_folder_path.mkdir()

        return nested_folder_path

    def create_nested_directory(
        self, ai_name: str, created_at: str, cycle_count: int
    ) -> Path:
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
        cycle_log_dir = self.create_nested_directory(ai_name, created_at, cycle_count)

        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        log_file_path = cycle_log_dir / f"{self.log_count_within_cycle}_{file_name}"

        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(json_data + "\n")

        self.log_count_within_cycle += 1
