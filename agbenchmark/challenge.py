import glob
import inspect
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pytest
from dotenv import load_dotenv

from agbenchmark.challenges.define_task_types import ChallengeData, Ground

load_dotenv()

mock_test_str = os.getenv("MOCK_TEST")
MOCK_TEST = mock_test_str.lower() == "true" if mock_test_str else False


class Challenge(ABC):
    """The parent class to all specific challenges classes.
    Defines helper methods for running a challenge"""

    _data_cache: Dict[str, ChallengeData] = {}

    @abstractmethod
    def get_file_path(self) -> str:
        """This should be implemented by any class which inherits from BasicChallenge"""
        pass

    @property
    def data(self) -> ChallengeData:
        "Check if the data is already loaded, if not load it"
        file_path = (
            self.get_file_path()
        )  # file_path serves as the key in the cache dictionary
        if file_path not in Challenge._data_cache:
            Challenge._data_cache[file_path] = ChallengeData.deserialize(file_path)
        return Challenge._data_cache[file_path]

    @property
    def mock(self) -> Optional[str]:
        return self.data.mock.mock_func if self.data.mock else None

    @property
    def task(self) -> Optional[str]:
        return (
            self.data.mock.mock_task if self.data.mock and MOCK_TEST else self.data.task
        )

    @property
    def dependencies(self) -> list:
        return self.data.dependencies

    def setup_challenge(self, config: Dict[str, Any]) -> None:
        from agbenchmark.agent_interface import run_agent

        self.copy_artifacts_into_workspace(config["workspace"])

        run_agent(self.task, self.mock, config)

    @property
    def name(self) -> str:
        return self.data.name

    @pytest.mark.parametrize(
        "challenge_data",
        [data],
        indirect=True,
    )
    def test_method(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

    @staticmethod
    def open_file(workspace: str, filename: str) -> str:
        script_dir = os.path.abspath(workspace)
        workspace_dir = os.path.join(script_dir, filename)
        with open(workspace_dir, "r") as f:
            return f.read()

    @staticmethod
    def open_files(workspace: str, file_patterns: list) -> List[str]:
        script_dir = os.path.abspath(workspace)
        files_contents = []

        for file_pattern in file_patterns:
            # Check if it is a file extension
            if file_pattern.startswith("."):
                # Find all files with the given extension in the workspace
                matching_files = glob.glob(os.path.join(script_dir, "*" + file_pattern))
            else:
                # Otherwise, it is a specific file
                matching_files = [os.path.join(script_dir, file_pattern)]

            for file_path in matching_files:
                with open(file_path, "r") as f:
                    files_contents.append(f.read())

        return files_contents

    @staticmethod
    def write_to_file(workspace: str, filename: str, content: str) -> None:
        script_dir = os.path.abspath(workspace)
        print("Writing file at", script_dir)
        workspace_dir = os.path.join(script_dir, filename)

        # Open the file in write mode.
        with open(workspace_dir, "w") as f:
            # Write the content to the file.
            f.write(content)

    def get_filenames_in_workspace(self, workspace: str) -> List[str]:
        return [
            filename
            for filename in os.listdir(workspace)
            if os.path.isfile(os.path.join(workspace, filename))
        ]

    def scoring(self, content: str, ground: Ground) -> float:
        if ground.should_contain:
            for should_contain_word in ground.should_contain:
                if should_contain_word not in content:
                    return 0.0
                else:
                    print(
                        f"Word that should exist: {should_contain_word} exists in the content"
                    )

        if ground.should_not_contain:
            for should_not_contain_word in ground.should_not_contain:
                if should_not_contain_word in content:
                    return 0.0
                else:
                    print(
                        f"Word that should not exist: {should_not_contain_word} does not exist in the content"
                    )

        return 1.0

    def copy_artifacts_into_workspace(self, workspace: str) -> None:
        curr_frame = inspect.currentframe()
        outer_frame = inspect.getouterframes(curr_frame)[2]
        caller_file_path = outer_frame.filename
        caller_dir_path = os.path.dirname(os.path.abspath(caller_file_path))
        source_dir = os.path.join(caller_dir_path, "artifacts")

        # Check if source_dir exists, if not then return immediately.
        if not os.path.exists(source_dir):
            return

        for file_name in os.listdir(source_dir):
            full_file_name = os.path.join(source_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, workspace)
