import os
import glob
import pytest
from abc import ABC, abstractmethod
from agbenchmark.challenges.define_task_types import Ground
from agbenchmark.challenges.define_task_types import ChallengeData
from dotenv import load_dotenv

load_dotenv()

mock_test_str = os.getenv("MOCK_TEST")
MOCK_TEST = mock_test_str.lower() == "true" if mock_test_str else False


class Challenge(ABC):
    """The parent class to all specific challenges classes.
    Defines helper methods for running a challenge"""

    @abstractmethod
    def get_file_path(self) -> str:
        """This should be implemented by any class which inherits from BasicChallenge"""
        pass

    @property
    def data(self) -> ChallengeData:
        # TODO: make it so that this is cached somewhere to just call self.deserialized_data
        return ChallengeData.deserialize(self.get_file_path())

    @property
    def mock(self):
        return self.data.mock.mock_func if self.data.mock else None

    @property
    def task(self):
        return (
            self.data.mock.mock_task if self.data.mock and MOCK_TEST else self.data.task
        )

    @property
    def dependencies(self) -> list:
        return self.data.dependencies

    def setup_challenge(self, config):
        from agbenchmark.agent_interface import run_agent

        run_agent(self.task, self.mock, config)

    @property
    def name(self) -> str:
        return self.data.name

    @pytest.mark.parametrize(
        "challenge_data",
        [data],
        indirect=True,
    )
    def test_method(self, config):
        raise NotImplementedError

    @staticmethod
    def open_file(workspace: str, filename: str):
        script_dir = os.path.abspath(workspace)
        workspace_dir = os.path.join(script_dir, filename)
        with open(workspace_dir, "r") as f:
            return f.read()

    @staticmethod
    def open_files(workspace: str, file_patterns: list):
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
    def write_to_file(workspace: str, filename: str, content: str):
        script_dir = os.path.abspath(workspace)
        print("Writing file at", script_dir)
        workspace_dir = os.path.join(script_dir, filename)

        # Open the file in write mode.
        with open(workspace_dir, "w") as f:
            # Write the content to the file.
            f.write(content)

    def get_filenames_in_workspace(self, workspace: str):
        return [
            filename
            for filename in os.listdir(workspace)
            if os.path.isfile(os.path.join(workspace, filename))
        ]

    def scoring(self, content: str, ground: Ground):
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
