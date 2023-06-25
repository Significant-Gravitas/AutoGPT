import os
import glob
from agbenchmark.challenges.define_task_types import Ground


class Challenge:
    """The parent class to all specific challenges classes.
    Defines helper methods for running a challenge"""

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
