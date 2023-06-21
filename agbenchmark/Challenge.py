import os
from typing import Optional


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
