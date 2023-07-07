import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from agbenchmark.mocks.mock_manager import MockManager

load_dotenv()

MOCK_FLAG = os.getenv("MOCK_TEST")


def run_agent(
    task: Optional[str],
    mock_func: Optional[str],
    config: Dict[str, Any],
    challenge_location: str,
) -> None:
    """Calling to get a response"""

    if MOCK_FLAG == "True":
        copy_artifacts_into_workspace(
            config["workspace"], "artifacts_out", challenge_location
        )
        if mock_func is None:
            print("No mock provided")
            return
        mock_manager = MockManager(
            task, config
        )  # workspace doesn't need to be passed in, stays the same
        print("Server unavailable, using mock", mock_func)
        mock_manager.delegate(mock_func)
    else:
        timeout = config["cutoff"]
        print(
            f"Running Python function '{config['entry_path']}' with timeout {timeout}"
        )

        # Get the current working directory
        cwd = os.path.join(os.getcwd(), config["home_path"])

        # Add current directory to Python's import path
        sys.path.append(cwd)

        command = [sys.executable, config["entry_path"], str(task)]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=cwd,
        )

        start_time = time.time()
        timeout = config["cutoff"]

        while True:
            if process.stdout is None:
                continue

            while output := process.stdout.readline():
                print(output.strip())

            # Check if process has ended
            if process.poll() is not None:
                print("The Python function has finished running.")
                break

            # Check if process has exceeded timeout
            if time.time() - start_time > timeout:
                print(
                    "The Python function has exceeded the time limit and was terminated."
                )
                # Terminate the process group
                process.terminate()
                break

            # Optional: sleep for a while
            time.sleep(0.1)

        # Wait for process to terminate, then get return code
        process.wait()


def copy_artifacts_into_workspace(
    workspace: str, artifact_folder_name: str, challenge_dir_path: str
) -> None:
    source_dir = os.path.join(challenge_dir_path, artifact_folder_name)

    # Check if source_dir exists, if not then return immediately.
    if not os.path.exists(source_dir):
        return

    for file_name in os.listdir(source_dir):
        full_file_name = os.path.join(source_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, workspace)


ENVIRONMENT = os.getenv("ENVIRONMENT") or "production"
