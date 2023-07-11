import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

from dotenv import load_dotenv

from agbenchmark.start_benchmark import CURRENT_DIRECTORY

load_dotenv()

mock_test_str = os.getenv("MOCK_TEST")
MOCK_FLAG = mock_test_str.lower() == "true" if mock_test_str else False


def run_agent(
    task: str,
    config: Dict[str, Any],
    challenge_location: str,
) -> None:
    """Calling to get a response"""

    if MOCK_FLAG:
        print("ITS A MOCK TEST", challenge_location)
        copy_artifacts_into_workspace(
            config["workspace"], "artifacts_out", challenge_location
        )
    else:
        timeout = config["cutoff"]
        print(
            f"Running Python function '{config['entry_path']}' with timeout {timeout}"
        )

        command = [sys.executable, config["entry_path"], str(task)]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=os.getcwd(),
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
    # this file is at agbenchmark\agent_interface.py
    source_dir = os.path.join(
        CURRENT_DIRECTORY, "..", challenge_dir_path, artifact_folder_name
    )

    # Check if source_dir exists, if not then return immediately.
    if not os.path.exists(source_dir):
        return

    for file_name in os.listdir(source_dir):
        full_file_name = os.path.join(source_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, workspace)
