import os
import select
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

from dotenv import load_dotenv

from agbenchmark.start_benchmark import CURRENT_DIRECTORY, HOME_DIRECTORY

load_dotenv()

mock_test_str = os.getenv("MOCK_TEST")
MOCK_FLAG = mock_test_str.lower() == "true" if mock_test_str else False


def run_agent(
    task: str, config: Dict[str, Any], artifacts_location: str, cutoff: int
) -> None:
    """Calling to get a response"""
    if task == "":
        return
    if MOCK_FLAG:
        print("Running mock agent")
        copy_artifacts_into_workspace(
            config["workspace"], "artifacts_out", artifacts_location
        )
        return
    entry_path = "agbenchmark.benchmarks"

    timeout = cutoff
    if "--nc" in sys.argv:
        timeout = 100000

    print(f"Running '{entry_path}' with timeout {timeout}")

    command = [sys.executable, "-m", entry_path, str(task)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=HOME_DIRECTORY,
        bufsize=1,
    )

    start_time = time.time()

    while True:
        try:
            # This checks if there's data to be read from stdout without blocking.
            if process.stdout and select.select([process.stdout], [], [], 0)[0]:
                output = process.stdout.readline()
                print(output.strip())
        except Exception as e:
            continue

        # Check if process has ended, has no more output, or exceeded timeout
        if process.poll() is not None or (time.time() - start_time > timeout):
            break

    if time.time() - start_time > timeout:
        print("The Python function has exceeded the time limit and was terminated.")
        process.kill()
    else:
        print("The Python function has finished running.")

    process.wait()

    if process.returncode != 0:
        print(f"The agent timed out")


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
