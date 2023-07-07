import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from agbenchmark.mocks.mock_manager import MockManager

load_dotenv()

MOCK_FLAG = os.getenv("MOCK_TEST")


def run_agent(
    task: Optional[str], mock_func: Optional[str], config: Dict[str, Any]
) -> None:
    """Calling to get a response"""

    if mock_func == None and MOCK_FLAG == "True":
        print("No mock provided")
    elif MOCK_FLAG == "True":
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
            output = process.stdout.readline()
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


ENVIRONMENT = os.getenv("ENVIRONMENT") or "production"
