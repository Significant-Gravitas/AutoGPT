import importlib

from agbenchmark.mocks.MockManager import MockManager
import os
import sys
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

MOCK_FLAG = os.getenv("MOCK_TEST")


def run_agent(task, mock_func, config):
    """Calling to get a response"""

    if mock_func == None and MOCK_FLAG == "True":
        print("No mock provided")
    elif MOCK_FLAG == "True":
        mock_manager = MockManager(
            task
        )  # workspace doesn't need to be passed in, stays the same
        print("Server unavailable, using mock", mock_func)
        mock_manager.delegate(mock_func)
    else:
        timeout = config["cutoff"]
        print(f"Running Python function '{config['func_path']}' with timeout {timeout}")

        # Get the current working directory
        cwd = os.getcwd()

        # Add current directory to Python's import path
        sys.path.append(cwd)


        module_name = config["func_path"].replace("/", ".").rstrip(".py")
        module = importlib.import_module(module_name)


        command = [sys.executable, "benchmarks.py", str(task)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=cwd)

        start_time = time.time()
        timeout = config["cutoff"]

        while True:
            output = process.stdout.readline()
            print(output.strip())

            # Check if process has ended
            if process.poll() is not None:
                print("The Python function has finished running.")
                break

            # Check if process has exceeded timeout
            if time.time() - start_time > timeout:
                print("The Python function has exceeded the time limit and was terminated.")
                process.terminate()
                break

            # Optional: sleep for a while
            time.sleep(0.1)

        # Wait for process to terminate, then get return code
        process.wait()



ENVIRONMENT = os.getenv("ENVIRONMENT") or "production"
