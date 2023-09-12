import os
import platform
import queue
import select
import shutil
import subprocess
import sys
import time
from threading import Thread
from typing import Any, List

import psutil
from dotenv import load_dotenv

import agbenchmark.start_benchmark

load_dotenv()

helicone_graphql_logs = os.getenv("HELICONE_GRAPHQL_LOGS")
HELICONE_GRAPHQL_LOGS = (
    helicone_graphql_logs.lower() == "true" if helicone_graphql_logs else False
)


def run_linux_env(process: Any, start_time: float, timeout: float) -> None:
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
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

    else:
        print("The Python function has finished running.")


def enqueue_output(out: Any, my_queue: Any) -> None:
    for line in iter(out.readline, b""):
        my_queue.put(line)
    out.close()


def run_windows_env(process: Any, start_time: float, timeout: float) -> None:
    my_queue: Any = queue.Queue()
    thread = Thread(target=enqueue_output, args=(process.stdout, my_queue))
    thread.daemon = True
    thread.start()

    while True:
        try:
            output = my_queue.get_nowait().strip()
            print(output)
        except queue.Empty:
            pass

        if process.poll() is not None or (time.time() - start_time > timeout):
            break

    if time.time() - start_time > timeout:
        print("The Python function has exceeded the time limit and was terminated.")
        process.terminate()


def run_agent(task: str, timeout: int) -> None:
    """Calling to get a response"""

    entry_path = "agbenchmark.benchmarks"

    print(f"Running '{entry_path}' with timeout {timeout}")

    command = [sys.executable, "-m", entry_path, str(task)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=agbenchmark.start_benchmark.HOME_DIRECTORY,
        bufsize=1,
    )

    start_time = time.time()

    if platform.system() == "Windows":
        run_windows_env(process, start_time, timeout)
    else:
        run_linux_env(process, start_time, timeout)

    process.wait()

    if process.returncode != 0:
        print(f"The agent timed out")


def get_list_of_file_paths(
    challenge_dir_path: str, artifact_folder_name: str
) -> List[str]:
    # this file is at agbenchmark\agent_interface.py
    source_dir = os.path.join(
        agbenchmark.start_benchmark.CURRENT_DIRECTORY,
        "..",
        challenge_dir_path,
        artifact_folder_name,
    )
    if not os.path.exists(source_dir):
        return []
    return [os.path.join(source_dir, file_name) for file_name in os.listdir(source_dir)]


def copy_artifacts_into_workspace(
    workspace: str | dict[str, str], artifact_folder_name: str, challenge_dir_path: str
) -> None:
    if isinstance(workspace, dict):
        if artifact_folder_name == "artifacts_in":
            workspace = workspace["input"]
        else:
            workspace = workspace["output"]
    file_paths = get_list_of_file_paths(challenge_dir_path, artifact_folder_name)
    for file_path in file_paths:
        if os.path.isfile(file_path):
            shutil.copy(file_path, workspace)
