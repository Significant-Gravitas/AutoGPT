import os
import sys
import subprocess
import time
from agbenchmark.mocks.MockManager import MockManager
from multiprocessing import Process, Pipe

from agent.hook import run_specific_agent

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
        if config["agent"]["type"] == "python":
            run_agent_function(config, task)
        elif config["agent"]["type"] == "script":
            run_agent_command(config, task)


ENVIRONMENT = os.getenv("ENVIRONMENT") or "production"


def run_agent_command(config, task):
    path = config["agent"]["path"]

    if ENVIRONMENT == "local":
        AGENT_NAME = os.getenv("AGENT_NAME")
        path = os.path.join(os.getcwd(), f"agent\\{AGENT_NAME}")

    timeout = config["agent"]["cutoff"] or sys.maxsize
    print(f"Running {task} with timeout {timeout}")

    command_from_config = config["agent"]["script"]
    command_list = command_from_config.split()

    # replace '{}' with the task
    command_list = [cmd if cmd != "{}" else task for cmd in command_list]
    print("path, command_list", path, command_list)
    start_time = time.time()
    proc = subprocess.Popen(
        command_list,
        cwd=path,
        shell=True,
    )

    while True:
        if time.time() - start_time > timeout:
            print("The subprocess has exceeded the time limit and was terminated.")
            proc.terminate()
            break

        if proc.poll() is not None:
            print("The subprocess has finished running.")
            break


def run_agent_function(config, task):
    timeout = (
        config["cutoff"]["count"] if config["cutoff"]["type"] == "time" else sys.maxsize
    )
    print(
        f"Running Python function '{config['agent']['function']}' with timeout {timeout}"
    )

    parent_conn, child_conn = Pipe()
    process = Process(target=run_specific_agent, args=(task, child_conn))
    process.start()
    start_time = time.time()

    while True:
        if parent_conn.poll():  # Check if there's a new message from the child process
            response, cycle_count = parent_conn.recv()
            print(f"Cycle {cycle_count}: {response}")

            if cycle_count >= config["cutoff"]["count"]:
                print(
                    f"Cycle count has reached the limit of {config['cutoff']['count']}. Terminating."
                )
                child_conn.send("terminate")
                break

        if time.time() - start_time > timeout:
            print("The Python function has exceeded the time limit and was terminated.")
            child_conn.send(
                "terminate"
            )  # Send a termination signal to the child process
            break

        if not process.is_alive():
            print("The Python function has finished running.")
            break

    process.join()
