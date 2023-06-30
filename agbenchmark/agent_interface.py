import os
import importlib
import time
from agbenchmark.mocks.MockManager import MockManager
from multiprocessing import Process, Pipe

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

        parent_conn, child_conn = Pipe()

        # Import the specific agent dynamically
        module_name = config["func_path"].replace("/", ".").rstrip(".py")
        module = importlib.import_module(module_name)
        run_specific_agent = getattr(module, "run_specific_agent")

        process = Process(target=run_specific_agent, args=(task, child_conn))
        process.start()
        start_time = time.time()

        while True:
            if (
                parent_conn.poll()
            ):  # Check if there's a new message from the child process
                response, cycle_count = parent_conn.recv()
                print(f"Cycle {cycle_count}: {response}")

                if cycle_count >= config["cutoff"]:
                    print(
                        f"Cycle count has reached the limit of {config['cutoff']}. Terminating."
                    )
                    child_conn.send("terminate")
                    break

            if time.time() - start_time > timeout:
                print(
                    "The Python function has exceeded the time limit and was terminated."
                )
                child_conn.send(
                    "terminate"
                )  # Send a termination signal to the child process
                break

            if not process.is_alive():
                print("The Python function has finished running.")
                break

        process.join()


ENVIRONMENT = os.getenv("ENVIRONMENT") or "production"
