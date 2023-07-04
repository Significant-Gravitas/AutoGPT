import os
import sys
from typing import Tuple
import pexpect


def run_specific_agent(task: str) -> Tuple[str, int]:
    # Ensure the directory for the project exists
    os.makedirs("workspace_path", exist_ok=True)

    # Run the agent command
    child = pexpect.spawn(f"python example.py {task}")

    # Create a loop to continuously read output
    while True:
        try:
            child.expect("\n")  # This waits until a newline appears
            print(child.before.decode())  # This prints the line
        except pexpect.EOF:
            break  # No more output, break the loop

    # Check the exit status
    child.close()  # Close the child process

    # Return child process's exit status and any error messages
    return child.before.decode(), child.exitstatus


if __name__ == "__main__":
    # The first argument is the script name itself, second is the task
    if len(sys.argv) != 2:
        print("Usage: python script.py <task>")
        sys.exit(1)
    task = sys.argv[1]
    run_specific_agent(task)
