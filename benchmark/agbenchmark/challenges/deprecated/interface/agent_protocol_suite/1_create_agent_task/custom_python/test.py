import subprocess
import sys


def call_agent_protocol() -> None:
    command = (
        "poetry run agent-protocol test --url=http://127.0.0.1:8000 -k test_create_agent_task"
    )
    try:
        result = subprocess.run(command, shell=True, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    call_agent_protocol()
