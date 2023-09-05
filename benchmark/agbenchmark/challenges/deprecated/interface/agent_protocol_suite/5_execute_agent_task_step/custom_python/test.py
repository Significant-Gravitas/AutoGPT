# mypy: ignore-errors

import subprocess


def call_agent_protocol() -> None:
    command = "poetry run agent-protocol test --url=http://127.0.0.1:8000 -k test_execute_agent_task_step"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    call_agent_protocol()
