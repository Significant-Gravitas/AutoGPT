from autogpt_server.agent_api import start_server
from autogpt_server.agent_executor import start_executors
from autogpt_server.data import ExecutionQueue


def main() -> None:
    queue = ExecutionQueue()
    start_executors(5, queue)
    start_server(queue)


if __name__ == "__main__":
    main()
