from autogpt_server.server import start_server
from autogpt_server.executor import start_executor_manager
from autogpt_server.data import ExecutionQueue


def main() -> None:
    queue = ExecutionQueue()
    start_executor_manager(5, queue)
    start_server(queue)


if __name__ == "__main__":
    main()
