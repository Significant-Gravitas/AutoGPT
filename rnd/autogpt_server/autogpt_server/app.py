from multiprocessing import freeze_support, set_start_method, get_start_method
from autogpt_server.server import start_server
from autogpt_server.executor import start_executor_manager
from autogpt_server.data import ExecutionQueue


def main() -> None:
    queue = ExecutionQueue()
    start_executor_manager(5, queue)
    start_server(queue)


if __name__ == "__main__":
    freeze_support()
    print(get_start_method())
    set_start_method("fork", force=True)
    print(get_start_method())
    main()
