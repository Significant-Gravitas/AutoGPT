from autogpt_server.agent_api import start_server
from autogpt_server.agent_executor import start_executor
from autogpt_server.data import ExecutionQueue

def main() -> None:
    queue = ExecutionQueue()
    start_executor(5, queue)
    start_server(queue)

if __name__ == "__main__":
    main()
