from multiprocessing import Process

from autogpt_server.agent_api import start_server
from autogpt_server.agent_executor import start_executor
from autogpt_server.data import ExecutionQueue

if __name__ == "__main__":
    queue = ExecutionQueue()
    executor_process = Process(target=start_executor, args=(5,queue))
    executor_process.start()
    start_server(queue)
