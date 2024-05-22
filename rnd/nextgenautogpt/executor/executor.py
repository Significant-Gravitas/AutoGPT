import multiprocessing as mp
import time
from typing import Any


def run_executor(manager_to_executor: mp.Queue) -> None:
    # Each executor process will run this initializer
    print("Executor process started")
    while True:
        if not manager_to_executor.empty():
            task = manager_to_executor.get()
            print(f"Executor processing: {task}")
            # Simulate executor work
            time.sleep(1)
