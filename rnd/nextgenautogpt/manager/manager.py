import multiprocessing as mp
import time

import nextgenautogpt.executor.executor as mod_executor


def run_manager(server_to_manager: mp.Queue, manager_to_server: mp.Queue) -> None:
    # Create queue for communication between manager and executors
    print("Manager process started")
    manager_to_executor = mp.Queue()
    executors_to_manager = mp.Queue()
    # Create and start a pool of executor processes
    with mp.Pool(
        processes=5,
        initializer=mod_executor.run_executor,
        initargs=(
            manager_to_executor,
            executors_to_manager,
        ),
    ):
        while True:
            if not server_to_manager.empty():
                message = server_to_manager.get()
                print(f"Manager received: {message}")
                manager_to_server.put("Manager: Received message from server")
                manager_to_executor.put("Task for executor")
                # Simulate manager work
                time.sleep(1)
            if not executors_to_manager.empty():
                message = executors_to_manager.get()
                print(f"Manager received: {message}")
                # Simulate manager work
                time.sleep(1)
