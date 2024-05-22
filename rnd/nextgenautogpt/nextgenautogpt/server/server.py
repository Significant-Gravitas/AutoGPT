import multiprocessing as mp
import time
from typing import Any


def run_server(server_to_manager: mp.Queue, manager_to_server: mp.Queue) -> None:
    print("Server process started")
    while True:
        message = "Message from server"
        server_to_manager.put(message)
        # Simulate server work
        time.sleep(1)
        if not manager_to_server.empty():
            message = manager_to_server.get()
            print(f"Server received: {message}")
            # Simulate server work
            time.sleep(1)
