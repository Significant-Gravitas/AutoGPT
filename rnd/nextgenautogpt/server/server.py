import multiprocessing as mp
import time
from typing import Any


def run_server(server_to_manager: mp.Queue) -> None:
    print("Server process started")
    while True:
        message = "Message from server"
        server_to_manager.put(message)
        # Simulate server work
        time.sleep(1)
