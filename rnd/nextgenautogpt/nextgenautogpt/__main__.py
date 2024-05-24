import multiprocessing as mp
from typing import Any

import nextgenautogpt.manager.manager as mod_manager
import nextgenautogpt.server.server as mod_server


def main() -> None:
    # Create queues/pipes for communication
    server_to_manager: mp.Queue[Any] = mp.Queue()
    manager_to_server: mp.Queue[Any] = mp.Queue()

    # Create and start server process
    server: mp.Process = mp.Process(
        target=mod_server.run_server,
        args=(
            server_to_manager,
            manager_to_server,
        ),
    )
    server.start()

    # Create and start manager process
    manager: mp.Process = mp.Process(
        target=mod_manager.run_manager,
        args=(
            server_to_manager,
            manager_to_server,
        ),
    )
    manager.start()

    server.join()
    manager.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
