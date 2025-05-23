from backend.app import run_processes
from backend.server.ws_api import WebsocketServer
from backend.util.logging import configure_logging


def main():
    """
    Run all the processes required for the AutoGPT-server WebSocket API.
    """
    configure_logging()
    run_processes(WebsocketServer())


if __name__ == "__main__":
    main()
