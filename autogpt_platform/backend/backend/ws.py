from backend.app import run_processes
from backend.server.ws_api import WebsocketServer


def main():
    """
    Run all the processes required for the AutoGPT-server WebSocket API.
    """
    run_processes(WebsocketServer())


if __name__ == "__main__":
    main()
