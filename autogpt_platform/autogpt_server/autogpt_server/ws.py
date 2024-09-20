from autogpt_server.app import run_processes
from autogpt_server.server.ws_api import WebsocketServer


def main():
    """
    Run all the processes required for the AutoGPT-server WebSocket API.
    """
    run_processes(WebsocketServer())


if __name__ == "__main__":
    main()
