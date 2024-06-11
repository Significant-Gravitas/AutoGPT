from multiprocessing import Process
import subprocess


def start_realtime(pool_size: int, queue) -> None:
    """
    Starts the livekit server and the realtime connector
    """

    # Start the livekit server
    lksp = subprocess.Popen(
        ["livekit-server", "--dev"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    # Start the connector
    realtime_process = Process(target=start_realtime_connector, args=(pool_size, queue))
    realtime_process.start()


def start_realtime_connector(pool_size, queue):
    """
    This needs to startup the connector.

    The connector needs to manage the livekit-server in response to events
    that come from the agent server-api, handle livekit data streams and
    send events to the executor manager.
    """
    pass
