import time
import subprocess
from multiprocessing import Process
from autogpt_server.data.execution import ExecutionQueue, Execution

# Reminder of types of todo notes I can add to the file...
# PERF: Performance
# HACK: Hack
# TODO: Todo
# NOTE: Note
# FIX: Fix me
# WARNING: Warning


def start_realtime(api_queue: ExecutionQueue, execution_queue: ExecutionQueue) -> bool:
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
    realtime_process = Process(
        target=start_realtime_connector, args=(api_queue, execution_queue)
    )
    realtime_process.start()
    return True


async def start_realtime_connector(
    api_queue: ExecutionQueue, execution_queue: ExecutionQueue
) -> None:
    """
    This needs to startup the connector.

    The connector needs to manage the livekit-server in response to events
    that come from the agent server-api, handle livekit data streams and
    send events to the executor manager.
    """

    # FIX: Comms channels currently they do not appear to be working
    # TODO: Define API Event types that the realtime system can consume.
    # TODO: Define the tirggering event trypes that can be sent to the executor
    api_queue.add(Execution(run_id="1", node_id="connector Started", data={}))
    while True:
        amsg = api_queue.get()
        if amsg:
            print(amsg)
            api_queue.add(Execution(run_id="1", node_id="hi", data={}))
        emsg = execution_queue.get()
        if emsg:
            print(emsg)
        time.sleep(0.1)


# TODO: Add function to setup a new room
# TODO: Add function to join agents room
# TODO: Add function to leave agents room
# TODO: Add function to get room details
# TODO: Add function to to delete room

# TODO: Add function to listen to room
# TODO: Add function to process audio
# TODO: Add function to process video
# TODO: Add function to process data channel, needs to be able to handle chat messages and files.

# TODO: Add function to send audio
# TODO: Add function to send video, possibly needs to be combined with audio?
# TODO: Add function to send data - messages and file
