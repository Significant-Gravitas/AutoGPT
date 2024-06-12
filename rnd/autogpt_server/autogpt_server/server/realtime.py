import time
import subprocess
from multiprocessing import Process
from autogpt_server.data.execution import ExecutionQueue, Execution


def start_realtime(api_queue: ExecutionQueue, execution_queue: ExecutionQueue) -> None:
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
    realtime_process = Process(target=start_realtime_connector, args=(api_queue, execution_queue))
    realtime_process.start()


async def start_realtime_connector(api_queue: ExecutionQueue, execution_queue: ExecutionQueue) -> None: 
    """
    This needs to startup the connector.

    The connector needs to manage the livekit-server in response to events
    that come from the agent server-api, handle livekit data streams and
    send events to the executor manager.
    """
    while True:
        amsg = api_queue.get()
        if amsg:
            print(amsg)
            api_queue.add(Execution(run_id="1", node_id="hi", data={}))
        emsg = execution_queue.get()
        if emsg:
            print(emsg)
        time.sleep(1)
