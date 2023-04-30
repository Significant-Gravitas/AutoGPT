import json
import threading
import time
import urllib.request
from typing import Dict

from autogpt.config import Config

# Retrieve configuration
CFG = Config()

# Build URL for target
url = f"{CFG.event_dispatcher_protocol}://{CFG.event_dispatcher_host}:{CFG.event_dispatcher_port}{CFG.event_dispatcher_endpoint}"


# Fires and forgets the data to the configured endpoint
def fire_and_forget(url: str, data: Dict[str, any], headers: Dict[str, any]):
    """
    Send a JSON payload to a RESTful endpoint asynchronously.

    Args:
    - url (str): the endpoint URL where the data is sent to
    - data (dict): a dictionary containing the payload data to be sent as JSON
    - headers (dict): a dictionary containing headers

    Returns:
    - None
    """

    # ensure later import (dependency injection)
    from autogpt.logs import logger

    json_payload = json.dumps(data).encode("utf-8")

    # Callback definition for thread
    def send_request():
        try:
            # send the request
            req = urllib.request.Request(
                url, data=json_payload, headers=headers, method="POST"
            )
            if CFG.debug_mode:
                logger.debug(f"Sending: {req}", "[Event Dispatcher]")
            with urllib.request.urlopen(req) as response:
                pass  # do nothing with the response
        except Exception as e:  # ensure catching all exceptions
            if CFG.debug_mode:
                logger.error("[Event Dispatcher]", f"Error: {e}")

    # Start a new thread to send the request
    t = threading.Thread(target=send_request)
    t.start()


# Helper method for passing data and config
def fire(data: Dict[str, any]):
    headers = {
        "Content-type": "application/json",
        "Event-time": str(int(time.time_ns())),
        "Event-origin": "AutoGPT",
    }
    fire_and_forget(url, data, headers)
