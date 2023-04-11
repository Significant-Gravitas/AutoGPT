#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from pinecone.core.api_action import ActionAPI, VersionResponse, WhoAmIResponse
from pinecone.config import Config

import time
import requests

__all__ = ["version", "whoami", "VersionResponse", "WhoAmIResponse"]


def _get_action_api():
    return ActionAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)


def version() -> VersionResponse:
    """Returns version information (client and server)."""
    api = _get_action_api()
    return api.version()


def whoami() -> WhoAmIResponse:
    """Returns the details of the currently authenticated API key."""
    api = _get_action_api()
    return api.whoami()


def wait_controller_ready(timeout: int = 30):
    connection = False
    max_time = time.time() + timeout
    while (not connection) and time.time() < max_time:
        try:
            version()
            time.sleep(3)
            connection = True
        except requests.exceptions.ConnectionError:
            time.sleep(1)