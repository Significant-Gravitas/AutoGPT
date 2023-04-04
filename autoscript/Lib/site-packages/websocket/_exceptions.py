"""
_exceptions.py
websocket - WebSocket client library for Python

Copyright 2022 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class WebSocketException(Exception):
    """
    WebSocket exception class.
    """
    pass


class WebSocketProtocolException(WebSocketException):
    """
    If the WebSocket protocol is invalid, this exception will be raised.
    """
    pass


class WebSocketPayloadException(WebSocketException):
    """
    If the WebSocket payload is invalid, this exception will be raised.
    """
    pass


class WebSocketConnectionClosedException(WebSocketException):
    """
    If remote host closed the connection or some network error happened,
    this exception will be raised.
    """
    pass


class WebSocketTimeoutException(WebSocketException):
    """
    WebSocketTimeoutException will be raised at socket timeout during read/write data.
    """
    pass


class WebSocketProxyException(WebSocketException):
    """
    WebSocketProxyException will be raised when proxy error occurred.
    """
    pass


class WebSocketBadStatusException(WebSocketException):
    """
    WebSocketBadStatusException will be raised when we get bad handshake status code.
    """

    def __init__(self, message, status_code, status_message=None, resp_headers=None):
        msg = message % (status_code, status_message)
        super().__init__(msg)
        self.status_code = status_code
        self.resp_headers = resp_headers


class WebSocketAddressException(WebSocketException):
    """
    If the websocket address info cannot be found, this exception will be raised.
    """
    pass
