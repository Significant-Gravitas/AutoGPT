#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import requests
from requests.exceptions import HTTPError


class BaseAPI:
    """Base class for HTTP API calls."""

    def __init__(self, host: str, api_key: str = None):
        self.host = host
        self.api_key = api_key

    @property
    def headers(self):
        return {"api-key": self.api_key}

    def _send_request(self, request_handler, url, **kwargs):
        response = request_handler('{0}{1}'.format(self.host, url), headers=self.headers, **kwargs)
        try:
            response.raise_for_status()
        except HTTPError as e:
            e.args = e.args + (response.text,)
            raise e
        return response.json()

    def get(self, url: str, params: dict = None):
        return self._send_request(requests.get, url, params=params)

    def post(self, url: str, json: dict = None):
        return self._send_request(requests.post, url, json=json)

    def patch(self, url: str, json: dict = None):
        return self._send_request(requests.patch, url, json=json)

    def delete(self, url: str):
        return self._send_request(requests.delete, url)

