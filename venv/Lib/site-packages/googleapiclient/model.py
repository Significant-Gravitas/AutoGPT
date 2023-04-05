# Copyright 2014 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model objects for requests and responses.

Each API may support one or more serializations, such
as JSON, Atom, etc. The model classes are responsible
for converting between the wire format and the Python
object representation.
"""
from __future__ import absolute_import

__author__ = "jcgregorio@google.com (Joe Gregorio)"

import json
import logging
import platform
import urllib

from googleapiclient import version as googleapiclient_version
from googleapiclient.errors import HttpError

_LIBRARY_VERSION = googleapiclient_version.__version__
_PY_VERSION = platform.python_version()

LOGGER = logging.getLogger(__name__)

dump_request_response = False


def _abstract():
    raise NotImplementedError("You need to override this function")


class Model(object):
    """Model base class.

    All Model classes should implement this interface.
    The Model serializes and de-serializes between a wire
    format such as JSON and a Python object representation.
    """

    def request(self, headers, path_params, query_params, body_value):
        """Updates outgoing requests with a serialized body.

        Args:
          headers: dict, request headers
          path_params: dict, parameters that appear in the request path
          query_params: dict, parameters that appear in the query
          body_value: object, the request body as a Python object, which must be
                      serializable.
        Returns:
          A tuple of (headers, path_params, query, body)

          headers: dict, request headers
          path_params: dict, parameters that appear in the request path
          query: string, query part of the request URI
          body: string, the body serialized in the desired wire format.
        """
        _abstract()

    def response(self, resp, content):
        """Convert the response wire format into a Python object.

        Args:
          resp: httplib2.Response, the HTTP response headers and status
          content: string, the body of the HTTP response

        Returns:
          The body de-serialized as a Python object.

        Raises:
          googleapiclient.errors.HttpError if a non 2xx response is received.
        """
        _abstract()


class BaseModel(Model):
    """Base model class.

    Subclasses should provide implementations for the "serialize" and
    "deserialize" methods, as well as values for the following class attributes.

    Attributes:
      accept: The value to use for the HTTP Accept header.
      content_type: The value to use for the HTTP Content-type header.
      no_content_response: The value to return when deserializing a 204 "No
          Content" response.
      alt_param: The value to supply as the "alt" query parameter for requests.
    """

    accept = None
    content_type = None
    no_content_response = None
    alt_param = None

    def _log_request(self, headers, path_params, query, body):
        """Logs debugging information about the request if requested."""
        if dump_request_response:
            LOGGER.info("--request-start--")
            LOGGER.info("-headers-start-")
            for h, v in headers.items():
                LOGGER.info("%s: %s", h, v)
            LOGGER.info("-headers-end-")
            LOGGER.info("-path-parameters-start-")
            for h, v in path_params.items():
                LOGGER.info("%s: %s", h, v)
            LOGGER.info("-path-parameters-end-")
            LOGGER.info("body: %s", body)
            LOGGER.info("query: %s", query)
            LOGGER.info("--request-end--")

    def request(self, headers, path_params, query_params, body_value):
        """Updates outgoing requests with a serialized body.

        Args:
          headers: dict, request headers
          path_params: dict, parameters that appear in the request path
          query_params: dict, parameters that appear in the query
          body_value: object, the request body as a Python object, which must be
                      serializable by json.
        Returns:
          A tuple of (headers, path_params, query, body)

          headers: dict, request headers
          path_params: dict, parameters that appear in the request path
          query: string, query part of the request URI
          body: string, the body serialized as JSON
        """
        query = self._build_query(query_params)
        headers["accept"] = self.accept
        headers["accept-encoding"] = "gzip, deflate"
        if "user-agent" in headers:
            headers["user-agent"] += " "
        else:
            headers["user-agent"] = ""
        headers["user-agent"] += "(gzip)"
        if "x-goog-api-client" in headers:
            headers["x-goog-api-client"] += " "
        else:
            headers["x-goog-api-client"] = ""
        headers["x-goog-api-client"] += "gdcl/%s gl-python/%s" % (
            _LIBRARY_VERSION,
            _PY_VERSION,
        )

        if body_value is not None:
            headers["content-type"] = self.content_type
            body_value = self.serialize(body_value)
        self._log_request(headers, path_params, query, body_value)
        return (headers, path_params, query, body_value)

    def _build_query(self, params):
        """Builds a query string.

        Args:
          params: dict, the query parameters

        Returns:
          The query parameters properly encoded into an HTTP URI query string.
        """
        if self.alt_param is not None:
            params.update({"alt": self.alt_param})
        astuples = []
        for key, value in params.items():
            if type(value) == type([]):
                for x in value:
                    x = x.encode("utf-8")
                    astuples.append((key, x))
            else:
                if isinstance(value, str) and callable(value.encode):
                    value = value.encode("utf-8")
                astuples.append((key, value))
        return "?" + urllib.parse.urlencode(astuples)

    def _log_response(self, resp, content):
        """Logs debugging information about the response if requested."""
        if dump_request_response:
            LOGGER.info("--response-start--")
            for h, v in resp.items():
                LOGGER.info("%s: %s", h, v)
            if content:
                LOGGER.info(content)
            LOGGER.info("--response-end--")

    def response(self, resp, content):
        """Convert the response wire format into a Python object.

        Args:
          resp: httplib2.Response, the HTTP response headers and status
          content: string, the body of the HTTP response

        Returns:
          The body de-serialized as a Python object.

        Raises:
          googleapiclient.errors.HttpError if a non 2xx response is received.
        """
        self._log_response(resp, content)
        # Error handling is TBD, for example, do we retry
        # for some operation/error combinations?
        if resp.status < 300:
            if resp.status == 204:
                # A 204: No Content response should be treated differently
                # to all the other success states
                return self.no_content_response
            return self.deserialize(content)
        else:
            LOGGER.debug("Content from bad request was: %r" % content)
            raise HttpError(resp, content)

    def serialize(self, body_value):
        """Perform the actual Python object serialization.

        Args:
          body_value: object, the request body as a Python object.

        Returns:
          string, the body in serialized form.
        """
        _abstract()

    def deserialize(self, content):
        """Perform the actual deserialization from response string to Python
        object.

        Args:
          content: string, the body of the HTTP response

        Returns:
          The body de-serialized as a Python object.
        """
        _abstract()


class JsonModel(BaseModel):
    """Model class for JSON.

    Serializes and de-serializes between JSON and the Python
    object representation of HTTP request and response bodies.
    """

    accept = "application/json"
    content_type = "application/json"
    alt_param = "json"

    def __init__(self, data_wrapper=False):
        """Construct a JsonModel.

        Args:
          data_wrapper: boolean, wrap requests and responses in a data wrapper
        """
        self._data_wrapper = data_wrapper

    def serialize(self, body_value):
        if (
            isinstance(body_value, dict)
            and "data" not in body_value
            and self._data_wrapper
        ):
            body_value = {"data": body_value}
        return json.dumps(body_value)

    def deserialize(self, content):
        try:
            content = content.decode("utf-8")
        except AttributeError:
            pass
        try:
            body = json.loads(content)
        except json.decoder.JSONDecodeError:
            body = content
        else:
            if self._data_wrapper and "data" in body:
                body = body["data"]
        return body

    @property
    def no_content_response(self):
        return {}


class RawModel(JsonModel):
    """Model class for requests that don't return JSON.

    Serializes and de-serializes between JSON and the Python
    object representation of HTTP request, and returns the raw bytes
    of the response body.
    """

    accept = "*/*"
    content_type = "application/json"
    alt_param = None

    def deserialize(self, content):
        return content

    @property
    def no_content_response(self):
        return ""


class MediaModel(JsonModel):
    """Model class for requests that return Media.

    Serializes and de-serializes between JSON and the Python
    object representation of HTTP request, and returns the raw bytes
    of the response body.
    """

    accept = "*/*"
    content_type = "application/json"
    alt_param = "media"

    def deserialize(self, content):
        return content

    @property
    def no_content_response(self):
        return ""


class ProtocolBufferModel(BaseModel):
    """Model class for protocol buffers.

    Serializes and de-serializes the binary protocol buffer sent in the HTTP
    request and response bodies.
    """

    accept = "application/x-protobuf"
    content_type = "application/x-protobuf"
    alt_param = "proto"

    def __init__(self, protocol_buffer):
        """Constructs a ProtocolBufferModel.

        The serialized protocol buffer returned in an HTTP response will be
        de-serialized using the given protocol buffer class.

        Args:
          protocol_buffer: The protocol buffer class used to de-serialize a
          response from the API.
        """
        self._protocol_buffer = protocol_buffer

    def serialize(self, body_value):
        return body_value.SerializeToString()

    def deserialize(self, content):
        return self._protocol_buffer.FromString(content)

    @property
    def no_content_response(self):
        return self._protocol_buffer()


def makepatch(original, modified):
    """Create a patch object.

    Some methods support PATCH, an efficient way to send updates to a resource.
    This method allows the easy construction of patch bodies by looking at the
    differences between a resource before and after it was modified.

    Args:
      original: object, the original deserialized resource
      modified: object, the modified deserialized resource
    Returns:
      An object that contains only the changes from original to modified, in a
      form suitable to pass to a PATCH method.

    Example usage:
      item = service.activities().get(postid=postid, userid=userid).execute()
      original = copy.deepcopy(item)
      item['object']['content'] = 'This is updated.'
      service.activities.patch(postid=postid, userid=userid,
        body=makepatch(original, item)).execute()
    """
    patch = {}
    for key, original_value in original.items():
        modified_value = modified.get(key, None)
        if modified_value is None:
            # Use None to signal that the element is deleted
            patch[key] = None
        elif original_value != modified_value:
            if type(original_value) == type({}):
                # Recursively descend objects
                patch[key] = makepatch(original_value, modified_value)
            else:
                # In the case of simple types or arrays we just replace
                patch[key] = modified_value
        else:
            # Don't add anything to patch if there's no change
            pass
    for key in modified:
        if key not in original:
            patch[key] = modified[key]

    return patch
