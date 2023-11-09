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

"""Classes to encapsulate a single HTTP request.

The classes implement a command pattern, with every
object supporting an execute() method that does the
actual HTTP request.
"""
from __future__ import absolute_import

__author__ = "jcgregorio@google.com (Joe Gregorio)"

import copy
import http.client as http_client
import io
import json
import logging
import mimetypes
import os
import random
import socket
import time
import urllib
import uuid

import httplib2

# TODO(issue 221): Remove this conditional import jibbajabba.
try:
    import ssl
except ImportError:
    _ssl_SSLError = object()
else:
    _ssl_SSLError = ssl.SSLError

from email.generator import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.parser import FeedParser

from googleapiclient import _auth
from googleapiclient import _helpers as util
from googleapiclient.errors import (
    BatchError,
    HttpError,
    InvalidChunkSizeError,
    ResumableUploadError,
    UnexpectedBodyError,
    UnexpectedMethodError,
)
from googleapiclient.model import JsonModel

LOGGER = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 100 * 1024 * 1024

MAX_URI_LENGTH = 2048

MAX_BATCH_LIMIT = 1000

_TOO_MANY_REQUESTS = 429

DEFAULT_HTTP_TIMEOUT_SEC = 60

_LEGACY_BATCH_URI = "https://www.googleapis.com/batch"


def _should_retry_response(resp_status, content):
    """Determines whether a response should be retried.

    Args:
      resp_status: The response status received.
      content: The response content body.

    Returns:
      True if the response should be retried, otherwise False.
    """
    reason = None

    # Retry on 5xx errors.
    if resp_status >= 500:
        return True

    # Retry on 429 errors.
    if resp_status == _TOO_MANY_REQUESTS:
        return True

    # For 403 errors, we have to check for the `reason` in the response to
    # determine if we should retry.
    if resp_status == http_client.FORBIDDEN:
        # If there's no details about the 403 type, don't retry.
        if not content:
            return False

        # Content is in JSON format.
        try:
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, dict):
                # There are many variations of the error json so we need
                # to determine the keyword which has the error detail. Make sure
                # that the order of the keywords below isn't changed as it can
                # break user code. If the "errors" key exists, we must use that
                # first.
                # See Issue #1243
                # https://github.com/googleapis/google-api-python-client/issues/1243
                error_detail_keyword = next(
                    (
                        kw
                        for kw in ["errors", "status", "message"]
                        if kw in data["error"]
                    ),
                    "",
                )

                if error_detail_keyword:
                    reason = data["error"][error_detail_keyword]

                    if isinstance(reason, list) and len(reason) > 0:
                        reason = reason[0]
                        if "reason" in reason:
                            reason = reason["reason"]
            else:
                reason = data[0]["error"]["errors"]["reason"]
        except (UnicodeDecodeError, ValueError, KeyError):
            LOGGER.warning("Invalid JSON content from response: %s", content)
            return False

        LOGGER.warning('Encountered 403 Forbidden with reason "%s"', reason)

        # Only retry on rate limit related failures.
        if reason in ("userRateLimitExceeded", "rateLimitExceeded"):
            return True

    # Everything else is a success or non-retriable so break.
    return False


def _retry_request(
    http, num_retries, req_type, sleep, rand, uri, method, *args, **kwargs
):
    """Retries an HTTP request multiple times while handling errors.

    If after all retries the request still fails, last error is either returned as
    return value (for HTTP 5xx errors) or thrown (for ssl.SSLError).

    Args:
      http: Http object to be used to execute request.
      num_retries: Maximum number of retries.
      req_type: Type of the request (used for logging retries).
      sleep, rand: Functions to sleep for random time between retries.
      uri: URI to be requested.
      method: HTTP method to be used.
      args, kwargs: Additional arguments passed to http.request.

    Returns:
      resp, content - Response from the http request (may be HTTP 5xx).
    """
    resp = None
    content = None
    exception = None
    for retry_num in range(num_retries + 1):
        if retry_num > 0:
            # Sleep before retrying.
            sleep_time = rand() * 2**retry_num
            LOGGER.warning(
                "Sleeping %.2f seconds before retry %d of %d for %s: %s %s, after %s",
                sleep_time,
                retry_num,
                num_retries,
                req_type,
                method,
                uri,
                resp.status if resp else exception,
            )
            sleep(sleep_time)

        try:
            exception = None
            resp, content = http.request(uri, method, *args, **kwargs)
        # Retry on SSL errors and socket timeout errors.
        except _ssl_SSLError as ssl_error:
            exception = ssl_error
        except socket.timeout as socket_timeout:
            # Needs to be before socket.error as it's a subclass of OSError
            # socket.timeout has no errorcode
            exception = socket_timeout
        except ConnectionError as connection_error:
            # Needs to be before socket.error as it's a subclass of OSError
            exception = connection_error
        except OSError as socket_error:
            # errno's contents differ by platform, so we have to match by name.
            # Some of these same errors may have been caught above, e.g. ECONNRESET *should* be
            # raised as a ConnectionError, but some libraries will raise it as a socket.error
            # with an errno corresponding to ECONNRESET
            if socket.errno.errorcode.get(socket_error.errno) not in {
                "WSAETIMEDOUT",
                "ETIMEDOUT",
                "EPIPE",
                "ECONNABORTED",
                "ECONNREFUSED",
                "ECONNRESET",
            }:
                raise
            exception = socket_error
        except httplib2.ServerNotFoundError as server_not_found_error:
            exception = server_not_found_error

        if exception:
            if retry_num == num_retries:
                raise exception
            else:
                continue

        if not _should_retry_response(resp.status, content):
            break

    return resp, content


class MediaUploadProgress(object):
    """Status of a resumable upload."""

    def __init__(self, resumable_progress, total_size):
        """Constructor.

        Args:
          resumable_progress: int, bytes sent so far.
          total_size: int, total bytes in complete upload, or None if the total
            upload size isn't known ahead of time.
        """
        self.resumable_progress = resumable_progress
        self.total_size = total_size

    def progress(self):
        """Percent of upload completed, as a float.

        Returns:
          the percentage complete as a float, returning 0.0 if the total size of
          the upload is unknown.
        """
        if self.total_size is not None and self.total_size != 0:
            return float(self.resumable_progress) / float(self.total_size)
        else:
            return 0.0


class MediaDownloadProgress(object):
    """Status of a resumable download."""

    def __init__(self, resumable_progress, total_size):
        """Constructor.

        Args:
          resumable_progress: int, bytes received so far.
          total_size: int, total bytes in complete download.
        """
        self.resumable_progress = resumable_progress
        self.total_size = total_size

    def progress(self):
        """Percent of download completed, as a float.

        Returns:
          the percentage complete as a float, returning 0.0 if the total size of
          the download is unknown.
        """
        if self.total_size is not None and self.total_size != 0:
            return float(self.resumable_progress) / float(self.total_size)
        else:
            return 0.0


class MediaUpload(object):
    """Describes a media object to upload.

    Base class that defines the interface of MediaUpload subclasses.

    Note that subclasses of MediaUpload may allow you to control the chunksize
    when uploading a media object. It is important to keep the size of the chunk
    as large as possible to keep the upload efficient. Other factors may influence
    the size of the chunk you use, particularly if you are working in an
    environment where individual HTTP requests may have a hardcoded time limit,
    such as under certain classes of requests under Google App Engine.

    Streams are io.Base compatible objects that support seek(). Some MediaUpload
    subclasses support using streams directly to upload data. Support for
    streaming may be indicated by a MediaUpload sub-class and if appropriate for a
    platform that stream will be used for uploading the media object. The support
    for streaming is indicated by has_stream() returning True. The stream() method
    should return an io.Base object that supports seek(). On platforms where the
    underlying httplib module supports streaming, for example Python 2.6 and
    later, the stream will be passed into the http library which will result in
    less memory being used and possibly faster uploads.

    If you need to upload media that can't be uploaded using any of the existing
    MediaUpload sub-class then you can sub-class MediaUpload for your particular
    needs.
    """

    def chunksize(self):
        """Chunk size for resumable uploads.

        Returns:
          Chunk size in bytes.
        """
        raise NotImplementedError()

    def mimetype(self):
        """Mime type of the body.

        Returns:
          Mime type.
        """
        return "application/octet-stream"

    def size(self):
        """Size of upload.

        Returns:
          Size of the body, or None of the size is unknown.
        """
        return None

    def resumable(self):
        """Whether this upload is resumable.

        Returns:
          True if resumable upload or False.
        """
        return False

    def getbytes(self, begin, end):
        """Get bytes from the media.

        Args:
          begin: int, offset from beginning of file.
          length: int, number of bytes to read, starting at begin.

        Returns:
          A string of bytes read. May be shorter than length if EOF was reached
          first.
        """
        raise NotImplementedError()

    def has_stream(self):
        """Does the underlying upload support a streaming interface.

        Streaming means it is an io.IOBase subclass that supports seek, i.e.
        seekable() returns True.

        Returns:
          True if the call to stream() will return an instance of a seekable io.Base
          subclass.
        """
        return False

    def stream(self):
        """A stream interface to the data being uploaded.

        Returns:
          The returned value is an io.IOBase subclass that supports seek, i.e.
          seekable() returns True.
        """
        raise NotImplementedError()

    @util.positional(1)
    def _to_json(self, strip=None):
        """Utility function for creating a JSON representation of a MediaUpload.

        Args:
          strip: array, An array of names of members to not include in the JSON.

        Returns:
           string, a JSON representation of this instance, suitable to pass to
           from_json().
        """
        t = type(self)
        d = copy.copy(self.__dict__)
        if strip is not None:
            for member in strip:
                del d[member]
        d["_class"] = t.__name__
        d["_module"] = t.__module__
        return json.dumps(d)

    def to_json(self):
        """Create a JSON representation of an instance of MediaUpload.

        Returns:
           string, a JSON representation of this instance, suitable to pass to
           from_json().
        """
        return self._to_json()

    @classmethod
    def new_from_json(cls, s):
        """Utility class method to instantiate a MediaUpload subclass from a JSON
        representation produced by to_json().

        Args:
          s: string, JSON from to_json().

        Returns:
          An instance of the subclass of MediaUpload that was serialized with
          to_json().
        """
        data = json.loads(s)
        # Find and call the right classmethod from_json() to restore the object.
        module = data["_module"]
        m = __import__(module, fromlist=module.split(".")[:-1])
        kls = getattr(m, data["_class"])
        from_json = getattr(kls, "from_json")
        return from_json(s)


class MediaIoBaseUpload(MediaUpload):
    """A MediaUpload for a io.Base objects.

    Note that the Python file object is compatible with io.Base and can be used
    with this class also.

      fh = BytesIO('...Some data to upload...')
      media = MediaIoBaseUpload(fh, mimetype='image/png',
        chunksize=1024*1024, resumable=True)
      farm.animals().insert(
          id='cow',
          name='cow.png',
          media_body=media).execute()

    Depending on the platform you are working on, you may pass -1 as the
    chunksize, which indicates that the entire file should be uploaded in a single
    request. If the underlying platform supports streams, such as Python 2.6 or
    later, then this can be very efficient as it avoids multiple connections, and
    also avoids loading the entire file into memory before sending it. Note that
    Google App Engine has a 5MB limit on request size, so you should never set
    your chunksize larger than 5MB, or to -1.
    """

    @util.positional(3)
    def __init__(self, fd, mimetype, chunksize=DEFAULT_CHUNK_SIZE, resumable=False):
        """Constructor.

        Args:
          fd: io.Base or file object, The source of the bytes to upload. MUST be
            opened in blocking mode, do not use streams opened in non-blocking mode.
            The given stream must be seekable, that is, it must be able to call
            seek() on fd.
          mimetype: string, Mime-type of the file.
          chunksize: int, File will be uploaded in chunks of this many bytes. Only
            used if resumable=True. Pass in a value of -1 if the file is to be
            uploaded as a single chunk. Note that Google App Engine has a 5MB limit
            on request size, so you should never set your chunksize larger than 5MB,
            or to -1.
          resumable: bool, True if this is a resumable upload. False means upload
            in a single request.
        """
        super(MediaIoBaseUpload, self).__init__()
        self._fd = fd
        self._mimetype = mimetype
        if not (chunksize == -1 or chunksize > 0):
            raise InvalidChunkSizeError()
        self._chunksize = chunksize
        self._resumable = resumable

        self._fd.seek(0, os.SEEK_END)
        self._size = self._fd.tell()

    def chunksize(self):
        """Chunk size for resumable uploads.

        Returns:
          Chunk size in bytes.
        """
        return self._chunksize

    def mimetype(self):
        """Mime type of the body.

        Returns:
          Mime type.
        """
        return self._mimetype

    def size(self):
        """Size of upload.

        Returns:
          Size of the body, or None of the size is unknown.
        """
        return self._size

    def resumable(self):
        """Whether this upload is resumable.

        Returns:
          True if resumable upload or False.
        """
        return self._resumable

    def getbytes(self, begin, length):
        """Get bytes from the media.

        Args:
          begin: int, offset from beginning of file.
          length: int, number of bytes to read, starting at begin.

        Returns:
          A string of bytes read. May be shorted than length if EOF was reached
          first.
        """
        self._fd.seek(begin)
        return self._fd.read(length)

    def has_stream(self):
        """Does the underlying upload support a streaming interface.

        Streaming means it is an io.IOBase subclass that supports seek, i.e.
        seekable() returns True.

        Returns:
          True if the call to stream() will return an instance of a seekable io.Base
          subclass.
        """
        return True

    def stream(self):
        """A stream interface to the data being uploaded.

        Returns:
          The returned value is an io.IOBase subclass that supports seek, i.e.
          seekable() returns True.
        """
        return self._fd

    def to_json(self):
        """This upload type is not serializable."""
        raise NotImplementedError("MediaIoBaseUpload is not serializable.")


class MediaFileUpload(MediaIoBaseUpload):
    """A MediaUpload for a file.

    Construct a MediaFileUpload and pass as the media_body parameter of the
    method. For example, if we had a service that allowed uploading images:

      media = MediaFileUpload('cow.png', mimetype='image/png',
        chunksize=1024*1024, resumable=True)
      farm.animals().insert(
          id='cow',
          name='cow.png',
          media_body=media).execute()

    Depending on the platform you are working on, you may pass -1 as the
    chunksize, which indicates that the entire file should be uploaded in a single
    request. If the underlying platform supports streams, such as Python 2.6 or
    later, then this can be very efficient as it avoids multiple connections, and
    also avoids loading the entire file into memory before sending it. Note that
    Google App Engine has a 5MB limit on request size, so you should never set
    your chunksize larger than 5MB, or to -1.
    """

    @util.positional(2)
    def __init__(
        self, filename, mimetype=None, chunksize=DEFAULT_CHUNK_SIZE, resumable=False
    ):
        """Constructor.

        Args:
          filename: string, Name of the file.
          mimetype: string, Mime-type of the file. If None then a mime-type will be
            guessed from the file extension.
          chunksize: int, File will be uploaded in chunks of this many bytes. Only
            used if resumable=True. Pass in a value of -1 if the file is to be
            uploaded in a single chunk. Note that Google App Engine has a 5MB limit
            on request size, so you should never set your chunksize larger than 5MB,
            or to -1.
          resumable: bool, True if this is a resumable upload. False means upload
            in a single request.
        """
        self._fd = None
        self._filename = filename
        self._fd = open(self._filename, "rb")
        if mimetype is None:
            # No mimetype provided, make a guess.
            mimetype, _ = mimetypes.guess_type(filename)
            if mimetype is None:
                # Guess failed, use octet-stream.
                mimetype = "application/octet-stream"
        super(MediaFileUpload, self).__init__(
            self._fd, mimetype, chunksize=chunksize, resumable=resumable
        )

    def __del__(self):
        if self._fd:
            self._fd.close()

    def to_json(self):
        """Creating a JSON representation of an instance of MediaFileUpload.

        Returns:
           string, a JSON representation of this instance, suitable to pass to
           from_json().
        """
        return self._to_json(strip=["_fd"])

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        return MediaFileUpload(
            d["_filename"],
            mimetype=d["_mimetype"],
            chunksize=d["_chunksize"],
            resumable=d["_resumable"],
        )


class MediaInMemoryUpload(MediaIoBaseUpload):
    """MediaUpload for a chunk of bytes.

    DEPRECATED: Use MediaIoBaseUpload with either io.TextIOBase or io.StringIO for
    the stream.
    """

    @util.positional(2)
    def __init__(
        self,
        body,
        mimetype="application/octet-stream",
        chunksize=DEFAULT_CHUNK_SIZE,
        resumable=False,
    ):
        """Create a new MediaInMemoryUpload.

        DEPRECATED: Use MediaIoBaseUpload with either io.TextIOBase or io.StringIO for
        the stream.

        Args:
          body: string, Bytes of body content.
          mimetype: string, Mime-type of the file or default of
            'application/octet-stream'.
          chunksize: int, File will be uploaded in chunks of this many bytes. Only
            used if resumable=True.
          resumable: bool, True if this is a resumable upload. False means upload
            in a single request.
        """
        fd = io.BytesIO(body)
        super(MediaInMemoryUpload, self).__init__(
            fd, mimetype, chunksize=chunksize, resumable=resumable
        )


class MediaIoBaseDownload(object):
    """ "Download media resources.

    Note that the Python file object is compatible with io.Base and can be used
    with this class also.


    Example:
      request = farms.animals().get_media(id='cow')
      fh = io.FileIO('cow.png', mode='wb')
      downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024)

      done = False
      while done is False:
        status, done = downloader.next_chunk()
        if status:
          print "Download %d%%." % int(status.progress() * 100)
      print "Download Complete!"
    """

    @util.positional(3)
    def __init__(self, fd, request, chunksize=DEFAULT_CHUNK_SIZE):
        """Constructor.

        Args:
          fd: io.Base or file object, The stream in which to write the downloaded
            bytes.
          request: googleapiclient.http.HttpRequest, the media request to perform in
            chunks.
          chunksize: int, File will be downloaded in chunks of this many bytes.
        """
        self._fd = fd
        self._request = request
        self._uri = request.uri
        self._chunksize = chunksize
        self._progress = 0
        self._total_size = None
        self._done = False

        # Stubs for testing.
        self._sleep = time.sleep
        self._rand = random.random

        self._headers = {}
        for k, v in request.headers.items():
            # allow users to supply custom headers by setting them on the request
            # but strip out the ones that are set by default on requests generated by
            # API methods like Drive's files().get(fileId=...)
            if not k.lower() in ("accept", "accept-encoding", "user-agent"):
                self._headers[k] = v

    @util.positional(1)
    def next_chunk(self, num_retries=0):
        """Get the next chunk of the download.

        Args:
          num_retries: Integer, number of times to retry with randomized
                exponential backoff. If all retries fail, the raised HttpError
                represents the last request. If zero (default), we attempt the
                request only once.

        Returns:
          (status, done): (MediaDownloadProgress, boolean)
             The value of 'done' will be True when the media has been fully
             downloaded or the total size of the media is unknown.

        Raises:
          googleapiclient.errors.HttpError if the response was not a 2xx.
          httplib2.HttpLib2Error if a transport error has occurred.
        """
        headers = self._headers.copy()
        headers["range"] = "bytes=%d-%d" % (
            self._progress,
            self._progress + self._chunksize - 1,
        )
        http = self._request.http

        resp, content = _retry_request(
            http,
            num_retries,
            "media download",
            self._sleep,
            self._rand,
            self._uri,
            "GET",
            headers=headers,
        )

        if resp.status in [200, 206]:
            if "content-location" in resp and resp["content-location"] != self._uri:
                self._uri = resp["content-location"]
            self._progress += len(content)
            self._fd.write(content)

            if "content-range" in resp:
                content_range = resp["content-range"]
                length = content_range.rsplit("/", 1)[1]
                self._total_size = int(length)
            elif "content-length" in resp:
                self._total_size = int(resp["content-length"])

            if self._total_size is None or self._progress == self._total_size:
                self._done = True
            return MediaDownloadProgress(self._progress, self._total_size), self._done
        elif resp.status == 416:
            # 416 is Range Not Satisfiable
            # This typically occurs with a zero byte file
            content_range = resp["content-range"]
            length = content_range.rsplit("/", 1)[1]
            self._total_size = int(length)
            if self._total_size == 0:
                self._done = True
                return (
                    MediaDownloadProgress(self._progress, self._total_size),
                    self._done,
                )
        raise HttpError(resp, content, uri=self._uri)


class _StreamSlice(object):
    """Truncated stream.

    Takes a stream and presents a stream that is a slice of the original stream.
    This is used when uploading media in chunks. In later versions of Python a
    stream can be passed to httplib in place of the string of data to send. The
    problem is that httplib just blindly reads to the end of the stream. This
    wrapper presents a virtual stream that only reads to the end of the chunk.
    """

    def __init__(self, stream, begin, chunksize):
        """Constructor.

        Args:
          stream: (io.Base, file object), the stream to wrap.
          begin: int, the seek position the chunk begins at.
          chunksize: int, the size of the chunk.
        """
        self._stream = stream
        self._begin = begin
        self._chunksize = chunksize
        self._stream.seek(begin)

    def read(self, n=-1):
        """Read n bytes.

        Args:
          n, int, the number of bytes to read.

        Returns:
          A string of length 'n', or less if EOF is reached.
        """
        # The data left available to read sits in [cur, end)
        cur = self._stream.tell()
        end = self._begin + self._chunksize
        if n == -1 or cur + n > end:
            n = end - cur
        return self._stream.read(n)


class HttpRequest(object):
    """Encapsulates a single HTTP request."""

    @util.positional(4)
    def __init__(
        self,
        http,
        postproc,
        uri,
        method="GET",
        body=None,
        headers=None,
        methodId=None,
        resumable=None,
    ):
        """Constructor for an HttpRequest.

        Args:
          http: httplib2.Http, the transport object to use to make a request
          postproc: callable, called on the HTTP response and content to transform
                    it into a data object before returning, or raising an exception
                    on an error.
          uri: string, the absolute URI to send the request to
          method: string, the HTTP method to use
          body: string, the request body of the HTTP request,
          headers: dict, the HTTP request headers
          methodId: string, a unique identifier for the API method being called.
          resumable: MediaUpload, None if this is not a resumbale request.
        """
        self.uri = uri
        self.method = method
        self.body = body
        self.headers = headers or {}
        self.methodId = methodId
        self.http = http
        self.postproc = postproc
        self.resumable = resumable
        self.response_callbacks = []
        self._in_error_state = False

        # The size of the non-media part of the request.
        self.body_size = len(self.body or "")

        # The resumable URI to send chunks to.
        self.resumable_uri = None

        # The bytes that have been uploaded.
        self.resumable_progress = 0

        # Stubs for testing.
        self._rand = random.random
        self._sleep = time.sleep

    @util.positional(1)
    def execute(self, http=None, num_retries=0):
        """Execute the request.

        Args:
          http: httplib2.Http, an http object to be used in place of the
                one the HttpRequest request object was constructed with.
          num_retries: Integer, number of times to retry with randomized
                exponential backoff. If all retries fail, the raised HttpError
                represents the last request. If zero (default), we attempt the
                request only once.

        Returns:
          A deserialized object model of the response body as determined
          by the postproc.

        Raises:
          googleapiclient.errors.HttpError if the response was not a 2xx.
          httplib2.HttpLib2Error if a transport error has occurred.
        """
        if http is None:
            http = self.http

        if self.resumable:
            body = None
            while body is None:
                _, body = self.next_chunk(http=http, num_retries=num_retries)
            return body

        # Non-resumable case.

        if "content-length" not in self.headers:
            self.headers["content-length"] = str(self.body_size)
        # If the request URI is too long then turn it into a POST request.
        # Assume that a GET request never contains a request body.
        if len(self.uri) > MAX_URI_LENGTH and self.method == "GET":
            self.method = "POST"
            self.headers["x-http-method-override"] = "GET"
            self.headers["content-type"] = "application/x-www-form-urlencoded"
            parsed = urllib.parse.urlparse(self.uri)
            self.uri = urllib.parse.urlunparse(
                (parsed.scheme, parsed.netloc, parsed.path, parsed.params, None, None)
            )
            self.body = parsed.query
            self.headers["content-length"] = str(len(self.body))

        # Handle retries for server-side errors.
        resp, content = _retry_request(
            http,
            num_retries,
            "request",
            self._sleep,
            self._rand,
            str(self.uri),
            method=str(self.method),
            body=self.body,
            headers=self.headers,
        )

        for callback in self.response_callbacks:
            callback(resp)
        if resp.status >= 300:
            raise HttpError(resp, content, uri=self.uri)
        return self.postproc(resp, content)

    @util.positional(2)
    def add_response_callback(self, cb):
        """add_response_headers_callback

        Args:
          cb: Callback to be called on receiving the response headers, of signature:

          def cb(resp):
            # Where resp is an instance of httplib2.Response
        """
        self.response_callbacks.append(cb)

    @util.positional(1)
    def next_chunk(self, http=None, num_retries=0):
        """Execute the next step of a resumable upload.

        Can only be used if the method being executed supports media uploads and
        the MediaUpload object passed in was flagged as using resumable upload.

        Example:

          media = MediaFileUpload('cow.png', mimetype='image/png',
                                  chunksize=1000, resumable=True)
          request = farm.animals().insert(
              id='cow',
              name='cow.png',
              media_body=media)

          response = None
          while response is None:
            status, response = request.next_chunk()
            if status:
              print "Upload %d%% complete." % int(status.progress() * 100)


        Args:
          http: httplib2.Http, an http object to be used in place of the
                one the HttpRequest request object was constructed with.
          num_retries: Integer, number of times to retry with randomized
                exponential backoff. If all retries fail, the raised HttpError
                represents the last request. If zero (default), we attempt the
                request only once.

        Returns:
          (status, body): (ResumableMediaStatus, object)
             The body will be None until the resumable media is fully uploaded.

        Raises:
          googleapiclient.errors.HttpError if the response was not a 2xx.
          httplib2.HttpLib2Error if a transport error has occurred.
        """
        if http is None:
            http = self.http

        if self.resumable.size() is None:
            size = "*"
        else:
            size = str(self.resumable.size())

        if self.resumable_uri is None:
            start_headers = copy.copy(self.headers)
            start_headers["X-Upload-Content-Type"] = self.resumable.mimetype()
            if size != "*":
                start_headers["X-Upload-Content-Length"] = size
            start_headers["content-length"] = str(self.body_size)

            resp, content = _retry_request(
                http,
                num_retries,
                "resumable URI request",
                self._sleep,
                self._rand,
                self.uri,
                method=self.method,
                body=self.body,
                headers=start_headers,
            )

            if resp.status == 200 and "location" in resp:
                self.resumable_uri = resp["location"]
            else:
                raise ResumableUploadError(resp, content)
        elif self._in_error_state:
            # If we are in an error state then query the server for current state of
            # the upload by sending an empty PUT and reading the 'range' header in
            # the response.
            headers = {"Content-Range": "bytes */%s" % size, "content-length": "0"}
            resp, content = http.request(self.resumable_uri, "PUT", headers=headers)
            status, body = self._process_response(resp, content)
            if body:
                # The upload was complete.
                return (status, body)

        if self.resumable.has_stream():
            data = self.resumable.stream()
            if self.resumable.chunksize() == -1:
                data.seek(self.resumable_progress)
                chunk_end = self.resumable.size() - self.resumable_progress - 1
            else:
                # Doing chunking with a stream, so wrap a slice of the stream.
                data = _StreamSlice(
                    data, self.resumable_progress, self.resumable.chunksize()
                )
                chunk_end = min(
                    self.resumable_progress + self.resumable.chunksize() - 1,
                    self.resumable.size() - 1,
                )
        else:
            data = self.resumable.getbytes(
                self.resumable_progress, self.resumable.chunksize()
            )

            # A short read implies that we are at EOF, so finish the upload.
            if len(data) < self.resumable.chunksize():
                size = str(self.resumable_progress + len(data))

            chunk_end = self.resumable_progress + len(data) - 1

        headers = {
            # Must set the content-length header here because httplib can't
            # calculate the size when working with _StreamSlice.
            "Content-Length": str(chunk_end - self.resumable_progress + 1),
        }

        # An empty file results in chunk_end = -1 and size = 0
        # sending "bytes 0--1/0" results in an invalid request
        # Only add header "Content-Range" if chunk_end != -1
        if chunk_end != -1:
            headers["Content-Range"] = "bytes %d-%d/%s" % (
                self.resumable_progress,
                chunk_end,
                size,
            )

        for retry_num in range(num_retries + 1):
            if retry_num > 0:
                self._sleep(self._rand() * 2**retry_num)
                LOGGER.warning(
                    "Retry #%d for media upload: %s %s, following status: %d"
                    % (retry_num, self.method, self.uri, resp.status)
                )

            try:
                resp, content = http.request(
                    self.resumable_uri, method="PUT", body=data, headers=headers
                )
            except:
                self._in_error_state = True
                raise
            if not _should_retry_response(resp.status, content):
                break

        return self._process_response(resp, content)

    def _process_response(self, resp, content):
        """Process the response from a single chunk upload.

        Args:
          resp: httplib2.Response, the response object.
          content: string, the content of the response.

        Returns:
          (status, body): (ResumableMediaStatus, object)
             The body will be None until the resumable media is fully uploaded.

        Raises:
          googleapiclient.errors.HttpError if the response was not a 2xx or a 308.
        """
        if resp.status in [200, 201]:
            self._in_error_state = False
            return None, self.postproc(resp, content)
        elif resp.status == 308:
            self._in_error_state = False
            # A "308 Resume Incomplete" indicates we are not done.
            try:
                self.resumable_progress = int(resp["range"].split("-")[1]) + 1
            except KeyError:
                # If resp doesn't contain range header, resumable progress is 0
                self.resumable_progress = 0
            if "location" in resp:
                self.resumable_uri = resp["location"]
        else:
            self._in_error_state = True
            raise HttpError(resp, content, uri=self.uri)

        return (
            MediaUploadProgress(self.resumable_progress, self.resumable.size()),
            None,
        )

    def to_json(self):
        """Returns a JSON representation of the HttpRequest."""
        d = copy.copy(self.__dict__)
        if d["resumable"] is not None:
            d["resumable"] = self.resumable.to_json()
        del d["http"]
        del d["postproc"]
        del d["_sleep"]
        del d["_rand"]

        return json.dumps(d)

    @staticmethod
    def from_json(s, http, postproc):
        """Returns an HttpRequest populated with info from a JSON object."""
        d = json.loads(s)
        if d["resumable"] is not None:
            d["resumable"] = MediaUpload.new_from_json(d["resumable"])
        return HttpRequest(
            http,
            postproc,
            uri=d["uri"],
            method=d["method"],
            body=d["body"],
            headers=d["headers"],
            methodId=d["methodId"],
            resumable=d["resumable"],
        )

    @staticmethod
    def null_postproc(resp, contents):
        return resp, contents


class BatchHttpRequest(object):
    """Batches multiple HttpRequest objects into a single HTTP request.

    Example:
      from googleapiclient.http import BatchHttpRequest

      def list_animals(request_id, response, exception):
        \"\"\"Do something with the animals list response.\"\"\"
        if exception is not None:
          # Do something with the exception.
          pass
        else:
          # Do something with the response.
          pass

      def list_farmers(request_id, response, exception):
        \"\"\"Do something with the farmers list response.\"\"\"
        if exception is not None:
          # Do something with the exception.
          pass
        else:
          # Do something with the response.
          pass

      service = build('farm', 'v2')

      batch = BatchHttpRequest()

      batch.add(service.animals().list(), list_animals)
      batch.add(service.farmers().list(), list_farmers)
      batch.execute(http=http)
    """

    @util.positional(1)
    def __init__(self, callback=None, batch_uri=None):
        """Constructor for a BatchHttpRequest.

        Args:
          callback: callable, A callback to be called for each response, of the
            form callback(id, response, exception). The first parameter is the
            request id, and the second is the deserialized response object. The
            third is an googleapiclient.errors.HttpError exception object if an HTTP error
            occurred while processing the request, or None if no error occurred.
          batch_uri: string, URI to send batch requests to.
        """
        if batch_uri is None:
            batch_uri = _LEGACY_BATCH_URI

        if batch_uri == _LEGACY_BATCH_URI:
            LOGGER.warning(
                "You have constructed a BatchHttpRequest using the legacy batch "
                "endpoint %s. This endpoint will be turned down on August 12, 2020. "
                "Please provide the API-specific endpoint or use "
                "service.new_batch_http_request(). For more details see "
                "https://developers.googleblog.com/2018/03/discontinuing-support-for-json-rpc-and.html"
                "and https://developers.google.com/api-client-library/python/guide/batch.",
                _LEGACY_BATCH_URI,
            )
        self._batch_uri = batch_uri

        # Global callback to be called for each individual response in the batch.
        self._callback = callback

        # A map from id to request.
        self._requests = {}

        # A map from id to callback.
        self._callbacks = {}

        # List of request ids, in the order in which they were added.
        self._order = []

        # The last auto generated id.
        self._last_auto_id = 0

        # Unique ID on which to base the Content-ID headers.
        self._base_id = None

        # A map from request id to (httplib2.Response, content) response pairs
        self._responses = {}

        # A map of id(Credentials) that have been refreshed.
        self._refreshed_credentials = {}

    def _refresh_and_apply_credentials(self, request, http):
        """Refresh the credentials and apply to the request.

        Args:
          request: HttpRequest, the request.
          http: httplib2.Http, the global http object for the batch.
        """
        # For the credentials to refresh, but only once per refresh_token
        # If there is no http per the request then refresh the http passed in
        # via execute()
        creds = None
        request_credentials = False

        if request.http is not None:
            creds = _auth.get_credentials_from_http(request.http)
            request_credentials = True

        if creds is None and http is not None:
            creds = _auth.get_credentials_from_http(http)

        if creds is not None:
            if id(creds) not in self._refreshed_credentials:
                _auth.refresh_credentials(creds)
                self._refreshed_credentials[id(creds)] = 1

        # Only apply the credentials if we are using the http object passed in,
        # otherwise apply() will get called during _serialize_request().
        if request.http is None or not request_credentials:
            _auth.apply_credentials(creds, request.headers)

    def _id_to_header(self, id_):
        """Convert an id to a Content-ID header value.

        Args:
          id_: string, identifier of individual request.

        Returns:
          A Content-ID header with the id_ encoded into it. A UUID is prepended to
          the value because Content-ID headers are supposed to be universally
          unique.
        """
        if self._base_id is None:
            self._base_id = uuid.uuid4()

        # NB: we intentionally leave whitespace between base/id and '+', so RFC2822
        # line folding works properly on Python 3; see
        # https://github.com/googleapis/google-api-python-client/issues/164
        return "<%s + %s>" % (self._base_id, urllib.parse.quote(id_))

    def _header_to_id(self, header):
        """Convert a Content-ID header value to an id.

        Presumes the Content-ID header conforms to the format that _id_to_header()
        returns.

        Args:
          header: string, Content-ID header value.

        Returns:
          The extracted id value.

        Raises:
          BatchError if the header is not in the expected format.
        """
        if header[0] != "<" or header[-1] != ">":
            raise BatchError("Invalid value for Content-ID: %s" % header)
        if "+" not in header:
            raise BatchError("Invalid value for Content-ID: %s" % header)
        base, id_ = header[1:-1].split(" + ", 1)

        return urllib.parse.unquote(id_)

    def _serialize_request(self, request):
        """Convert an HttpRequest object into a string.

        Args:
          request: HttpRequest, the request to serialize.

        Returns:
          The request as a string in application/http format.
        """
        # Construct status line
        parsed = urllib.parse.urlparse(request.uri)
        request_line = urllib.parse.urlunparse(
            ("", "", parsed.path, parsed.params, parsed.query, "")
        )
        status_line = request.method + " " + request_line + " HTTP/1.1\n"
        major, minor = request.headers.get("content-type", "application/json").split(
            "/"
        )
        msg = MIMENonMultipart(major, minor)
        headers = request.headers.copy()

        if request.http is not None:
            credentials = _auth.get_credentials_from_http(request.http)
            if credentials is not None:
                _auth.apply_credentials(credentials, headers)

        # MIMENonMultipart adds its own Content-Type header.
        if "content-type" in headers:
            del headers["content-type"]

        for key, value in headers.items():
            msg[key] = value
        msg["Host"] = parsed.netloc
        msg.set_unixfrom(None)

        if request.body is not None:
            msg.set_payload(request.body)
            msg["content-length"] = str(len(request.body))

        # Serialize the mime message.
        fp = io.StringIO()
        # maxheaderlen=0 means don't line wrap headers.
        g = Generator(fp, maxheaderlen=0)
        g.flatten(msg, unixfrom=False)
        body = fp.getvalue()

        return status_line + body

    def _deserialize_response(self, payload):
        """Convert string into httplib2 response and content.

        Args:
          payload: string, headers and body as a string.

        Returns:
          A pair (resp, content), such as would be returned from httplib2.request.
        """
        # Strip off the status line
        status_line, payload = payload.split("\n", 1)
        protocol, status, reason = status_line.split(" ", 2)

        # Parse the rest of the response
        parser = FeedParser()
        parser.feed(payload)
        msg = parser.close()
        msg["status"] = status

        # Create httplib2.Response from the parsed headers.
        resp = httplib2.Response(msg)
        resp.reason = reason
        resp.version = int(protocol.split("/", 1)[1].replace(".", ""))

        content = payload.split("\r\n\r\n", 1)[1]

        return resp, content

    def _new_id(self):
        """Create a new id.

        Auto incrementing number that avoids conflicts with ids already used.

        Returns:
           string, a new unique id.
        """
        self._last_auto_id += 1
        while str(self._last_auto_id) in self._requests:
            self._last_auto_id += 1
        return str(self._last_auto_id)

    @util.positional(2)
    def add(self, request, callback=None, request_id=None):
        """Add a new request.

        Every callback added will be paired with a unique id, the request_id. That
        unique id will be passed back to the callback when the response comes back
        from the server. The default behavior is to have the library generate it's
        own unique id. If the caller passes in a request_id then they must ensure
        uniqueness for each request_id, and if they are not an exception is
        raised. Callers should either supply all request_ids or never supply a
        request id, to avoid such an error.

        Args:
          request: HttpRequest, Request to add to the batch.
          callback: callable, A callback to be called for this response, of the
            form callback(id, response, exception). The first parameter is the
            request id, and the second is the deserialized response object. The
            third is an googleapiclient.errors.HttpError exception object if an HTTP error
            occurred while processing the request, or None if no errors occurred.
          request_id: string, A unique id for the request. The id will be passed
            to the callback with the response.

        Returns:
          None

        Raises:
          BatchError if a media request is added to a batch.
          KeyError is the request_id is not unique.
        """

        if len(self._order) >= MAX_BATCH_LIMIT:
            raise BatchError(
                "Exceeded the maximum calls(%d) in a single batch request."
                % MAX_BATCH_LIMIT
            )
        if request_id is None:
            request_id = self._new_id()
        if request.resumable is not None:
            raise BatchError("Media requests cannot be used in a batch request.")
        if request_id in self._requests:
            raise KeyError("A request with this ID already exists: %s" % request_id)
        self._requests[request_id] = request
        self._callbacks[request_id] = callback
        self._order.append(request_id)

    def _execute(self, http, order, requests):
        """Serialize batch request, send to server, process response.

        Args:
          http: httplib2.Http, an http object to be used to make the request with.
          order: list, list of request ids in the order they were added to the
            batch.
          requests: list, list of request objects to send.

        Raises:
          httplib2.HttpLib2Error if a transport error has occurred.
          googleapiclient.errors.BatchError if the response is the wrong format.
        """
        message = MIMEMultipart("mixed")
        # Message should not write out it's own headers.
        setattr(message, "_write_headers", lambda self: None)

        # Add all the individual requests.
        for request_id in order:
            request = requests[request_id]

            msg = MIMENonMultipart("application", "http")
            msg["Content-Transfer-Encoding"] = "binary"
            msg["Content-ID"] = self._id_to_header(request_id)

            body = self._serialize_request(request)
            msg.set_payload(body)
            message.attach(msg)

        # encode the body: note that we can't use `as_string`, because
        # it plays games with `From ` lines.
        fp = io.StringIO()
        g = Generator(fp, mangle_from_=False)
        g.flatten(message, unixfrom=False)
        body = fp.getvalue()

        headers = {}
        headers["content-type"] = (
            "multipart/mixed; " 'boundary="%s"'
        ) % message.get_boundary()

        resp, content = http.request(
            self._batch_uri, method="POST", body=body, headers=headers
        )

        if resp.status >= 300:
            raise HttpError(resp, content, uri=self._batch_uri)

        # Prepend with a content-type header so FeedParser can handle it.
        header = "content-type: %s\r\n\r\n" % resp["content-type"]
        # PY3's FeedParser only accepts unicode. So we should decode content
        # here, and encode each payload again.
        content = content.decode("utf-8")
        for_parser = header + content

        parser = FeedParser()
        parser.feed(for_parser)
        mime_response = parser.close()

        if not mime_response.is_multipart():
            raise BatchError(
                "Response not in multipart/mixed format.", resp=resp, content=content
            )

        for part in mime_response.get_payload():
            request_id = self._header_to_id(part["Content-ID"])
            response, content = self._deserialize_response(part.get_payload())
            # We encode content here to emulate normal http response.
            if isinstance(content, str):
                content = content.encode("utf-8")
            self._responses[request_id] = (response, content)

    @util.positional(1)
    def execute(self, http=None):
        """Execute all the requests as a single batched HTTP request.

        Args:
          http: httplib2.Http, an http object to be used in place of the one the
            HttpRequest request object was constructed with. If one isn't supplied
            then use a http object from the requests in this batch.

        Returns:
          None

        Raises:
          httplib2.HttpLib2Error if a transport error has occurred.
          googleapiclient.errors.BatchError if the response is the wrong format.
        """
        # If we have no requests return
        if len(self._order) == 0:
            return None

        # If http is not supplied use the first valid one given in the requests.
        if http is None:
            for request_id in self._order:
                request = self._requests[request_id]
                if request is not None:
                    http = request.http
                    break

        if http is None:
            raise ValueError("Missing a valid http object.")

        # Special case for OAuth2Credentials-style objects which have not yet been
        # refreshed with an initial access_token.
        creds = _auth.get_credentials_from_http(http)
        if creds is not None:
            if not _auth.is_valid(creds):
                LOGGER.info("Attempting refresh to obtain initial access_token")
                _auth.refresh_credentials(creds)

        self._execute(http, self._order, self._requests)

        # Loop over all the requests and check for 401s. For each 401 request the
        # credentials should be refreshed and then sent again in a separate batch.
        redo_requests = {}
        redo_order = []

        for request_id in self._order:
            resp, content = self._responses[request_id]
            if resp["status"] == "401":
                redo_order.append(request_id)
                request = self._requests[request_id]
                self._refresh_and_apply_credentials(request, http)
                redo_requests[request_id] = request

        if redo_requests:
            self._execute(http, redo_order, redo_requests)

        # Now process all callbacks that are erroring, and raise an exception for
        # ones that return a non-2xx response? Or add extra parameter to callback
        # that contains an HttpError?

        for request_id in self._order:
            resp, content = self._responses[request_id]

            request = self._requests[request_id]
            callback = self._callbacks[request_id]

            response = None
            exception = None
            try:
                if resp.status >= 300:
                    raise HttpError(resp, content, uri=request.uri)
                response = request.postproc(resp, content)
            except HttpError as e:
                exception = e

            if callback is not None:
                callback(request_id, response, exception)
            if self._callback is not None:
                self._callback(request_id, response, exception)


class HttpRequestMock(object):
    """Mock of HttpRequest.

    Do not construct directly, instead use RequestMockBuilder.
    """

    def __init__(self, resp, content, postproc):
        """Constructor for HttpRequestMock

        Args:
          resp: httplib2.Response, the response to emulate coming from the request
          content: string, the response body
          postproc: callable, the post processing function usually supplied by
                    the model class. See model.JsonModel.response() as an example.
        """
        self.resp = resp
        self.content = content
        self.postproc = postproc
        if resp is None:
            self.resp = httplib2.Response({"status": 200, "reason": "OK"})
        if "reason" in self.resp:
            self.resp.reason = self.resp["reason"]

    def execute(self, http=None):
        """Execute the request.

        Same behavior as HttpRequest.execute(), but the response is
        mocked and not really from an HTTP request/response.
        """
        return self.postproc(self.resp, self.content)


class RequestMockBuilder(object):
    """A simple mock of HttpRequest

    Pass in a dictionary to the constructor that maps request methodIds to
    tuples of (httplib2.Response, content, opt_expected_body) that should be
    returned when that method is called. None may also be passed in for the
    httplib2.Response, in which case a 200 OK response will be generated.
    If an opt_expected_body (str or dict) is provided, it will be compared to
    the body and UnexpectedBodyError will be raised on inequality.

    Example:
      response = '{"data": {"id": "tag:google.c...'
      requestBuilder = RequestMockBuilder(
        {
          'plus.activities.get': (None, response),
        }
      )
      googleapiclient.discovery.build("plus", "v1", requestBuilder=requestBuilder)

    Methods that you do not supply a response for will return a
    200 OK with an empty string as the response content or raise an excpetion
    if check_unexpected is set to True. The methodId is taken from the rpcName
    in the discovery document.

    For more details see the project wiki.
    """

    def __init__(self, responses, check_unexpected=False):
        """Constructor for RequestMockBuilder

        The constructed object should be a callable object
        that can replace the class HttpResponse.

        responses - A dictionary that maps methodIds into tuples
                    of (httplib2.Response, content). The methodId
                    comes from the 'rpcName' field in the discovery
                    document.
        check_unexpected - A boolean setting whether or not UnexpectedMethodError
                           should be raised on unsupplied method.
        """
        self.responses = responses
        self.check_unexpected = check_unexpected

    def __call__(
        self,
        http,
        postproc,
        uri,
        method="GET",
        body=None,
        headers=None,
        methodId=None,
        resumable=None,
    ):
        """Implements the callable interface that discovery.build() expects
        of requestBuilder, which is to build an object compatible with
        HttpRequest.execute(). See that method for the description of the
        parameters and the expected response.
        """
        if methodId in self.responses:
            response = self.responses[methodId]
            resp, content = response[:2]
            if len(response) > 2:
                # Test the body against the supplied expected_body.
                expected_body = response[2]
                if bool(expected_body) != bool(body):
                    # Not expecting a body and provided one
                    # or expecting a body and not provided one.
                    raise UnexpectedBodyError(expected_body, body)
                if isinstance(expected_body, str):
                    expected_body = json.loads(expected_body)
                body = json.loads(body)
                if body != expected_body:
                    raise UnexpectedBodyError(expected_body, body)
            return HttpRequestMock(resp, content, postproc)
        elif self.check_unexpected:
            raise UnexpectedMethodError(methodId=methodId)
        else:
            model = JsonModel(False)
            return HttpRequestMock(None, "{}", model.response)


class HttpMock(object):
    """Mock of httplib2.Http"""

    def __init__(self, filename=None, headers=None):
        """
        Args:
          filename: string, absolute filename to read response from
          headers: dict, header to return with response
        """
        if headers is None:
            headers = {"status": "200"}
        if filename:
            with open(filename, "rb") as f:
                self.data = f.read()
        else:
            self.data = None
        self.response_headers = headers
        self.headers = None
        self.uri = None
        self.method = None
        self.body = None
        self.headers = None

    def request(
        self,
        uri,
        method="GET",
        body=None,
        headers=None,
        redirections=1,
        connection_type=None,
    ):
        self.uri = uri
        self.method = method
        self.body = body
        self.headers = headers
        return httplib2.Response(self.response_headers), self.data

    def close(self):
        return None


class HttpMockSequence(object):
    """Mock of httplib2.Http

    Mocks a sequence of calls to request returning different responses for each
    call. Create an instance initialized with the desired response headers
    and content and then use as if an httplib2.Http instance.

      http = HttpMockSequence([
        ({'status': '401'}, ''),
        ({'status': '200'}, '{"access_token":"1/3w","expires_in":3600}'),
        ({'status': '200'}, 'echo_request_headers'),
        ])
      resp, content = http.request("http://examples.com")

    There are special values you can pass in for content to trigger
    behavours that are helpful in testing.

    'echo_request_headers' means return the request headers in the response body
    'echo_request_headers_as_json' means return the request headers in
       the response body
    'echo_request_body' means return the request body in the response body
    'echo_request_uri' means return the request uri in the response body
    """

    def __init__(self, iterable):
        """
        Args:
          iterable: iterable, a sequence of pairs of (headers, body)
        """
        self._iterable = iterable
        self.follow_redirects = True
        self.request_sequence = list()

    def request(
        self,
        uri,
        method="GET",
        body=None,
        headers=None,
        redirections=1,
        connection_type=None,
    ):
        # Remember the request so after the fact this mock can be examined
        self.request_sequence.append((uri, method, body, headers))
        resp, content = self._iterable.pop(0)
        if isinstance(content, str):
            content = content.encode("utf-8")

        if content == b"echo_request_headers":
            content = headers
        elif content == b"echo_request_headers_as_json":
            content = json.dumps(headers)
        elif content == b"echo_request_body":
            if hasattr(body, "read"):
                content = body.read()
            else:
                content = body
        elif content == b"echo_request_uri":
            content = uri
        if isinstance(content, str):
            content = content.encode("utf-8")
        return httplib2.Response(resp), content


def set_user_agent(http, user_agent):
    """Set the user-agent on every request.

    Args:
       http - An instance of httplib2.Http
           or something that acts like it.
       user_agent: string, the value for the user-agent header.

    Returns:
       A modified instance of http that was passed in.

    Example:

      h = httplib2.Http()
      h = set_user_agent(h, "my-app-name/6.0")

    Most of the time the user-agent will be set doing auth, this is for the rare
    cases where you are accessing an unauthenticated endpoint.
    """
    request_orig = http.request

    # The closure that will replace 'httplib2.Http.request'.
    def new_request(
        uri,
        method="GET",
        body=None,
        headers=None,
        redirections=httplib2.DEFAULT_MAX_REDIRECTS,
        connection_type=None,
    ):
        """Modify the request headers to add the user-agent."""
        if headers is None:
            headers = {}
        if "user-agent" in headers:
            headers["user-agent"] = user_agent + " " + headers["user-agent"]
        else:
            headers["user-agent"] = user_agent
        resp, content = request_orig(
            uri,
            method=method,
            body=body,
            headers=headers,
            redirections=redirections,
            connection_type=connection_type,
        )
        return resp, content

    http.request = new_request
    return http


def tunnel_patch(http):
    """Tunnel PATCH requests over POST.
    Args:
       http - An instance of httplib2.Http
           or something that acts like it.

    Returns:
       A modified instance of http that was passed in.

    Example:

      h = httplib2.Http()
      h = tunnel_patch(h, "my-app-name/6.0")

    Useful if you are running on a platform that doesn't support PATCH.
    Apply this last if you are using OAuth 1.0, as changing the method
    will result in a different signature.
    """
    request_orig = http.request

    # The closure that will replace 'httplib2.Http.request'.
    def new_request(
        uri,
        method="GET",
        body=None,
        headers=None,
        redirections=httplib2.DEFAULT_MAX_REDIRECTS,
        connection_type=None,
    ):
        """Modify the request headers to add the user-agent."""
        if headers is None:
            headers = {}
        if method == "PATCH":
            if "oauth_token" in headers.get("authorization", ""):
                LOGGER.warning(
                    "OAuth 1.0 request made with Credentials after tunnel_patch."
                )
            headers["x-http-method-override"] = "PATCH"
            method = "POST"
        resp, content = request_orig(
            uri,
            method=method,
            body=body,
            headers=headers,
            redirections=redirections,
            connection_type=connection_type,
        )
        return resp, content

    http.request = new_request
    return http


def build_http():
    """Builds httplib2.Http object

    Returns:
    A httplib2.Http object, which is used to make http requests, and which has timeout set by default.
    To override default timeout call

      socket.setdefaulttimeout(timeout_in_sec)

    before interacting with this method.
    """
    if socket.getdefaulttimeout() is not None:
        http_timeout = socket.getdefaulttimeout()
    else:
        http_timeout = DEFAULT_HTTP_TIMEOUT_SEC
    http = httplib2.Http(timeout=http_timeout)
    # 308's are used by several Google APIs (Drive, YouTube)
    # for Resumable Uploads rather than Permanent Redirects.
    # This asks httplib2 to exclude 308s from the status codes
    # it treats as redirects
    try:
        http.redirect_codes = http.redirect_codes - {308}
    except AttributeError:
        # Apache Beam tests depend on this library and cannot
        # currently upgrade their httplib2 version
        # http.redirect_codes does not exist in previous versions
        # of httplib2, so pass
        pass

    return http
