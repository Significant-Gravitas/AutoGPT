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

"""Errors for the library.

All exceptions defined by the library
should be defined in this file.
"""
from __future__ import absolute_import

__author__ = "jcgregorio@google.com (Joe Gregorio)"

import json

from googleapiclient import _helpers as util


class Error(Exception):
    """Base error for this module."""

    pass


class HttpError(Error):
    """HTTP data was invalid or unexpected."""

    @util.positional(3)
    def __init__(self, resp, content, uri=None):
        self.resp = resp
        if not isinstance(content, bytes):
            raise TypeError("HTTP content should be bytes")
        self.content = content
        self.uri = uri
        self.error_details = ""
        self.reason = self._get_reason()

    @property
    def status_code(self):
        """Return the HTTP status code from the response content."""
        return self.resp.status

    def _get_reason(self):
        """Calculate the reason for the error from the response content."""
        reason = self.resp.reason
        try:
            try:
                data = json.loads(self.content.decode("utf-8"))
            except json.JSONDecodeError:
                # In case it is not json
                data = self.content.decode("utf-8")
            if isinstance(data, dict):
                reason = data["error"]["message"]
                error_detail_keyword = next(
                    (
                        kw
                        for kw in ["detail", "details", "errors", "message"]
                        if kw in data["error"]
                    ),
                    "",
                )
                if error_detail_keyword:
                    self.error_details = data["error"][error_detail_keyword]
            elif isinstance(data, list) and len(data) > 0:
                first_error = data[0]
                reason = first_error["error"]["message"]
                if "details" in first_error["error"]:
                    self.error_details = first_error["error"]["details"]
            else:
                self.error_details = data
        except (ValueError, KeyError, TypeError):
            pass
        if reason is None:
            reason = ""
        return reason.strip()

    def __repr__(self):
        if self.error_details:
            return '<HttpError %s when requesting %s returned "%s". Details: "%s">' % (
                self.resp.status,
                self.uri,
                self.reason,
                self.error_details,
            )
        elif self.uri:
            return '<HttpError %s when requesting %s returned "%s">' % (
                self.resp.status,
                self.uri,
                self.reason,
            )
        else:
            return '<HttpError %s "%s">' % (self.resp.status, self.reason)

    __str__ = __repr__


class InvalidJsonError(Error):
    """The JSON returned could not be parsed."""

    pass


class UnknownFileType(Error):
    """File type unknown or unexpected."""

    pass


class UnknownLinkType(Error):
    """Link type unknown or unexpected."""

    pass


class UnknownApiNameOrVersion(Error):
    """No API with that name and version exists."""

    pass


class UnacceptableMimeTypeError(Error):
    """That is an unacceptable mimetype for this operation."""

    pass


class MediaUploadSizeError(Error):
    """Media is larger than the method can accept."""

    pass


class ResumableUploadError(HttpError):
    """Error occurred during resumable upload."""

    pass


class InvalidChunkSizeError(Error):
    """The given chunksize is not valid."""

    pass


class InvalidNotificationError(Error):
    """The channel Notification is invalid."""

    pass


class BatchError(HttpError):
    """Error occurred during batch operations."""

    @util.positional(2)
    def __init__(self, reason, resp=None, content=None):
        self.resp = resp
        self.content = content
        self.reason = reason

    def __repr__(self):
        if getattr(self.resp, "status", None) is None:
            return '<BatchError "%s">' % (self.reason)
        else:
            return '<BatchError %s "%s">' % (self.resp.status, self.reason)

    __str__ = __repr__


class UnexpectedMethodError(Error):
    """Exception raised by RequestMockBuilder on unexpected calls."""

    @util.positional(1)
    def __init__(self, methodId=None):
        """Constructor for an UnexpectedMethodError."""
        super(UnexpectedMethodError, self).__init__(
            "Received unexpected call %s" % methodId
        )


class UnexpectedBodyError(Error):
    """Exception raised by RequestMockBuilder on unexpected bodies."""

    def __init__(self, expected, provided):
        """Constructor for an UnexpectedMethodError."""
        super(UnexpectedBodyError, self).__init__(
            "Expected: [%s] - Provided: [%s]" % (expected, provided)
        )
