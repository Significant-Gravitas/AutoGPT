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

"""Channel notifications support.

Classes and functions to support channel subscriptions and notifications
on those channels.

Notes:
  - This code is based on experimental APIs and is subject to change.
  - Notification does not do deduplication of notification ids, that's up to
    the receiver.
  - Storing the Channel between calls is up to the caller.


Example setting up a channel:

  # Create a new channel that gets notifications via webhook.
  channel = new_webhook_channel("https://example.com/my_web_hook")

  # Store the channel, keyed by 'channel.id'. Store it before calling the
  # watch method because notifications may start arriving before the watch
  # method returns.
  ...

  resp = service.objects().watchAll(
    bucket="some_bucket_id", body=channel.body()).execute()
  channel.update(resp)

  # Store the channel, keyed by 'channel.id'. Store it after being updated
  # since the resource_id value will now be correct, and that's needed to
  # stop a subscription.
  ...


An example Webhook implementation using webapp2. Note that webapp2 puts
headers in a case insensitive dictionary, as headers aren't guaranteed to
always be upper case.

  id = self.request.headers[X_GOOG_CHANNEL_ID]

  # Retrieve the channel by id.
  channel = ...

  # Parse notification from the headers, including validating the id.
  n = notification_from_headers(channel, self.request.headers)

  # Do app specific stuff with the notification here.
  if n.resource_state == 'sync':
    # Code to handle sync state.
  elif n.resource_state == 'exists':
    # Code to handle the exists state.
  elif n.resource_state == 'not_exists':
    # Code to handle the not exists state.


Example of unsubscribing.

  service.channels().stop(channel.body()).execute()
"""
from __future__ import absolute_import

import datetime
import uuid

from googleapiclient import _helpers as util
from googleapiclient import errors

# The unix time epoch starts at midnight 1970.
EPOCH = datetime.datetime.utcfromtimestamp(0)

# Map the names of the parameters in the JSON channel description to
# the parameter names we use in the Channel class.
CHANNEL_PARAMS = {
    "address": "address",
    "id": "id",
    "expiration": "expiration",
    "params": "params",
    "resourceId": "resource_id",
    "resourceUri": "resource_uri",
    "type": "type",
    "token": "token",
}

X_GOOG_CHANNEL_ID = "X-GOOG-CHANNEL-ID"
X_GOOG_MESSAGE_NUMBER = "X-GOOG-MESSAGE-NUMBER"
X_GOOG_RESOURCE_STATE = "X-GOOG-RESOURCE-STATE"
X_GOOG_RESOURCE_URI = "X-GOOG-RESOURCE-URI"
X_GOOG_RESOURCE_ID = "X-GOOG-RESOURCE-ID"


def _upper_header_keys(headers):
    new_headers = {}
    for k, v in headers.items():
        new_headers[k.upper()] = v
    return new_headers


class Notification(object):
    """A Notification from a Channel.

    Notifications are not usually constructed directly, but are returned
    from functions like notification_from_headers().

    Attributes:
      message_number: int, The unique id number of this notification.
      state: str, The state of the resource being monitored.
      uri: str, The address of the resource being monitored.
      resource_id: str, The unique identifier of the version of the resource at
        this event.
    """

    @util.positional(5)
    def __init__(self, message_number, state, resource_uri, resource_id):
        """Notification constructor.

        Args:
          message_number: int, The unique id number of this notification.
          state: str, The state of the resource being monitored. Can be one
            of "exists", "not_exists", or "sync".
          resource_uri: str, The address of the resource being monitored.
          resource_id: str, The identifier of the watched resource.
        """
        self.message_number = message_number
        self.state = state
        self.resource_uri = resource_uri
        self.resource_id = resource_id


class Channel(object):
    """A Channel for notifications.

    Usually not constructed directly, instead it is returned from helper
    functions like new_webhook_channel().

    Attributes:
      type: str, The type of delivery mechanism used by this channel. For
        example, 'web_hook'.
      id: str, A UUID for the channel.
      token: str, An arbitrary string associated with the channel that
        is delivered to the target address with each event delivered
        over this channel.
      address: str, The address of the receiving entity where events are
        delivered. Specific to the channel type.
      expiration: int, The time, in milliseconds from the epoch, when this
        channel will expire.
      params: dict, A dictionary of string to string, with additional parameters
        controlling delivery channel behavior.
      resource_id: str, An opaque id that identifies the resource that is
        being watched. Stable across different API versions.
      resource_uri: str, The canonicalized ID of the watched resource.
    """

    @util.positional(5)
    def __init__(
        self,
        type,
        id,
        token,
        address,
        expiration=None,
        params=None,
        resource_id="",
        resource_uri="",
    ):
        """Create a new Channel.

        In user code, this Channel constructor will not typically be called
        manually since there are functions for creating channels for each specific
        type with a more customized set of arguments to pass.

        Args:
          type: str, The type of delivery mechanism used by this channel. For
            example, 'web_hook'.
          id: str, A UUID for the channel.
          token: str, An arbitrary string associated with the channel that
            is delivered to the target address with each event delivered
            over this channel.
          address: str,  The address of the receiving entity where events are
            delivered. Specific to the channel type.
          expiration: int, The time, in milliseconds from the epoch, when this
            channel will expire.
          params: dict, A dictionary of string to string, with additional parameters
            controlling delivery channel behavior.
          resource_id: str, An opaque id that identifies the resource that is
            being watched. Stable across different API versions.
          resource_uri: str, The canonicalized ID of the watched resource.
        """
        self.type = type
        self.id = id
        self.token = token
        self.address = address
        self.expiration = expiration
        self.params = params
        self.resource_id = resource_id
        self.resource_uri = resource_uri

    def body(self):
        """Build a body from the Channel.

        Constructs a dictionary that's appropriate for passing into watch()
        methods as the value of body argument.

        Returns:
          A dictionary representation of the channel.
        """
        result = {
            "id": self.id,
            "token": self.token,
            "type": self.type,
            "address": self.address,
        }
        if self.params:
            result["params"] = self.params
        if self.resource_id:
            result["resourceId"] = self.resource_id
        if self.resource_uri:
            result["resourceUri"] = self.resource_uri
        if self.expiration:
            result["expiration"] = self.expiration

        return result

    def update(self, resp):
        """Update a channel with information from the response of watch().

        When a request is sent to watch() a resource, the response returned
        from the watch() request is a dictionary with updated channel information,
        such as the resource_id, which is needed when stopping a subscription.

        Args:
          resp: dict, The response from a watch() method.
        """
        for json_name, param_name in CHANNEL_PARAMS.items():
            value = resp.get(json_name)
            if value is not None:
                setattr(self, param_name, value)


def notification_from_headers(channel, headers):
    """Parse a notification from the webhook request headers, validate
      the notification, and return a Notification object.

    Args:
      channel: Channel, The channel that the notification is associated with.
      headers: dict, A dictionary like object that contains the request headers
        from the webhook HTTP request.

    Returns:
      A Notification object.

    Raises:
      errors.InvalidNotificationError if the notification is invalid.
      ValueError if the X-GOOG-MESSAGE-NUMBER can't be converted to an int.
    """
    headers = _upper_header_keys(headers)
    channel_id = headers[X_GOOG_CHANNEL_ID]
    if channel.id != channel_id:
        raise errors.InvalidNotificationError(
            "Channel id mismatch: %s != %s" % (channel.id, channel_id)
        )
    else:
        message_number = int(headers[X_GOOG_MESSAGE_NUMBER])
        state = headers[X_GOOG_RESOURCE_STATE]
        resource_uri = headers[X_GOOG_RESOURCE_URI]
        resource_id = headers[X_GOOG_RESOURCE_ID]
        return Notification(message_number, state, resource_uri, resource_id)


@util.positional(2)
def new_webhook_channel(url, token=None, expiration=None, params=None):
    """Create a new webhook Channel.

    Args:
      url: str, URL to post notifications to.
      token: str, An arbitrary string associated with the channel that
        is delivered to the target address with each notification delivered
        over this channel.
      expiration: datetime.datetime, A time in the future when the channel
        should expire. Can also be None if the subscription should use the
        default expiration. Note that different services may have different
        limits on how long a subscription lasts. Check the response from the
        watch() method to see the value the service has set for an expiration
        time.
      params: dict, Extra parameters to pass on channel creation. Currently
        not used for webhook channels.
    """
    expiration_ms = 0
    if expiration:
        delta = expiration - EPOCH
        expiration_ms = (
            delta.microseconds / 1000 + (delta.seconds + delta.days * 24 * 3600) * 1000
        )
        if expiration_ms < 0:
            expiration_ms = 0

    return Channel(
        "web_hook",
        str(uuid.uuid4()),
        token,
        url,
        expiration=expiration_ms,
        params=params,
    )
