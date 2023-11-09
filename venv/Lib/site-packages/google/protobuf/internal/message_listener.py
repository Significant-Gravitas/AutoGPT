# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Defines a listener interface for observing certain
state transitions on Message objects.

Also defines a null implementation of this interface.
"""

__author__ = 'robinson@google.com (Will Robinson)'


class MessageListener(object):

  """Listens for modifications made to a message.  Meant to be registered via
  Message._SetListener().

  Attributes:
    dirty:  If True, then calling Modified() would be a no-op.  This can be
            used to avoid these calls entirely in the common case.
  """

  def Modified(self):
    """Called every time the message is modified in such a way that the parent
    message may need to be updated.  This currently means either:
    (a) The message was modified for the first time, so the parent message
        should henceforth mark the message as present.
    (b) The message's cached byte size became dirty -- i.e. the message was
        modified for the first time after a previous call to ByteSize().
        Therefore the parent should also mark its byte size as dirty.
    Note that (a) implies (b), since new objects start out with a client cached
    size (zero).  However, we document (a) explicitly because it is important.

    Modified() will *only* be called in response to one of these two events --
    not every time the sub-message is modified.

    Note that if the listener's |dirty| attribute is true, then calling
    Modified at the moment would be a no-op, so it can be skipped.  Performance-
    sensitive callers should check this attribute directly before calling since
    it will be true most of the time.
    """

    raise NotImplementedError


class NullMessageListener(object):

  """No-op MessageListener implementation."""

  def Modified(self):
    pass
