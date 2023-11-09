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

"""File based cache for the discovery document.

The cache is stored in a single file so that multiple processes can
share the same cache. It locks the file whenever accessing to the
file. When the cache content is corrupted, it will be initialized with
an empty cache.
"""

from __future__ import division

import datetime
import json
import logging
import os
import tempfile

try:
    from oauth2client.contrib.locked_file import LockedFile
except ImportError:
    # oauth2client < 2.0.0
    try:
        from oauth2client.locked_file import LockedFile
    except ImportError:
        # oauth2client > 4.0.0 or google-auth
        raise ImportError(
            "file_cache is unavailable when using oauth2client >= 4.0.0 or google-auth"
        )

from . import base
from ..discovery_cache import DISCOVERY_DOC_MAX_AGE

LOGGER = logging.getLogger(__name__)

FILENAME = "google-api-python-client-discovery-doc.cache"
EPOCH = datetime.datetime.utcfromtimestamp(0)


def _to_timestamp(date):
    try:
        return (date - EPOCH).total_seconds()
    except AttributeError:
        # The following is the equivalent of total_seconds() in Python2.6.
        # See also: https://docs.python.org/2/library/datetime.html
        delta = date - EPOCH
        return (
            delta.microseconds + (delta.seconds + delta.days * 24 * 3600) * 10**6
        ) / 10**6


def _read_or_initialize_cache(f):
    f.file_handle().seek(0)
    try:
        cache = json.load(f.file_handle())
    except Exception:
        # This means it opens the file for the first time, or the cache is
        # corrupted, so initializing the file with an empty dict.
        cache = {}
        f.file_handle().truncate(0)
        f.file_handle().seek(0)
        json.dump(cache, f.file_handle())
    return cache


class Cache(base.Cache):
    """A file based cache for the discovery documents."""

    def __init__(self, max_age):
        """Constructor.

        Args:
          max_age: Cache expiration in seconds.
        """
        self._max_age = max_age
        self._file = os.path.join(tempfile.gettempdir(), FILENAME)
        f = LockedFile(self._file, "a+", "r")
        try:
            f.open_and_lock()
            if f.is_locked():
                _read_or_initialize_cache(f)
            # If we can not obtain the lock, other process or thread must
            # have initialized the file.
        except Exception as e:
            LOGGER.warning(e, exc_info=True)
        finally:
            f.unlock_and_close()

    def get(self, url):
        f = LockedFile(self._file, "r+", "r")
        try:
            f.open_and_lock()
            if f.is_locked():
                cache = _read_or_initialize_cache(f)
                if url in cache:
                    content, t = cache.get(url, (None, 0))
                    if _to_timestamp(datetime.datetime.now()) < t + self._max_age:
                        return content
                return None
            else:
                LOGGER.debug("Could not obtain a lock for the cache file.")
                return None
        except Exception as e:
            LOGGER.warning(e, exc_info=True)
        finally:
            f.unlock_and_close()

    def set(self, url, content):
        f = LockedFile(self._file, "r+", "r")
        try:
            f.open_and_lock()
            if f.is_locked():
                cache = _read_or_initialize_cache(f)
                cache[url] = (content, _to_timestamp(datetime.datetime.now()))
                # Remove stale cache.
                for k, (_, timestamp) in list(cache.items()):
                    if (
                        _to_timestamp(datetime.datetime.now())
                        >= timestamp + self._max_age
                    ):
                        del cache[k]
                f.file_handle().truncate(0)
                f.file_handle().seek(0)
                json.dump(cache, f.file_handle())
            else:
                LOGGER.debug("Could not obtain a lock for the cache file.")
        except Exception as e:
            LOGGER.warning(e, exc_info=True)
        finally:
            f.unlock_and_close()


cache = Cache(max_age=DISCOVERY_DOC_MAX_AGE)
