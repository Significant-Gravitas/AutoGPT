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

"""An abstract class for caching the discovery document."""

import abc


class Cache(object):
    """A base abstract cache class."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get(self, url):
        """Gets the content from the memcache with a given key.

        Args:
          url: string, the key for the cache.

        Returns:
          object, the value in the cache for the given key, or None if the key is
          not in the cache.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(self, url, content):
        """Sets the given key and content in the cache.

        Args:
          url: string, the key for the cache.
          content: string, the discovery document.
        """
        raise NotImplementedError()
