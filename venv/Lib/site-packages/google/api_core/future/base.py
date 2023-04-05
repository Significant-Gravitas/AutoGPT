# Copyright 2017, Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract and helper bases for Future implementations."""

import abc


class Future(object, metaclass=abc.ABCMeta):
    # pylint: disable=missing-docstring
    # We inherit the interfaces here from concurrent.futures.

    """Future interface.

    This interface is based on :class:`concurrent.futures.Future`.
    """

    @abc.abstractmethod
    def cancel(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def cancelled(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def running(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def done(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def result(self, timeout=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def exception(self, timeout=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def add_done_callback(self, fn):
        # pylint: disable=invalid-name
        raise NotImplementedError()

    @abc.abstractmethod
    def set_result(self, result):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_exception(self, exception):
        raise NotImplementedError()
