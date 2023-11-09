# Copyright 2022 Google LLC
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

import random
import time

import six

# The default amount of retry attempts
_DEFAULT_RETRY_TOTAL_ATTEMPTS = 3

# The default initial backoff period (1.0 second).
_DEFAULT_INITIAL_INTERVAL_SECONDS = 1.0

# The default randomization factor (0.1 which results in a random period ranging
# between 10% below and 10% above the retry interval).
_DEFAULT_RANDOMIZATION_FACTOR = 0.1

# The default multiplier value (2 which is 100% increase per back off).
_DEFAULT_MULTIPLIER = 2.0

"""Exponential Backoff Utility

This is a private module that implements the exponential back off algorithm.
It can be used as a utility for code that needs to retry on failure, for example
an HTTP request.
"""


class ExponentialBackoff(six.Iterator):
    """An exponential backoff iterator. This can be used in a for loop to
    perform requests with exponential backoff.

    Args:
        total_attempts Optional[int]:
            The maximum amount of retries that should happen.
            The default value is 3 attempts.
        initial_wait_seconds Optional[int]:
            The amount of time to sleep in the first backoff. This parameter
            should be in seconds.
            The default value is 1 second.
        randomization_factor Optional[float]:
            The amount of jitter that should be in each backoff. For example,
            a value of 0.1 will introduce a jitter range of 10% to the
            current backoff period.
            The default value is 0.1.
        multiplier Optional[float]:
            The backoff multipler. This adjusts how much each backoff will
            increase. For example a value of 2.0 leads to a 200% backoff
            on each attempt. If the initial_wait is 1.0 it would look like
            this sequence [1.0, 2.0, 4.0, 8.0].
            The default value is 2.0.
    """

    def __init__(
        self,
        total_attempts=_DEFAULT_RETRY_TOTAL_ATTEMPTS,
        initial_wait_seconds=_DEFAULT_INITIAL_INTERVAL_SECONDS,
        randomization_factor=_DEFAULT_RANDOMIZATION_FACTOR,
        multiplier=_DEFAULT_MULTIPLIER,
    ):
        self._total_attempts = total_attempts
        self._initial_wait_seconds = initial_wait_seconds

        self._current_wait_in_seconds = self._initial_wait_seconds

        self._randomization_factor = randomization_factor
        self._multiplier = multiplier
        self._backoff_count = 0

    def __iter__(self):
        self._backoff_count = 0
        self._current_wait_in_seconds = self._initial_wait_seconds
        return self

    def __next__(self):
        if self._backoff_count >= self._total_attempts:
            raise StopIteration
        self._backoff_count += 1

        jitter_variance = self._current_wait_in_seconds * self._randomization_factor
        jitter = random.uniform(
            self._current_wait_in_seconds - jitter_variance,
            self._current_wait_in_seconds + jitter_variance,
        )

        time.sleep(jitter)

        self._current_wait_in_seconds *= self._multiplier
        return self._backoff_count

    @property
    def total_attempts(self):
        """The total amount of backoff attempts that will be made."""
        return self._total_attempts

    @property
    def backoff_count(self):
        """The current amount of backoff attempts that have been made."""
        return self._backoff_count
