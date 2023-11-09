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

"""Private helpers for futures."""

import logging
import threading


_LOGGER = logging.getLogger(__name__)


def start_daemon_thread(*args, **kwargs):
    """Starts a thread and marks it as a daemon thread."""
    thread = threading.Thread(*args, **kwargs)
    thread.daemon = True
    thread.start()
    return thread


def safe_invoke_callback(callback, *args, **kwargs):
    """Invoke a callback, swallowing and logging any exceptions."""
    # pylint: disable=bare-except
    # We intentionally want to swallow all exceptions.
    try:
        return callback(*args, **kwargs)
    except Exception:
        _LOGGER.exception("Error while executing Future callback.")
