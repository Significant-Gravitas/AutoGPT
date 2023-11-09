# Copyright 2020 Google LLC
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
"""AsyncIO helpers for wrapping gRPC methods with common functionality.

This is used by gapic clients to provide common error mapping, retry, timeout,
pagination, and long-running operations to gRPC methods.
"""

import functools

from google.api_core import grpc_helpers_async
from google.api_core.gapic_v1 import client_info
from google.api_core.gapic_v1.method import _GapicCallable
from google.api_core.gapic_v1.method import DEFAULT  # noqa: F401
from google.api_core.gapic_v1.method import USE_DEFAULT_METADATA  # noqa: F401


def wrap_method(
    func,
    default_retry=None,
    default_timeout=None,
    client_info=client_info.DEFAULT_CLIENT_INFO,
):
    """Wrap an async RPC method with common behavior.

    Returns:
        Callable: A new callable that takes optional ``retry`` and ``timeout``
            arguments and applies the common error mapping, retry, timeout,
            and metadata behavior to the low-level RPC method.
    """
    func = grpc_helpers_async.wrap_errors(func)

    metadata = [client_info.to_grpc_metadata()] if client_info is not None else None

    return functools.wraps(func)(
        _GapicCallable(func, default_retry, default_timeout, metadata=metadata)
    )
