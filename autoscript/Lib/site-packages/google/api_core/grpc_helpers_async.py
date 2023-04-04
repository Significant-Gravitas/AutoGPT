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

"""AsyncIO helpers for :mod:`grpc` supporting 3.7+.

Please combine more detailed docstring in grpc_helpers.py to use following
functions. This module is implementing the same surface with AsyncIO semantics.
"""

import asyncio
import functools

import grpc
from grpc import aio

from google.api_core import exceptions, grpc_helpers


# NOTE(lidiz) Alternatively, we can hack "__getattribute__" to perform
# automatic patching for us. But that means the overhead of creating an
# extra Python function spreads to every single send and receive.


class _WrappedCall(aio.Call):
    def __init__(self):
        self._call = None

    def with_call(self, call):
        """Supplies the call object separately to keep __init__ clean."""
        self._call = call
        return self

    async def initial_metadata(self):
        return await self._call.initial_metadata()

    async def trailing_metadata(self):
        return await self._call.trailing_metadata()

    async def code(self):
        return await self._call.code()

    async def details(self):
        return await self._call.details()

    def cancelled(self):
        return self._call.cancelled()

    def done(self):
        return self._call.done()

    def time_remaining(self):
        return self._call.time_remaining()

    def cancel(self):
        return self._call.cancel()

    def add_done_callback(self, callback):
        self._call.add_done_callback(callback)

    async def wait_for_connection(self):
        try:
            await self._call.wait_for_connection()
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error


class _WrappedUnaryResponseMixin(_WrappedCall):
    def __await__(self):
        try:
            response = yield from self._call.__await__()
            return response
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error


class _WrappedStreamResponseMixin(_WrappedCall):
    def __init__(self):
        self._wrapped_async_generator = None

    async def read(self):
        try:
            return await self._call.read()
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error

    async def _wrapped_aiter(self):
        try:
            # NOTE(lidiz) coverage doesn't understand the exception raised from
            # __anext__ method. It is covered by test case:
            #     test_wrap_stream_errors_aiter_non_rpc_error
            async for response in self._call:  # pragma: no branch
                yield response
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error

    def __aiter__(self):
        if not self._wrapped_async_generator:
            self._wrapped_async_generator = self._wrapped_aiter()
        return self._wrapped_async_generator


class _WrappedStreamRequestMixin(_WrappedCall):
    async def write(self, request):
        try:
            await self._call.write(request)
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error

    async def done_writing(self):
        try:
            await self._call.done_writing()
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error


# NOTE(lidiz) Implementing each individual class separately, so we don't
# expose any API that should not be seen. E.g., __aiter__ in unary-unary
# RPC, or __await__ in stream-stream RPC.
class _WrappedUnaryUnaryCall(_WrappedUnaryResponseMixin, aio.UnaryUnaryCall):
    """Wrapped UnaryUnaryCall to map exceptions."""


class _WrappedUnaryStreamCall(_WrappedStreamResponseMixin, aio.UnaryStreamCall):
    """Wrapped UnaryStreamCall to map exceptions."""


class _WrappedStreamUnaryCall(
    _WrappedUnaryResponseMixin, _WrappedStreamRequestMixin, aio.StreamUnaryCall
):
    """Wrapped StreamUnaryCall to map exceptions."""


class _WrappedStreamStreamCall(
    _WrappedStreamRequestMixin, _WrappedStreamResponseMixin, aio.StreamStreamCall
):
    """Wrapped StreamStreamCall to map exceptions."""


def _wrap_unary_errors(callable_):
    """Map errors for Unary-Unary async callables."""
    grpc_helpers._patch_callable_name(callable_)

    @functools.wraps(callable_)
    def error_remapped_callable(*args, **kwargs):
        call = callable_(*args, **kwargs)
        return _WrappedUnaryUnaryCall().with_call(call)

    return error_remapped_callable


def _wrap_stream_errors(callable_):
    """Map errors for streaming RPC async callables."""
    grpc_helpers._patch_callable_name(callable_)

    @functools.wraps(callable_)
    async def error_remapped_callable(*args, **kwargs):
        call = callable_(*args, **kwargs)

        if isinstance(call, aio.UnaryStreamCall):
            call = _WrappedUnaryStreamCall().with_call(call)
        elif isinstance(call, aio.StreamUnaryCall):
            call = _WrappedStreamUnaryCall().with_call(call)
        elif isinstance(call, aio.StreamStreamCall):
            call = _WrappedStreamStreamCall().with_call(call)
        else:
            raise TypeError("Unexpected type of call %s" % type(call))

        await call.wait_for_connection()
        return call

    return error_remapped_callable


def wrap_errors(callable_):
    """Wrap a gRPC async callable and map :class:`grpc.RpcErrors` to
    friendly error classes.

    Errors raised by the gRPC callable are mapped to the appropriate
    :class:`google.api_core.exceptions.GoogleAPICallError` subclasses. The
    original `grpc.RpcError` (which is usually also a `grpc.Call`) is
    available from the ``response`` property on the mapped exception. This
    is useful for extracting metadata from the original error.

    Args:
        callable_ (Callable): A gRPC callable.

    Returns: Callable: The wrapped gRPC callable.
    """
    if isinstance(callable_, aio.UnaryUnaryMultiCallable):
        return _wrap_unary_errors(callable_)
    else:
        return _wrap_stream_errors(callable_)


def create_channel(
    target,
    credentials=None,
    scopes=None,
    ssl_credentials=None,
    credentials_file=None,
    quota_project_id=None,
    default_scopes=None,
    default_host=None,
    **kwargs
):
    """Create an AsyncIO secure channel with credentials.

    Args:
        target (str): The target service address in the format 'hostname:port'.
        credentials (google.auth.credentials.Credentials): The credentials. If
            not specified, then this function will attempt to ascertain the
            credentials from the environment using :func:`google.auth.default`.
        scopes (Sequence[str]): A optional list of scopes needed for this
            service. These are only used when credentials are not specified and
            are passed to :func:`google.auth.default`.
        ssl_credentials (grpc.ChannelCredentials): Optional SSL channel
            credentials. This can be used to specify different certificates.
        credentials_file (str): A file with credentials that can be loaded with
            :func:`google.auth.load_credentials_from_file`. This argument is
            mutually exclusive with credentials.
        quota_project_id (str): An optional project to use for billing and quota.
        default_scopes (Sequence[str]): Default scopes passed by a Google client
            library. Use 'scopes' for user-defined scopes.
        default_host (str): The default endpoint. e.g., "pubsub.googleapis.com".
        kwargs: Additional key-word args passed to :func:`aio.secure_channel`.

    Returns:
        aio.Channel: The created channel.

    Raises:
        google.api_core.DuplicateCredentialArgs: If both a credentials object and credentials_file are passed.
    """

    composite_credentials = grpc_helpers._create_composite_credentials(
        credentials=credentials,
        credentials_file=credentials_file,
        scopes=scopes,
        default_scopes=default_scopes,
        ssl_credentials=ssl_credentials,
        quota_project_id=quota_project_id,
        default_host=default_host,
    )

    return aio.secure_channel(target, composite_credentials, **kwargs)


class FakeUnaryUnaryCall(_WrappedUnaryUnaryCall):
    """Fake implementation for unary-unary RPCs.

    It is a dummy object for response message. Supply the intended response
    upon the initialization, and the coroutine will return the exact response
    message.
    """

    def __init__(self, response=object()):
        self.response = response
        self._future = asyncio.get_event_loop().create_future()
        self._future.set_result(self.response)

    def __await__(self):
        response = yield from self._future.__await__()
        return response


class FakeStreamUnaryCall(_WrappedStreamUnaryCall):
    """Fake implementation for stream-unary RPCs.

    It is a dummy object for response message. Supply the intended response
    upon the initialization, and the coroutine will return the exact response
    message.
    """

    def __init__(self, response=object()):
        self.response = response
        self._future = asyncio.get_event_loop().create_future()
        self._future.set_result(self.response)

    def __await__(self):
        response = yield from self._future.__await__()
        return response

    async def wait_for_connection(self):
        pass
