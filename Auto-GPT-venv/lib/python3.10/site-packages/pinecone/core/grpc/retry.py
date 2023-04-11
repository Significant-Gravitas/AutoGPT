#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import abc
import logging
import random
import time
from typing import Optional, Tuple, NamedTuple

import grpc


_logger = logging.getLogger(__name__)


class SleepPolicy(abc.ABC):
    @abc.abstractmethod
    def sleep(self, try_i: int):
        """
        How long to sleep in milliseconds.
        :param try_i: the number of retry (starting from zero)
        """
        assert try_i >= 0


class ExponentialBackoff(SleepPolicy):
    def __init__(self, *, init_backoff_ms: int, max_backoff_ms: int, multiplier: int):
        self.init_backoff = random.randint(0, init_backoff_ms)
        self.max_backoff = max_backoff_ms
        self.multiplier = multiplier

    def sleep(self, try_i: int):
        sleep_range = min(self.init_backoff * self.multiplier ** try_i, self.max_backoff)
        sleep_ms = random.randint(0, sleep_range)
        _logger.debug(f"gRPC retry. Sleeping for {sleep_ms}ms")
        time.sleep(sleep_ms / 1000)


class RetryOnRpcErrorClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """gRPC retry.

    Referece: https://github.com/grpc/grpc/issues/19514#issuecomment-531700657
    """

    def __init__(self, retry_config: 'RetryConfig'):
        self.max_attempts = retry_config.max_attempts
        self.sleep_policy = retry_config.sleep_policy
        self.retryable_status = retry_config.retryable_status

    def _is_retryable_error(self, response_or_error):
        """Determine if a response is a retryable error."""
        return (
                isinstance(response_or_error, grpc.RpcError)
                and "_MultiThreadedRendezvous" not in response_or_error.__class__.__name__
                and response_or_error.code() in self.retryable_status
        )

    def _intercept_call(self, continuation, client_call_details, request_or_iterator):
        response = None
        for try_i in range(self.max_attempts):
            response = continuation(client_call_details, request_or_iterator)
            if not self._is_retryable_error(response):
                break
            self.sleep_policy.sleep(try_i)
        return response

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return self._intercept_call(continuation, client_call_details, request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return self._intercept_call(continuation, client_call_details, request_iterator)


class RetryConfig(NamedTuple):
    """Config settings related to retry"""

    max_attempts: int = 4
    sleep_policy: SleepPolicy = ExponentialBackoff(init_backoff_ms=100, max_backoff_ms=1600, multiplier=2)
    retryable_status: Optional[Tuple[grpc.StatusCode, ...]] = (grpc.StatusCode.UNAVAILABLE,)
