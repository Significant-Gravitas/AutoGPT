import asyncio
import concurrent
import concurrent.futures
import inspect
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from functools import update_wrapper
from typing import (
    Any,
    Awaitable,
    Callable,
    Concatenate,
    Coroutine,
    Optional,
    ParamSpec,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import httpx
import uvicorn
from fastapi import FastAPI, Request, responses
from pydantic import BaseModel, TypeAdapter, create_model

import backend.util.exceptions as exceptions
from backend.monitoring.instrumentation import instrument_fastapi
from backend.util.json import to_dict
from backend.util.metrics import sentry_init
from backend.util.process import AppProcess, get_service_name
from backend.util.retry import conn_retry, create_retry_decorator
from backend.util.settings import Config

logger = logging.getLogger(__name__)
T = TypeVar("T")
C = TypeVar("C", bound=Callable)

config = Config()
api_host = config.pyro_host
api_comm_retry = config.pyro_client_comm_retry
api_comm_timeout = config.pyro_client_comm_timeout
api_call_timeout = config.rpc_client_call_timeout
api_comm_max_wait = config.pyro_client_max_wait


def _validate_no_prisma_objects(obj: Any, path: str = "result") -> None:
    """
    Recursively validate that no Prisma objects are being returned from service methods.
    This enforces proper separation of layers - only application models should cross service boundaries.
    """
    if obj is None:
        return

    # Check if it's a Prisma model object
    if hasattr(obj, "__class__") and hasattr(obj.__class__, "__module__"):
        module_name = obj.__class__.__module__
        if module_name and "prisma.models" in module_name:
            raise ValueError(
                f"Prisma object {obj.__class__.__name__} found in {path}. "
                "Service methods must return application models, not Prisma objects. "
                f"Use {obj.__class__.__name__}.from_db() to convert to application model."
            )

    # Recursively check collections
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _validate_no_prisma_objects(item, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _validate_no_prisma_objects(value, f"{path}['{key}']")


P = ParamSpec("P")
R = TypeVar("R")
EXPOSED_FLAG = "__exposed__"


def expose(func: C) -> C:
    func = getattr(func, "__func__", func)
    setattr(func, EXPOSED_FLAG, True)
    return func


# --------------------------------------------------
# AppService for IPC service based on HTTP request through FastAPI
# --------------------------------------------------
class BaseAppService(AppProcess, ABC):
    shared_event_loop: asyncio.AbstractEventLoop

    @classmethod
    @abstractmethod
    def get_port(cls) -> int:
        pass

    @classmethod
    def get_host(cls) -> str:
        source_host = os.environ.get(f"{get_service_name().upper()}_HOST", api_host)
        target_host = os.environ.get(f"{cls.__name__.upper()}_HOST", api_host)

        if source_host == target_host and source_host != api_host:
            logger.warning(
                f"Service {cls.__name__} is the same host as the source service."
                f"Use the localhost of {api_host} instead."
            )
            return api_host

        return target_host

    def run_service(self) -> None:
        while True:
            time.sleep(10)

    def run_and_wait(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self.shared_event_loop).result()

    def run(self):
        self.shared_event_loop = asyncio.get_event_loop()


class RemoteCallError(BaseModel):
    type: str = "RemoteCallError"
    args: Optional[Tuple[Any, ...]] = None


class UnhealthyServiceError(ValueError):
    def __init__(
        self, message: str = "Service is unhealthy or not ready", log: bool = True
    ):
        msg = f"[{get_service_name()}] - {message}"
        super().__init__(msg)
        self.message = msg
        if log:
            logger.error(self.message)

    def __str__(self):
        return self.message


class HTTPClientError(Exception):
    """Exception for HTTP client errors (4xx status codes) that should not be retried."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class HTTPServerError(Exception):
    """Exception for HTTP server errors (5xx status codes) that can be retried."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


EXCEPTION_MAPPING = {
    e.__name__: e
    for e in [
        ValueError,
        RuntimeError,
        TimeoutError,
        ConnectionError,
        UnhealthyServiceError,
        HTTPClientError,
        HTTPServerError,
        *[
            ErrorType
            for _, ErrorType in inspect.getmembers(exceptions)
            if inspect.isclass(ErrorType)
            and issubclass(ErrorType, Exception)
            and ErrorType.__module__ == exceptions.__name__
        ],
    ]
}


class AppService(BaseAppService, ABC):
    fastapi_app: FastAPI
    log_level: str = "info"

    def set_log_level(self, log_level: str):
        """Set the uvicorn log level. Returns self for chaining."""
        self.log_level = log_level
        return self

    @staticmethod
    def _handle_internal_http_error(status_code: int = 500, log_error: bool = True):
        def handler(request: Request, exc: Exception):
            if log_error:
                if status_code == 500:
                    log = logger.exception
                else:
                    log = logger.error
                log(f"{request.method} {request.url.path} failed: {exc}")
            return responses.JSONResponse(
                status_code=status_code,
                content=RemoteCallError(
                    type=str(exc.__class__.__name__),
                    args=exc.args or (str(exc),),
                ).model_dump(),
            )

        return handler

    def _create_fastapi_endpoint(self, func: Callable) -> Callable:
        """
        Generates a FastAPI endpoint for the given function, handling default and optional parameters.

        :param func: The original function (sync/async, bound or unbound)
        :return: A FastAPI endpoint function.
        """
        sig = inspect.signature(func)
        fields = {}

        is_bound_method = False
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                is_bound_method = True
                continue

            # Use the provided annotation or fallback to str if not specified
            annotation = (
                param.annotation if param.annotation != inspect.Parameter.empty else str
            )

            # If a default value is provided, use it; otherwise, mark the field as required with '...'
            default = param.default if param.default != inspect.Parameter.empty else ...

            fields[name] = (annotation, default)

        # Dynamically create a Pydantic model for the request body
        RequestBodyModel = create_model("RequestBodyModel", **fields)
        f = func.__get__(self) if is_bound_method else func

        if asyncio.iscoroutinefunction(f):

            async def async_endpoint(body: RequestBodyModel):  # type: ignore #RequestBodyModel being variable
                result = await f(
                    **{name: getattr(body, name) for name in type(body).model_fields}
                )
                _validate_no_prisma_objects(result, f"{func.__name__} result")
                return result

            return async_endpoint
        else:

            def sync_endpoint(body: RequestBodyModel):  # type: ignore #RequestBodyModel being variable
                result = f(
                    **{name: getattr(body, name) for name in type(body).model_fields}
                )
                _validate_no_prisma_objects(result, f"{func.__name__} result")
                return result

            return sync_endpoint

    @conn_retry("FastAPI server", "Starting FastAPI server")
    def __start_fastapi(self):
        logger.info(
            f"[{self.service_name}] Starting RPC server at http://{api_host}:{self.get_port()}"
        )

        server = uvicorn.Server(
            uvicorn.Config(
                self.fastapi_app,
                host=api_host,
                port=self.get_port(),
                log_config=None,  # Explicitly None to avoid uvicorn replacing the logger.
                log_level=self.log_level,
            )
        )
        self.shared_event_loop.run_until_complete(server.serve())

    async def health_check(self) -> str:
        """
        A method to check the health of the process.
        """
        return "OK"

    def run(self):
        sentry_init()
        super().run()
        self.fastapi_app = FastAPI()

        # Add Prometheus instrumentation to all services
        try:
            instrument_fastapi(
                self.fastapi_app,
                service_name=self.service_name,
                expose_endpoint=True,
                endpoint="/metrics",
                include_in_schema=False,
            )
        except ImportError:
            logger.warning(
                f"Prometheus instrumentation not available for {self.service_name}"
            )
        except Exception as e:
            logger.error(
                f"Failed to instrument {self.service_name} with Prometheus: {e}"
            )

        # Register the exposed API routes.
        for attr_name, attr in vars(type(self)).items():
            if getattr(attr, EXPOSED_FLAG, False):
                route_path = f"/{attr_name}"
                self.fastapi_app.add_api_route(
                    route_path,
                    self._create_fastapi_endpoint(attr),
                    methods=["POST"],
                )
        self.fastapi_app.add_api_route(
            "/health_check", self.health_check, methods=["POST", "GET"]
        )
        self.fastapi_app.add_api_route(
            "/health_check_async", self.health_check, methods=["POST", "GET"]
        )
        self.fastapi_app.add_exception_handler(
            ValueError, self._handle_internal_http_error(400)
        )
        self.fastapi_app.add_exception_handler(
            Exception, self._handle_internal_http_error(500)
        )

        # Start the FastAPI server in a separate thread.
        api_thread = threading.Thread(target=self.__start_fastapi, daemon=True)
        api_thread.start()

        # Run the main service loop (blocking).
        self.run_service()


# --------------------------------------------------
# HTTP Client utilities for dynamic service client abstraction
# --------------------------------------------------
AS = TypeVar("AS", bound=AppService)


class AppServiceClient(ABC):
    @classmethod
    @abstractmethod
    def get_service_type(cls) -> Type[AppService]:
        pass

    def health_check(self):
        pass

    async def health_check_async(self):
        pass

    def close(self):
        pass


ASC = TypeVar("ASC", bound=AppServiceClient)


@conn_retry("AppService client", "Creating service client", max_retry=api_comm_retry)
def get_service_client(
    service_client_type: Type[ASC],
    call_timeout: int | None = api_call_timeout,
    request_retry: bool = False,
) -> ASC:

    def _maybe_retry(fn: Callable[..., R]) -> Callable[..., R]:
        """Decorate *fn* with tenacity retry when enabled."""
        if not request_retry:
            return fn

        # Use preconfigured retry decorator for service communication
        return create_retry_decorator(
            max_attempts=api_comm_retry,
            max_wait=api_comm_max_wait,
            context="Service communication",
            exclude_exceptions=(
                # Don't retry these specific exceptions that won't be fixed by retrying
                ValueError,  # Invalid input/parameters
                KeyError,  # Missing required data
                TypeError,  # Wrong data types
                AttributeError,  # Missing attributes
                asyncio.CancelledError,  # Task was cancelled
                concurrent.futures.CancelledError,  # Future was cancelled
                HTTPClientError,  # HTTP 4xx client errors - don't retry
            ),
        )(fn)

    class DynamicClient:
        def __init__(self) -> None:
            service_type = service_client_type.get_service_type()
            host = service_type.get_host()
            port = service_type.get_port()
            self.base_url = f"http://{host}:{port}".rstrip("/")
            self._connection_failure_count = 0
            self._last_client_reset = 0
            self._async_clients = {}  # None key for default async client
            self._sync_clients = {}  # For sync clients (no event loop concept)

        def _create_sync_client(self) -> httpx.Client:
            return httpx.Client(
                base_url=self.base_url,
                timeout=call_timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=200,  # 10x default for async concurrent calls
                    max_connections=500,  # High limit for burst handling
                    keepalive_expiry=30.0,  # Keep connections alive longer
                ),
            )

        def _create_async_client(self) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                base_url=self.base_url,
                timeout=call_timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=200,  # 10x default for async concurrent calls
                    max_connections=500,  # High limit for burst handling
                    keepalive_expiry=30.0,  # Keep connections alive longer
                ),
            )

        @property
        def sync_client(self) -> httpx.Client:
            """Get the sync client (thread-safe singleton)."""
            # Use service name as key for better identification
            service_name = service_client_type.get_service_type().__name__
            if client := self._sync_clients.get(service_name):
                return client
            return self._sync_clients.setdefault(
                service_name, self._create_sync_client()
            )

        @property
        def async_client(self) -> httpx.AsyncClient:
            """Get the appropriate async client for the current context.

            Returns per-event-loop client when in async context,
            falls back to default client otherwise.
            """
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop, use None as default key
                loop = None

            if client := self._async_clients.get(loop):
                return client
            return self._async_clients.setdefault(loop, self._create_async_client())

        def _handle_connection_error(self, error: Exception) -> None:
            """Handle connection errors and implement self-healing"""
            self._connection_failure_count += 1
            current_time = time.time()

            # If we've had 3+ failures, and it's been more than 30 seconds since last reset
            if (
                self._connection_failure_count >= 3
                and current_time - self._last_client_reset > 30
            ):

                logger.warning(
                    f"Connection failures detected ({self._connection_failure_count}), recreating HTTP clients"
                )

                # Clear cached clients to force recreation on next access
                # Only recreate when there's actually a problem
                self._sync_clients.clear()
                self._async_clients.clear()

                # Reset counters
                self._connection_failure_count = 0
                self._last_client_reset = current_time

        def _handle_call_method_response(
            self, *, response: httpx.Response, method_name: str
        ) -> Any:
            try:
                response.raise_for_status()
                # Reset failure count on successful response
                self._connection_failure_count = 0
                return response.json()
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Try to parse the error response as RemoteCallError for mapped exceptions
                error_response = None
                try:
                    error_response = RemoteCallError.model_validate(e.response.json())
                except Exception:
                    pass

                # If we successfully parsed a mapped exception type, re-raise it
                if error_response and error_response.type in EXCEPTION_MAPPING:
                    exception_class = EXCEPTION_MAPPING[error_response.type]
                    args = error_response.args or [str(e)]
                    raise exception_class(*args)

                # Otherwise categorize by HTTP status code
                if 400 <= status_code < 500:
                    # Client errors (4xx) - wrap to prevent retries
                    raise HTTPClientError(status_code, str(e))
                elif 500 <= status_code < 600:
                    # Server errors (5xx) - wrap but allow retries
                    raise HTTPServerError(status_code, str(e))
                else:
                    # Other status codes (1xx, 2xx, 3xx) - re-raise original error
                    raise e

        @_maybe_retry
        def _call_method_sync(self, method_name: str, **kwargs: Any) -> Any:
            try:
                return self._handle_call_method_response(
                    method_name=method_name,
                    response=self.sync_client.post(method_name, json=to_dict(kwargs)),
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                self._handle_connection_error(e)
                raise

        @_maybe_retry
        async def _call_method_async(self, method_name: str, **kwargs: Any) -> Any:
            try:
                return self._handle_call_method_response(
                    method_name=method_name,
                    response=await self.async_client.post(
                        method_name, json=to_dict(kwargs)
                    ),
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                self._handle_connection_error(e)
                raise

        async def aclose(self) -> None:
            # Close all sync clients
            for client in self._sync_clients.values():
                client.close()
            self._sync_clients.clear()

            # Close all async clients (including default with None key)
            for client in self._async_clients.values():
                await client.aclose()
            self._async_clients.clear()

        def close(self) -> None:
            # Close all sync clients
            for client in self._sync_clients.values():
                client.close()
            self._sync_clients.clear()
            # Note: Cannot close async clients synchronously
            # They will be cleaned up by garbage collection

        def __del__(self):
            """Cleanup HTTP clients on garbage collection to prevent resource leaks."""
            try:
                # Close any remaining sync clients
                for client in self._sync_clients.values():
                    client.close()

                # Warn if async clients weren't properly closed
                if self._async_clients:
                    import warnings

                    warnings.warn(
                        "DynamicClient async clients not explicitly closed. "
                        "Call aclose() before destroying the client.",
                        ResourceWarning,
                        stacklevel=2,
                    )
            except Exception:
                # Silently ignore cleanup errors in __del__
                pass

        async def __aenter__(self):
            """Async context manager entry."""
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Async context manager exit."""
            await self.aclose()

        def _get_params(
            self, signature: inspect.Signature, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            if args:
                arg_names = list(signature.parameters.keys())
                if arg_names and arg_names[0] in ("self", "cls"):
                    arg_names = arg_names[1:]
                kwargs.update(dict(zip(arg_names, args)))
            return kwargs

        def _get_return(self, expected_return: TypeAdapter | None, result: Any) -> Any:
            if expected_return:
                return expected_return.validate_python(result)
            return result

        def __getattr__(self, name: str) -> Callable[..., Any]:
            original_func = getattr(service_client_type, name, None)
            if original_func is None:
                raise AttributeError(
                    f"Method {name} not found in {service_client_type}"
                )

            rpc_name = original_func.__name__
            sig = inspect.signature(original_func)
            ret_ann = sig.return_annotation
            expected_return = (
                None if ret_ann is inspect.Signature.empty else TypeAdapter(ret_ann)
            )

            if inspect.iscoroutinefunction(original_func):

                async def async_method(*args: P.args, **kwargs: P.kwargs):
                    params = self._get_params(sig, *args, **kwargs)
                    result = await self._call_method_async(rpc_name, **params)
                    return self._get_return(expected_return, result)

                return async_method

            else:

                def sync_method(*args: P.args, **kwargs: P.kwargs):
                    params = self._get_params(sig, *args, **kwargs)
                    result = self._call_method_sync(rpc_name, **params)
                    return self._get_return(expected_return, result)

                return sync_method

    client = cast(ASC, DynamicClient())

    return client


def endpoint_to_sync(
    func: Callable[Concatenate[Any, P], Awaitable[R]],
) -> Callable[Concatenate[Any, P], R]:
    """
    Produce a *typed* stub that **looks** synchronous to the type‑checker.
    """

    def _stub(*args: P.args, **kwargs: P.kwargs) -> R:  # pragma: no cover
        raise RuntimeError("should be intercepted by __getattr__")

    update_wrapper(_stub, func)
    return cast(Callable[Concatenate[Any, P], R], _stub)


def endpoint_to_async(
    func: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], Awaitable[R]]:
    """
    The async mirror of `to_sync`.
    """

    async def _stub(*args: P.args, **kwargs: P.kwargs) -> R:  # pragma: no cover
        raise RuntimeError("should be intercepted by __getattr__")

    update_wrapper(_stub, func)
    return cast(Callable[Concatenate[Any, P], Awaitable[R]], _stub)
