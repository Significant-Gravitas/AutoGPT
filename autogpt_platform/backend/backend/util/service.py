import asyncio
import inspect
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from functools import cached_property, update_wrapper
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
from autogpt_libs.logging.utils import generate_uvicorn_config
from fastapi import FastAPI, Request, responses
from pydantic import BaseModel, TypeAdapter, create_model
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

import backend.util.exceptions as exceptions
from backend.util.json import to_dict
from backend.util.metrics import sentry_init
from backend.util.process import AppProcess, get_service_name
from backend.util.retry import conn_retry
from backend.util.settings import Config

logger = logging.getLogger(__name__)
T = TypeVar("T")
C = TypeVar("C", bound=Callable)

config = Config()
api_host = config.pyro_host
api_comm_retry = config.pyro_client_comm_retry
api_comm_timeout = config.pyro_client_comm_timeout
api_call_timeout = config.rpc_client_call_timeout

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
        target_host = os.environ.get(f"{cls.service_name.upper()}_HOST", api_host)

        if source_host == target_host and source_host != api_host:
            logger.warning(
                f"Service {cls.service_name} is the same host as the source service."
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


EXCEPTION_MAPPING = {
    e.__name__: e
    for e in [
        ValueError,
        TimeoutError,
        ConnectionError,
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
                return await f(
                    **{name: getattr(body, name) for name in type(body).model_fields}
                )

            return async_endpoint
        else:

            def sync_endpoint(body: RequestBodyModel):  # type: ignore #RequestBodyModel being variable
                return f(
                    **{name: getattr(body, name) for name in type(body).model_fields}
                )

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
                log_config=generate_uvicorn_config(),
                log_level=self.log_level,
            )
        )
        self.shared_event_loop.run_until_complete(server.serve())

    def run(self):
        sentry_init()
        super().run()
        self.fastapi_app = FastAPI()

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
    health_check: bool = True,
    request_retry: bool | int = False,
) -> ASC:

    def _maybe_retry(fn: Callable[..., R]) -> Callable[..., R]:
        """Decorate *fn* with tenacity retry when enabled."""
        nonlocal request_retry

        if isinstance(request_retry, int):
            retry_attempts = request_retry
            request_retry = True
        else:
            retry_attempts = api_comm_retry

        if not request_retry:
            return fn

        return retry(
            reraise=True,
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential_jitter(max=4.0),
            retry=retry_if_exception_type(
                (
                    httpx.ConnectError,
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                    httpx.ConnectTimeout,
                    httpx.RemoteProtocolError,
                )
            ),
        )(fn)

    class DynamicClient:
        def __init__(self) -> None:
            service_type = service_client_type.get_service_type()
            host = service_type.get_host()
            port = service_type.get_port()
            self.base_url = f"http://{host}:{port}".rstrip("/")

        @cached_property
        def sync_client(self) -> httpx.Client:
            return httpx.Client(
                base_url=self.base_url,
                timeout=call_timeout,
            )

        @cached_property
        def async_client(self) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                base_url=self.base_url,
                timeout=call_timeout,
            )

        def _handle_call_method_response(
            self, *, response: httpx.Response, method_name: str
        ) -> Any:
            try:
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error in {method_name}: {e.response.text}")
                error = RemoteCallError.model_validate(e.response.json())
                # DEBUG HELP: if you made a custom exception, make sure you override self.args to be how to make your exception
                raise EXCEPTION_MAPPING.get(error.type, Exception)(
                    *(error.args or [str(e)])
                )

        @_maybe_retry
        def _call_method_sync(self, method_name: str, **kwargs: Any) -> Any:
            return self._handle_call_method_response(
                method_name=method_name,
                response=self.sync_client.post(method_name, json=to_dict(kwargs)),
            )

        @_maybe_retry
        async def _call_method_async(self, method_name: str, **kwargs: Any) -> Any:
            return self._handle_call_method_response(
                method_name=method_name,
                response=await self.async_client.post(
                    method_name, json=to_dict(kwargs)
                ),
            )

        async def aclose(self) -> None:
            self.sync_client.close()
            await self.async_client.aclose()

        def close(self) -> None:
            self.sync_client.close()

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
    if health_check and hasattr(client, "health_check"):
        client.health_check()

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
