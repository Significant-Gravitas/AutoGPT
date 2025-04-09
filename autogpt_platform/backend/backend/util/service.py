import asyncio
import inspect
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
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

from backend.util.exceptions import InsufficientBalanceError
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


def expose(func: C) -> C:
    func = getattr(func, "__func__", func)
    setattr(func, "__exposed__", True)
    return func


def exposed_run_and_wait(
    f: Callable[P, Coroutine[None, None, R]]
) -> Callable[Concatenate[object, P], R]:
    # TODO:
    #  This function lies about its return type to make the DynamicClient
    #  call the function synchronously, fix this when DynamicClient can choose
    #  to call a function synchronously or asynchronously.
    return expose(f)  # type: ignore


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
        InsufficientBalanceError,
    ]
}


class AppService(BaseAppService, ABC):
    fastapi_app: FastAPI

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
                log_level="warning",
            )
        )
        self.shared_event_loop.run_until_complete(server.serve())

    def run(self):
        sentry_init()
        super().run()
        self.fastapi_app = FastAPI()

        # Register the exposed API routes.
        for attr_name, attr in vars(type(self)).items():
            if getattr(attr, "__exposed__", False):
                route_path = f"/{attr_name}"
                self.fastapi_app.add_api_route(
                    route_path,
                    self._create_fastapi_endpoint(attr),
                    methods=["POST"],
                )
        self.fastapi_app.add_api_route(
            "/health_check", self.health_check, methods=["POST"]
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


def close_service_client(client: Any) -> None:
    if hasattr(client, "close"):
        client.close()
    else:
        logger.warning(f"Client {client} is not closable")


@conn_retry("FastAPI client", "Creating service client", max_retry=api_comm_retry)
def get_service_client(
    service_type: Type[AS],
    call_timeout: int | None = api_call_timeout,
) -> AS:
    class DynamicClient:
        def __init__(self):
            host = service_type.get_host()
            port = service_type.get_port()
            self.base_url = f"http://{host}:{port}".rstrip("/")
            self.client = httpx.Client(
                base_url=self.base_url,
                timeout=call_timeout,
            )

        def _call_method(self, method_name: str, **kwargs) -> Any:
            try:
                response = self.client.post(method_name, json=to_dict(kwargs))
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error in {method_name}: {e.response.text}")
                error = RemoteCallError.model_validate(e.response.json())
                # DEBUG HELP: if you made a custom exception, make sure you override self.args to be how to make your exception
                raise EXCEPTION_MAPPING.get(error.type, Exception)(
                    *(error.args or [str(e)])
                )

        def close(self):
            self.client.close()

        def __getattr__(self, name: str) -> Callable[..., Any]:
            # Try to get the original function from the service type.
            orig_func = getattr(service_type, name, None)
            if orig_func is None:
                raise AttributeError(f"Method {name} not found in {service_type}")

            sig = inspect.signature(orig_func)
            ret_ann = sig.return_annotation
            if ret_ann != inspect.Signature.empty:
                expected_return = TypeAdapter(ret_ann)
            else:
                expected_return = None

            def method(*args, **kwargs) -> Any:
                if args:
                    arg_names = list(sig.parameters.keys())
                    if arg_names[0] in ("self", "cls"):
                        arg_names = arg_names[1:]
                    kwargs.update(dict(zip(arg_names, args)))
                result = self._call_method(name, **kwargs)
                if expected_return:
                    return expected_return.validate_python(result)
                return result

            return method

    client = cast(AS, DynamicClient())
    client.health_check()

    return cast(AS, client)
