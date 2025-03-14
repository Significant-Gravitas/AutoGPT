import asyncio
import builtins
import inspect
import logging
import os
import threading
import time
import typing
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from types import NoneType, UnionType
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Concatenate,
    Coroutine,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import httpx
import Pyro5.api
import uvicorn
from fastapi import FastAPI, Request, responses
from pydantic import BaseModel, TypeAdapter, create_model
from Pyro5 import api as pyro
from Pyro5 import config as pyro_config

from backend.data import db, rabbitmq, redis
from backend.util.exceptions import InsufficientBalanceError
from backend.util.json import to_dict
from backend.util.process import AppProcess
from backend.util.retry import conn_retry
from backend.util.settings import Config, Secrets

logger = logging.getLogger(__name__)
T = TypeVar("T")
C = TypeVar("C", bound=Callable)

config = Config()
api_host = config.pyro_host
api_comm_retry = config.pyro_client_comm_retry
api_comm_timeout = config.pyro_client_comm_timeout
api_call_timeout = config.rpc_client_call_timeout
pyro_config.MAX_RETRIES = api_comm_retry  # type: ignore
pyro_config.COMMTIMEOUT = api_comm_timeout  # type: ignore


P = ParamSpec("P")
R = TypeVar("R")


def fastapi_expose(func: C) -> C:
    func = getattr(func, "__func__", func)
    setattr(func, "__exposed__", True)
    return func


def fastapi_exposed_run_and_wait(
    f: Callable[P, Coroutine[None, None, R]]
) -> Callable[Concatenate[object, P], R]:
    # TODO:
    #  This function lies about its return type to make the DynamicClient
    #  call the function synchronously, fix this when DynamicClient can choose
    #  to call a function synchronously or asynchronously.
    return expose(f)  # type: ignore


# ----- Begin Pyro Expose Block ---- #
def pyro_expose(func: C) -> C:
    """
    Decorator to mark a method or class to be exposed for remote calls.

    ## ⚠️ Gotcha
    Aside from "simple" types, only Pydantic models are passed unscathed *if annotated*.
    Any other passed or returned class objects are converted to dictionaries by Pyro.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"Error in {func.__name__}: {e}"
            if isinstance(e, ValueError):
                logger.warning(msg)
            else:
                logger.exception(msg)
            raise

    register_pydantic_serializers(func)

    return pyro.expose(wrapper)  # type: ignore


def register_pydantic_serializers(func: Callable):
    """Register custom serializers and deserializers for annotated Pydantic models"""
    for name, annotation in func.__annotations__.items():
        try:
            pydantic_types = _pydantic_models_from_type_annotation(annotation)
        except Exception as e:
            raise TypeError(f"Error while exposing {func.__name__}: {e}")

        for model in pydantic_types:
            logger.debug(
                f"Registering Pyro (de)serializers for {func.__name__} annotation "
                f"'{name}': {model.__qualname__}"
            )
            pyro.register_class_to_dict(model, _make_custom_serializer(model))
            pyro.register_dict_to_class(
                model.__qualname__, _make_custom_deserializer(model)
            )


def _make_custom_serializer(model: Type[BaseModel]):
    def custom_class_to_dict(obj):
        data = {
            "__class__": obj.__class__.__qualname__,
            **obj.model_dump(),
        }
        logger.debug(f"Serializing {obj.__class__.__qualname__} with data: {data}")
        return data

    return custom_class_to_dict


def _make_custom_deserializer(model: Type[BaseModel]):
    def custom_dict_to_class(qualname, data: dict):
        logger.debug(f"Deserializing {model.__qualname__} from data: {data}")
        return model(**data)

    return custom_dict_to_class


def pyro_exposed_run_and_wait(
    f: Callable[P, Coroutine[None, None, R]]
) -> Callable[Concatenate[object, P], R]:
    @expose
    @wraps(f)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        coroutine = f(*args, **kwargs)
        res = self.run_and_wait(coroutine)
        return res

    # Register serializers for annotations on bare function
    register_pydantic_serializers(f)

    return wrapper


if config.use_http_based_rpc:
    expose = fastapi_expose
    exposed_run_and_wait = fastapi_exposed_run_and_wait
else:
    expose = pyro_expose
    exposed_run_and_wait = pyro_exposed_run_and_wait

# ----- End Pyro Expose Block ---- #


# --------------------------------------------------
# AppService for IPC service based on HTTP request through FastAPI
# --------------------------------------------------
class BaseAppService(AppProcess, ABC):
    shared_event_loop: asyncio.AbstractEventLoop
    use_db: bool = False
    use_redis: bool = False
    rabbitmq_config: Optional[rabbitmq.RabbitMQConfig] = None
    rabbitmq_service: Optional[rabbitmq.AsyncRabbitMQ] = None
    use_supabase: bool = False

    @classmethod
    @abstractmethod
    def get_port(cls) -> int:
        pass

    @classmethod
    def get_host(cls) -> str:
        return os.environ.get(f"{cls.service_name.upper()}_HOST", api_host)

    @property
    def rabbit(self) -> rabbitmq.AsyncRabbitMQ:
        """Access the RabbitMQ service. Will raise if not configured."""
        if not self.rabbitmq_service:
            raise RuntimeError("RabbitMQ not configured for this service")
        return self.rabbitmq_service

    @property
    def rabbit_config(self) -> rabbitmq.RabbitMQConfig:
        """Access the RabbitMQ config. Will raise if not configured."""
        if not self.rabbitmq_config:
            raise RuntimeError("RabbitMQ not configured for this service")
        return self.rabbitmq_config

    def run_service(self) -> None:
        while True:
            time.sleep(10)

    def run_and_wait(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self.shared_event_loop).result()

    def run(self):
        self.shared_event_loop = asyncio.get_event_loop()
        if self.use_db:
            self.shared_event_loop.run_until_complete(db.connect())
        if self.use_redis:
            redis.connect()
        if self.rabbitmq_config:
            logger.info(f"[{self.__class__.__name__}] ⏳ Configuring RabbitMQ...")
            self.rabbitmq_service = rabbitmq.AsyncRabbitMQ(self.rabbitmq_config)
            self.shared_event_loop.run_until_complete(self.rabbitmq_service.connect())
        if self.use_supabase:
            from supabase import create_client

            secrets = Secrets()
            self.supabase = create_client(
                secrets.supabase_url, secrets.supabase_service_role_key
            )

    def cleanup(self):
        if self.use_db:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting DB...")
            self.run_and_wait(db.disconnect())
        if self.use_redis:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting Redis...")
            redis.disconnect()
        if self.rabbitmq_config:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting RabbitMQ...")


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


class FastApiAppService(BaseAppService, ABC):
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
                    **{name: getattr(body, name) for name in body.model_fields}
                )

            return async_endpoint
        else:

            def sync_endpoint(body: RequestBodyModel):  # type: ignore #RequestBodyModel being variable
                return f(**{name: getattr(body, name) for name in body.model_fields})

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


# ----- Begin Pyro AppService Block ---- #


class PyroAppService(BaseAppService, ABC):

    @conn_retry("Pyro", "Starting Pyro Service")
    def __start_pyro(self):
        maximum_connection_thread_count = max(
            Pyro5.config.THREADPOOL_SIZE,
            config.num_node_workers * config.num_graph_workers,
        )

        Pyro5.config.THREADPOOL_SIZE = maximum_connection_thread_count  # type: ignore
        daemon = Pyro5.api.Daemon(host=api_host, port=self.get_port())
        self.uri = daemon.register(self, objectId=self.service_name)
        logger.info(f"[{self.service_name}] Connected to Pyro; URI = {self.uri}")
        daemon.requestLoop()

    def run(self):
        super().run()

        # Initialize the async loop.
        async_thread = threading.Thread(target=self.shared_event_loop.run_forever)
        async_thread.daemon = True
        async_thread.start()

        # Initialize pyro service
        daemon_thread = threading.Thread(target=self.__start_pyro)
        daemon_thread.daemon = True
        daemon_thread.start()

        # Run the main service loop (blocking).
        self.run_service()


if config.use_http_based_rpc:

    class AppService(FastApiAppService, ABC):  # type: ignore #AppService defined twice
        pass

else:

    class AppService(PyroAppService, ABC):
        pass


# ----- End Pyro AppService Block ---- #


# --------------------------------------------------
# HTTP Client utilities for dynamic service client abstraction
# --------------------------------------------------
AS = TypeVar("AS", bound=AppService)


def fastapi_close_service_client(client: Any) -> None:
    if hasattr(client, "close"):
        client.close()
    else:
        logger.warning(f"Client {client} is not closable")


@conn_retry("FastAPI client", "Creating service client", max_retry=api_comm_retry)
def fastapi_get_service_client(
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


# ----- Begin Pyro Client Block ---- #
class PyroClient:
    proxy: Pyro5.api.Proxy


def pyro_close_service_client(client: BaseAppService) -> None:
    if isinstance(client, PyroClient):
        client.proxy._pyroRelease()
    else:
        raise RuntimeError(f"Client {client.__class__} is not a Pyro client.")


def pyro_get_service_client(service_type: Type[AS]) -> AS:
    service_name = service_type.service_name

    class DynamicClient(PyroClient):
        @conn_retry("Pyro", f"Connecting to [{service_name}]")
        def __init__(self):
            uri = f"PYRO:{service_type.service_name}@{service_type.get_host()}:{service_type.get_port()}"
            logger.debug(f"Connecting to service [{service_name}]. URI = {uri}")
            self.proxy = Pyro5.api.Proxy(uri)
            # Attempt to bind to ensure the connection is established
            self.proxy._pyroBind()
            logger.debug(f"Successfully connected to service [{service_name}]")

        def __getattr__(self, name: str) -> Callable[..., Any]:
            res = getattr(self.proxy, name)
            return res

    return cast(AS, DynamicClient())


builtin_types = [*vars(builtins).values(), NoneType, Enum]


def _pydantic_models_from_type_annotation(annotation) -> Iterator[type[BaseModel]]:
    # Peel Annotated parameters
    if (origin := get_origin(annotation)) and origin is Annotated:
        annotation = get_args(annotation)[0]

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (
        Union,
        UnionType,
        list,
        List,
        tuple,
        Tuple,
        set,
        Set,
        frozenset,
        FrozenSet,
    ):
        for arg in args:
            yield from _pydantic_models_from_type_annotation(arg)
    elif origin in (dict, Dict):
        key_type, value_type = args
        yield from _pydantic_models_from_type_annotation(key_type)
        yield from _pydantic_models_from_type_annotation(value_type)
    elif origin in (Awaitable, Coroutine):
        # For coroutines and awaitables, check the return type
        return_type = args[-1]
        yield from _pydantic_models_from_type_annotation(return_type)
    else:
        annotype = annotation if origin is None else origin

        # Exclude generic types and aliases
        if (
            annotype is not None
            and not hasattr(typing, getattr(annotype, "__name__", ""))
            and isinstance(annotype, type)
        ):
            if issubclass(annotype, BaseModel):
                yield annotype
            elif annotype not in builtin_types and not issubclass(annotype, Enum):
                raise TypeError(f"Unsupported type encountered: {annotype}")


if config.use_http_based_rpc:
    close_service_client = fastapi_close_service_client
    get_service_client = fastapi_get_service_client
else:
    close_service_client = pyro_close_service_client
    get_service_client = pyro_get_service_client

# ----- End Pyro Client Block ---- #
