import asyncio
import logging
import os
import threading
import time
import typing
from abc import abstractmethod
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Coroutine,
    Iterator,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import Pyro5.api
from pydantic import BaseModel
from Pyro5 import api as pyro

from backend.data import db
from backend.data.queue import AbstractEventQueue, RedisEventQueue
from backend.util.process import AppProcess
from backend.util.retry import conn_retry
from backend.util.settings import Config, Secrets

logger = logging.getLogger(__name__)
T = TypeVar("T")
C = TypeVar("C", bound=Callable)

pyro_host = Config().pyro_host


def expose(func: C) -> C:
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
            msg = f"Error in {func.__name__}: {e.__str__()}"
            logger.exception(msg)
            raise Exception(msg, e)

    # Register custom serializers and deserializers for annotated Pydantic models
    for name, annotation in func.__annotations__.items():
        for model in _pydantic_models_from_type_annotation(annotation):
            logger.debug(
                f"Registering Pyro (de)serializers for {func.__name__} annotation "
                f"'{name}': {model.__qualname__}"
            )
            pyro.register_class_to_dict(
                model,
                lambda obj: {
                    "__class__": obj.__class__.__qualname__,
                    **obj.model_dump(),
                },
            )
            pyro.register_dict_to_class(
                model.__qualname__, _make_pyrodantic_parser(model)
            )

    return pyro.expose(wrapper)  # type: ignore


def _make_pyrodantic_parser(model: type[BaseModel]):
    def parse_data_into_model(qualname, data: dict):
        logger.debug(f"Parsing Pyroed {model.__qualname__} from data {data}")
        return model(**data)

    return parse_data_into_model


class AppService(AppProcess):
    shared_event_loop: asyncio.AbstractEventLoop
    event_queue: AbstractEventQueue = RedisEventQueue()
    use_db: bool = False
    use_queue: bool = False
    use_supabase: bool = False

    def __init__(self, port):
        self.port = port
        self.uri = None

    @abstractmethod
    def run_service(self):
        while True:
            time.sleep(10)

    def __run_async(self, coro: Coroutine[Any, Any, T]):
        return asyncio.run_coroutine_threadsafe(coro, self.shared_event_loop)

    def run_and_wait(self, coro: Coroutine[Any, Any, T]) -> T:
        future = self.__run_async(coro)
        return future.result()

    def run(self):
        self.shared_event_loop = asyncio.get_event_loop()
        if self.use_db:
            self.shared_event_loop.run_until_complete(db.connect())
        if self.use_queue:
            self.event_queue.connect()
        if self.use_supabase:
            from supabase import create_client

            secrets = Secrets()
            self.supabase = create_client(
                secrets.supabase_url, secrets.supabase_service_role_key
            )

        # Initialize the async loop.
        async_thread = threading.Thread(target=self.__start_async_loop)
        async_thread.daemon = True
        async_thread.start()

        # Initialize pyro service
        daemon_thread = threading.Thread(target=self.__start_pyro)
        daemon_thread.daemon = True
        daemon_thread.start()

        # Run the main service (if it's not implemented, just sleep).
        self.run_service()

    def cleanup(self):
        if self.use_db:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting DB...")
            self.run_and_wait(db.disconnect())
        if self.use_queue:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting Redis...")
            self.event_queue.close()

    @conn_retry("Pyro", "Starting Pyro Service")
    def __start_pyro(self):
        host = Config().pyro_host
        daemon = Pyro5.api.Daemon(host=host, port=self.port)
        self.uri = daemon.register(self, objectId=self.service_name)
        logger.info(f"[{self.service_name}] Connected to Pyro; URI = {self.uri}")
        daemon.requestLoop()

    def __start_async_loop(self):
        self.shared_event_loop.run_forever()


AS = TypeVar("AS", bound=AppService)


def get_service_client(service_type: Type[AS], port: int) -> AS:
    service_name = service_type.service_name

    class DynamicClient:
        @conn_retry("Pyro", f"Connecting to [{service_name}]")
        def __init__(self):
            host = os.environ.get(f"{service_name.upper()}_HOST", "localhost")
            uri = f"PYRO:{service_type.service_name}@{host}:{port}"
            logger.debug(f"Connecting to service [{service_name}]. URI = {uri}")
            self.proxy = Pyro5.api.Proxy(uri)
            # Attempt to bind to ensure the connection is established
            self.proxy._pyroBind()
            logger.debug(f"Successfully connected to service [{service_name}]")

        def __getattr__(self, name: str) -> Callable[..., Any]:
            return getattr(self.proxy, name)

    return cast(AS, DynamicClient())


# --------- UTILITIES --------- #


def _pydantic_models_from_type_annotation(annotation) -> Iterator[type[BaseModel]]:
    # Peel Annotated parameters
    if (origin := get_origin(annotation)) and origin is Annotated:
        annotation = get_args(annotation)[0]

    if origin := get_origin(annotation):
        if origin is UnionType:
            types = get_args(annotation)
        else:
            types = [origin]
    else:
        types = [annotation]

    for annotype in types:
        if (
            annotype is not None
            and not hasattr(typing, annotype.__name__)  # avoid generics and aliases
            and issubclass(annotype, BaseModel)
        ):
            yield annotype
