import asyncio
import logging
import os
import threading
import time
from abc import abstractmethod
from typing import Any, Callable, Coroutine, Type, TypeVar, cast

import Pyro5.api
from Pyro5 import api as pyro

from autogpt_server.data import db
from autogpt_server.data.queue import AsyncEventQueue, AsyncRedisEventQueue
from autogpt_server.util.process import AppProcess
from autogpt_server.util.retry import conn_retry
from autogpt_server.util.settings import Config

logger = logging.getLogger(__name__)
T = TypeVar("T")
C = TypeVar("C", bound=Callable)

pyro_host = Config().pyro_host


def expose(func: C) -> C:
    """
    Decorator to mark a method or class to be exposed for remote calls.

    ## ⚠️ Gotcha
    The types on the exposed function signature are respected **as long as they are
    fully picklable**. This is not the case for Pydantic models, so if you really need
    to pass a model, try dumping the model and passing the resulting dict instead.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"Error in {func.__name__}: {e.__str__()}"
            logger.exception(msg)
            raise Exception(msg, e)

    return pyro.expose(wrapper)  # type: ignore


class AppService(AppProcess):
    shared_event_loop: asyncio.AbstractEventLoop
    event_queue: AsyncEventQueue = AsyncRedisEventQueue()
    use_db: bool = False
    use_redis: bool = False

    def __init__(self, port):
        self.port = port
        self.uri = None

    @classmethod
    @property
    def service_name(cls) -> str:
        return cls.__name__

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
        if self.use_redis:
            self.shared_event_loop.run_until_complete(self.event_queue.connect())

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
        if self.use_redis:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting Redis...")
            self.run_and_wait(self.event_queue.close())

    @conn_retry
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
        @conn_retry
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
