import asyncio
import logging
import threading
import time
from abc import abstractmethod
from typing import Any, Callable, Coroutine, Type, TypeVar, cast

from Pyro5 import api as pyro
from Pyro5 import nameserver
from tenacity import retry, stop_after_attempt, wait_exponential

from autogpt_server.data import db
from autogpt_server.data.queue import AsyncEventQueue, AsyncRedisEventQueue
from autogpt_server.util.process import AppProcess
from autogpt_server.util.settings import Config

logger = logging.getLogger(__name__)
conn_retry = retry(
    stop=stop_after_attempt(30), wait=wait_exponential(multiplier=1, min=1, max=30)
)
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


class PyroNameServer(AppProcess):
    def run(self):
        try:
            print("Starting NameServer loop")
            nameserver.start_ns_loop(host=pyro_host, port=9090)
        except KeyboardInterrupt:
            print("Shutting down NameServer")


class AppService(AppProcess):
    shared_event_loop: asyncio.AbstractEventLoop
    event_queue: AsyncEventQueue = AsyncRedisEventQueue()
    use_db: bool = True
    use_redis: bool = False

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

    @conn_retry
    def __start_pyro(self):
        daemon = pyro.Daemon(host=pyro_host)
        ns = pyro.locate_ns(host=pyro_host, port=9090)
        uri = daemon.register(self)
        ns.register(self.service_name, uri)
        logger.warning(f"Service [{self.service_name}] Ready. Object URI = {uri}")
        daemon.requestLoop()

    def __start_async_loop(self):
        self.shared_event_loop.run_forever()


AS = TypeVar("AS", bound=AppService)


def get_service_client(service_type: Type[AS]) -> AS:
    service_name = service_type.service_name

    class DynamicClient:
        @conn_retry
        def __init__(self):
            ns = pyro.locate_ns()
            uri = ns.lookup(service_name)
            self.proxy = pyro.Proxy(uri)
            self.proxy._pyroBind()

        def __getattr__(self, name: str) -> Callable[..., Any]:
            return getattr(self.proxy, name)

    return cast(AS, DynamicClient())
