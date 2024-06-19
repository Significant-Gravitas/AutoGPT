import time
import asyncio
import logging
import threading

from abc import abstractmethod
from typing import Any, Callable, Type, TypeVar, cast, Coroutine

from Pyro5 import api as pyro
from Pyro5 import nameserver
from tenacity import retry, stop_after_delay, wait_exponential

from autogpt_server.data import db
from autogpt_server.util.process import AppProcess

logger = logging.getLogger(__name__)
conn_retry = retry(stop=stop_after_delay(5), wait=wait_exponential(multiplier=0.1))
expose = pyro.expose


class PyroNameServer(AppProcess):
    def run(self):
        try:
            print("Starting NameServer loop")
            nameserver.start_ns_loop()
        except KeyboardInterrupt:
            print("Shutting down NameServer")


class AppService(AppProcess):

    shared_event_loop: asyncio.AbstractEventLoop

    @classmethod
    @property
    def service_name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def run_service(self):
        while True:
            time.sleep(10)

    def run_async(self, coro: Coroutine):
        return asyncio.run_coroutine_threadsafe(coro, self.shared_event_loop)

    def run_and_wait(self, coro: Coroutine):
        future = self.run_async(coro)
        return future.result()

    def run(self):
        self.shared_event_loop = asyncio.get_event_loop()
        self.shared_event_loop.run_until_complete(db.connect())

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
        daemon = pyro.Daemon()
        ns = pyro.locate_ns()
        uri = daemon.register(self)
        ns.register(self.service_name, uri)
        logger.warning(f"Service [{self.service_name}] Ready. Object URI = {uri}")
        daemon.requestLoop()

    def __start_async_loop(self):
        # asyncio.set_event_loop(self.shared_event_loop)
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
