import logging
import threading
from abc import abstractmethod
from typing import Any, Callable, Type, TypeVar, cast

from Pyro5 import api as pyro
from Pyro5 import nameserver
from tenacity import retry, stop_after_delay, wait_exponential

from autogpt_server.util.process import AppProcess

logger = logging.getLogger(__name__)
conn_retry = retry(stop=stop_after_delay(5), wait=wait_exponential(multiplier=0.1))
expose = pyro.expose


class PyroNameServer(AppProcess):
    def run(self):
        nameserver.start_ns_loop()


class AppService(AppProcess):
    @classmethod
    @property
    def service_name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def run(self):
        pass

    @conn_retry
    def start_pyro(self):
        daemon = pyro.Daemon()
        ns = pyro.locate_ns()
        uri = daemon.register(self)
        ns.register(self.service_name, uri)
        logger.warning(f"Service [{self.service_name}] Ready. Object URI = {uri}")
        daemon.requestLoop()

    def execute_run_command(self, *args, **kwargs):
        daemon_thread = threading.Thread(target=self.start_pyro)
        daemon_thread.daemon = True
        daemon_thread.start()
        super().execute_run_command(*args, **kwargs)


T = TypeVar("T", bound=AppService)


def get_service_client(service_type: Type[T]) -> T:
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

    return cast(T, DynamicClient())
