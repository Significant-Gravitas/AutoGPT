import asyncio
import inspect
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Optional, Type, TypeVar, cast

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, create_model

from backend.data import db, rabbitmq, redis
from backend.util.process import AppProcess
from backend.util.settings import Config, Secrets

logger = logging.getLogger(__name__)
T = TypeVar("T")
C = TypeVar("C", bound=Callable)

config = Config()
api_host = config.pyro_host


def expose(func: C) -> C:
    func = getattr(func, "__func__", func)
    setattr(func, "__exposed__", True)
    return func


# --------------------------------------------------
# AppService for IPC service based on HTTP request through FastAPI
# --------------------------------------------------
APP_SERVICES: dict[str, "AppService"] = {}


class AppService(AppProcess, ABC):
    shared_event_loop: asyncio.AbstractEventLoop
    use_db: bool = False
    use_redis: bool = False
    rabbitmq_config: Optional[rabbitmq.RabbitMQConfig] = None
    rabbitmq_service: Optional[rabbitmq.AsyncRabbitMQ] = None
    use_supabase: bool = False
    fastapi_app: FastAPI

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

    def _create_fastapi_endpoint(self, func: Callable) -> Callable:
        """
        Generates a FastAPI endpoint for the given function, handling default and optional parameters.

        :param func: The original function (sync/async, bound or unbound)
        :return: A FastAPI endpoint function.
        """
        sig = inspect.signature(func)
        fields = {}

        # Build fields by skipping 'self' or 'cls' and checking for defaults
        is_bounded_method = False
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                is_bounded_method = True
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
        f = func.__get__(self) if is_bounded_method else func

        if asyncio.iscoroutinefunction(f):

            async def async_endpoint(body: RequestBodyModel):  # type: ignore
                return await f(**body.model_dump())

            return async_endpoint
        else:

            def sync_endpoint(body: RequestBodyModel):  # type: ignore
                return f(**body.model_dump())

            return sync_endpoint

    def run(self):
        self.fastapi_app = FastAPI()
        # Register the exposed API routes.
        for attr_name, attr in type(self).__dict__.items():
            if getattr(attr, "__exposed__", False):
                route_path = f"/{attr_name}"
                self.fastapi_app.add_api_route(
                    route_path,
                    self._create_fastapi_endpoint(attr),
                    methods=["POST"],
                )

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

        # Start the FastAPI server in a separate thread.
        api_thread = threading.Thread(target=self.__start_fastapi, daemon=True)
        api_thread.start()

        # Run the main service loop (blocking).
        self.run_service()

    def cleanup(self):
        if self.use_db:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting DB...")
            self.run_and_wait(db.disconnect())
        if self.use_redis:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting Redis...")
            redis.disconnect()
        if self.rabbitmq_config:
            logger.info(f"[{self.__class__.__name__}] ⏳ Disconnecting RabbitMQ...")

    def __start_fastapi(self):
        port = self.get_port()
        host = self.get_host()
        logger.info(
            f"[{self.service_name}] Starting FastAPI server at http://{host}:{port}"
        )
        server = uvicorn.Server(uvicorn.Config(self.fastapi_app, host=host, port=port))
        self.shared_event_loop.run_until_complete(server.serve())


# --------------------------------------------------
# HTTP Client utilities for dynamic service client abstraction
# --------------------------------------------------
AS = TypeVar("AS", bound=AppService)


class HttpClient:
    """
    A simple HTTP client abstraction for making remote calls to a FastAPI service.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        import httpx

        self.client = httpx.Client()

    def call_method(self, method_name: str, **kwargs) -> Any:
        url = f"{self.base_url}/{method_name}"
        response = self.client.post(url, json=kwargs)
        response.raise_for_status()
        return response.json()


def close_service_client(client: Any) -> None:
    if hasattr(client, "client") and hasattr(client.client, "close"):
        client.client.close()
    else:
        raise RuntimeError(f"Client {client.__class__} is not a valid HTTP client.")


def get_service_client(service_type: Type[AS]) -> AS:
    service_name = service_type.service_name

    class DynamicClient:
        def __init__(self):
            host = os.environ.get(f"{service_name.upper()}_HOST", api_host)
            port = service_type.get_port()
            self.base_url = f"http://{host}:{port}"
            self.http_client = HttpClient(self.base_url)

        def __getattr__(self, name: str) -> Callable[..., Any]:
            # Try to get the original function from the service type.
            orig_func = getattr(service_type, name, None)
            expected_return = None
            if orig_func is not None:
                sig = inspect.signature(orig_func)
                ret_ann = sig.return_annotation
                if (
                    ret_ann != inspect.Signature.empty
                    and isinstance(ret_ann, type)
                    and issubclass(ret_ann, BaseModel)
                ):
                    expected_return = ret_ann

            def method(**kwargs) -> Any:
                result = self.http_client.call_method(name, **kwargs)
                if expected_return is not None and isinstance(result, dict):
                    return expected_return.model_validate(result)
                return result

            return method

    return cast(AS, DynamicClient())
