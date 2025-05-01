import pytest

from backend.util.service import (
    AppService,
    AppServiceClient,
    endpoint_to_async,
    expose,
    get_service_client,
)

TEST_SERVICE_PORT = 8765


class ServiceTest(AppService):
    def __init__(self):
        super().__init__()

    def cleanup(self):
        pass

    @classmethod
    def get_port(cls) -> int:
        return TEST_SERVICE_PORT

    @expose
    def add(self, a: int, b: int) -> int:
        return a + b

    @expose
    def subtract(self, a: int, b: int) -> int:
        return a - b

    @expose
    def fun_with_async(self, a: int, b: int) -> int:
        async def add_async(a: int, b: int) -> int:
            return a + b

        return self.run_and_wait(add_async(a, b))


class ServiceTestClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return ServiceTest

    add = ServiceTest.add
    subtract = ServiceTest.subtract
    fun_with_async = ServiceTest.fun_with_async

    add_async = endpoint_to_async(ServiceTest.add)
    subtract_async = endpoint_to_async(ServiceTest.subtract)


@pytest.mark.asyncio(loop_scope="session")
async def test_service_creation(server):
    with ServiceTest():
        client = get_service_client(ServiceTestClient)
        assert client.add(5, 3) == 8
        assert client.subtract(10, 4) == 6
        assert client.fun_with_async(5, 3) == 8
        assert await client.add_async(5, 3) == 8
        assert await client.subtract_async(10, 4) == 6
