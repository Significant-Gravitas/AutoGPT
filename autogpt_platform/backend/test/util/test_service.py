import pytest

from backend.util.service import AppService, expose, get_service_client

TEST_SERVICE_PORT = 8765


class ServiceTest(AppService):
    def __init__(self):
        super().__init__()

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


@pytest.mark.asyncio(scope="session")
async def test_service_creation(server):
    with ServiceTest():
        client = get_service_client(ServiceTest)
        assert client.add(5, 3) == 8
        assert client.subtract(10, 4) == 6
        assert client.fun_with_async(5, 3) == 8
