import time

from autogpt_server.util.service import (
    AppService,
    PyroNameServer,
    expose,
    get_service_client,
)


class TestService(AppService):

    def run_service(self):
        super().run_service()

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


def test_service_creation():
    with PyroNameServer():
        time.sleep(0.5)
        with TestService():
            client = get_service_client(TestService)
            assert client.add(5, 3) == 8
            assert client.subtract(10, 4) == 6
            assert client.fun_with_async(5, 3) == 8
