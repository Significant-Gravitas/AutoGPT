import time

from autogpt_server.util.service import (
    AppService,
    PyroNameServer,
    expose,
    get_service_client,
)


class TestService(AppService):

    def run(self):
        while True:
            pass  # does nothing

    @expose
    def add(self, a: int, b: int) -> int:
        return a + b

    @expose
    def subtract(self, a: int, b: int) -> int:
        return a - b


def test_service_creation():
    print("Starting TestService...")
    with PyroNameServer():
        time.sleep(0.3)
        with TestService():
            client = get_service_client(TestService)
            assert client.add(5, 3) == 8
            assert client.subtract(10, 4) == 6
