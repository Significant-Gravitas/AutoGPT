"""Local servers for the ``Requests`` tests, and the clients that reach them."""

from unittest.mock import patch

import pytest
from aiohttp import web

from backend.util import request as request_module
from backend.util.request import Requests

# Every request in these tests talks to a local server, so its attempts take
# milliseconds; the bound only trips when the retries never stop.
ENDS_WITHIN_SECONDS = 10

# The two clients that retry a throttle status: one raising its errors, with a
# cap, and one checking statuses itself, on the default.
RETRYING_CLIENTS = pytest.mark.parametrize(
    "client_kwargs",
    [{"retry_max_attempts": 3}, {"raise_for_status": False}],
    ids=["raise_for_status-with-a-cap", "status-checked-by-the-caller"],
)


async def serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


def local_requests(**kwargs) -> Requests:
    return Requests(trusted_origins=["127.0.0.1"], **kwargs)


# Far more waits than any cap in these tests asks for.
MOST_RETRY_WAITS = 10_000


async def retry_waits(client: Requests) -> list[float]:
    """The waits between the attempts of one of *client*'s requests that meets
    a throttle status every time: what its retries would sleep, noted instead.

    Fails at ``MOST_RETRY_WAITS``: noting a wait never yields to the event
    loop, so a budget that never runs out would otherwise hang the test."""
    waits: list[float] = []

    async def note(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) >= MOST_RETRY_WAITS:
            raise AssertionError(
                f"The retry budget never ran out: {len(waits)} waits, so it "
                "is unbounded."
            )

    with patch.object(request_module, "sleep", note):
        budget = client._retry_budget()
        while await budget.send_again():
            pass
    return waits


def scripted(*statuses: int) -> tuple[web.Application, list[int]]:
    """An app answering *statuses* in turn, repeating the last for good, and
    the statuses it has served. Each body names its attempt."""
    served: list[int] = []

    async def answer(request: web.Request) -> web.Response:
        status = statuses[min(len(served), len(statuses) - 1)]
        served.append(status)
        return web.Response(status=status, text=f"attempt {len(served)}")

    app = web.Application()
    app.router.add_route("*", "/", answer)
    return app, served
