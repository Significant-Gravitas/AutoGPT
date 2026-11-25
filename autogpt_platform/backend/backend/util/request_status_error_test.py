import io
import logging
import traceback

import pytest
from aiohttp import web

from backend.util.request import HTTPClientError, HTTPServerError
from backend.util.request_test_helpers import local_requests, serve

KEY = "not-a-real-key-7731"
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "status, error_type", [(404, HTTPClientError), (500, HTTPServerError)]
)
async def test_a_status_error_and_its_log_carry_nothing_of_the_request_url(
    status: int, error_type: type, service_output: io.StringIO
) -> None:
    async def answer(_: web.Request) -> web.Response:
        return web.Response(status=status, text="refused")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", answer)
    runner, base = await serve(app)
    try:
        with pytest.raises((HTTPClientError, HTTPServerError)) as raised:
            await local_requests().get(f"{base}/v1/{KEY}/items?key={KEY}")
    finally:
        await runner.cleanup()

    error = raised.value
    assert type(error) is error_type
    assert error.status_code == status
    assert error.__cause__ is None and error.__context__ is None
    assert KEY not in "".join(traceback.format_exception(error))
    # The local server's own access log quotes the URL, so only the record counts.
    start = len(service_output.getvalue())
    logger.error("The request failed", exc_info=error)
    record = service_output.getvalue()[start:]
    assert f"HTTP {status} Error" in record
    assert KEY not in record
