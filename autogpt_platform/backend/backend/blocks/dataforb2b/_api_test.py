from backend.blocks.dataforb2b._api import api_error_message
from backend.util.request import HTTPClientError


def test_api_error_message_does_not_expose_response_body():
    error = HTTPClientError(
        "HTTP 400 Error: Bad Request, Body: secret upstream response", 400
    )

    message = api_error_message(error, "people search")

    assert message == "DataForB2B people search failed with HTTP status 400."
    assert "secret upstream response" not in message
