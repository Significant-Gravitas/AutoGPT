import logging

import pytest
from openai.error import APIError, RateLimitError, ServiceUnavailableError

from autogpt.llm.providers import openai
from autogpt.logs.config import USER_FRIENDLY_OUTPUT_LOGGER


@pytest.fixture(params=[RateLimitError, ServiceUnavailableError, APIError])
def error(request):
    if request.param == APIError:
        return request.param("Error", http_status=502)
    else:
        return request.param("Error")


def error_factory(error_instance, error_count, retry_count, warn_user=True):
    """Creates errors"""

    class RaisesError:
        def __init__(self):
            self.count = 0

        @openai.retry_api(
            max_retries=retry_count, backoff_base=0.001, warn_user=warn_user
        )
        def __call__(self):
            self.count += 1
            if self.count <= error_count:
                raise error_instance
            return self.count

    return RaisesError()


def test_retry_open_api_no_error(caplog: pytest.LogCaptureFixture):
    """Tests the retry functionality with no errors expected"""

    @openai.retry_api()
    def f():
        return 1

    result = f()
    assert result == 1

    output = caplog.text
    assert output == ""
    assert output == ""


@pytest.mark.parametrize(
    "error_count, retry_count, failure",
    [(2, 10, False), (2, 2, False), (10, 2, True), (3, 2, True), (1, 0, True)],
    ids=["passing", "passing_edge", "failing", "failing_edge", "failing_no_retries"],
)
def test_retry_open_api_passing(
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    error_count: int,
    retry_count: int,
    failure: bool,
):
    """Tests the retry with simulated errors [RateLimitError, ServiceUnavailableError, APIError], but should ulimately pass"""

    # Add capture handler to non-propagating logger
    logging.getLogger(USER_FRIENDLY_OUTPUT_LOGGER).addHandler(caplog.handler)

    call_count = min(error_count, retry_count) + 1

    raises = error_factory(error, error_count, retry_count)
    if failure:
        with pytest.raises(type(error)):
            raises()
    else:
        result = raises()
        assert result == call_count

    assert raises.count == call_count

    output = caplog.text

    if error_count and retry_count:
        if type(error) == RateLimitError:
            assert "Reached rate limit" in output
            assert "Please double check" in output
        if type(error) == ServiceUnavailableError:
            assert "The OpenAI API engine is currently overloaded" in output
            assert "Please double check" in output
    else:
        assert output == ""


def test_retry_open_api_rate_limit_no_warn(caplog: pytest.LogCaptureFixture):
    """Tests the retry logic with a rate limit error"""
    error_count = 2
    retry_count = 10

    raises = error_factory(RateLimitError, error_count, retry_count, warn_user=False)
    result = raises()
    call_count = min(error_count, retry_count) + 1
    assert result == call_count
    assert raises.count == call_count

    output = caplog.text

    assert "Reached rate limit" in output
    assert "Please double check" not in output


def test_retry_open_api_service_unavairable_no_warn(caplog: pytest.LogCaptureFixture):
    """Tests the retry logic with a service unavairable error"""
    error_count = 2
    retry_count = 10

    raises = error_factory(
        ServiceUnavailableError, error_count, retry_count, warn_user=False
    )
    result = raises()
    call_count = min(error_count, retry_count) + 1
    assert result == call_count
    assert raises.count == call_count

    output = caplog.text

    assert "The OpenAI API engine is currently overloaded" in output
    assert "Please double check" not in output


def test_retry_openapi_other_api_error(caplog: pytest.LogCaptureFixture):
    """Tests the Retry logic with a non rate limit error such as HTTP500"""
    error_count = 2
    retry_count = 10

    raises = error_factory(APIError("Error", http_status=500), error_count, retry_count)

    with pytest.raises(APIError):
        raises()
    call_count = 1
    assert raises.count == call_count

    output = caplog.text
    assert output == ""
