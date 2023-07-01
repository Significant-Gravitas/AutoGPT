import pytest
from openai.error import APIError, RateLimitError, ServiceUnavailableError

from autogpt.llm.providers import openai


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
            num_retries=retry_count, backoff_base=0.001, warn_user=warn_user
        )
        def __call__(self):
            self.count += 1
            if self.count <= error_count:
                raise error_instance
            return self.count

    return RaisesError()


def test_retry_open_api_no_error(capsys):
    """Tests the retry functionality with no errors expected"""

    @openai.retry_api()
    def f():
        return 1

    result = f()
    assert result == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


@pytest.mark.parametrize(
    "error_count, retry_count, failure",
    [(2, 10, False), (2, 2, False), (10, 2, True), (3, 2, True), (1, 0, True)],
    ids=["passing", "passing_edge", "failing", "failing_edge", "failing_no_retries"],
)
def test_retry_open_api_passing(capsys, error, error_count, retry_count, failure):
    """Tests the retry with simulated errors [RateLimitError, ServiceUnavailableError, APIError], but should ulimately pass"""
    call_count = min(error_count, retry_count) + 1

    raises = error_factory(error, error_count, retry_count)
    if failure:
        with pytest.raises(type(error)):
            raises()
    else:
        result = raises()
        assert result == call_count

    assert raises.count == call_count

    output = capsys.readouterr()

    if error_count and retry_count:
        if type(error) == RateLimitError:
            assert "Reached rate limit, passing..." in output.out
            assert "Please double check" in output.out
        if type(error) == ServiceUnavailableError:
            assert (
                "The OpenAI API engine is currently overloaded, passing..."
                in output.out
            )
            assert "Please double check" in output.out
        if type(error) == APIError:
            assert "API Bad gateway" in output.out
    else:
        assert output.out == ""


def test_retry_open_api_rate_limit_no_warn(capsys):
    """Tests the retry logic with a rate limit error"""
    error_count = 2
    retry_count = 10

    raises = error_factory(RateLimitError, error_count, retry_count, warn_user=False)
    result = raises()
    call_count = min(error_count, retry_count) + 1
    assert result == call_count
    assert raises.count == call_count

    output = capsys.readouterr()

    assert "Reached rate limit, passing..." in output.out
    assert "Please double check" not in output.out


def test_retry_open_api_service_unavairable_no_warn(capsys):
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

    output = capsys.readouterr()

    assert "The OpenAI API engine is currently overloaded, passing..." in output.out
    assert "Please double check" not in output.out


def test_retry_openapi_other_api_error(capsys):
    """Tests the Retry logic with a non rate limit error such as HTTP500"""
    error_count = 2
    retry_count = 10

    raises = error_factory(APIError("Error", http_status=500), error_count, retry_count)

    with pytest.raises(APIError):
        raises()
    call_count = 1
    assert raises.count == call_count

    output = capsys.readouterr()
    assert output.out == ""
